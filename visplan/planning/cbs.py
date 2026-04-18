"""CBS and MMD-PP multi-piece trajectory planning algorithms.

This module implements two CBS-inspired planning algorithms adapted from the
MMD paper (Shaoul et al., ICLR 2025) for coordinating Voronoi puzzle pieces:

CBS (Conflict-Based Search)
---------------------------
A two-level search algorithm:
  High level: Constraint Tree (CT) — a priority queue of CTNodes ordered by
              number of pairwise conflicts.  The node with the fewest conflicts
              is always expanded first.
  Low level:  LowLevelPlanner — a guided diffusion model that samples a batch
              of B trajectories for one piece under a set of sphere constraints,
              returning the representative (fewest conflicts).

Algorithm sketch (Algorithm 1 of MMD paper):
  1. Create root node; plan all pieces independently (no constraints).
  2. Pop node N with fewest conflicts from the CT heap.
  3. If N has zero conflicts, return N.trajectories_world.
  4. Find one conflict ⟨piece_i, piece_j, timestep t⟩ in N.
  5. Create two child nodes:
       child_i: copy of N + SphereConstraint on piece_i at time t
       child_j: copy of N + SphereConstraint on piece_j at time t
  6. Replan only the constrained piece in each child (other pieces unchanged).
  7. Update representatives and conflict counts; push children to heap.
  8. If max_nodes exceeded, return the best node seen so far.

MMD-PP (Prioritised Planning)
------------------------------
Simpler: plan pieces in a fixed order.  Each piece plans under constraints
derived from all previously planned pieces' trajectories.  Single pass, no CT.
No completeness guarantee but very fast.

Termination
-----------
CBS terminates when either a conflict-free solution is found or ``max_nodes``
CT nodes have been expanded (whichever comes first).  The best solution seen
(minimum conflicts) is always returned so the caller always gets *something*.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from visplan.planning.conflict_detector import (
    detect_conflicts,
    conflict_to_sphere_constraint,
    count_conflicts,
)
from visplan.planning.constraints import SphereConstraint
from visplan.planning.ct_node import CTNode
from visplan.planning.low_level_planner import LowLevelPlannerProtocol

# Backward-compatibility alias so external code importing LowLevelPlanner still works
from visplan.planning.low_level_planner import LowLevelPlanner  # noqa: F401

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Return type
# ---------------------------------------------------------------------------

@dataclass
class PlanningResult:
    """Output of a CBS or MMD-PP planning call.

    Attributes
    ----------
    trajectories_world : np.ndarray
        (N, T, 3) SE(2) trajectories in world metres [x, y, theta].
    num_conflicts : int
        Number of pairwise piece-timestep collisions in the returned plan.
        Zero means collision-free.
    success : bool
        True iff num_conflicts == 0.
    nodes_expanded : int
        Number of CT nodes expanded (CBS only; 0 for MMD-PP).
    planning_time_s : float
        Wall-clock seconds spent planning.
    """

    trajectories_world: np.ndarray
    num_conflicts: int
    success: bool
    nodes_expanded: int
    planning_time_s: float


# ---------------------------------------------------------------------------
#  Scene batch builder (helper used by both algorithms)
# ---------------------------------------------------------------------------

def _build_scene_batch(
    episode_data: dict,
    piece_idx: int,
    device,
) -> Dict:
    """Build the planner input dict for a single piece.

    Returns a dict usable by both DiffusionLowLevel (needs image tensors) and
    RRTLowLevel (needs start_pose, goal_pose, outline, piece_idx).

    Args:
        episode_data: Raw dict loaded from a .npz episode file (or equivalent
                      dict with the same keys).
        piece_idx:    Which piece to build the batch for.
        device:       Torch device (or "cpu" string for RRT).

    Returns:
        Dict with keys:
          start_image  (1,3,H,W) float32 tensor on device
          goal_image   (1,3,H,W) float32 tensor on device
          piece_mask   (1,1,H,W) float32 tensor on device
          start_pose   (3,) float32 numpy [x, y, theta] world metres
          goal_pose    (3,) float32 numpy [x, y, theta] world metres
          piece_idx    int
          outline      (V,2) float32 numpy local-frame polygon vertices
    """
    import torch

    # Images are optional — absent in synthetic episodes (no data_dir).
    # RRT planners never read image tensors; only diffusion does.
    if "start_image" in episode_data:
        start_img = episode_data["start_image"].astype(np.float32) / 255.0
        start_img = np.transpose(start_img, (2, 0, 1))   # (3, H, W)
        goal_img = episode_data["goal_image"].astype(np.float32) / 255.0
        goal_img = np.transpose(goal_img, (2, 0, 1))
        mask = episode_data["piece_masks"][piece_idx].astype(np.float32)[None]
        img_tensors = {
            "start_image": torch.from_numpy(start_img).unsqueeze(0).to(device),
            "goal_image":  torch.from_numpy(goal_img).unsqueeze(0).to(device),
            "piece_mask":  torch.from_numpy(mask).unsqueeze(0).to(device),
        }
    else:
        img_tensors = {"start_image": None, "goal_image": None, "piece_mask": None}

    return {
        **img_tensors,
        "start_pose": episode_data["start_poses"][piece_idx].astype(np.float32),
        "goal_pose":  episode_data["goal_poses"][piece_idx].astype(np.float32),
        "piece_idx":  piece_idx,
        "outline":    episode_data[f"outline_{piece_idx}"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
#  CBS
# ---------------------------------------------------------------------------

def run_cbs(
    episode_data: dict,
    outlines: List[np.ndarray],
    planner: LowLevelPlannerProtocol,
    visible_range: float,
    constraint_margin: float = 1.2,
    constraint_epsilon: float = 1.2,
    max_nodes: int = 1000,
) -> PlanningResult:
    """Run Conflict-Based Search to find collision-free reassembly trajectories.

    Args:
        episode_data:       Dict with keys from a .npz episode file.  At minimum:
                            ``start_image``, ``goal_image``, ``piece_masks``,
                            ``num_pieces``.
        outlines:           List of N (V_i, 2) local-frame polygon outlines (metres).
        planner:            LowLevelPlanner wrapping a trained diffusion model.
        visible_range:      Camera half-extent in metres.
        constraint_margin:  Multiplier for bounding radius when sizing constraints.
        constraint_epsilon: Epsilon (padding) inside SphereConstraint cost.
        max_nodes:          Maximum CT nodes to expand before giving up and
                            returning the best solution found.

    Returns:
        PlanningResult with collision-free or best-effort trajectories.
    """
    t_start = time.time()

    N = int(episode_data["num_pieces"])
    node_counter = [0]  # mutable counter shared across helpers

    def _next_id() -> int:
        node_counter[0] += 1
        return node_counter[0]

    # ------------------------------------------------------------------
    # 1. Root node: plan all pieces without any constraints
    # ------------------------------------------------------------------
    root = CTNode.make_root(num_pieces=N, node_id=0)

    logger.info("CBS root: planning %d pieces (no constraints)", N)
    for i in range(N):
        scene_batch = _build_scene_batch(episode_data, i, planner.device)
        batch_world = planner.plan_piece(scene_batch, constraints=[])
        # batch_world: (B, T, 3)
        root.update_representative(i, batch_world, outlines)
        logger.debug("  Root piece %d: batch_size=%d", i, batch_world.shape[0])

    logger.info("Root conflicts: %d", root.num_conflicts)

    # Best node seen so far (in case we exhaust max_nodes without success)
    best_node = root

    # ------------------------------------------------------------------
    # 2. CBS main loop
    # ------------------------------------------------------------------
    # Heap: list of (num_conflicts, CTNode).  Python heapq is a min-heap,
    # and CTNode implements __lt__ by num_conflicts, so we use (count, node).
    heap: List[Tuple[int, CTNode]] = [(root.num_conflicts, root)]
    nodes_expanded = 0

    while heap and nodes_expanded < max_nodes:
        _, node = heapq.heappop(heap)
        nodes_expanded += 1

        logger.debug(
            "CBS expand node %d (conflicts=%d, nodes_expanded=%d)",
            node.node_id, node.num_conflicts, nodes_expanded,
        )

        # Track best node for early-exit / max-nodes fallback
        if node.num_conflicts < best_node.num_conflicts:
            best_node = node

        # Success: no conflicts
        if node.num_conflicts == 0:
            logger.info(
                "CBS: found collision-free solution after %d node expansions",
                nodes_expanded,
            )
            return PlanningResult(
                trajectories_world=node.trajectories_world,
                num_conflicts=0,
                success=True,
                nodes_expanded=nodes_expanded,
                planning_time_s=time.time() - t_start,
            )

        # Find one conflict to branch on
        conflicts = detect_conflicts(
            node.trajectories_world,
            outlines,
            visible_range=visible_range,
            constraint_margin=constraint_margin,
            max_conflicts=1,
        )
        if not conflicts:
            # count_conflicts found collisions but detect_conflicts found none —
            # can happen if count_conflicts has a bug; treat as conflict-free.
            logger.warning(
                "Node %d: count_conflicts=%d but detect_conflicts found none; "
                "treating as conflict-free.",
                node.node_id, node.num_conflicts,
            )
            return PlanningResult(
                trajectories_world=node.trajectories_world,
                num_conflicts=0,
                success=True,
                nodes_expanded=nodes_expanded,
                planning_time_s=time.time() - t_start,
            )

        conflict = conflicts[0]
        logger.debug(
            "  Conflict: pieces (%d, %d) at t=%d, pos=(%.3f, %.3f)",
            conflict.piece_i, conflict.piece_j, conflict.timestep,
            conflict.world_pos[0], conflict.world_pos[1],
        )

        # ------------------------------------------------------------------
        # 3. Branch: constrain piece_i in child_i; constrain piece_j in child_j
        # ------------------------------------------------------------------
        for constrained_piece in (conflict.piece_i, conflict.piece_j):
            new_constraint = conflict_to_sphere_constraint(
                conflict,
                piece_idx=constrained_piece,
                visible_range=visible_range,
                epsilon=constraint_epsilon,
            )

            child = node.child_with_constraint(
                piece_idx=constrained_piece,
                new_constraint=new_constraint,
                node_id=_next_id(),
            )

            # Replan only the constrained piece with the new constraint set
            scene_batch = _build_scene_batch(
                episode_data, constrained_piece, planner.device
            )
            new_batch = planner.plan_piece(
                scene_batch,
                constraints=child.constraints[constrained_piece],
            )
            child.update_representative(constrained_piece, new_batch, outlines)

            logger.debug(
                "  Child %d (constrain piece %d): conflicts=%d",
                child.node_id, constrained_piece, child.num_conflicts,
            )

            heapq.heappush(heap, (child.num_conflicts, child))

    # ------------------------------------------------------------------
    # 4. Exhausted budget — return best solution found
    # ------------------------------------------------------------------
    logger.warning(
        "CBS: max_nodes=%d reached. Best solution has %d conflicts.",
        max_nodes, best_node.num_conflicts,
    )
    return PlanningResult(
        trajectories_world=best_node.trajectories_world,
        num_conflicts=best_node.num_conflicts,
        success=best_node.num_conflicts == 0,
        nodes_expanded=nodes_expanded,
        planning_time_s=time.time() - t_start,
    )


# ---------------------------------------------------------------------------
#  Independent planning (no coordination between pieces)
# ---------------------------------------------------------------------------

def run_independent(
    episode_data: dict,
    outlines: List[np.ndarray],
    planner: LowLevelPlannerProtocol,
    visible_range: float,
) -> PlanningResult:
    """Plan each piece independently with empty constraints (no CBS, no PP).

    This is the simplest baseline: each piece plans without any knowledge of
    the other pieces.  The first trajectory from the returned batch (index 0)
    is selected as the representative for each piece.

    Args:
        episode_data: Dict with keys from a .npz episode file.
        outlines:     List of N (V_i, 2) local-frame polygon outlines.
        planner:      Low-level planner (diffusion or RRT).
        visible_range: Camera half-extent in metres (unused here but kept for
                       API consistency with run_cbs / run_pp).

    Returns:
        PlanningResult with nodes_expanded=0 (no search tree).
    """
    from visplan.planning.conflict_detector import count_conflicts

    t_start = time.time()
    N = int(episode_data["num_pieces"])
    planned_trajs: List[np.ndarray] = []

    for i in range(N):
        scene_batch = _build_scene_batch(episode_data, i, planner.device)
        batch_world = planner.plan_piece(scene_batch, constraints=[])
        # batch_world: (B, T, 3); take first sample as representative
        planned_trajs.append(batch_world[0])
        logger.debug("Independent piece %d planned.", i)

    trajectories_world = np.stack(planned_trajs, axis=0)  # (N, T, 3)
    num_conflicts = count_conflicts(trajectories_world, outlines)

    logger.info("Independent finished. Conflicts: %d", num_conflicts)

    return PlanningResult(
        trajectories_world=trajectories_world,
        num_conflicts=num_conflicts,
        success=num_conflicts == 0,
        nodes_expanded=0,
        planning_time_s=time.time() - t_start,
    )


# ---------------------------------------------------------------------------
#  MMD-PP (Prioritised Planning)
# ---------------------------------------------------------------------------

def run_pp(
    episode_data: dict,
    outlines: List[np.ndarray],
    planner: LowLevelPlannerProtocol,
    visible_range: float,
    constraint_margin: float = 1.2,
    constraint_epsilon: float = 1.2,
) -> PlanningResult:
    """Run MMD-Prioritised Planning (single-pass, no constraint tree).

    Pieces are planned in order 0..N-1.  Each piece is constrained to avoid
    all previously planned pieces' representative trajectories.  Soft sphere
    constraints are placed at every timestep where a future piece's centroid
    would overlap a previous piece.

    This is cheaper than CBS (one diffusion call per piece) but provides no
    completeness guarantee: the soft constraints may be violated if the
    guidance weight is too low.

    Args:
        episode_data, outlines, planner, visible_range,
        constraint_margin, constraint_epsilon:
            Same semantics as run_cbs.

    Returns:
        PlanningResult.
    """
    t_start = time.time()

    N = int(episode_data["num_pieces"])
    T = planner.traj_len

    # Representative trajectories in world coords for each planned piece
    planned_trajs: List[np.ndarray] = []   # each: (T, 3)
    planned_indices: List[int] = []

    for i in range(N):
        # Build constraints: avoid every previously planned piece at every timestep
        constraints: List[SphereConstraint] = []

        for j_idx, j in enumerate(planned_indices):
            prev_traj = planned_trajs[j_idx]  # (T, 3)
            r_world = constraint_margin * _bounding_radius(outlines[i])

            for t in range(T):
                cx_w, cy_w = float(prev_traj[t, 0]), float(prev_traj[t, 1])
                cx_norm = cx_w / visible_range
                cy_norm = cy_w / visible_range
                r_norm = r_world / visible_range

                constraints.append(SphereConstraint(
                    piece_idx=i,
                    center_norm=(cx_norm, cy_norm),
                    radius_norm=r_norm,
                    t_start=t,
                    t_end=t,
                    epsilon=constraint_epsilon,
                ))

        scene_batch = _build_scene_batch(episode_data, i, planner.device)
        batch_world = planner.plan_piece(scene_batch, constraints=constraints)
        # batch_world: (B, T, 3)

        # Pick the representative: trajectory with fewest conflicts against
        # all already-planned pieces
        best_rep = _pick_representative_pp(batch_world, planned_trajs,
                                           planned_indices, outlines)

        planned_trajs.append(best_rep)
        planned_indices.append(i)

        logger.debug(
            "PP piece %d planned. Constraints from %d prior pieces.",
            i, len(planned_indices) - 1,
        )

    # Assemble final trajectory array
    trajectories_world = np.stack(planned_trajs, axis=0)  # (N, T, 3)
    num_conflicts = count_conflicts(trajectories_world, outlines)

    logger.info("PP finished. Conflicts: %d", num_conflicts)

    return PlanningResult(
        trajectories_world=trajectories_world,
        num_conflicts=num_conflicts,
        success=num_conflicts == 0,
        nodes_expanded=0,
        planning_time_s=time.time() - t_start,
    )


# ---------------------------------------------------------------------------
#  MMD-PP internal helpers
# ---------------------------------------------------------------------------

def _bounding_radius(outline: np.ndarray) -> float:
    """Circumscribed radius of a local-frame polygon outline (metres)."""
    return float(np.max(np.linalg.norm(outline, axis=1)))


def _pick_representative_pp(
    batch_world: np.ndarray,
    planned_trajs: List[np.ndarray],
    planned_indices: List[int],
    outlines: List[np.ndarray],
) -> np.ndarray:
    """Select the trajectory from ``batch_world`` with the fewest conflicts
    against all already-planned pieces.

    Args:
        batch_world:     (B, T, 3) candidate trajectories for the current piece.
        planned_trajs:   List of (T, 3) representatives for prior pieces.
        planned_indices: Original piece indices for each entry in planned_trajs.
        outlines:        All piece outlines.

    Returns:
        (T, 3) best representative trajectory.
    """
    if not planned_trajs:
        # No prior pieces → pick first sample
        return batch_world[0]

    B = batch_world.shape[0]
    best_idx = 0
    best_count = int(1e9)

    current_piece_idx = len(planned_indices)  # the piece currently being planned

    for b in range(B):
        candidate = batch_world[b]  # (T, 3)

        # Build a temporary (M+1, T, 3) for conflict checking
        tmp_list = list(planned_trajs) + [candidate]
        tmp_array = np.stack(tmp_list, axis=0)  # (M+1, T, 3)

        tmp_outlines = [outlines[k] for k in planned_indices] + [
            outlines[current_piece_idx]
        ]

        n_conflicts = count_conflicts(tmp_array, tmp_outlines)
        if n_conflicts < best_count:
            best_count = n_conflicts
            best_idx = b

    return batch_world[best_idx]
