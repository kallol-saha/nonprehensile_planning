"""Constraint Tree node for CBS planning.

Each CTNode stores:
  - A batch of B trajectories per piece (world SE(2), metres).
    The batch is produced by the low-level diffusion planner and allows CBS to
    pick the trajectory with the fewest conflicts as the *representative*.
  - The accumulated constraint set per piece (list of SphereConstraints).
  - The cached number of conflicts in the representative trajectories, used as
    the CT heap key so that the most-promising (least-conflict) node is always
    expanded first.

Design notes
------------
* Trajectories are stored in world SE(2) metres (N, B, T, 3) so that the
  conflict detector (which works in metres) can operate without extra conversion.
* Constraints are stored in normalised coordinates (as SphereConstraint holds
  them) so they can be passed directly to the guided sampler.
* A CTNode is immutable after construction; CBS creates new child nodes by
  copying a parent and adding one constraint.
* The node is comparable via __lt__ so it can be placed in a heapq.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from visplan.planning.constraints import SphereConstraint


@dataclass
class CTNode:
    """A single node in the CBS constraint tree.

    Attributes
    ----------
    constraints : Dict[int, List[SphereConstraint]]
        Per-piece constraint lists (normalised coords).  piece_idx → constraints.
    trajectories_world : np.ndarray | None
        (N, T, 3) representative SE(2) trajectories in world metres for all
        pieces.  ``None`` until the node has been expanded by the planner.
    trajectory_batches_world : Dict[int, np.ndarray]
        piece_idx → (B, T, 3) batch of candidate trajectories in world metres.
        CBS picks the representative from this batch (fewest conflicts).
    num_conflicts : int
        Cached conflict count for the representative trajectories.  Used as the
        heap key.  Set by ``update_representative()``.
    node_id : int
        Unique identifier for debugging / logging.
    parent_id : int | None
        ID of the parent node (None for root).
    """

    constraints: Dict[int, List[SphereConstraint]]
    trajectories_world: Optional[np.ndarray]  # (N, T, 3)
    trajectory_batches_world: Dict[int, np.ndarray]  # piece_idx → (B, T, 3)
    num_conflicts: int
    node_id: int
    parent_id: Optional[int]

    # ------------------------------------------------------------------
    # Heap ordering: CBS always expands the node with the fewest conflicts
    # ------------------------------------------------------------------

    def __lt__(self, other: "CTNode") -> bool:
        return self.num_conflicts < other.num_conflicts

    def __le__(self, other: "CTNode") -> bool:
        return self.num_conflicts <= other.num_conflicts

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def make_root(cls, num_pieces: int, node_id: int = 0) -> "CTNode":
        """Create an empty root node with no constraints."""
        return cls(
            constraints={i: [] for i in range(num_pieces)},
            trajectories_world=None,
            trajectory_batches_world={},
            num_conflicts=0,
            node_id=node_id,
            parent_id=None,
        )

    def child_with_constraint(
        self,
        piece_idx: int,
        new_constraint: SphereConstraint,
        node_id: int,
    ) -> "CTNode":
        """Return a deep-copied child node with one additional constraint.

        All trajectory batches are copied from the parent so that only the
        affected piece needs to be replanned.  The ``trajectories_world`` and
        ``num_conflicts`` fields are set to ``None`` / ``-1`` to signal that
        the child has not yet been expanded.

        Args:
            piece_idx:      The piece that receives the new constraint.
            new_constraint: SphereConstraint to add.
            node_id:        Unique ID for the new child node.

        Returns:
            New CTNode (child of self).
        """
        # Deep-copy constraints and batches so mutations are independent
        new_constraints: Dict[int, List[SphereConstraint]] = {
            k: list(v) for k, v in self.constraints.items()
        }
        new_constraints[piece_idx] = new_constraints[piece_idx] + [new_constraint]

        # Copy trajectory batches; the affected piece will be replanned
        new_batches: Dict[int, np.ndarray] = dict(self.trajectory_batches_world)

        # Copy representative trajectories; the affected piece will be updated
        new_traj = (
            self.trajectories_world.copy()
            if self.trajectories_world is not None
            else None
        )

        return CTNode(
            constraints=new_constraints,
            trajectories_world=new_traj,
            trajectory_batches_world=new_batches,
            num_conflicts=-1,          # stale until update_representative()
            node_id=node_id,
            parent_id=self.node_id,
        )

    # ------------------------------------------------------------------
    # Representative selection
    # ------------------------------------------------------------------

    def update_representative(
        self,
        piece_idx: int,
        batch_world: np.ndarray,
        outlines: List[np.ndarray],
    ) -> None:
        """Select the trajectory from ``batch_world`` with the fewest conflicts
        against the current representative trajectories of all other pieces,
        and update ``self.trajectories_world`` and ``self.num_conflicts``.

        This implements the batch trajectory generation strategy from §A.1 of
        the MMD paper: CBS generates a batch of B trajectories per planning
        call and marks the one with the fewest collisions as the representative.

        Args:
            piece_idx:   The piece whose batch was just replanned.
            batch_world: (B, T, 3) candidate trajectories in world metres.
            outlines:    List of N (V_i, 2) local-frame polygon outlines.
        """
        from visplan.planning.conflict_detector import count_conflicts

        self.trajectory_batches_world[piece_idx] = batch_world

        # Build a full (N, T, 3) array using the best-so-far representative for
        # all *other* pieces (if available) so we can score each candidate.
        N = len(outlines)
        B, T, _ = batch_world.shape

        # Assemble the current representatives for all pieces except piece_idx
        other_trajs: Dict[int, np.ndarray] = {}
        for k in range(N):
            if k == piece_idx:
                continue
            if k in self.trajectory_batches_world:
                # Use the existing representative for piece k (first element of
                # the stored batch, or the representative slice from
                # trajectories_world if available).
                if self.trajectories_world is not None:
                    other_trajs[k] = self.trajectories_world[k]
                else:
                    # Fall back to first sample in the batch
                    other_trajs[k] = self.trajectory_batches_world[k][0]
            # If a piece has no trajectory yet (very first planning round for
            # some pieces), it is omitted from conflict counting.

        best_idx = 0
        best_conflicts = int(1e9)

        for b in range(B):
            candidate = batch_world[b]  # (T, 3)

            # Assemble a temporary (N, T, 3) by inserting this candidate
            tmp_pieces = []
            tmp_indices = []
            for k in range(N):
                if k == piece_idx:
                    tmp_pieces.append(candidate)
                    tmp_indices.append(k)
                elif k in other_trajs:
                    tmp_pieces.append(other_trajs[k])
                    tmp_indices.append(k)

            if len(tmp_pieces) < 2:
                # Cannot check conflicts with fewer than 2 pieces; pick first
                best_idx = 0
                break

            tmp_array = np.stack(tmp_pieces, axis=0)  # (M, T, 3)
            tmp_outlines = [outlines[k] for k in tmp_indices]

            n_conflicts = count_conflicts(tmp_array, tmp_outlines)
            if n_conflicts < best_conflicts:
                best_conflicts = n_conflicts
                best_idx = b

        # Install the chosen representative
        if self.trajectories_world is None:
            # First time: allocate
            # Use placeholder zeros for pieces not yet planned
            placeholder = np.zeros((T, 3), dtype=np.float32)
            traj = np.stack(
                [
                    batch_world[best_idx] if k == piece_idx
                    else other_trajs.get(k, placeholder)
                    for k in range(N)
                ],
                axis=0,
            )
            self.trajectories_world = traj
        else:
            self.trajectories_world[piece_idx] = batch_world[best_idx]

        # Recompute total conflict count with the new representative set
        self.num_conflicts = count_conflicts(
            self.trajectories_world, outlines
        )
