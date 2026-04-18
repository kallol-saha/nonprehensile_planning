"""Space-time RRT-Connect low-level planner for CBS.

Implements bidirectional space-time RRT-Connect in SE(2) for a single puzzle
piece.  Hard sphere constraints (from CBS conflict resolution) are treated as
spatiotemporal obstacles that reject sampled nodes whose (xy, timestep) pair
falls inside any active exclusion disk.

This module exposes ``RRTLowLevel``, which satisfies the same
``LowLevelPlannerProtocol`` interface as ``DiffusionLowLevel`` so that CBS can
use either planner interchangeably.

Coordinate system
-----------------
All planning is done in **world metres** SE(2) = (x, y, theta).
SphereConstraints are stored in normalised coords (center_norm, radius_norm),
so they are converted to world metres internally by multiplying by visible_range.

Space-time dimension
--------------------
The RRT tree nodes carry an associated **trajectory timestep** t ∈ {0…T-1}.
A node at timestep t must not violate any constraint whose active window
[t_start, t_end] contains t.

The timestep is estimated from the cumulative arc-length of the path from the
root divided by the straight-line start→goal distance, then scaled to [0, T-1]:
  t_forward  = round( d(start, q_new) / d_sg * (T-1) )
  t_backward = T-1 - round( d(goal, q_new) / d_sg * (T-1) )

Note: this is an approximation because RRT paths are not straight lines, but
it provides a monotone time estimate that is consistent with the constraint
windows produced by CBS (which are per-timestep sphere constraints).

Distance metric
---------------
  d(q1, q2) = || Δxy ||₂  +  W_THETA * |Δθ|_wrapped
where W_THETA = 0.1 (angle contribution kept small relative to translation).

Collision model
---------------
Option A (centroid-in-disk): a node (q, t) violates constraint c if
  t ∈ [c.t_start, c.t_end]  AND
  ||q_xy − center_world||₂ < c.radius_norm * visible_range
The sphere radius already incorporates a 1.2× margin from CBS.

Algorithm
---------
Bidirectional RRT-Connect (Kuffner & LaValle 2000):
  1. Sample q_rand uniformly in the state space (with GOAL_BIAS toward goal).
  2. Extend tree A toward q_rand (advance by STEP_SIZE or until q_rand).
  3. If EXTEND didn't trap, connect tree B toward the newly added node
     (greedily advance until reached or trapped).
  4. If trees connect, extract path and return.
  5. Swap roles and repeat.

Budget: max_samples total tree nodes across both trees, OR max_time_s wall
clock seconds.  Fallback: straight-line path resampled to T waypoints.

Tree storage
------------
Each tree uses preallocated numpy arrays (per_tree = max_samples // 2 rows).
A node at row k has state (x,y,θ), parent index, and estimated timestep.

Output
------
``plan_piece`` returns a (1, T, 3) numpy float32 array (B=1).
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from visplan.planning.constraints import SphereConstraint


# ---------------------------------------------------------------------------
#  Constants
# ---------------------------------------------------------------------------

W_THETA: float = 0.1      # weight on angular distance in SE(2) metric
STEP_SIZE: float = 0.05   # max advance per RRT step (metres equivalent)
GOAL_BIAS: float = 0.10   # probability of sampling the goal state directly
REACH_TOL: float = 1e-9   # distance to declare trees connected

# State-space bounds (slightly larger than visible_range=0.6 to give margin)
XY_BOUND: float = 0.65
THETA_BOUND: float = np.pi


# ---------------------------------------------------------------------------
#  Geometry helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


def _se2_dist(q1: np.ndarray, q2: np.ndarray) -> float:
    """SE(2) distance: ||Δxy||₂ + W_THETA * |Δθ|_wrapped."""
    dxy = float(np.linalg.norm(q1[:2] - q2[:2]))
    dtheta = abs(_wrap_angle(float(q1[2] - q2[2])))
    return dxy + W_THETA * dtheta


def _se2_steer(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """Return a state advanced from q_from toward q_to by at most STEP_SIZE."""
    d = _se2_dist(q_from, q_to)
    if d < REACH_TOL:
        return q_to.copy()
    frac = min(1.0, STEP_SIZE / d)
    dtheta = _wrap_angle(float(q_to[2]) - float(q_from[2]))
    return np.array([
        q_from[0] + frac * (q_to[0] - q_from[0]),
        q_from[1] + frac * (q_to[1] - q_from[1]),
        _wrap_angle(float(q_from[2]) + frac * dtheta),
    ], dtype=np.float64)


def _resample_path(path: List[np.ndarray], T: int) -> np.ndarray:
    """Resample a SE(2) waypoint list to exactly T points by linear interpolation.

    Args:
        path: List of (3,) SE(2) arrays.  Must have at least 1 element.
        T:    Number of output waypoints.

    Returns:
        (T, 3) float32 array.
    """
    if len(path) == 1:
        return np.tile(path[0], (T, 1)).astype(np.float32)

    # Cumulative arc lengths
    cum = [0.0]
    for k in range(1, len(path)):
        cum.append(cum[-1] + _se2_dist(path[k - 1], path[k]))
    total = cum[-1]

    if total < REACH_TOL:
        return np.tile(path[0], (T, 1)).astype(np.float32)

    path_arr = np.array(path, dtype=np.float64)
    cum_arr = np.array(cum, dtype=np.float64)
    targets = np.linspace(0.0, total, T)
    result = np.empty((T, 3), dtype=np.float64)

    for i, s in enumerate(targets):
        idx = int(np.searchsorted(cum_arr, s, side="right")) - 1
        idx = min(max(idx, 0), len(path) - 2)
        seg = cum_arr[idx + 1] - cum_arr[idx]
        if seg < REACH_TOL:
            result[i] = path_arr[idx]
        else:
            t_seg = (s - cum_arr[idx]) / seg
            p0, p1 = path_arr[idx], path_arr[idx + 1]
            dtheta = _wrap_angle(p1[2] - p0[2])
            result[i, 0] = p0[0] + t_seg * (p1[0] - p0[0])
            result[i, 1] = p0[1] + t_seg * (p1[1] - p0[1])
            result[i, 2] = _wrap_angle(p0[2] + t_seg * dtheta)

    return result.astype(np.float32)


# ---------------------------------------------------------------------------
#  Constraint checking
# ---------------------------------------------------------------------------

def _violates_constraints(
    q: np.ndarray,
    timestep: int,
    constraints: List[SphereConstraint],
    visible_range: float,
) -> bool:
    """Return True if (q, timestep) violates any hard sphere constraint.

    Option A: centroid-in-disk check.  The sphere radius (radius_norm * visible_range)
    already includes the CBS margin.
    """
    for c in constraints:
        if timestep < c.t_start or timestep > c.t_end:
            continue
        cx = c.center_norm[0] * visible_range
        cy = c.center_norm[1] * visible_range
        r = c.radius_norm * visible_range
        if float(np.hypot(q[0] - cx, q[1] - cy)) < r:
            return True
    return False


# ---------------------------------------------------------------------------
#  Bidirectional space-time RRT-Connect
# ---------------------------------------------------------------------------

_TRAPPED = 0
_ADVANCED = 1
_REACHED = 2


class _SpaceTimeRRTConnect:
    """Bidirectional space-time RRT-Connect for a single SE(2) piece.

    Internal implementation — use RRTLowLevel.plan_piece() externally.

    Trees are stored as preallocated numpy arrays.  Each tree can hold at most
    (max_samples // 2) nodes.  Trying to add a node when the tree is full
    silently returns TRAPPED.

    Tree A is rooted at start (timestep 0).
    Tree B is rooted at goal  (timestep T-1).

    Args:
        start, goal:    (3,) float64 SE(2) in world metres.
        constraints:    SphereConstraints in normalised coords.
        visible_range:  Camera half-extent in metres.
        traj_len:       T — number of trajectory timesteps in the output.
        rng:            numpy RandomState.
        max_samples:    Total node budget (both trees combined).
        max_time_s:     Wall-clock budget per call.
    """

    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        constraints: List[SphereConstraint],
        visible_range: float,
        traj_len: int,
        rng: np.random.RandomState,
        max_samples: int,
        max_time_s: float,
    ):
        self.start = np.array(start, dtype=np.float64)
        self.goal = np.array(goal, dtype=np.float64)
        self.constraints = constraints
        self.visible_range = visible_range
        self.T = traj_len
        self.rng = rng
        self.max_samples = max_samples
        self.max_time_s = max_time_s

        self._d_sg = max(_se2_dist(self.start, self.goal), REACH_TOL)

        per_tree = max(max_samples // 2, 1)

        # Tree A: rooted at start
        self._a_states = np.empty((per_tree, 3), dtype=np.float64)
        self._a_parents = np.full(per_tree, -1, dtype=np.int32)
        self._a_times = np.zeros(per_tree, dtype=np.int32)
        self._a_len = 0

        # Tree B: rooted at goal
        self._b_states = np.empty((per_tree, 3), dtype=np.float64)
        self._b_parents = np.full(per_tree, -1, dtype=np.int32)
        self._b_times = np.zeros(per_tree, dtype=np.int32)
        self._b_len = 0

        # Insert roots
        self._a_states[0] = self.start
        self._a_times[0] = 0
        self._a_len = 1

        self._b_states[0] = self.goal
        self._b_times[0] = self.T - 1
        self._b_len = 1

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def plan(self) -> np.ndarray:
        """Run RRT-Connect and return a (T, 3) float32 SE(2) trajectory.

        Returns a straight-line fallback if the node/time budget is exhausted.
        """
        deadline = time.time() + self.max_time_s

        if _se2_dist(self.start, self.goal) < REACH_TOL:
            return _resample_path([self.start, self.goal], self.T)

        while (self._a_len + self._b_len) < self.max_samples:
            if time.time() > deadline:
                break

            # ---- Extend A toward random sample, then connect B toward new A node ----
            q_rand = self._sample(bias_toward=self._b_states[0])
            a_status, _ = self._extend(
                q_rand,
                self._a_states, self._a_parents, self._a_times, self._a_len,
                forward=True,
            )
            if a_status != _TRAPPED:
                self._a_len += 1
                q_new_a = self._a_states[self._a_len - 1].copy()
                b_status, b_added = self._connect(
                    q_new_a,
                    self._b_states, self._b_parents, self._b_times, self._b_len,
                    forward=False,
                )
                self._b_len += b_added
                if b_status == _REACHED:
                    return _resample_path(
                        self._extract_path(self._a_len - 1, self._b_len - 1),
                        self.T,
                    )

            # ---- Extend B toward random sample, then connect A toward new B node ----
            q_rand2 = self._sample(bias_toward=self._a_states[0])
            b_status2, _ = self._extend(
                q_rand2,
                self._b_states, self._b_parents, self._b_times, self._b_len,
                forward=False,
            )
            if b_status2 != _TRAPPED:
                self._b_len += 1
                q_new_b = self._b_states[self._b_len - 1].copy()
                a_status2, a_added = self._connect(
                    q_new_b,
                    self._a_states, self._a_parents, self._a_times, self._a_len,
                    forward=True,
                )
                self._a_len += a_added
                if a_status2 == _REACHED:
                    return _resample_path(
                        self._extract_path(self._a_len - 1, self._b_len - 1),
                        self.T,
                    )

        # Budget exhausted — straight-line fallback
        return _resample_path([self.start, self.goal], self.T)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, bias_toward: np.ndarray) -> np.ndarray:
        """Uniform random state with GOAL_BIAS toward bias_toward."""
        if self.rng.random() < GOAL_BIAS:
            return bias_toward.copy()
        return np.array([
            self.rng.uniform(-XY_BOUND, XY_BOUND),
            self.rng.uniform(-XY_BOUND, XY_BOUND),
            self.rng.uniform(-THETA_BOUND, THETA_BOUND),
        ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Tree primitives
    # ------------------------------------------------------------------

    def _nearest_idx(self, states: np.ndarray, n: int, q: np.ndarray) -> int:
        """Index of the nearest node in states[:n] to q."""
        best_idx, best_d = 0, float("inf")
        for k in range(n):
            d = _se2_dist(states[k], q)
            if d < best_d:
                best_d = d
                best_idx = k
        return best_idx

    def _estimate_time(self, q: np.ndarray, forward: bool) -> int:
        """Estimate the trajectory timestep for state q.

        Forward tree:  t = round(d(start, q) / d_sg * (T-1)), clamped to [0, T-1].
        Backward tree: t = T-1 - round(d(goal, q) / d_sg * (T-1)), clamped to [0, T-1].
        """
        if forward:
            t = int(round(_se2_dist(self.start, q) / self._d_sg * (self.T - 1)))
        else:
            t = (self.T - 1) - int(round(_se2_dist(self.goal, q) / self._d_sg * (self.T - 1)))
        return max(0, min(self.T - 1, t))

    def _try_add(
        self,
        q: np.ndarray,
        parent: int,
        t: int,
        states: np.ndarray,
        parents: np.ndarray,
        times: np.ndarray,
        n: int,
    ) -> bool:
        """Write node at position n.  Returns False if tree full or constraint violated."""
        if n >= states.shape[0]:
            return False
        if _violates_constraints(q, t, self.constraints, self.visible_range):
            return False
        states[n] = q
        parents[n] = parent
        times[n] = t
        return True

    def _extend(
        self,
        q_rand: np.ndarray,
        states: np.ndarray,
        parents: np.ndarray,
        times: np.ndarray,
        n: int,
        forward: bool,
    ) -> Tuple[int, int]:
        """One EXTEND step toward q_rand.

        Finds the nearest node, steers toward q_rand by at most STEP_SIZE,
        checks constraints, and writes to states[n] if valid.

        Returns:
            (_TRAPPED | _ADVANCED | _REACHED, n)
            Caller must increment n on non-TRAPPED result.
        """
        near_idx = self._nearest_idx(states, n, q_rand)
        q_new = _se2_steer(states[near_idx], q_rand)
        t_new = self._estimate_time(q_new, forward=forward)

        if not self._try_add(q_new, near_idx, t_new, states, parents, times, n):
            return _TRAPPED, n

        if _se2_dist(q_new, q_rand) < REACH_TOL:
            return _REACHED, n
        return _ADVANCED, n

    def _connect(
        self,
        q_target: np.ndarray,
        states: np.ndarray,
        parents: np.ndarray,
        times: np.ndarray,
        n: int,
        forward: bool,
    ) -> Tuple[int, int]:
        """Greedily extend toward q_target until REACHED or TRAPPED.

        Returns:
            (status, n_added) — final status and number of nodes written.
        """
        local_n = n
        n_added = 0
        while True:
            status, _ = self._extend(
                q_target, states, parents, times, local_n, forward=forward
            )
            if status == _TRAPPED:
                return _TRAPPED, n_added
            # Successfully added at local_n
            local_n += 1
            n_added += 1
            if status == _REACHED:
                return _REACHED, n_added
            # Check for convergence even when ADVANCED
            if _se2_dist(states[local_n - 1], q_target) < REACH_TOL:
                return _REACHED, n_added

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def _extract_path(
        self,
        node_a_idx: int,
        node_b_idx: int,
    ) -> List[np.ndarray]:
        """Extract the full SE(2) path from start to goal.

        Tree A is rooted at start.  Tree B is rooted at goal.
        - Trace A backward from node_a_idx to root → reverse → [start … meeting_A].
        - Trace B backward from node_b_idx to root → gives [meeting_B … goal].
          (No reversal needed: tracing leaf→root in a goal-rooted tree naturally
          yields meeting→goal order.)
        - Concatenate, skipping the duplicate meeting point.
        """
        # Forward path: start → node_a
        path_a: List[np.ndarray] = []
        idx = node_a_idx
        while idx != -1:
            path_a.append(self._a_states[idx].copy())
            idx = int(self._a_parents[idx])
        path_a.reverse()  # now: [start, ..., meeting_point_in_A]

        # Path from node_b to goal root (B is rooted at goal, so
        # tracing leaf→root gives meeting_B → ... → goal)
        path_b: List[np.ndarray] = []
        idx = node_b_idx
        while idx != -1:
            path_b.append(self._b_states[idx].copy())
            idx = int(self._b_parents[idx])
        # path_b is now [meeting_B, ..., goal] — correct forward order from
        # meeting point to goal, no reversal needed.

        # Combine: drop path_b[0] since it's approximately equal to path_a[-1]
        combined = path_a + (path_b[1:] if len(path_b) > 1 else [])
        if len(combined) < 2:
            combined = [self.start.copy(), self.goal.copy()]
        return combined


# ---------------------------------------------------------------------------
#  Public planner class
# ---------------------------------------------------------------------------

class RRTLowLevel:
    """Space-time RRT-Connect low-level planner satisfying LowLevelPlannerProtocol.

    Produces B=1 trajectory per planning call.  Does not use the neural
    network or GPU — ``device = "cpu"`` for compatibility with _build_scene_batch.

    Args:
        visible_range:  Camera half-extent in metres.
        traj_len:       T — output trajectory length (number of timesteps).
        seed:           Base random seed; incremented per plan_piece call.
        max_samples:    Total tree-node budget per call (both trees).
        max_time_s:     Wall-clock budget per call in seconds.
    """

    # Must be a string so that _build_scene_batch can call tensor.to(device)
    device: str = "cpu"

    def __init__(
        self,
        visible_range: float = 0.6,
        traj_len: int = 32,
        seed: int = 0,
        max_samples: int = 5000,
        max_time_s: float = 10.0,
    ):
        self.visible_range = visible_range
        self.traj_len = traj_len
        self._base_seed = seed
        self.max_samples = max_samples
        self.max_time_s = max_time_s
        self._call_count = 0

    def plan_piece(
        self,
        scene_batch: dict,
        constraints: List[SphereConstraint],
    ) -> np.ndarray:
        """Plan one trajectory using space-time RRT-Connect.

        Reads ``start_pose`` and ``goal_pose`` from scene_batch.  All other
        keys (images, masks) are ignored — RRT does not use vision.

        Args:
            scene_batch: Dict with keys:
                ``start_pose`` (3,) float32 [x, y, theta] world metres
                ``goal_pose``  (3,) float32 [x, y, theta] world metres
                (other keys ignored)
            constraints: Hard sphere constraints (normalised coords).

        Returns:
            (1, T, 3) float32 array — single trajectory in world metres SE(2).
        """
        start = np.asarray(scene_batch["start_pose"], dtype=np.float64).flatten()[:3]
        goal = np.asarray(scene_batch["goal_pose"], dtype=np.float64).flatten()[:3]

        rng = np.random.RandomState(self._base_seed + self._call_count)
        self._call_count += 1

        rrt = _SpaceTimeRRTConnect(
            start=start,
            goal=goal,
            constraints=constraints,
            visible_range=self.visible_range,
            traj_len=self.traj_len,
            rng=rng,
            max_samples=self.max_samples,
            max_time_s=self.max_time_s,
        )
        traj = rrt.plan()  # (T, 3) float32

        return traj[np.newaxis, ...]  # (1, T, 3)
