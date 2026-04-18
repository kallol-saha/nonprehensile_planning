"""CBS-based multi-piece trajectory planning for Voronoi reassembly.

Package layout
--------------
  constraints.py       – SphereConstraint dataclass and soft-guidance cost/gradient
  conflict_detector.py – Pairwise polygon collision detection → Conflict objects
                         Includes compute_conflict_time_ratio() metric.
  ct_node.py           – CTNode: constraint tree node (trajectories + constraints)
  guided_sampler.py    – Guidance-augmented DDPM/DDIM sampling loops
  low_level_planner.py – LowLevelPlannerProtocol (Protocol), DiffusionLowLevel
  rrt_low_level.py     – RRTLowLevel: space-time RRT-Connect low-level planner
  cbs.py               – run_cbs, run_pp, run_independent planning algorithms
"""
