"""CBS-based multi-piece trajectory planning for Voronoi reassembly.

Package layout
--------------
  constraints.py       – SphereConstraint dataclass and soft-guidance cost/gradient
  conflict_detector.py – Pairwise polygon collision detection → Conflict objects
  ct_node.py           – CTNode: constraint tree node (trajectories + constraints)
  guided_sampler.py    – Guidance-augmented DDPM/DDIM sampling loops
  low_level_planner.py – LowLevelPlanner: wraps a diffusion model, plan_piece()
  cbs.py               – CBS and MMD-PP planning algorithms
"""
