"""Polygon shape factories.

Each factory returns a :class:`ConvexPolygon` instance ready for use
with the push simulator.
"""

from __future__ import annotations

import numpy as np

from visplan.push_sim.geometry import ConvexPolygon


def regular_polygon(n_sides: int, radius: float = 0.05) -> ConvexPolygon:
    """Regular *n*-sided polygon inscribed in a circle of given radius."""
    angles = np.linspace(0, 2 * np.pi, n_sides, endpoint=False)
    verts = np.column_stack([np.cos(angles), np.sin(angles)]) * radius
    return ConvexPolygon(verts)


def random_convex_polygon(
    n_verts: int = 6,
    scale: float = 0.05,
    rng: np.random.RandomState | None = None,
) -> ConvexPolygon:
    """Random convex polygon by taking the convex hull of *n_verts* random points.

    Points are sampled uniformly in a disk of the given ``scale`` (radius).
    The actual vertex count may be fewer than *n_verts* after the hull.
    """
    if rng is None:
        rng = np.random.RandomState()
    angles = rng.uniform(0, 2 * np.pi, size=n_verts)
    radii = scale * np.sqrt(rng.uniform(0, 1, size=n_verts))
    pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return ConvexPolygon(pts)


def from_voronoi_cell(vertices: np.ndarray) -> ConvexPolygon:
    """Wrap an existing (V, 2) Voronoi cell vertex array as a ConvexPolygon.

    This bridges the existing ``generation_utils.generate_voronoi_meshes``
    output — pass the XY slice of a mesh's vertices.
    """
    return ConvexPolygon(np.asarray(vertices)[:, :2])
