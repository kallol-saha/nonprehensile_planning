"""Convex polygon geometry utilities.

All computations assume a 2D convex polygon with vertices in
counter-clockwise (CCW) order, centered at its centroid.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull


class ConvexPolygon:
    """Immutable convex polygon defined by CCW vertices in a local frame.

    The local frame origin is at the polygon's centroid — vertices are
    automatically re-centred on construction.

    Attributes:
        vertices:     (V, 2) CCW vertices, centroid at origin.
        num_vertices: number of vertices / edges.
        area:         polygon area (m²).
        moment_of_inertia: second moment of area about centroid (m⁴),
                      equivalent to I/m for unit-density uniform slab.
    """

    def __init__(self, vertices: np.ndarray):
        """Create a ConvexPolygon from a (V, 2) array of 2D points.

        Points need not be ordered or convex — the convex hull is taken
        and vertices are stored in CCW order, centred at the centroid.
        """
        pts = np.asarray(vertices, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"Expected (V, 2) array, got {pts.shape}")

        # Convex hull → CCW ordering
        if len(pts) < 3:
            raise ValueError("Need at least 3 points")
        hull = ConvexHull(pts)
        ordered = pts[hull.vertices]  # CCW by scipy convention

        # Re-centre at centroid
        cx, cy = self._centroid(ordered)
        ordered = ordered - np.array([cx, cy])

        self._vertices = ordered
        self._area: float | None = None
        self._moi: float | None = None
        self._edges: np.ndarray | None = None
        self._edge_normals: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    #  Core properties (lazy-computed, cached)
    # ------------------------------------------------------------------ #

    @property
    def vertices(self) -> np.ndarray:
        """(V, 2) CCW vertices with centroid at origin."""
        return self._vertices

    @property
    def num_vertices(self) -> int:
        return len(self._vertices)

    @property
    def edges(self) -> np.ndarray:
        """(V, 2, 2) array — edges[i] = [start, end] for edge i."""
        if self._edges is None:
            v = self._vertices
            self._edges = np.stack([v, np.roll(v, -1, axis=0)], axis=1)
        return self._edges

    @property
    def edge_normals(self) -> np.ndarray:
        """(V, 2) outward unit normals, one per edge."""
        if self._edge_normals is None:
            d = self.edges[:, 1] - self.edges[:, 0]  # (V, 2) edge vectors
            # Outward normal for CCW polygon: rotate edge 90° clockwise
            normals = np.column_stack([d[:, 1], -d[:, 0]])
            normals /= np.linalg.norm(normals, axis=1, keepdims=True)
            self._edge_normals = normals
        return self._edge_normals

    @property
    def edge_lengths(self) -> np.ndarray:
        """(V,) length of each edge."""
        d = self.edges[:, 1] - self.edges[:, 0]
        return np.linalg.norm(d, axis=1)

    @property
    def area(self) -> float:
        if self._area is None:
            self._area = self._shoelace_area(self._vertices)
        return self._area

    @property
    def moment_of_inertia(self) -> float:
        """Second moment of area about centroid (uniform density, unit mass).

        For a uniform slab of mass m and density ρ = m/A, the rotational
        inertia is I = ρ · J where J is the second moment of area.
        This returns J/A so that I/m = moment_of_inertia.
        """
        if self._moi is None:
            self._moi = self._compute_moi(self._vertices)
        return self._moi

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def sample_edge_point(self, edge_idx: int, t: float) -> np.ndarray:
        """Return the point at parameter t ∈ [0, 1] along edge `edge_idx` (local frame)."""
        e = self.edges[edge_idx]
        return (1.0 - t) * e[0] + t * e[1]

    def transform(self, x: float, y: float, theta: float) -> np.ndarray:
        """Return (V, 2) vertices transformed to world frame by pose (x, y, θ)."""
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        return (R @ self._vertices.T).T + np.array([x, y])

    def contains_point(self, point: np.ndarray) -> bool:
        """Test whether a 2D point lies inside (or on boundary of) the polygon."""
        p = np.asarray(point, dtype=np.float64)
        v = self._vertices
        n = len(v)
        for i in range(n):
            edge = v[(i + 1) % n] - v[i]
            to_point = p - v[i]
            cross = edge[0] * to_point[1] - edge[1] * to_point[0]
            if cross < -1e-12:
                return False
        return True

    # ------------------------------------------------------------------ #
    #  Static geometry computations
    # ------------------------------------------------------------------ #

    @staticmethod
    def _centroid(vertices: np.ndarray) -> tuple[float, float]:
        """Centroid of a simple polygon (vertices in order)."""
        v = vertices
        n = len(v)
        cx = cy = 0.0
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            cross = v[i, 0] * v[j, 1] - v[j, 0] * v[i, 1]
            area += cross
            cx += (v[i, 0] + v[j, 0]) * cross
            cy += (v[i, 1] + v[j, 1]) * cross
        area *= 0.5
        cx /= (6.0 * area)
        cy /= (6.0 * area)
        return cx, cy

    @staticmethod
    def _shoelace_area(vertices: np.ndarray) -> float:
        """Unsigned area via the shoelace formula."""
        v = vertices
        n = len(v)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += v[i, 0] * v[j, 1] - v[j, 0] * v[i, 1]
        return abs(area) * 0.5

    @staticmethod
    def _compute_moi(vertices: np.ndarray) -> float:
        """Second moment of area / area (= I/m for unit mass) about centroid.

        Uses the standard triangulation formula for a polygon with centroid
        at the origin.
        """
        v = vertices
        n = len(v)
        numer = 0.0
        denom = 0.0
        for i in range(n):
            j = (i + 1) % n
            cross = abs(v[i, 0] * v[j, 1] - v[j, 0] * v[i, 1])
            numer += cross * (
                np.dot(v[i], v[i]) + np.dot(v[i], v[j]) + np.dot(v[j], v[j])
            )
            denom += cross
        if denom < 1e-15:
            return 0.0
        return numer / (6.0 * denom)

    def __repr__(self) -> str:
        return (f"ConvexPolygon(V={self.num_vertices}, "
                f"area={self.area:.6f}, moi={self.moment_of_inertia:.6f})")
