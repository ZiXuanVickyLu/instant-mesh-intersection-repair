"""
ACCD (Additive Continuous Collision Detection) for edge-edge and vertex-triangle pairs.

Port of the CUDA ACCD implementation (accd.cu) to NVIDIA Warp for
GPU-accelerated batched CCD queries.

Usage:
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd, batch_vertex_triangle_ccd

    toi = batch_edge_edge_ccd(
        a0_start, a1_start, b0_start, b1_start,
        a0_end, a1_end, b0_end, b1_end,
    )
    # toi: (N,) float array, 1.0 = no collision

    toi = batch_vertex_triangle_ccd(
        p_start, t0_start, t1_start, t2_start,
        p_end, t0_end, t1_end, t2_end,
    )
    # toi: (N,) float array, 1.0 = no collision
"""

import warp as wp
import torch
import numpy as np

# ACCD default parameters
ACCD_MAX_ITER = 500
ACCD_XI = 1e-3   # thickness / minimum separation
ACCD_S = 0.1     # conservative advancement factor
ACCD_TAU = 1.0   # max time horizon


# ---------------------------------------------------------------------------
# Warp functions — edge-edge
# ---------------------------------------------------------------------------

EE_EPS = 1.0e-6  # epsilon for warp closest_point_edge_edge


@wp.func
def _edge_edge_dist_sq(
    p0: wp.vec3, p1: wp.vec3,
    q0: wp.vec3, q1: wp.vec3,
) -> float:
    """Squared minimum distance between segments (p0,p1) and (q0,q1).

    Uses warp's built-in closest_point_edge_edge which returns parametric
    coordinates (s, t) along each edge as a vec3 (s, t, _).
    """
    st = wp.closest_point_edge_edge(p0, p1, q0, q1, EE_EPS)
    s = st[0]
    t = st[1]
    c1 = p0 + (p1 - p0) * s
    c2 = q0 + (q1 - q0) * t
    diff = c1 - c2
    return wp.dot(diff, diff)


@wp.func
def _ee_accd(
    p0: wp.vec3, p1: wp.vec3, q0: wp.vec3, q1: wp.vec3,
    p0t: wp.vec3, p1t: wp.vec3, q0t: wp.vec3, q1t: wp.vec3,
    max_iter: int, xi: float, s: float, tau: float,
) -> float:
    """
    ACCD for two edges sweeping linearly.

    Edge A: (p0,p1) at t=0 -> (p0t,p1t) at t=1
    Edge B: (q0,q1) at t=0 -> (q0t,q1t) at t=1

    Returns TOI in [0, 1].  1.0 means no collision within [0, tau].
    """
    # Current positions (advanced each iteration)
    v0 = p0
    v1 = p1
    v2 = q0
    v3 = q1

    # Full displacements
    disp0 = p0t - p0
    disp1 = p1t - p1
    disp2 = q0t - q0
    disp3 = q1t - q1

    # Mean (rigid) displacement -- factored out for tighter lp bound
    mean_d = (disp0 + disp1 + disp2 + disp3) * 0.25

    # Relative (deformation) displacements
    de0 = disp0 - mean_d
    de1 = disp1 - mean_d
    de2 = disp2 - mean_d
    de3 = disp3 - mean_d

    # Upper bound on deformation speed
    lp = wp.max(wp.length(de0), wp.length(de1)) + wp.max(wp.length(de2), wp.length(de3))

    if lp <= 0.0:
        return 1.0  # pure rigid motion -- distance is constant

    # Initial squared distance
    dsqr = _edge_edge_dist_sq(v0, v1, v2, v3)
    dFunc = dsqr - xi * xi

    # Already interpenetrating beyond xi -- fall back to vertex-vertex distances
    if dFunc <= 0.0:
        d00 = wp.length_sq(p0 - q0)
        d01 = wp.length_sq(p0 - q1)
        d10 = wp.length_sq(p1 - q0)
        d11 = wp.length_sq(p1 - q1)
        dsqr = wp.min(wp.min(d00, d01), wp.min(d10, d11))
        dFunc = dsqr - xi * xi

    dis_cur = wp.sqrt(dsqr)
    g = s * dFunc / (dis_cur + xi)
    toi = float(0.0)
    done = int(0)

    it = int(0)
    while it < max_iter and done == 0:
        tl = (1.0 - s) * dFunc / (lp * (dis_cur + xi))
        it = it + 1

        # Advance vertices by the safe step
        v0 = v0 + tl * de0
        v1 = v1 + tl * de1
        v2 = v2 + tl * de2
        v3 = v3 + tl * de3

        # Recompute distance
        dsqr = _edge_edge_dist_sq(v0, v1, v2, v3)
        dFunc = dsqr - xi * xi

        if dFunc <= 0.0:
            d00 = wp.length_sq(v0 - v2)
            d01 = wp.length_sq(v1 - v2)
            d10 = wp.length_sq(v0 - v3)
            d11 = wp.length_sq(v1 - v3)
            dsqr = wp.min(wp.min(d00, d01), wp.min(d10, d11))
            dFunc = dsqr - xi * xi

        dis_cur = wp.sqrt(dsqr)
        g_cur = dFunc / (dis_cur + xi)

        # Gap function started increasing -> we passed the minimum
        if toi > 0.0 and g_cur < g:
            done = 1
        else:
            toi = toi + tl
            if toi >= tau:
                toi = 1.0
                done = 1

    return wp.clamp(toi, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Warp functions — point-triangle distance (Geometric Tools algorithm)
# ---------------------------------------------------------------------------

@wp.func
def _point_triangle_dist_sq(
    point: wp.vec3,
    v0: wp.vec3, v1: wp.vec3, v2: wp.vec3,
) -> float:
    """Squared distance from point to triangle (v0, v1, v2).

    Port of the Geometric Tools DCPQuery<Point3, Triangle3> algorithm
    by David Eberly. Uses the 7-region classification approach.
    """
    diff = v0 - point
    edge0 = v1 - v0
    edge1 = v2 - v0
    a00 = wp.dot(edge0, edge0)
    a01 = wp.dot(edge0, edge1)
    a11 = wp.dot(edge1, edge1)
    b0 = wp.dot(diff, edge0)
    b1 = wp.dot(diff, edge1)
    det = wp.max(a00 * a11 - a01 * a01, 0.0)
    s_val = a01 * b1 - a11 * b0
    t_val = a01 * b0 - a00 * b1

    if s_val + t_val <= det:
        if s_val < 0.0:
            if t_val < 0.0:
                # region 4
                if b0 < 0.0:
                    t_val = 0.0
                    if -b0 >= a00:
                        s_val = 1.0
                    else:
                        s_val = -b0 / a00
                else:
                    s_val = 0.0
                    if b1 >= 0.0:
                        t_val = 0.0
                    elif -b1 >= a11:
                        t_val = 1.0
                    else:
                        t_val = -b1 / a11
            else:
                # region 3
                s_val = 0.0
                if b1 >= 0.0:
                    t_val = 0.0
                elif -b1 >= a11:
                    t_val = 1.0
                else:
                    t_val = -b1 / a11
        elif t_val < 0.0:
            # region 5
            t_val = 0.0
            if b0 >= 0.0:
                s_val = 0.0
            elif -b0 >= a00:
                s_val = 1.0
            else:
                s_val = -b0 / a00
        else:
            # region 0 — interior
            if det > 0.0:
                s_val = s_val / det
                t_val = t_val / det
            else:
                s_val = 0.0
                t_val = 0.0
    else:
        if s_val < 0.0:
            # region 2
            tmp0 = a01 + b0
            tmp1 = a11 + b1
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a00 - 2.0 * a01 + a11
                if numer >= denom:
                    s_val = 1.0
                    t_val = 0.0
                else:
                    s_val = numer / denom
                    t_val = 1.0 - s_val
            else:
                s_val = 0.0
                if tmp1 <= 0.0:
                    t_val = 1.0
                elif b1 >= 0.0:
                    t_val = 0.0
                else:
                    t_val = -b1 / a11
        elif t_val < 0.0:
            # region 6
            tmp0 = a01 + b1
            tmp1 = a00 + b0
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a00 - 2.0 * a01 + a11
                if numer >= denom:
                    t_val = 1.0
                    s_val = 0.0
                else:
                    t_val = numer / denom
                    s_val = 1.0 - t_val
            else:
                t_val = 0.0
                if tmp1 <= 0.0:
                    s_val = 1.0
                elif b0 >= 0.0:
                    s_val = 0.0
                else:
                    s_val = -b0 / a00
        else:
            # region 1
            numer = a11 + b1 - a01 - b0
            if numer <= 0.0:
                s_val = 0.0
                t_val = 1.0
            else:
                denom = a00 - 2.0 * a01 + a11
                if numer >= denom:
                    s_val = 1.0
                    t_val = 0.0
                else:
                    s_val = numer / denom
                    t_val = 1.0 - s_val

    closest = v0 + s_val * edge0 + t_val * edge1
    d = point - closest
    return wp.dot(d, d)


# ---------------------------------------------------------------------------
# Warp functions — vertex-triangle ACCD
# ---------------------------------------------------------------------------

@wp.func
def _vt_accd(
    p0: wp.vec3, q0: wp.vec3, q1: wp.vec3, q2: wp.vec3,
    p0t: wp.vec3, q0t: wp.vec3, q1t: wp.vec3, q2t: wp.vec3,
    max_iter: int, xi: float, s: float, tau: float,
) -> float:
    """
    ACCD for a vertex sweeping toward a triangle.

    Vertex: p0 at t=0 -> p0t at t=1
    Triangle: (q0,q1,q2) at t=0 -> (q0t,q1t,q2t) at t=1

    Returns TOI in [0, 1].  1.0 means no collision within [0, tau].

    Port of vertex_triangle_ccd from accd.cu.
    """
    # Current positions
    v0 = p0
    v1 = q0
    v2 = q1
    v3 = q2

    # Full displacements
    disp0 = p0t - p0
    disp1 = q0t - q0
    disp2 = q1t - q1
    disp3 = q2t - q2

    # Mean (rigid) displacement
    mean_d = (disp0 + disp1 + disp2 + disp3) * 0.25

    # Relative (deformation) displacements
    de0 = disp0 - mean_d
    de1 = disp1 - mean_d
    de2 = disp2 - mean_d
    de3 = disp3 - mean_d

    # Upper bound on deformation speed (vertex + max of triangle vertices)
    lp = wp.length(de0) + wp.max(wp.max(wp.length(de1), wp.length(de2)), wp.length(de3))

    if lp <= 0.0:
        return 1.0  # pure rigid motion

    # Initial squared distance
    dsqr = _point_triangle_dist_sq(v0, v1, v2, v3)
    g = s * (dsqr - xi * xi) / (wp.sqrt(dsqr) + xi)

    toi = float(0.0)
    tl = (1.0 - s) * (dsqr - xi * xi) / (lp * (wp.sqrt(dsqr) + xi))
    done = int(0)

    it = int(0)
    while it < max_iter and done == 0:
        it = it + 1

        # Advance vertices by the safe step
        v0 = v0 + tl * de0
        v1 = v1 + tl * de1
        v2 = v2 + tl * de2
        v3 = v3 + tl * de3

        # Recompute distance
        dsqr = _point_triangle_dist_sq(v0, v1, v2, v3)
        g_cur = (dsqr - xi * xi) / (wp.sqrt(dsqr) + xi)

        if toi > 0.0 and g_cur < g:
            done = 1
        else:
            toi = toi + tl
            if toi >= tau:
                toi = 1.0
                done = 1
            else:
                tl = 0.9 * g_cur / lp

    return wp.clamp(toi, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _ee_ccd_kernel(
    a0s: wp.array(dtype=wp.vec3),
    a1s: wp.array(dtype=wp.vec3),
    b0s: wp.array(dtype=wp.vec3),
    b1s: wp.array(dtype=wp.vec3),
    a0e: wp.array(dtype=wp.vec3),
    a1e: wp.array(dtype=wp.vec3),
    b0e: wp.array(dtype=wp.vec3),
    b1e: wp.array(dtype=wp.vec3),
    max_iter: int,
    xi: float,
    s_param: float,
    tau: float,
    toi_out: wp.array(dtype=float),
):
    tid = wp.tid()
    toi_out[tid] = _ee_accd(
        a0s[tid], a1s[tid], b0s[tid], b1s[tid],
        a0e[tid], a1e[tid], b0e[tid], b1e[tid],
        max_iter, xi, s_param, tau,
    )


@wp.kernel
def _vt_ccd_kernel(
    ps: wp.array(dtype=wp.vec3),
    t0s: wp.array(dtype=wp.vec3),
    t1s: wp.array(dtype=wp.vec3),
    t2s: wp.array(dtype=wp.vec3),
    pe: wp.array(dtype=wp.vec3),
    t0e: wp.array(dtype=wp.vec3),
    t1e: wp.array(dtype=wp.vec3),
    t2e: wp.array(dtype=wp.vec3),
    max_iter: int,
    xi: float,
    s_param: float,
    tau: float,
    toi_out: wp.array(dtype=float),
):
    tid = wp.tid()
    toi_out[tid] = _vt_accd(
        ps[tid], t0s[tid], t1s[tid], t2s[tid],
        pe[tid], t0e[tid], t1e[tid], t2e[tid],
        max_iter, xi, s_param, tau,
    )


# ---------------------------------------------------------------------------
# Python API
# ---------------------------------------------------------------------------

def _init_and_wrap(tensors, device_str):
    """Initialize warp and wrap torch tensors as warp arrays."""
    wp.init()
    wp_device = "cuda:0" if "cuda" in device_str else "cpu"

    def _to_wp(t: torch.Tensor) -> wp.array:
        t = t.contiguous().float()
        return wp.from_torch(t.view(-1, 3), dtype=wp.vec3)

    return [_to_wp(t) for t in tensors], wp_device


def batch_edge_edge_ccd(
    a0_start: torch.Tensor,
    a1_start: torch.Tensor,
    b0_start: torch.Tensor,
    b1_start: torch.Tensor,
    a0_end: torch.Tensor,
    a1_end: torch.Tensor,
    b0_end: torch.Tensor,
    b1_end: torch.Tensor,
    xi: float = ACCD_XI,
    s: float = ACCD_S,
    tau: float = ACCD_TAU,
    max_iter: int = ACCD_MAX_ITER,
) -> torch.Tensor:
    """
    Batched edge-edge CCD using ACCD on GPU via Warp.

    Args:
        a0_start, a1_start: (N, 3) start positions of edge A endpoints
        b0_start, b1_start: (N, 3) start positions of edge B endpoints
        a0_end, a1_end:     (N, 3) end positions of edge A endpoints
        b0_end, b1_end:     (N, 3) end positions of edge B endpoints
        xi:       thickness / minimum separation (default 1e-3)
        s:        ACCD step factor (default 0.1)
        tau:      max time horizon (default 1.0)
        max_iter: iteration cap (default 500)

    Returns:
        toi: (N,) float tensor on same device. 1.0 = no collision.
    """
    n = a0_start.shape[0]
    if n == 0:
        return torch.ones(0, device=a0_start.device, dtype=torch.float32)

    arrays, wp_device = _init_and_wrap(
        [a0_start, a1_start, b0_start, b1_start,
         a0_end, a1_end, b0_end, b1_end],
        str(a0_start.device),
    )

    toi_wp = wp.zeros(n, dtype=float, device=wp_device)

    wp.launch(
        kernel=_ee_ccd_kernel,
        dim=n,
        inputs=arrays + [max_iter, xi, s, tau],
        outputs=[toi_wp],
        device=wp_device,
    )

    return wp.to_torch(toi_wp)


def batch_vertex_triangle_ccd(
    p_start: torch.Tensor,
    t0_start: torch.Tensor,
    t1_start: torch.Tensor,
    t2_start: torch.Tensor,
    p_end: torch.Tensor,
    t0_end: torch.Tensor,
    t1_end: torch.Tensor,
    t2_end: torch.Tensor,
    xi: float = ACCD_XI,
    s: float = ACCD_S,
    tau: float = ACCD_TAU,
    max_iter: int = ACCD_MAX_ITER,
) -> torch.Tensor:
    """
    Batched vertex-triangle CCD using ACCD on GPU via Warp.

    Args:
        p_start:            (N, 3) start positions of moving vertex
        t0_start, t1_start, t2_start: (N, 3) start positions of triangle verts
        p_end:              (N, 3) end positions of moving vertex
        t0_end, t1_end, t2_end: (N, 3) end positions of triangle verts

    Returns:
        toi: (N,) float tensor on same device. 1.0 = no collision.
    """
    n = p_start.shape[0]
    if n == 0:
        return torch.ones(0, device=p_start.device, dtype=torch.float32)

    arrays, wp_device = _init_and_wrap(
        [p_start, t0_start, t1_start, t2_start,
         p_end, t0_end, t1_end, t2_end],
        str(p_start.device),
    )

    toi_wp = wp.zeros(n, dtype=float, device=wp_device)

    wp.launch(
        kernel=_vt_ccd_kernel,
        dim=n,
        inputs=arrays + [max_iter, xi, s, tau],
        outputs=[toi_wp],
        device=wp_device,
    )

    return wp.to_torch(toi_wp)
