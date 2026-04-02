"""
XPBD Guide Solver — GPU-accelerated position-based dynamics for guide curves.

Replaces Laplacian diffusion with physically-motivated constraints:
  - Stretch: preserve edge lengths between consecutive guide points
  - Bending: preserve angles via ghost-edge (skip-one) distance constraints
  - Pin: root vertex (layer 0) fixed

Uses Warp for GPU kernels with Jacobi-style parallel constraint projection.
"""

import numpy as np
import torch
import warp as wp

# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _predict_kernel(
    x: wp.array(dtype=wp.vec3),
    disp: wp.array(dtype=wp.vec3),
    x_pred: wp.array(dtype=wp.vec3),
):
    """x_pred = x + displacement."""
    i = wp.tid()
    x_pred[i] = x[i] + disp[i]


@wp.kernel
def _zero_corrections_kernel(
    corrections: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
):
    """Reset correction accumulators to zero."""
    i = wp.tid()
    corrections[i] = wp.vec3(0.0, 0.0, 0.0)
    counts[i] = wp.int32(0)


@wp.kernel
def _project_distance_kernel(
    x_pred: wp.array(dtype=wp.vec3),
    pairs_i: wp.array(dtype=wp.int32),
    pairs_j: wp.array(dtype=wp.int32),
    rest_dists: wp.array(dtype=wp.float32),
    alpha_tilde: wp.float32,
    corrections: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
):
    """Project a distance constraint C = ||xi - xj|| - d_rest.

    Used for both stretch (consecutive) and bending (skip-one) constraints.
    Jacobi-style: accumulate corrections via atomics, apply later.
    """
    tid = wp.tid()
    i = pairs_i[tid]
    j = pairs_j[tid]

    d = x_pred[j] - x_pred[i]
    dist = wp.length(d)
    if dist < 1.0e-10:
        return

    C = dist - wp.float32(rest_dists[tid])
    # XPBD: delta_lambda = -C / (w_i + w_j + alpha_tilde)
    # w_i = w_j = 1 (unit mass), so denominator = 2 + alpha_tilde
    denom = 2.0 + alpha_tilde
    delta_lambda = -C / denom
    corr = delta_lambda * d / dist

    # Accumulate (Jacobi): vertex i gets -corr, vertex j gets +corr
    wp.atomic_add(corrections, i, -corr)
    wp.atomic_add(corrections, j, corr)
    wp.atomic_add(counts, i, wp.int32(1))
    wp.atomic_add(counts, j, wp.int32(1))


@wp.kernel
def _apply_corrections_kernel(
    x_pred: wp.array(dtype=wp.vec3),
    corrections: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
):
    """Apply averaged Jacobi corrections."""
    i = wp.tid()
    c = counts[i]
    if c > 0:
        x_pred[i] = x_pred[i] + corrections[i] / wp.float32(c)


@wp.kernel
def _project_target_kernel(
    x_pred: wp.array(dtype=wp.vec3),
    x_target: wp.array(dtype=wp.vec3),
    alpha_tilde: wp.float32,
    corrections: wp.array(dtype=wp.vec3),
    counts: wp.array(dtype=wp.int32),
):
    """Project soft position-target constraint: C = ||x - x_target||.

    Pulls each vertex toward its target (guide + displacement) with compliance.
    This encodes the tube collision response as an energy term in the XPBD loop.
    """
    i = wp.tid()
    d = x_pred[i] - x_target[i]
    dist = wp.length(d)
    if dist < 1.0e-10:
        return

    # Unilateral: only one vertex participates, so w = 1
    C = dist
    denom = 1.0 + alpha_tilde
    delta_lambda = -C / denom
    corr = delta_lambda * d / dist

    wp.atomic_add(corrections, i, corr)
    wp.atomic_add(counts, i, wp.int32(1))


@wp.kernel
def _pin_kernel(
    x_pred: wp.array(dtype=wp.vec3),
    pin_indices: wp.array(dtype=wp.int32),
    pin_positions: wp.array(dtype=wp.vec3),
):
    """Pin vertices to fixed positions."""
    tid = wp.tid()
    idx = pin_indices[tid]
    x_pred[idx] = pin_positions[tid]


# ---------------------------------------------------------------------------
# Solver class
# ---------------------------------------------------------------------------

class XPBDGuideSolver:
    """XPBD solver for guide polyline deformation on GPU.

    Build topology once from guide structure, then call solve() each step
    with current positions and displacements.

    Args:
        n_guides: number of guide curves.
        n_layers: number of vertices per guide (same for all).
        device: torch/warp device string.
        n_iters: number of XPBD iterations per solve.
        stretch_stiffness: stiffness for stretch constraints (higher = stiffer).
        bend_stiffness: stiffness for bending ghost-edge constraints.
        target_stiffness: stiffness for target position constraints.
        dt: pseudo time-step for XPBD compliance scaling.
    """

    def __init__(
        self,
        n_guides: int,
        n_layers: int,
        device: str = "cuda:0",
        n_iters: int = 15,
        stretch_stiffness: float = 1e3,
        bend_stiffness: float = 1e5,
        target_stiffness: float = 1e4,
        dt: float = 1.0,
    ):
        wp.init()
        self.n_guides = n_guides
        self.n_layers = n_layers
        self.n_iters = n_iters
        self.dt = dt
        # Convert stiffness to compliance: compliance = 1/stiffness
        self.stretch_compliance = 1.0 / stretch_stiffness if stretch_stiffness > 0 else 0.0
        self.bend_compliance = 1.0 / bend_stiffness if bend_stiffness > 0 else 0.0
        self.target_compliance = 1.0 / target_stiffness if target_stiffness > 0 else 0.0

        self.wp_device = "cuda:0" if "cuda" in device else "cpu"
        self.total_verts = n_guides * n_layers

        # --- Build constraint topology (fixed across all solves) ---
        L = n_layers

        # Pin: first vertex of each guide
        pin_ids = np.array([gi * L for gi in range(n_guides)], dtype=np.int32)

        # Stretch: consecutive pairs within each guide
        stretch_i = []
        stretch_j = []
        for gi in range(n_guides):
            base = gi * L
            for li in range(L - 1):
                stretch_i.append(base + li)
                stretch_j.append(base + li + 1)
        stretch_i_np = np.array(stretch_i, dtype=np.int32)
        stretch_j_np = np.array(stretch_j, dtype=np.int32)

        # Bending: ghost-edge (skip-one) pairs within each guide
        bend_i = []
        bend_j = []
        for gi in range(n_guides):
            base = gi * L
            for li in range(L - 2):
                bend_i.append(base + li)
                bend_j.append(base + li + 2)
        bend_i_np = np.array(bend_i, dtype=np.int32)
        bend_j_np = np.array(bend_j, dtype=np.int32)

        # Keep CPU copies for rest-length computation
        self._stretch_i_np = stretch_i_np
        self._stretch_j_np = stretch_j_np
        self._bend_i_np = bend_i_np
        self._bend_j_np = bend_j_np

        # Upload topology to GPU
        self.pin_indices = wp.array(pin_ids, dtype=wp.int32, device=self.wp_device)
        self.stretch_i = wp.array(stretch_i_np, dtype=wp.int32, device=self.wp_device)
        self.stretch_j = wp.array(stretch_j_np, dtype=wp.int32, device=self.wp_device)
        self.bend_i = wp.array(bend_i_np, dtype=wp.int32, device=self.wp_device)
        self.bend_j = wp.array(bend_j_np, dtype=wp.int32, device=self.wp_device)
        self.n_stretch = len(stretch_i_np)
        self.n_bend = len(bend_i_np)
        self.n_pins = len(pin_ids)

        # Pre-allocate working buffers (reused every solve)
        self.x_pred = wp.zeros(self.total_verts, dtype=wp.vec3, device=self.wp_device)
        self.corrections = wp.zeros(self.total_verts, dtype=wp.vec3, device=self.wp_device)
        self.counts = wp.zeros(self.total_verts, dtype=wp.int32, device=self.wp_device)

        # Pre-allocate input/output GPU buffers (avoid wp.array alloc per solve)
        self._x_wp = wp.zeros(self.total_verts, dtype=wp.vec3, device=self.wp_device)
        self._disp_wp = wp.zeros(self.total_verts, dtype=wp.vec3, device=self.wp_device)
        self._target_wp = wp.zeros(self.total_verts, dtype=wp.vec3, device=self.wp_device)
        self._stretch_rest = wp.zeros(self.n_stretch, dtype=wp.float32, device=self.wp_device)
        self._bend_rest = wp.zeros(self.n_bend, dtype=wp.float32, device=self.wp_device)
        self._pin_pos = wp.zeros(self.n_pins, dtype=wp.vec3, device=self.wp_device)

        # Pre-allocate CPU buffers
        self._cpu_x = np.empty((self.total_verts, 3), dtype=np.float32)
        self._cpu_disp = np.empty((self.total_verts, 3), dtype=np.float32)
        self._cpu_pin = np.empty((self.n_pins, 3), dtype=np.float32)

    def solve(
        self,
        guides: list,
        displacements: list,
    ) -> list:
        """Run XPBD solve: predict from displacement, then iteratively project constraints.

        Args:
            guides: list of (L, 3) float32 numpy arrays — current guide positions.
            displacements: list of (L, 3) float64 numpy arrays — per-layer displacement.

        Returns:
            list of (L, 3) float32 numpy arrays — new guide positions.
        """
        L = self.n_layers

        # Fill pre-allocated CPU buffers (no concat allocation)
        offset = 0
        for gi in range(self.n_guides):
            g = guides[gi]
            d = displacements[gi]
            n = g.shape[0]
            self._cpu_x[offset:offset + n] = g.astype(np.float32)
            self._cpu_disp[offset:offset + n] = d.astype(np.float32)
            self._cpu_pin[gi] = g[0].astype(np.float32)
            offset += n

        # Compute rest lengths from current positions (CPU, using cached indices)
        x64 = self._cpu_x.astype(np.float64)
        stretch_rest = np.linalg.norm(
            x64[self._stretch_j_np] - x64[self._stretch_i_np], axis=1
        ).astype(np.float32)
        bend_rest = np.linalg.norm(
            x64[self._bend_j_np] - x64[self._bend_i_np], axis=1
        ).astype(np.float32)

        # Compute target: x + disp
        target = (x64 + self._cpu_disp.astype(np.float64)).astype(np.float32)

        # Upload to pre-allocated GPU buffers (wp.copy avoids new allocation)
        wp.copy(self._x_wp, wp.array(self._cpu_x, dtype=wp.vec3, device=self.wp_device))
        wp.copy(self._disp_wp, wp.array(self._cpu_disp, dtype=wp.vec3, device=self.wp_device))
        wp.copy(self._target_wp, wp.array(target, dtype=wp.vec3, device=self.wp_device))
        wp.copy(self._stretch_rest, wp.array(stretch_rest, dtype=wp.float32, device=self.wp_device))
        wp.copy(self._bend_rest, wp.array(bend_rest, dtype=wp.float32, device=self.wp_device))
        wp.copy(self._pin_pos, wp.array(self._cpu_pin, dtype=wp.vec3, device=self.wp_device))

        # 1. Predict: x_pred = x + disp
        wp.launch(
            kernel=_predict_kernel,
            dim=self.total_verts,
            inputs=[self._x_wp, self._disp_wp, self.x_pred],
            device=self.wp_device,
        )

        # Compliance -> alpha_tilde
        stretch_alpha = wp.float32(self.stretch_compliance / (self.dt * self.dt))
        bend_alpha = wp.float32(self.bend_compliance / (self.dt * self.dt))
        target_alpha = wp.float32(self.target_compliance / (self.dt * self.dt))

        # 2. Iterative constraint projection
        for _ in range(self.n_iters):
            # Pin roots first
            wp.launch(
                kernel=_pin_kernel,
                dim=self.n_pins,
                inputs=[self.x_pred, self.pin_indices, self._pin_pos],
                device=self.wp_device,
            )

            # --- Stretch constraints ---
            wp.launch(
                kernel=_zero_corrections_kernel,
                dim=self.total_verts,
                inputs=[self.corrections, self.counts],
                device=self.wp_device,
            )
            wp.launch(
                kernel=_project_distance_kernel,
                dim=self.n_stretch,
                inputs=[
                    self.x_pred,
                    self.stretch_i, self.stretch_j,
                    self._stretch_rest, stretch_alpha,
                    self.corrections, self.counts,
                ],
                device=self.wp_device,
            )
            wp.launch(
                kernel=_apply_corrections_kernel,
                dim=self.total_verts,
                inputs=[self.x_pred, self.corrections, self.counts],
                device=self.wp_device,
            )

            # --- Bending (ghost-edge) constraints ---
            wp.launch(
                kernel=_zero_corrections_kernel,
                dim=self.total_verts,
                inputs=[self.corrections, self.counts],
                device=self.wp_device,
            )
            wp.launch(
                kernel=_project_distance_kernel,
                dim=self.n_bend,
                inputs=[
                    self.x_pred,
                    self.bend_i, self.bend_j,
                    self._bend_rest, bend_alpha,
                    self.corrections, self.counts,
                ],
                device=self.wp_device,
            )
            wp.launch(
                kernel=_apply_corrections_kernel,
                dim=self.total_verts,
                inputs=[self.x_pred, self.corrections, self.counts],
                device=self.wp_device,
            )

            # --- Target position constraints (tube collision response) ---
            wp.launch(
                kernel=_zero_corrections_kernel,
                dim=self.total_verts,
                inputs=[self.corrections, self.counts],
                device=self.wp_device,
            )
            wp.launch(
                kernel=_project_target_kernel,
                dim=self.total_verts,
                inputs=[
                    self.x_pred, self._target_wp, target_alpha,
                    self.corrections, self.counts,
                ],
                device=self.wp_device,
            )
            wp.launch(
                kernel=_apply_corrections_kernel,
                dim=self.total_verts,
                inputs=[self.x_pred, self.corrections, self.counts],
                device=self.wp_device,
            )

        # Final pin
        wp.launch(
            kernel=_pin_kernel,
            dim=self.n_pins,
            inputs=[self.x_pred, self.pin_indices, self._pin_pos],
            device=self.wp_device,
        )

        wp.synchronize()

        # Read back results and split into per-guide arrays
        result_np = self.x_pred.numpy().copy()  # (total_verts, 3) float32

        new_guides = []
        for gi in range(self.n_guides):
            start = gi * L
            new_guides.append(result_np[start:start + L].astype(np.float32))

        return new_guides
