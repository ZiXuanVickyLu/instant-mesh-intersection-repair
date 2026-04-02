"""
Iterative tube expansion with CCD clamping, visualized in Polyscope.

Loads real tube data, then expands from tip_scale=0.1 outward in steps
of 0.05 (no upper limit). Each step runs BVH + ACCD to clamp collisions.
A "Step" button in Polyscope advances one iteration.

Usage:
    python scripts/test_iterative_expansion.py
    python scripts/test_iterative_expansion.py --max-tubes 50
    python scripts/test_iterative_expansion.py --step-size 0.1
    python scripts/test_iterative_expansion.py --synthetic   # use synthetic data
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hair_tube_relaxation.ccd_tube_expansion import TubeExpander


def make_circle_cross_section(N, radius=0.02):
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)


def make_guide(origin, direction, length, L):
    t = np.linspace(0, length, L)
    d = direction / np.linalg.norm(direction)
    return origin[None, :] + t[:, None] * d[None, :]


def build_synthetic_tube(guide, N, radius, tip_scale):
    L = len(guide)
    cross_2d = make_circle_cross_section(N, radius)
    verts = np.zeros((L * N + 1, 3), dtype=np.float64)

    tangent = guide[-1] - guide[0]
    tangent /= np.linalg.norm(tangent)
    up = np.array([1.0, 0.0, 0.0]) if abs(tangent[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    normal = np.cross(tangent, up)
    normal /= np.linalg.norm(normal)
    bitangent = np.cross(tangent, normal)

    for li in range(L):
        t = li / (L - 1) if L > 1 else 0.0
        scale = 1.0 + t * (tip_scale - 1.0)
        center = guide[li]
        for ni in range(N):
            u, v = cross_2d[ni]
            verts[li * N + ni] = center + scale * (u * bitangent + v * normal)

    verts[L * N] = guide[-1]

    faces = []
    for li in range(L - 1):
        for ni in range(N):
            ni_next = (ni + 1) % N
            a, b = li * N + ni, li * N + ni_next
            c, d = (li + 1) * N + ni, (li + 1) * N + ni_next
            faces.append([a, c, b])
            faces.append([b, c, d])
    tip_idx = L * N
    last_layer = (L - 1) * N
    for ni in range(N):
        faces.append([last_layer + ni, tip_idx, last_layer + (ni + 1) % N])

    return verts, np.array(faces, dtype=np.int32)


def load_real_tubes(max_tubes=None):
    """Load real tube meshes using the full voronoi pipeline."""
    from hair_tube_relaxation.standalone_voronoi_tube import (
        build_tube_meshes, load_obj_with_uv, triangulate_faces,
        load_guides_obj_grouped, compute_voronoi_labels,
        fix_penetrating_guides, extract_boundary_segments,
        extract_mesh_boundary_edges, build_cell_boundary_polylines,
    )
    import toml

    project_root = os.path.join(os.path.dirname(__file__), "..")
    config_path = os.path.join(project_root, "config", "conf_00352.toml")
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        return None, None

    cfg = toml.load(config_path)
    data_base = os.path.join(project_root, cfg["data"]["base_path"])
    output_base = os.path.join(project_root, cfg["output"]["base_path"])

    # Load scalp mesh
    scalp_path = os.path.join(data_base, cfg["scalp"]["scalp_mesh"])
    print(f"  Loading scalp: {scalp_path}")
    scalp_data = load_obj_with_uv(scalp_path)
    scalp_verts = scalp_data["vertices"]
    scalp_faces = triangulate_faces(scalp_data["faces_v"])

    # Load guide strands
    guides_obj = os.path.join(output_base, "grown_strands_segmented.obj")
    print(f"  Loading guides: {guides_obj}")
    guides, guide_group_ids = load_guides_obj_grouped(guides_obj)
    roots = np.array([g[0] for g in guides], dtype=np.float64)

    # Voronoi segmentation
    print(f"  Computing Voronoi ({len(guides)} guides)...")
    vert_labels = compute_voronoi_labels(scalp_verts, roots)

    # Penetration fix
    body_mesh_path = os.path.join(data_base, "TSR/body_clean.obj")
    thickness = float(cfg.get("strand", {}).get("penetration_thickness", 0.001))
    fix_penetrating_guides(
        guides, body_mesh_path, scalp_verts, scalp_faces, vert_labels,
        thickness=thickness,
    )
    roots = np.array([g[0] for g in guides], dtype=np.float64)
    vert_labels = compute_voronoi_labels(scalp_verts, roots)

    # Boundary extraction
    print("  Extracting boundaries...")
    boundary_segments, segment_cells = extract_boundary_segments(
        scalp_verts, scalp_faces, vert_labels, roots,
    )
    mesh_boundary = extract_mesh_boundary_edges(scalp_verts, scalp_faces)
    cell_polylines = build_cell_boundary_polylines(
        boundary_segments, segment_cells, len(roots),
        mesh_boundary=mesh_boundary,
        vert_labels=vert_labels,
        vertices=scalp_verts,
        faces=scalp_faces,
    )

    # Build tubes
    print("  Building tube meshes...")
    tube_meshes = build_tube_meshes(
        guides, cell_polylines, scalp_verts, scalp_faces,
        tip_scale=0.1, resample_n=0,
    )

    # Filter and limit
    filtered_meshes = []
    filtered_guides = []
    for i, m in enumerate(tube_meshes):
        if m is not None:
            filtered_meshes.append(m)
            filtered_guides.append(guides[i])
            if max_tubes and len(filtered_meshes) >= max_tubes:
                break

    # Boundary mesh paths
    body_mesh_path = os.path.join(data_base, "TSR/body_clean.obj")
    wrapper_mesh_path = os.path.join(output_base, "boundary_wrap.obj")

    print(f"  {len(filtered_meshes)} tubes ready")
    return filtered_meshes, filtered_guides, body_mesh_path, wrapper_mesh_path


class IterativeExpander:
    """Manages iterative expansion with Polyscope visualization."""

    def __init__(self, tube_meshes, guides, initial_scale=0.1,
                 step_size=0.003, max_collisions=128, xi=1e-4,
                 smooth_iters=20, smooth_alpha=0.5,
                 guide_smooth_iters=20, guide_smooth_alpha=0.5,
                 guide_penalty_threshold=0.01, guide_penalty_strength=1.0,
                 spline_smoothing=0.0,
                 xpbd_iters=15, xpbd_stretch_stiffness=1e5,
                 xpbd_bend_stiffness=1e5, xpbd_target_stiffness=1e0,
                 body_mesh_path=None, wrapper_mesh_path=None,
                 device="cuda:0"):
        self.guides = guides
        self.initial_scale = initial_scale
        self.step_size = step_size
        self.max_collisions = max_collisions
        self.xi = xi
        self.smooth_iters = smooth_iters
        self.smooth_alpha = smooth_alpha
        self.guide_smooth_iters = guide_smooth_iters
        self.guide_smooth_alpha = guide_smooth_alpha
        self.guide_penalty_threshold = guide_penalty_threshold
        self.guide_penalty_strength = guide_penalty_strength
        self.spline_smoothing = spline_smoothing
        self.xpbd_iters = xpbd_iters
        self.xpbd_stretch_stiffness = xpbd_stretch_stiffness
        self.xpbd_bend_stiffness = xpbd_bend_stiffness
        self.xpbd_target_stiffness = xpbd_target_stiffness

        # Load boundary meshes
        self.body_mesh = self._load_boundary_mesh(body_mesh_path, "body")
        self.wrapper_mesh = self._load_boundary_mesh(wrapper_mesh_path, "wrapper")
        self.device = device

        self.step_count = 0

        # Store the full-scale boundary positions (scale=1.0) for each tube.
        # Computed once from the initial meshes.
        self.n_tubes = len(tube_meshes)
        self.v_boundary = []  # per-tube: (n_verts, 3)
        self.n_cross_list = []
        self.n_layers_list = []

        for ti, (verts, faces) in enumerate(tube_meshes):
            guide = guides[ti]
            # Detect N
            n_verts = verts.shape[0]
            n_faces = faces.shape[0]
            N = 2 * (n_verts - 1) - n_faces
            L = (n_verts - 1) // N
            self.n_cross_list.append(N)
            self.n_layers_list.append(L)

            # Compute boundary (scale=1.0) from current positions.
            # The generated tube has per-layer scale:
            #   layer 0: scale=1.0, layers 1..L-1: scale = 1 + (li/(L-1))*(tip_scale-1)
            v_bnd = verts.copy().astype(np.float64)
            # Layer 0 is already at scale=1.0, v_bnd = verts (no change)
            for li in range(1, L):
                t = li / (L - 1)
                gen_scale = 1.0 + t * (initial_scale - 1.0)
                center = guide[li]
                for ni in range(N):
                    idx = li * N + ni
                    offset = verts[idx] - center
                    v_bnd[idx] = center + offset / gen_scale
            # Tip vertex boundary = guide[-1] (always)
            self.v_boundary.append(v_bnd)

        # Recompute initial meshes: layer 0 stays at full scale (pinned to
        # scalp), layers 1..L-1 start at uniform initial_scale
        self.current_meshes = []
        for ti, (verts, faces) in enumerate(tube_meshes):
            guide = guides[ti]
            v_bnd = self.v_boundary[ti]
            L = self.n_layers_list[ti]
            N = self.n_cross_list[ti]

            v_init = v_bnd.copy()
            # Layer 0: keep at boundary (scale=1.0) — pinned on scalp
            # Layers 1..L-1: scale to initial_scale
            for li in range(1, L):
                center = guide[li]
                for ni in range(N):
                    idx = li * N + ni
                    v_init[idx] = center + initial_scale * (v_bnd[idx] - center)
            # Tip stays at guide[-1]
            v_init[L * N] = guide[-1]
            self.current_meshes.append((v_init, faces.copy()))

        # Store original root positions for pinning layer 0
        self.layer0_root = []
        for ti in range(self.n_tubes):
            guide = guides[ti]
            g0 = guide[0].astype(np.float64)
            self.layer0_root.append(g0.copy())

        # Cache a TubeExpander with topology + structural tensors (edges, faces, etc.)
        # Only v_start/v_end change each step — everything else is reused.
        from hair_tube_relaxation.ccd_tube_expansion import TubeExpander
        self._cached_expander = TubeExpander(
            tube_meshes=self.current_meshes,
            guides=self.guides,
            current_scale=0.0, target_scale=1.0,
            device=device,
        )
        self._cached_expander.parse_topology()
        # Build initial expansion data (will be overwritten each step)
        self._build_expansion_data(
            self._cached_expander, self.current_meshes, self.current_meshes)

        # XPBD guide solver (topology cached, reused across steps)
        from hair_tube_relaxation.xpbd_guide_solver import XPBDGuideSolver
        n_layers = self.n_layers_list[0]  # all guides have same L
        self._xpbd_solver = XPBDGuideSolver(
            n_guides=self.n_tubes,
            n_layers=n_layers,
            device=device,
            n_iters=self.xpbd_iters,
            stretch_stiffness=self.xpbd_stretch_stiffness,
            bend_stiffness=self.xpbd_bend_stiffness,
            target_stiffness=self.xpbd_target_stiffness,
        )

        # Cumulative expansion distance tracked for display
        self.cumulative_expand = 0.0

        # Store per-step t_vertex for visualization
        self.last_t_vertex = None

        # History for replay: list of (meshes, guides, t_vertex) per step
        self.history = []
        # Save initial state as step 0
        self.history.append((
            [(v.copy(), f.copy()) for v, f in self.current_meshes],
            [g.copy() for g in self.guides],
            None,  # no t_vertex for initial state
        ))

    @staticmethod
    def _load_boundary_mesh(path, name):
        """Load a boundary mesh OBJ, return (verts, faces) or None."""
        if path is None or not os.path.exists(path):
            if path is not None:
                print(f"  Warning: {name} mesh not found: {path}")
            return None
        import trimesh
        mesh = trimesh.load(path, process=False)
        verts = np.array(mesh.vertices, dtype=np.float32)
        faces = np.array(mesh.faces, dtype=np.int64)
        print(f"  Loaded {name} mesh: {verts.shape[0]} verts, {faces.shape[0]} faces")
        return (verts, faces)

    def compute_target_meshes(self, step_dist):
        """Compute target positions by moving each non-pinned vertex outward
        by step_dist (absolute distance) from the CURRENT position along
        its radial direction from the guide center."""
        target_meshes = []
        for ti in range(self.n_tubes):
            guide = self.guides[ti]
            L = self.n_layers_list[ti]
            N = self.n_cross_list[ti]
            v_cur, faces = self.current_meshes[ti]

            v_end = v_cur.copy().astype(np.float64)
            for li in range(1, L):  # skip layer 0 (pinned)
                center = guide[li].astype(np.float64)
                for ni in range(N):
                    idx = li * N + ni
                    offset = v_end[idx] - center
                    dist = np.linalg.norm(offset)
                    if dist > 1e-12:
                        v_end[idx] = v_end[idx] + step_dist * (offset / dist)
            # Layer 0 and tip: stay pinned
            target_meshes.append((v_end, faces))
        return target_meshes

    def _build_expansion_data(self, expander, start_meshes, end_meshes):
        """Build ExpansionData from start/end mesh lists."""
        from hair_tube_relaxation.ccd_tube_expansion import ExpansionData
        all_v_start, all_v_end, all_pinned = [], [], []
        all_edges, all_edge_tube_id = [], []
        vert_offsets, edge_offsets = [0], [0]
        v_off = 0

        for ti in range(expander.n_tubes):
            v_s, _ = start_meshes[ti]
            v_e, _ = end_meshes[ti]
            topo = expander.topologies[ti]

            all_v_start.append(v_s.astype(np.float32))
            all_v_end.append(v_e.astype(np.float32))
            all_pinned.append(topo.pinned_mask)

            edges_local = np.concatenate([
                topo.connection_edges, topo.ring_edges, topo.tip_edges,
            ], axis=0)
            all_edges.append(edges_local + v_off)
            n_e = edges_local.shape[0]
            all_edge_tube_id.append(np.full(n_e, ti, dtype=np.int32))
            v_off += topo.n_verts
            vert_offsets.append(v_off)
            edge_offsets.append(edge_offsets[-1] + n_e)

        expander.expansion = ExpansionData(
            v_start=torch.from_numpy(np.concatenate(all_v_start)).to(self.device),
            v_end=torch.from_numpy(np.concatenate(all_v_end)).to(self.device),
            pinned=torch.from_numpy(np.concatenate(all_pinned)).to(self.device),
            all_edges=torch.from_numpy(np.concatenate(all_edges)).long().to(self.device),
            edge_tube_id=torch.from_numpy(np.concatenate(all_edge_tube_id)).to(self.device),
            vert_offsets=vert_offsets,
            edge_offsets=edge_offsets,
        )
        return vert_offsets

    def _run_ccd_pass(self, start_meshes, end_meshes, label="CCD"):
        """Update cached expander positions, run CCD, return (t_vertex, vert_offsets, expander)."""
        expander = self._cached_expander
        # Only update v_start/v_end — topology, edges, pinned mask are cached
        expander.update_positions(start_meshes, end_meshes)

        vert_offsets = expander.expansion.vert_offsets

        print(f"  {label}:")
        t_vertex = expander.run_ccd(
            max_collisions=self.max_collisions,
            xi=self.xi,
            body_mesh=self.body_mesh,
            wrapper_mesh=self.wrapper_mesh,
            smooth_iters=self.smooth_iters,
            smooth_alpha=self.smooth_alpha,
        )

        n_clamped = (t_vertex < 1.0).sum().item()
        n_total = t_vertex.shape[0]
        print(f"    Clamped: {n_clamped}/{n_total} vertices")
        if n_clamped > 0:
            t_c = t_vertex[t_vertex < 1.0]
            print(f"    t_min={t_c.min().item():.4f}, "
                  f"t_mean={t_c.mean().item():.4f}")

        return t_vertex, vert_offsets, expander

    def step(self):
        """Run one expansion step.

        Order:
          1. Radial expansion + CCD → expand tubes, clamp collisions
          2. Guide displacement from clamping → negated blocked displacement
             pushes guides AWAY from collision side, smoothed along guide
          3. Guide strand CCD → prevent guides from crossing
          4. Tube snap → tubes follow guides with CCD safety
          5. Re-match guides → convex combination weights from layer 0
        """
        from hair_tube_relaxation.ccd_tube_expansion import (
            TubeExpander, ExpansionData, GuideDeformer,
        )

        self.cumulative_expand += self.step_size

        print(f"\n--- Step {self.step_count + 1}: "
              f"total expand {self.cumulative_expand:.4f} ---")

        # === Phase 1: Radial expansion + CCD ===
        target_meshes = self.compute_target_meshes(self.step_size)

        t_vertex, vert_offsets, expander = self._run_ccd_pass(
            self.current_meshes, target_meshes, label="Expansion")

        self.last_t_vertex = t_vertex
        clamped_meshes = expander.get_expanded_meshes(t_vertex)

        n_clamped = (t_vertex < 1.0).sum().item()

        # === Phase 2: Guide deformation from tube clamping (XPBD) ===
        if n_clamped > 0:
            deformer = GuideDeformer(
                smooth_iters=self.guide_smooth_iters,
                smooth_alpha=self.guide_smooth_alpha,
            )

            # Compute guide displacement from negated blocked tube displacement
            layer_disps = deformer.compute_layer_displacements(
                guides=self.guides,
                clamped_meshes=clamped_meshes,
                target_meshes=target_meshes,
                t_vertex=t_vertex,
                vert_offsets=vert_offsets,
                n_cross_list=self.n_cross_list,
                n_layers_list=self.n_layers_list,
            )

            total_disp = sum(np.linalg.norm(d, axis=1).sum() for d in layer_disps)
            print(f"  Guide displacement total: {total_disp:.6f}")

            if total_disp > 1e-10:
                old_guides = [g.copy() for g in self.guides]

                # XPBD solve: predict from displacement, project stretch + bending
                proposed_guides = self._xpbd_solver.solve(
                    self.guides, layer_disps,
                )

                # Guide strand CCD: prevent guides from crossing
                clamped_guides = GuideDeformer.guide_strand_ccd(
                    old_guides, proposed_guides,
                    device=self.device,
                    max_collisions=self.max_collisions,
                )

                # Guide boundary CCD: prevent guides from crossing boundary meshes
                if self.body_mesh is not None:
                    bv, bf = self.body_mesh
                    clamped_guides = GuideDeformer.guide_boundary_ccd(
                        old_guides, clamped_guides,
                        bv, bf, device=self.device, xi=self.xi,
                    )
                if self.wrapper_mesh is not None:
                    wv, wf = self.wrapper_mesh
                    clamped_guides = GuideDeformer.guide_boundary_ccd(
                        old_guides, clamped_guides,
                        wv, wf, device=self.device, xi=self.xi,
                    )

                max_guide_disp = max(
                    np.linalg.norm(clamped_guides[ti] - old_guides[ti], axis=1).max()
                    for ti in range(len(self.guides))
                )
                print(f"  Guide max displacement: {max_guide_disp:.6f}")

                # Tube snap: tubes follow guides with CCD safety
                snap_targets = GuideDeformer.compute_guide_following_positions(
                    clamped_meshes, old_guides, clamped_guides,
                    self.n_cross_list, self.n_layers_list,
                )

                snap_t, _, snap_exp = self._run_ccd_pass(
                    clamped_meshes, snap_targets, label="Tube snap")

                self.current_meshes = snap_exp.get_expanded_meshes(snap_t)

                # Re-match guides as ring centroids (pinned root)
                self.guides = GuideDeformer.rematch_guides_to_tubes(
                    self.current_meshes,
                    self.n_cross_list, self.n_layers_list,
                    layer0_root=self.layer0_root,
                )

                # Spline smooth guides to remove zigzag, then snap tubes
                old_guides_pre = [g.copy() for g in self.guides]
                smoothed = GuideDeformer.smooth_guides_spline(
                    self.guides, knot_spacing_factor=self.spline_smoothing,
                    pin_root=True,
                )
                n_spline_skipped = 0
                for ti in range(len(smoothed)):
                    if np.any(np.isnan(smoothed[ti])) or np.any(np.isinf(smoothed[ti])):
                        smoothed[ti] = old_guides_pre[ti]
                        n_spline_skipped += 1
                spline_devs = []
                for ti in range(len(smoothed)):
                    dev = np.linalg.norm(
                        smoothed[ti].astype(np.float64) - old_guides_pre[ti].astype(np.float64),
                        axis=1,
                    )
                    spline_devs.append(dev.max())
                max_spline_dev = max(spline_devs) if spline_devs else 0.0
                avg_spline_dev = np.mean(spline_devs) if spline_devs else 0.0
                print(f"  Spline smooth: max_dev={max_spline_dev:.6f}, "
                      f"avg_dev={avg_spline_dev:.6f}, "
                      f"skipped={n_spline_skipped}/{len(smoothed)}")
                self.guides = smoothed
                # Update tube positions to follow smoothed guides
                self.current_meshes = GuideDeformer.update_tubes_from_guides(
                    self.current_meshes, old_guides_pre, self.guides,
                    self.n_cross_list, self.n_layers_list,
                )

                # Final mesh NaN guard
                for ti in range(len(self.current_meshes)):
                    v, f = self.current_meshes[ti]
                    if np.any(np.isnan(v)) or np.any(np.isinf(v)):
                        # Revert this tube to pre-snap state
                        self.guides[ti] = old_guides[ti]
                        self.current_meshes[ti] = (
                            snap_exp.get_expanded_meshes(snap_t)[ti]
                        )
            else:
                self.current_meshes = clamped_meshes
        else:
            self.current_meshes = clamped_meshes

        self.step_count += 1

        # Save state for replay
        self.history.append((
            [(v.copy(), f.copy()) for v, f in self.current_meshes],
            [g.copy() for g in self.guides],
            self.last_t_vertex.cpu().clone() if self.last_t_vertex is not None else None,
        ))

        return True


def main():
    parser = argparse.ArgumentParser(description="Iterative tube expansion")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic tubes instead of real data")
    parser.add_argument("--max-tubes", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=0.001)
    parser.add_argument("--max-collisions", type=int, default=64,
                        help="BVH per-triangle collision buffer size")
    parser.add_argument("--xi", type=float, default=1e-3,
                        help="ACCD thickness / minimum separation")
    parser.add_argument("--smooth-iters", type=int, default=20,
                        help="Laplacian smoothing iterations for t values (0=disable)")
    parser.add_argument("--smooth-alpha", type=float, default=0.5,
                        help="Smoothing blend weight (0=none, 1=full neighbor avg)")
    parser.add_argument("--guide-smooth-iters", type=int, default=40,
                        help="Laplacian smoothing iterations for guide displacement (0=disable)")
    parser.add_argument("--guide-smooth-alpha", type=float, default=0.3,
                        help="Guide smoothing blend weight (0=none, 1=full neighbor avg)")
    parser.add_argument("--guide-penalty-threshold", type=float, default=0.01,
                        help="Distance below which guide penalty force activates")
    parser.add_argument("--guide-penalty-strength", type=float, default=1.0,
                        help="Multiplier on guide penalty displacement")
    parser.add_argument("--spline-smoothing", type=float, default=4.0,
                        help="Knot spacing factor: knots placed every factor*median_edge arc length (higher=more smooth)")
    parser.add_argument("--xpbd-iters", type=int, default=15,
                        help="XPBD solver iterations for guide deformation")
    parser.add_argument("--xpbd-stretch-stiffness", type=float, default=1e3,
                        help="XPBD stretch stiffness (higher=stiffer)")
    parser.add_argument("--xpbd-bend-stiffness", type=float, default=1e5,
                        help="XPBD bending stiffness (higher=stiffer)")
    parser.add_argument("--xpbd-target-stiffness", type=float, default=1e4,
                        help="XPBD target position stiffness (tube collision response, higher=stronger pull)")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    body_mesh_path = None
    wrapper_mesh_path = None

    if args.synthetic:
        print("Generating synthetic tubes...")
        L, N = 12, 8
        tip_scale = 0.1
        radius = 0.025
        spacing = 0.04
        origins = [
            np.array([0, 0, 0.0]),
            np.array([spacing, 0, 0.0]),
            np.array([0, spacing, 0.0]),
            np.array([spacing, spacing, 0.0]),
        ]
        guides = [make_guide(o, np.array([0, 0, 1.0]), 0.5, L) for o in origins]
        tube_meshes = [build_synthetic_tube(g, N, radius, tip_scale) for g in guides]
    else:
        print("Loading real tube data...")
        result = load_real_tubes(max_tubes=args.max_tubes)
        if result[0] is None:
            print("Failed to load real data, falling back to synthetic")
            return
        tube_meshes, guides, body_mesh_path, wrapper_mesh_path = result

    print(f"Loaded {len(tube_meshes)} tubes on {device}")

    # Create iterative expander
    expander = IterativeExpander(
        tube_meshes, guides,
        initial_scale=0.1,
        step_size=args.step_size,
        max_collisions=args.max_collisions,
        xi=args.xi,
        smooth_iters=args.smooth_iters,
        smooth_alpha=args.smooth_alpha,
        guide_smooth_iters=args.guide_smooth_iters,
        guide_smooth_alpha=args.guide_smooth_alpha,
        guide_penalty_threshold=args.guide_penalty_threshold,
        guide_penalty_strength=args.guide_penalty_strength,
        spline_smoothing=args.spline_smoothing,
        xpbd_iters=args.xpbd_iters,
        xpbd_stretch_stiffness=args.xpbd_stretch_stiffness,
        xpbd_bend_stiffness=args.xpbd_bend_stiffness,
        xpbd_target_stiffness=args.xpbd_target_stiffness,
        body_mesh_path=body_mesh_path,
        wrapper_mesh_path=wrapper_mesh_path,
        device=device,
    )

    # --- Polyscope setup ---
    import polyscope as ps
    import polyscope.imgui as psim

    ps.init()
    ps.set_up_dir("y_up")
    ps.set_ground_plane_mode("none")

    # Register boundary meshes once (they don't change)
    def register_boundary_meshes():
        if expander.body_mesh is not None:
            bv, bf = expander.body_mesh
            m = ps.register_surface_mesh("body_mesh", bv, bf)
            m.set_color((0.8, 0.4, 0.4))
            m.set_transparency(0.5)
        if expander.wrapper_mesh is not None:
            wv, wf = expander.wrapper_mesh
            m = ps.register_surface_mesh("wrapper_mesh", wv, wf)
            m.set_color((0.4, 0.4, 0.8))
            m.set_transparency(0.5)

    def update_vis(meshes, guides, t_vertex):
        """Re-register tube meshes in Polyscope from given state."""
        for ti in range(expander.n_tubes):
            if ps.has_surface_mesh(f"tube_{ti}"):
                ps.remove_surface_mesh(f"tube_{ti}")
            if ps.has_curve_network(f"guide_{ti}"):
                ps.remove_curve_network(f"guide_{ti}")

        for ti in range(expander.n_tubes):
            verts, faces = meshes[ti]
            name = f"tube_{ti}"
            mesh = ps.register_surface_mesh(name, verts, faces)
            mesh.set_color((0.2, 0.8, 0.2))
            if vis_state["show_guides_only"]:
                mesh.set_enabled(False)

            if t_vertex is not None:
                off = 0
                for j in range(ti):
                    off += meshes[j][0].shape[0]
                n_v = verts.shape[0]
                t_np = t_vertex[off:off + n_v]
                if isinstance(t_np, torch.Tensor):
                    t_np = t_np.cpu().numpy()
                mesh.add_scalar_quantity("t_vertex", t_np, enabled=True,
                                        vminmax=(0.0, 1.0), cmap="coolwarm")

        for ti in range(expander.n_tubes):
            guide = guides[ti]
            L = len(guide)
            edges = np.array([[i, i + 1] for i in range(L - 1)])
            net = ps.register_curve_network(f"guide_{ti}", guide, edges)
            net.set_color((1.0, 1.0, 0.0))
            net.set_radius(0.0005)

    # Mutable state for the slider
    vis_state = {"view_step": 0, "show_guides_only": False}

    def callback():
        n_history = len(expander.history)

        psim.TextUnformatted(f"Simulated: {expander.step_count} steps  "
                             f"(+{expander.step_size:.4f}/step)")

        # Step buttons
        if psim.Button("Step"):
            expander.step()
            vis_state["view_step"] = expander.step_count
            meshes, guides, t_v = expander.history[-1]
            update_vis(meshes, guides, t_v)

        psim.SameLine()
        if psim.Button("Step 10"):
            for _ in range(10):
                expander.step()
            vis_state["view_step"] = expander.step_count
            meshes, guides, t_v = expander.history[-1]
            update_vis(meshes, guides, t_v)

        psim.SameLine()
        if psim.Button("Show Guides" if not vis_state["show_guides_only"]
                        else "Show Tubes"):
            vis_state["show_guides_only"] = not vis_state["show_guides_only"]
            for ti in range(expander.n_tubes):
                if ps.has_surface_mesh(f"tube_{ti}"):
                    ps.get_surface_mesh(f"tube_{ti}").set_enabled(
                        not vis_state["show_guides_only"])

        # Replay slider
        if n_history > 1:
            psim.Separator()
            changed, new_val = psim.SliderInt(
                "Replay", vis_state["view_step"], 0, n_history - 1)
            if changed:
                vis_state["view_step"] = new_val
                meshes, guides, t_v = expander.history[new_val]
                update_vis(meshes, guides, t_v)

            psim.TextUnformatted(f"Viewing step {vis_state['view_step']}")

    # Initial visualization
    register_boundary_meshes()
    meshes0, guides0, t0 = expander.history[0]
    update_vis(meshes0, guides0, t0)
    ps.set_user_callback(callback)
    ps.show()


if __name__ == "__main__":
    main()
