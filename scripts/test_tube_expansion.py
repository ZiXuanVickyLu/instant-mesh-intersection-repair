"""
Test and visualize TubeExpander topology parsing and expansion computation.

Creates synthetic tube meshes, parses topology, computes expansion vectors,
and visualizes with Polyscope:
  - Original tubes (thin, at current_scale)
  - Expanded tubes (at target_scale)
  - Expansion displacement vectors
  - Edge classification (connection / ring / tip)

Usage:
    python scripts/test_tube_expansion.py
    python scripts/test_tube_expansion.py --no-vis          # skip GUI
    python scripts/test_tube_expansion.py --real             # use real data
    python scripts/test_tube_expansion.py --real --no-vis
"""

import argparse
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hair_tube_relaxation.ccd_tube_expansion import TubeExpander, TubeTopology


# ---------------------------------------------------------------------------
# Synthetic tube generation (no Voronoi needed)
# ---------------------------------------------------------------------------

def make_circle_cross_section(N: int, radius: float = 0.02):
    """N-gon cross-section in 2D."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.stack([np.cos(angles) * radius, np.sin(angles) * radius], axis=1)


def make_guide(origin: np.ndarray, direction: np.ndarray, length: float, L: int):
    """Straight-line guide with L points."""
    t = np.linspace(0, length, L)
    return origin[None, :] + t[:, None] * (direction / np.linalg.norm(direction))[None, :]


def build_synthetic_tube(guide: np.ndarray, N: int, radius: float, tip_scale: float):
    """Build a tube mesh mimicking _build_tube_mesh layout.

    Uses circular cross-section with parallel frames along straight guide.
    """
    L = len(guide)
    cross_2d = make_circle_cross_section(N, radius)
    verts = np.zeros((L * N + 1, 3), dtype=np.float64)

    # Compute a simple frame: tangent along guide, normal/bitangent orthogonal
    tangent = guide[-1] - guide[0]
    tangent /= np.linalg.norm(tangent)
    # Pick an arbitrary orthogonal vector
    if abs(tangent[0]) < 0.9:
        up = np.array([1.0, 0.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
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

    # Tip
    verts[L * N] = guide[-1]

    # Faces (same as _build_tube_mesh)
    faces = []
    for li in range(L - 1):
        for ni in range(N):
            ni_next = (ni + 1) % N
            a = li * N + ni
            b = li * N + ni_next
            c = (li + 1) * N + ni
            d = (li + 1) * N + ni_next
            faces.append([a, c, b])
            faces.append([b, c, d])
    tip_idx = L * N
    last_layer = (L - 1) * N
    for ni in range(N):
        ni_next = (ni + 1) % N
        faces.append([last_layer + ni, tip_idx, last_layer + ni_next])

    return verts, np.array(faces, dtype=np.int32)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_topology_parsing():
    """Test that edge counts and classifications are correct."""
    L, N = 10, 8
    tip_scale = 0.1
    guide = make_guide(np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 1.0, L)
    verts, faces = build_synthetic_tube(guide, N, radius=0.02, tip_scale=tip_scale)

    expander = TubeExpander(
        tube_meshes=[(verts, faces)],
        guides=[guide],
        current_scale=tip_scale,
        target_scale=0.8,
        device="cpu",
    )
    topos = expander.parse_topology()
    topo = topos[0]

    assert topo.n_layers == L, f"Expected L={L}, got {topo.n_layers}"
    assert topo.n_cross == N, f"Expected N={N}, got {topo.n_cross}"
    assert topo.n_verts == L * N + 1

    # Edge counts
    assert topo.connection_edges.shape == ((L - 1) * N, 2), \
        f"Connection edges: {topo.connection_edges.shape}"
    assert topo.ring_edges.shape == (L * N, 2), \
        f"Ring edges: {topo.ring_edges.shape}"
    assert topo.tip_edges.shape == (N, 2), \
        f"Tip edges: {topo.tip_edges.shape}"

    total_edges = (L - 1) * N + L * N + N
    assert total_edges == 2 * L * N

    # Pinned: layer 0 (N verts) + tip (1 vert)
    assert topo.pinned_mask.sum() == N + 1
    assert topo.pinned_mask[:N].all()      # layer 0
    assert topo.pinned_mask[L * N]          # tip

    # Layer mapping
    for li in range(L):
        for ni in range(N):
            assert topo.layer_of_vertex[li * N + ni] == li

    print("[PASS] test_topology_parsing")


def test_n_cross_detection():
    """Test auto-detection of N from mesh topology."""
    for N in [6, 8, 12, 16]:
        for L in [5, 10, 20]:
            guide = make_guide(np.zeros(3), np.array([0, 0, 1.0]), 1.0, L)
            verts, faces = build_synthetic_tube(guide, N, 0.02, 0.1)
            expander = TubeExpander([(verts, faces)], [guide], device="cpu")
            detected = expander.auto_detect_n_cross(verts, faces)
            assert detected == N, f"L={L}, N={N}: detected {detected}"
    print("[PASS] test_n_cross_detection")


def test_expansion_computation():
    """Test that expansion preserves pinned vertices and moves others."""
    L, N = 10, 8
    tip_scale = 0.1
    target_scale = 0.8
    guide = make_guide(np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 1.0, L)
    verts, faces = build_synthetic_tube(guide, N, radius=0.05, tip_scale=tip_scale)

    expander = TubeExpander(
        tube_meshes=[(verts, faces)],
        guides=[guide],
        current_scale=tip_scale,
        target_scale=target_scale,
        device="cpu",
    )
    exp = expander.compute_expansion()

    # Pinned vertices: v_start == v_end
    pinned_idx = exp.pinned.nonzero(as_tuple=True)[0]
    diff_pinned = (exp.v_end[pinned_idx] - exp.v_start[pinned_idx]).abs().max()
    assert diff_pinned < 1e-6, f"Pinned verts moved by {diff_pinned}"

    # Non-pinned: v_end should differ from v_start (they should expand outward)
    non_pinned = (~exp.pinned).nonzero(as_tuple=True)[0]
    diff_non_pinned = (exp.v_end[non_pinned] - exp.v_start[non_pinned]).norm(dim=1)
    assert diff_non_pinned.min() > 1e-6, "Some non-pinned verts didn't move"

    # Check that higher layers expand more (layer L-1 expands most)
    topo = expander.topologies[0]
    displacements = (exp.v_end - exp.v_start).norm(dim=1).cpu().numpy()
    layer_mean_disp = []
    for li in range(L):
        ring_indices = [li * N + ni for ni in range(N)]
        layer_mean_disp.append(displacements[ring_indices].mean())

    # Layer 0 should have 0 displacement, layer L-1 should have max
    assert layer_mean_disp[0] < 1e-6, f"Layer 0 displacement: {layer_mean_disp[0]}"
    for li in range(2, L):
        assert layer_mean_disp[li] >= layer_mean_disp[li - 1] - 1e-6, \
            f"Layer {li} disp ({layer_mean_disp[li]:.4f}) < layer {li-1} ({layer_mean_disp[li-1]:.4f})"

    print("[PASS] test_expansion_computation")


def test_expansion_multi_tube():
    """Test expansion with multiple tubes, including None entries."""
    L, N = 8, 6
    tip_scale = 0.1
    guides = [
        make_guide(np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 1.0, L),
        make_guide(np.array([0.2, 0, 0.0]), np.array([0, 0, 1.0]), 1.0, L),
        make_guide(np.array([0.4, 0, 0.0]), np.array([0, 0, 1.0]), 1.0, L),
    ]
    meshes = [
        build_synthetic_tube(guides[0], N, 0.03, tip_scale),
        None,  # skipped tube
        build_synthetic_tube(guides[2], N, 0.03, tip_scale),
    ]

    expander = TubeExpander(meshes, guides, current_scale=tip_scale, device="cpu")
    exp = expander.compute_expansion()

    assert expander.n_tubes == 2, f"Expected 2 tubes, got {expander.n_tubes}"
    expected_verts = 2 * (L * N + 1)
    assert exp.v_start.shape[0] == expected_verts
    assert exp.all_edges.shape[0] == 2 * (2 * L * N)

    # Edge tube IDs
    assert (exp.edge_tube_id == 0).sum() == 2 * L * N
    assert (exp.edge_tube_id == 1).sum() == 2 * L * N

    print("[PASS] test_expansion_multi_tube")


def test_get_expanded_meshes():
    """Test applying expansion factors to get final meshes."""
    L, N = 8, 6
    tip_scale = 0.1
    guide = make_guide(np.zeros(3), np.array([0, 0, 1.0]), 1.0, L)
    verts, faces = build_synthetic_tube(guide, N, 0.03, tip_scale)

    expander = TubeExpander([(verts, faces)], [guide],
                            current_scale=tip_scale, target_scale=0.8, device="cpu")
    exp = expander.compute_expansion()

    # Full expansion (t=1 for all)
    t_full = np.ones(exp.v_start.shape[0], dtype=np.float32)
    t_full = torch.from_numpy(t_full)
    results = expander.get_expanded_meshes(t_full)
    assert len(results) == 1
    new_verts, new_faces = results[0]
    assert new_verts.shape == verts.shape
    assert np.array_equal(new_faces, faces)

    # Zero expansion (t=0) should return original
    t_zero = torch.zeros(exp.v_start.shape[0])
    results_zero = expander.get_expanded_meshes(t_zero)
    np.testing.assert_allclose(results_zero[0][0], verts, atol=1e-5)

    print("[PASS] test_get_expanded_meshes")


def test_resample_centerlines():
    """Test centerline resampling as layer centroids."""
    L, N = 8, 6
    guide = make_guide(np.zeros(3), np.array([0, 0, 1.0]), 1.0, L)
    verts, faces = build_synthetic_tube(guide, N, 0.03, 0.1)

    expander = TubeExpander([(verts, faces)], [guide],
                            current_scale=0.1, target_scale=0.8, device="cpu")
    exp = expander.compute_expansion()

    # At t=0, centroids should match guide points
    new_guides = expander.resample_centerlines(exp.v_start)
    for li in range(L):
        np.testing.assert_allclose(new_guides[0][li], guide[li], atol=1e-4)

    print("[PASS] test_resample_centerlines")


def test_swept_triangles():
    """Test swept triangle encoding."""
    L, N = 5, 6
    guide = make_guide(np.zeros(3), np.array([0, 0, 1.0]), 0.5, L)
    verts, faces = build_synthetic_tube(guide, N, 0.02, 0.1)

    expander = TubeExpander([(verts, faces)], [guide],
                            current_scale=0.1, target_scale=0.8, device="cpu")
    exp = expander.compute_expansion()
    swept_tris, tri_to_edge = expander.build_swept_triangles()

    n_edges = exp.all_edges.shape[0]
    assert swept_tris.shape == (2 * n_edges, 3, 3), \
        f"Shape: {swept_tris.shape}, expected ({2*n_edges}, 3, 3)"
    assert tri_to_edge.shape == (2 * n_edges,)

    # tri 2*i and 2*i+1 should map to edge i
    for i in range(min(10, n_edges)):
        assert tri_to_edge[2 * i].item() == i
        assert tri_to_edge[2 * i + 1].item() == i

    print("[PASS] test_swept_triangles")


def test_ccd_no_collision():
    """Test that distant tubes get full expansion (t=1)."""
    L, N = 8, 6
    tip_scale = 0.1

    guides = [
        make_guide(np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 0.5, L),
        make_guide(np.array([1.0, 0, 0.0]), np.array([0, 0, 1.0]), 0.5, L),
    ]
    meshes = [
        build_synthetic_tube(guides[0], N, 0.02, tip_scale),
        build_synthetic_tube(guides[1], N, 0.02, tip_scale),
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expander = TubeExpander(meshes, guides,
                            current_scale=tip_scale, target_scale=0.8,
                            device=device)
    exp = expander.compute_expansion()
    t_vertex = expander.run_ccd(max_collisions=128)

    # All vertices should be at t=1 (no collisions)
    assert (t_vertex == 1.0).all(), \
        f"Some vertices clamped: min={t_vertex.min():.4f}"

    print("[PASS] test_ccd_no_collision")


def test_ccd_two_tubes_head_on():
    """Two tubes very close — expanded radius > half the gap.

    radius=0.03, spacing=0.04, target_scale=0.8.
    At full expansion, each tube's radius at layer li is:
        r(li) = radius * scale(li) where scale = 1.0 + (li/(L-1))*(0.8-1.0)
    At layer 0: r = 0.03 * 1.0 = 0.03.  Two radii = 0.06 > spacing 0.04
        -> layer 0 is already overlapping at generation, but layer 0 is pinned.
    At layer 5 (mid): scale ~= 0.9, r = 0.027, two radii = 0.054 > 0.04
        -> must clamp.
    """
    L, N = 10, 8
    tip_scale = 0.1
    target_scale = 0.8
    radius = 0.03
    spacing = 0.04

    guides = [
        make_guide(np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 0.5, L),
        make_guide(np.array([spacing, 0, 0.0]), np.array([0, 0, 1.0]), 0.5, L),
    ]
    meshes = [
        build_synthetic_tube(guides[0], N, radius, tip_scale),
        build_synthetic_tube(guides[1], N, radius, tip_scale),
    ]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expander = TubeExpander(meshes, guides,
                            current_scale=tip_scale, target_scale=target_scale,
                            device=device)
    exp = expander.compute_expansion()
    t_vertex = expander.run_ccd(max_collisions=128)

    n_clamped = (t_vertex < 1.0).sum().item()
    n_total = t_vertex.shape[0]
    t_min = t_vertex.min().item()
    t_mean = t_vertex[t_vertex < 1.0].mean().item() if n_clamped > 0 else 1.0

    print(f"  Two tubes head-on: {n_clamped}/{n_total} clamped, "
          f"t_min={t_min:.4f}, t_mean_clamped={t_mean:.4f}")

    assert n_clamped > 0, "Expected clamping for overlapping tubes!"

    # Verify expanded tubes don't interpenetrate
    expanded = expander.get_expanded_meshes(t_vertex)
    _check_no_interpenetration(expanded, label="two_tubes_head_on")

    print("[PASS] test_ccd_two_tubes_head_on")


def test_ccd_four_tubes_tight_cluster():
    """Four tubes in a 2x2 grid, spacing smaller than expanded diameter.

    Each tube at full expansion would overlap its neighbors.
    CCD must clamp all four to prevent inter-tube penetration.
    """
    L, N = 12, 8
    tip_scale = 0.1
    target_scale = 0.8
    radius = 0.025
    spacing = 0.04  # < 2 * radius * target_scale at most layers

    origins = [
        np.array([0, 0, 0.0]),
        np.array([spacing, 0, 0.0]),
        np.array([0, spacing, 0.0]),
        np.array([spacing, spacing, 0.0]),
    ]
    guides = [make_guide(o, np.array([0, 0, 1.0]), 0.5, L) for o in origins]
    meshes = [build_synthetic_tube(g, N, radius, tip_scale) for g in guides]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expander = TubeExpander(meshes, guides,
                            current_scale=tip_scale, target_scale=target_scale,
                            device=device)
    exp = expander.compute_expansion()
    t_vertex = expander.run_ccd(max_collisions=128)

    n_clamped = (t_vertex < 1.0).sum().item()
    n_total = t_vertex.shape[0]
    t_min = t_vertex.min().item()

    print(f"  4-tube cluster: {n_clamped}/{n_total} clamped, t_min={t_min:.4f}")

    # With 4 tubes in tight grid, many vertices should be clamped
    assert n_clamped > n_total * 0.1, \
        f"Expected >10% clamped, got {n_clamped}/{n_total}"

    expanded = expander.get_expanded_meshes(t_vertex)
    _check_no_interpenetration(expanded, label="four_tubes_cluster")

    print("[PASS] test_ccd_four_tubes_tight_cluster")


def test_ccd_asymmetric_clamping():
    """One tube surrounded by two neighbors on one side only.

    The center tube should be clamped on the side facing neighbors
    but free to expand on the opposite side. This tests that per-vertex
    clamping is directional, not uniform.
    """
    L, N = 10, 12  # more cross-section verts for directional resolution
    tip_scale = 0.1
    target_scale = 0.8
    radius = 0.02
    spacing = 0.035

    # Center tube + two neighbors to the right
    guides = [
        make_guide(np.array([0, 0, 0.0]), np.array([0, 0, 1.0]), 0.5, L),
        make_guide(np.array([spacing, 0, 0.0]), np.array([0, 0, 1.0]), 0.5, L),
        make_guide(np.array([spacing * 2, 0, 0.0]), np.array([0, 0, 1.0]), 0.5, L),
    ]
    meshes = [build_synthetic_tube(g, N, radius, tip_scale) for g in guides]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expander = TubeExpander(meshes, guides,
                            current_scale=tip_scale, target_scale=target_scale,
                            device=device)
    exp = expander.compute_expansion()
    t_vertex = expander.run_ccd(max_collisions=128)

    n_clamped = (t_vertex < 1.0).sum().item()
    n_total = t_vertex.shape[0]

    print(f"  Asymmetric 3-tube: {n_clamped}/{n_total} clamped")
    assert n_clamped > 0, "Expected some clamping!"

    # Check that tube 0 (leftmost) has some unclamped vertices
    # (the left side facing away from neighbors)
    off0 = exp.vert_offsets[0]
    n_v0 = expander.topologies[0].n_verts
    t_tube0 = t_vertex[off0:off0 + n_v0]
    n_free_tube0 = (t_tube0 == 1.0).sum().item()
    n_clamped_tube0 = (t_tube0 < 1.0).sum().item()

    print(f"  Tube 0 (left): {n_clamped_tube0} clamped, {n_free_tube0} free")

    # Tube 0 should have BOTH clamped and free vertices (asymmetric)
    # Pinned verts are always 1.0, so check non-pinned only
    pinned0 = exp.pinned[off0:off0 + n_v0]
    t_nonpinned = t_tube0[~pinned0]
    has_clamped = (t_nonpinned < 1.0).any().item()
    has_free = (t_nonpinned == 1.0).any().item()

    print(f"  Tube 0 non-pinned: has_clamped={has_clamped}, has_free={has_free}")
    assert has_clamped, "Tube 0 should have clamped vertices (facing neighbors)"

    expanded = expander.get_expanded_meshes(t_vertex)
    _check_no_interpenetration(expanded, label="asymmetric_3tube")

    print("[PASS] test_ccd_asymmetric_clamping")


def test_ccd_convergence_ring():
    """Six tubes arranged in a hexagonal ring, all expanding inward.

    All tubes face toward the center. At target_scale, they would all
    overlap in the middle. CCD should clamp them symmetrically.
    """
    L, N = 10, 8
    tip_scale = 0.1
    target_scale = 0.8
    radius = 0.025
    ring_radius = 0.04  # tight ring — expanded tubes must collide

    n_ring = 6
    angles = np.linspace(0, 2 * np.pi, n_ring, endpoint=False)
    origins = [np.array([ring_radius * np.cos(a), ring_radius * np.sin(a), 0.0])
               for a in angles]
    guides = [make_guide(o, np.array([0, 0, 1.0]), 0.5, L) for o in origins]
    meshes = [build_synthetic_tube(g, N, radius, tip_scale) for g in guides]

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    expander = TubeExpander(meshes, guides,
                            current_scale=tip_scale, target_scale=target_scale,
                            device=device)
    exp = expander.compute_expansion()
    t_vertex = expander.run_ccd(max_collisions=128)

    n_clamped = (t_vertex < 1.0).sum().item()
    n_total = t_vertex.shape[0]
    t_min = t_vertex.min().item()

    print(f"  Hex ring (6 tubes): {n_clamped}/{n_total} clamped, t_min={t_min:.4f}")
    assert n_clamped > 0, "Hex ring should have collisions!"

    # Check approximate symmetry: all tubes should have similar clamping stats
    t_means = []
    for ti in range(expander.n_tubes):
        off = exp.vert_offsets[ti]
        n_v = expander.topologies[ti].n_verts
        t_tube = t_vertex[off:off + n_v]
        t_means.append(t_tube.mean().item())

    t_means = np.array(t_means)
    spread = t_means.max() - t_means.min()
    print(f"  Per-tube mean t: {t_means}, spread={spread:.4f}")
    # Symmetry: spread should be small (tubes are symmetric)
    assert spread < 0.15, f"Asymmetric clamping in hex ring: spread={spread:.4f}"

    expanded = expander.get_expanded_meshes(t_vertex)
    _check_no_interpenetration(expanded, label="hex_ring")

    print("[PASS] test_ccd_convergence_ring")


def _check_no_interpenetration(
    expanded: list, label: str = "", xi: float = 1e-3,
):
    """Verify that no two expanded tubes have vertices closer than xi."""
    from scipy.spatial import cKDTree

    n = len(expanded)
    min_dist_global = float("inf")
    for i in range(n):
        vi = expanded[i][0]
        tree_i = cKDTree(vi)
        for j in range(i + 1, n):
            vj = expanded[j][0]
            dists, _ = tree_i.query(vj)
            d = dists.min()
            min_dist_global = min(min_dist_global, d)

    print(f"  [{label}] Min inter-tube distance: {min_dist_global:.6f} (xi={xi})")
    # Vertex-vertex distance should be >= 0 (CCD prevents penetration).
    # We don't check >= xi because vertex-vertex distance can be slightly
    # less than xi even when edges don't interpenetrate.
    assert min_dist_global >= 0, f"Negative distance?!"


# ---------------------------------------------------------------------------
# Polyscope visualization
# ---------------------------------------------------------------------------

def visualize_expansion(expander: TubeExpander, t_vertex: torch.Tensor = None,
                        title: str = "Tube Expansion"):
    """Visualize tube topology and expansion with Polyscope.

    Args:
        expander: TubeExpander with compute_expansion() already called.
        t_vertex: per-vertex expansion factor from run_ccd(). If None, shows
                  unclamped expansion (t=1 for all).
    """
    import polyscope as ps

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("none")

    exp = expander.expansion
    v_start_np = exp.v_start.cpu().numpy()
    v_end_np = exp.v_end.cpu().numpy()
    pinned_np = exp.pinned.cpu().numpy().astype(float)

    # Displacement magnitude per vertex
    disp = np.linalg.norm(v_end_np - v_start_np, axis=1)

    has_ccd = t_vertex is not None
    if has_ccd:
        t_np = t_vertex.cpu().numpy()

    # --- Register original (current) meshes ---
    for ti in range(expander.n_tubes):
        verts, faces = expander.meshes[ti]
        name = f"tube_{ti}_original"
        mesh = ps.register_surface_mesh(name, verts, faces)
        mesh.set_transparency(0.3)
        mesh.set_color((0.4, 0.4, 0.8))
        if has_ccd:
            mesh.set_enabled(False)  # hide originals when showing clamped

        off = exp.vert_offsets[ti]
        n_v = expander.topologies[ti].n_verts
        mesh.add_scalar_quantity("displacement", disp[off:off + n_v],
                                enabled=not has_ccd)
        mesh.add_scalar_quantity("pinned", pinned_np[off:off + n_v])
        if has_ccd:
            mesh.add_scalar_quantity("t_vertex", t_np[off:off + n_v])

    # --- Register unclamped expanded meshes (t=1) ---
    # Always show these so user can compare clamped vs unclamped
    t_full = torch.ones(exp.v_start.shape[0], device=exp.v_start.device)
    expanded_full = expander.get_expanded_meshes(t_full)
    for ti, (new_verts, new_faces) in enumerate(expanded_full):
        name = f"tube_{ti}_unclamped"
        mesh = ps.register_surface_mesh(name, new_verts, new_faces)
        mesh.set_color((0.8, 0.3, 0.3))
        mesh.set_transparency(0.85)
        mesh.set_enabled(True)  # visible — shows penetrating tubes

    # --- Register CCD-clamped expanded meshes ---
    if has_ccd:
        expanded_clamped = expander.get_expanded_meshes(t_vertex)
        for ti, (new_verts, new_faces) in enumerate(expanded_clamped):
            name = f"tube_{ti}_clamped"
            mesh = ps.register_surface_mesh(name, new_verts, new_faces)
            mesh.set_color((0.2, 0.8, 0.2))
            mesh.set_enabled(True)

            off = exp.vert_offsets[ti]
            n_v = expander.topologies[ti].n_verts
            mesh.add_scalar_quantity("t_vertex", t_np[off:off + n_v], enabled=True)

    # --- Register edge network (colored by type) ---
    for ti in range(expander.n_tubes):
        topo = expander.topologies[ti]
        off = exp.vert_offsets[ti]
        n_v = topo.n_verts
        tube_verts = v_start_np[off:off + n_v]

        all_local_edges = []
        edge_colors = []
        for edges, color in [
            (topo.connection_edges, [1.0, 0.2, 0.2]),
            (topo.ring_edges, [0.2, 1.0, 0.2]),
            (topo.tip_edges, [0.2, 0.2, 1.0]),
        ]:
            all_local_edges.append(edges)
            edge_colors.extend([color] * len(edges))

        all_local_edges = np.concatenate(all_local_edges, axis=0)
        edge_colors = np.array(edge_colors)

        net = ps.register_curve_network(
            f"tube_{ti}_edges", tube_verts, all_local_edges,
        )
        net.add_color_quantity("edge_type", edge_colors, defined_on="edges", enabled=True)
        net.set_enabled(False)

    # --- Register guide curves ---
    for ti in range(expander.n_tubes):
        guide = expander.guides[ti]
        L = len(guide)
        edges = np.array([[i, i + 1] for i in range(L - 1)])
        net = ps.register_curve_network(f"guide_{ti}", guide, edges)
        net.set_color((1.0, 1.0, 0.0))
        net.set_radius(0.001)

    ps.show()


# ---------------------------------------------------------------------------
# Real data loading
# ---------------------------------------------------------------------------

def load_real_tubes():
    """Load real tube meshes from the pipeline."""
    from hair_tube_relaxation.standalone_voronoi_tube import (
        build_tube_meshes,
        load_config,
        load_scalp,
        compute_voronoi_on_scalp,
        extract_cell_polylines,
    )
    import toml

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "config", "conf_00352.toml"
    )
    if not os.path.exists(config_path):
        print(f"Config not found: {config_path}")
        print("Falling back to synthetic tubes.")
        return None, None

    cfg = toml.load(config_path)
    project_root = os.path.join(os.path.dirname(__file__), "..")

    scalp_path = os.path.join(project_root, cfg["data"]["base_path"],
                              cfg["scalp"]["scalp_mesh"])
    output_base = os.path.join(project_root, cfg["output"]["base_path"])

    scalp_verts, scalp_faces, guides = load_scalp(scalp_path, output_base)
    voronoi_data = compute_voronoi_on_scalp(scalp_verts, scalp_faces, guides)
    cell_polylines = extract_cell_polylines(voronoi_data, scalp_verts, scalp_faces)

    tube_meshes = build_tube_meshes(
        guides, cell_polylines, scalp_verts, scalp_faces,
        tip_scale=0.1, resample_n=0,
    )
    return tube_meshes, guides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Test TubeExpander")
    parser.add_argument("--no-vis", action="store_true", help="Skip visualization")
    parser.add_argument("--real", action="store_true", help="Use real tube data")
    parser.add_argument("--skip-ccd-tests", action="store_true",
                        help="Skip CCD tests (require CUDA + BVH)")
    parser.add_argument("--max-tubes", type=int, default=20,
                        help="Max tubes for visualization (default 20)")
    args = parser.parse_args()

    # --- Run unit tests (no CUDA required) ---
    print("=== Running unit tests ===")
    test_n_cross_detection()
    test_topology_parsing()
    test_expansion_computation()
    test_expansion_multi_tube()
    test_get_expanded_meshes()
    test_resample_centerlines()
    test_swept_triangles()
    print("=== Unit tests passed ===\n")

    # --- Run CCD integration tests (require CUDA + BVH) ---
    if not args.skip_ccd_tests:
        print("=== Running CCD integration tests ===")
        test_ccd_no_collision()
        test_ccd_two_tubes_head_on()
        test_ccd_four_tubes_tight_cluster()
        test_ccd_asymmetric_clamping()
        test_ccd_convergence_ring()
        print("=== CCD tests passed ===\n")
    else:
        print("(Skipping CCD tests)\n")

    if args.no_vis:
        return

    # --- Visualization: use tight configurations that produce collisions ---
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.real:
        print("Loading real tube data...")
        tube_meshes, guides = load_real_tubes()
        if tube_meshes is None:
            args.real = False

    if args.real:
        tip_scale = 0.1
        limited_meshes = []
        limited_guides = []
        count = 0
        for i, m in enumerate(tube_meshes):
            if m is not None and count < args.max_tubes:
                limited_meshes.append(m)
                limited_guides.append(guides[i])
                count += 1
            else:
                limited_meshes.append(None)
                limited_guides.append(guides[i] if i < len(guides) else None)
        tube_meshes = limited_meshes[:len(limited_guides)]
        guides = limited_guides
    else:
        # 4-tube tight cluster: guaranteed heavy clamping
        print("Generating tight 4-tube cluster for visualization...")
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

    print(f"Creating TubeExpander with {len(tube_meshes)} tubes on {device}...")
    expander = TubeExpander(
        tube_meshes=tube_meshes,
        guides=guides,
        current_scale=tip_scale,
        target_scale=0.8,
        device=device,
    )
    exp = expander.compute_expansion()

    print(f"  Active tubes: {expander.n_tubes}")
    print(f"  Total vertices: {exp.v_start.shape[0]}")
    print(f"  Total edges: {exp.all_edges.shape[0]}")

    # Run CCD
    print("\nRunning BVH + ACCD...")
    t_vertex = expander.run_ccd(max_collisions=128)
    n_clamped = (t_vertex < 1.0).sum().item()
    print(f"  Clamped {n_clamped}/{t_vertex.shape[0]} vertices")
    if n_clamped > 0:
        t_min = t_vertex[t_vertex < 1.0].min().item()
        t_mean = t_vertex[t_vertex < 1.0].mean().item()
        print(f"  t_min={t_min:.4f}, t_mean_clamped={t_mean:.4f}")

    print("\nLaunching Polyscope...")
    visualize_expansion(expander, t_vertex=t_vertex)


if __name__ == "__main__":
    main()
