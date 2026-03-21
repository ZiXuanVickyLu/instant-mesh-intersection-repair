"""Standalone Voronoi scalp segmentation + hair tube generation.

No project-internal imports — only depends on numpy, scipy, trimesh,
toml, and open3d.

Usage:
    python standalone_voronoi_tube.py --config config/conf_00352.toml
    python standalone_voronoi_tube.py --config config/conf_00352.toml --skip-viz
"""

import os
import argparse

import toml
import numpy as np
import trimesh
from scipy.spatial import cKDTree

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ===================================================================
# OBJ loading
# ===================================================================


def load_obj_with_uv(obj_path: str) -> dict:
    """Load OBJ file and return vertices, faces, and per-face-vertex UVs."""
    vertices = []
    tex_coords = []
    faces_v = []
    faces_vt = []
    face_sizes = {}

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("vt "):
                parts = line.split()
                tex_coords.append([float(parts[1]), float(parts[2])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                fv = []
                ft = []
                has_uv = True
                for p in parts:
                    indices = p.split("/")
                    fv.append(int(indices[0]) - 1)
                    if len(indices) >= 2 and indices[1]:
                        ft.append(int(indices[1]) - 1)
                    else:
                        has_uv = False
                faces_v.append(fv)
                faces_vt.append(ft if has_uv else None)
                n = len(fv)
                face_sizes[n] = face_sizes.get(n, 0) + 1

    size_names = {3: "triangles", 4: "quads"}
    parts_str = ", ".join(
        f"{count} {size_names.get(n, f'{n}-gons')}" for n, count in sorted(face_sizes.items())
    )
    print(f"  Face types: {parts_str}")

    return {
        "vertices": np.array(vertices, dtype=np.float64),
        "tex_coords": np.array(tex_coords, dtype=np.float64) if tex_coords else np.empty((0, 2)),
        "faces_v": faces_v,
        "faces_vt": faces_vt,
    }


def load_guides_obj_grouped(obj_path: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Load guide polylines from OBJ, preserving per-guide card group IDs."""
    all_vertices = []
    guides = []
    guide_group_ids = []
    current_group_id = -1

    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("g card_"):
                current_group_id = int(line.split("_", 1)[1])
            elif line.startswith("v "):
                parts = line.split()
                all_vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("l "):
                indices = [int(x) - 1 for x in line.split()[1:]]
                polyline = np.array([all_vertices[i] for i in indices], dtype=np.float64)
                guides.append(polyline)
                guide_group_ids.append(current_group_id)

    guide_group_ids = np.array(guide_group_ids, dtype=np.int32)
    n_groups = len(set(guide_group_ids))
    print(f"  Loaded {len(guides)} guide polylines in {n_groups} groups "
          f"({len(all_vertices)} vertices) from {obj_path}")
    return guides, guide_group_ids


# ===================================================================
# Geometry helpers
# ===================================================================


def compute_smooth_normals(vertices: np.ndarray, faces_v) -> np.ndarray:
    """Compute area-weighted smooth vertex normals from face geometry."""
    normals = np.zeros_like(vertices)
    for face in faces_v:
        v0 = vertices[face[0]]
        for i in range(1, len(face) - 1):
            v1 = vertices[face[i]]
            v2 = vertices[face[i + 1]]
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            for vi_idx in [face[0], face[i], face[i + 1]]:
                normals[vi_idx] += face_normal
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.maximum(lengths, 1e-10)
    normals /= lengths
    return normals


def triangulate_faces(faces_v: list[list[int]]) -> np.ndarray:
    """Fan-triangulate polygon faces into triangles."""
    triangles = []
    for face in faces_v:
        for k in range(1, len(face) - 1):
            triangles.append([face[0], face[k], face[k + 1]])
    return np.array(triangles, dtype=np.int32)


# ===================================================================
# Voronoi segmentation
# ===================================================================


def compute_voronoi_labels(vertices: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Label each vertex by its nearest seed (L2 distance)."""
    tree = cKDTree(seeds)
    _, labels = tree.query(vertices)
    return labels.astype(np.int32)


def _plane_triangle_intersection(tri_pts, plane_pt, plane_normal):
    """Intersect a plane with a triangle, returning a segment or None."""
    dists = [np.dot(p - plane_pt, plane_normal) for p in tri_pts]
    crossings = []
    for i in range(3):
        j = (i + 1) % 3
        di, dj = dists[i], dists[j]
        if abs(di) < 1e-10:
            crossings.append(tri_pts[i].copy())
        if di * dj < -1e-20:
            t = di / (di - dj)
            crossings.append(tri_pts[i] + t * (tri_pts[j] - tri_pts[i]))
    unique = []
    for pt in crossings:
        if not any(np.linalg.norm(pt - u) < 1e-10 for u in unique):
            unique.append(pt)
    if len(unique) >= 2:
        return (unique[0], unique[1])
    return None


def _clip_segment_by_halfplane(p1, p2, plane_pt, plane_normal):
    """Clip segment (p1, p2) to half-space dot(p - plane_pt, normal) <= 0."""
    d1 = np.dot(p1 - plane_pt, plane_normal)
    d2 = np.dot(p2 - plane_pt, plane_normal)
    if d1 <= 1e-10 and d2 <= 1e-10:
        return (p1, p2)
    if d1 > 1e-10 and d2 > 1e-10:
        return None
    t = d1 / (d1 - d2)
    intersection = p1 + t * (p2 - p1)
    if d1 <= 1e-10:
        return (p1, intersection)
    else:
        return (intersection, p2)


def _find_candidate_labels(face_vert_indices, vertices, vert_labels, seed_tree, k_neighbors):
    """Find all seed labels that could have a Voronoi boundary in this face."""
    a, b, c = face_vert_indices
    candidates = {int(vert_labels[a]), int(vert_labels[b]), int(vert_labels[c])}
    k = min(k_neighbors, len(seed_tree.data))
    query_pts = np.array([
        vertices[a], vertices[b], vertices[c],
        (vertices[a] + vertices[b] + vertices[c]) / 3.0,
    ])
    _, knn = seed_tree.query(query_pts, k=k)
    for row in knn:
        for idx in row:
            candidates.add(int(idx))
    return sorted(candidates)


def _process_face_boundaries(tri_pts, candidate_labels, seeds):
    """Compute all Voronoi boundary segments within a single triangle face."""
    segments = []
    segment_cells = []
    n = len(candidate_labels)
    for i in range(n):
        for j in range(i + 1, n):
            label_a = candidate_labels[i]
            label_b = candidate_labels[j]
            mid_ab = 0.5 * (seeds[label_a] + seeds[label_b])
            normal_ab = seeds[label_b] - seeds[label_a]
            nlen = np.linalg.norm(normal_ab)
            if nlen < 1e-15:
                continue
            normal_ab /= nlen
            seg = _plane_triangle_intersection(tri_pts, mid_ab, normal_ab)
            if seg is None:
                continue
            p1, p2 = seg
            for k in range(n):
                label_c = candidate_labels[k]
                if label_c == label_a or label_c == label_b:
                    continue
                mid_ac = 0.5 * (seeds[label_a] + seeds[label_c])
                normal_ac = seeds[label_c] - seeds[label_a]
                nlen_ac = np.linalg.norm(normal_ac)
                if nlen_ac < 1e-15:
                    continue
                normal_ac /= nlen_ac
                result = _clip_segment_by_halfplane(p1, p2, mid_ac, normal_ac)
                if result is None:
                    p1 = p2 = None
                    break
                p1, p2 = result
            if p1 is not None and np.linalg.norm(p2 - p1) > 1e-10:
                segments.append((p1.copy(), p2.copy()))
                segment_cells.append((label_a, label_b))
    return segments, segment_cells


def extract_boundary_segments(vertices, faces, vert_labels, seeds, k_neighbors=8):
    """Extract Voronoi boundary line segments on the mesh surface."""
    seed_tree = cKDTree(seeds)
    segments = []
    segment_cells = []
    for face in faces:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        tri_pts = [vertices[a], vertices[b], vertices[c]]
        candidate_labels = _find_candidate_labels(
            (a, b, c), vertices, vert_labels, seed_tree, k_neighbors,
        )
        if len(candidate_labels) < 2:
            continue
        face_segs, face_cells = _process_face_boundaries(
            tri_pts, candidate_labels, seeds,
        )
        segments.extend(face_segs)
        segment_cells.extend(face_cells)
    return segments, segment_cells


def extract_mesh_boundary_edges(vertices, faces):
    """Find mesh boundary edges (edges belonging to only one face)."""
    edge_count = {}
    for face in faces:
        n = len(face)
        for i in range(n):
            v0, v1 = int(face[i]), int(face[(i + 1) % n])
            key = (min(v0, v1), max(v0, v1))
            edge_count[key] = edge_count.get(key, 0) + 1
    boundary = []
    for (v0, v1), count in edge_count.items():
        if count == 1:
            boundary.append((vertices[v0].copy(), vertices[v1].copy()))
    return boundary


def _chain_segments(segments, weld_tol):
    """Chain a set of line segments into ordered polylines."""
    if not segments:
        return []
    all_pts = []
    for p1, p2 in segments:
        all_pts.append(p1)
        all_pts.append(p2)
    all_pts = np.array(all_pts)
    tree = cKDTree(all_pts)
    n = len(all_pts)
    canonical = np.arange(n)
    visited = np.zeros(n, dtype=bool)
    for i in range(n):
        if visited[i]:
            continue
        neighbors = tree.query_ball_point(all_pts[i], weld_tol)
        rep = min(neighbors)
        for j in neighbors:
            canonical[j] = rep
            visited[j] = True
    adj = {}
    edge_set = set()
    for si in range(len(segments)):
        u = int(canonical[2 * si])
        v = int(canonical[2 * si + 1])
        if u == v:
            continue
        edge_key = (min(u, v), max(u, v))
        if edge_key in edge_set:
            continue
        edge_set.add(edge_key)
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    used_nodes = set()
    polylines = []
    for start in adj:
        if start in used_nodes:
            continue
        node = start
        if len(adj.get(node, [])) != 1:
            queue = [node]
            component = set()
            while queue:
                cur = queue.pop()
                if cur in component:
                    continue
                component.add(cur)
                for nb in adj.get(cur, []):
                    if nb not in component:
                        queue.append(nb)
            endpoints = [n for n in component if len(adj.get(n, [])) == 1]
            if endpoints:
                node = endpoints[0]
        chain = [node]
        used_nodes.add(node)
        prev = -1
        while True:
            neighbors = [n for n in adj.get(chain[-1], []) if n != prev]
            if not neighbors:
                break
            nxt = neighbors[0]
            if nxt in used_nodes and nxt != chain[0]:
                break
            prev = chain[-1]
            chain.append(nxt)
            if nxt == chain[0]:
                break
            used_nodes.add(nxt)
        if len(chain) >= 2:
            pts = np.array([all_pts[c] for c in chain])
            polylines.append(pts)
    # Simplify polylines by merging collinear points
    polylines = [_simplify_polyline(p) for p in polylines]
    return polylines


def _simplify_polyline(pts, angle_tol=1e-3):
    """Remove interior points that are collinear with their neighbors.

    Keeps endpoints and any point where the direction changes beyond
    *angle_tol* (measured as 1 - |cos(angle)|, so 0 = perfectly straight).
    """
    if len(pts) <= 2:
        return pts

    is_closed = np.linalg.norm(pts[0] - pts[-1]) < 1e-9

    keep = [True] * len(pts)

    for i in range(1, len(pts) - 1):
        d_prev = pts[i] - pts[i - 1]
        d_next = pts[i + 1] - pts[i]
        len_prev = np.linalg.norm(d_prev)
        len_next = np.linalg.norm(d_next)
        if len_prev < 1e-12 or len_next < 1e-12:
            keep[i] = False
            continue
        cos_angle = np.dot(d_prev, d_next) / (len_prev * len_next)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        if 1.0 - abs(cos_angle) < angle_tol:
            keep[i] = False

    # For closed loops, also check the wrap-around point
    if is_closed and len(pts) >= 3:
        i = len(pts) - 2
        if keep[i]:
            d_prev = pts[i] - pts[i - 1]
            d_next = pts[1] - pts[i]
            len_prev = np.linalg.norm(d_prev)
            len_next = np.linalg.norm(d_next)
            if len_prev > 1e-12 and len_next > 1e-12:
                cos_angle = np.dot(d_prev, d_next) / (len_prev * len_next)
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                if 1.0 - abs(cos_angle) < angle_tol:
                    keep[i] = False

    result = pts[keep]
    if len(result) < 2:
        return pts[[0, -1]]
    return result


def build_cell_boundary_polylines(
    segments, segment_cells, n_seeds,
    mesh_boundary=None, vert_labels=None, vertices=None, faces=None,
    weld_tol=1e-6,
):
    """Chain boundary segments into ordered polylines per Voronoi cell."""
    cell_segments = {i: [] for i in range(n_seeds)}
    for seg, (ca, cb) in zip(segments, segment_cells):
        cell_segments[ca].append(seg)
        cell_segments[cb].append(seg)
    if mesh_boundary is not None and vert_labels is not None:
        tree = cKDTree(vertices) if vertices is not None else None
        for p1, p2 in mesh_boundary:
            mid = 0.5 * (p1 + p2)
            if tree is not None:
                _, idx = tree.query(mid)
                cell_id = int(vert_labels[idx])
                cell_segments[cell_id].append((p1, p2))
    result = {}
    for cell_id, segs in cell_segments.items():
        if not segs:
            continue
        polylines = _chain_segments(segs, weld_tol)
        if polylines:
            result[cell_id] = polylines
    return result


# ===================================================================
# Penetration detection & repair
# ===================================================================


def find_boundary_root_ids(vert_labels, faces, n_seeds):
    """Return seed IDs whose Voronoi cell touches the mesh boundary."""
    edge_face_count = {}
    for face in faces:
        for i in range(3):
            v0, v1 = int(face[i]), int(face[(i + 1) % 3])
            key = (min(v0, v1), max(v0, v1))
            edge_face_count[key] = edge_face_count.get(key, 0) + 1
    boundary_ids = set()
    for (v0, v1), count in edge_face_count.items():
        if count == 1:
            boundary_ids.add(int(vert_labels[v0]))
            boundary_ids.add(int(vert_labels[v1]))
    return boundary_ids


def _smooth_repaired_guides(guides, fixed_guide_ids, inside_mask,
                            interior_indices, lengths,
                            smooth_iterations=5, smooth_margin=3):
    """Laplacian smooth around projected vertices to remove kinks."""
    offset = 0
    for local_i, global_i in enumerate(interior_indices):
        length = lengths[local_i]
        if int(global_i) not in fixed_guide_ids:
            offset += length
            continue
        guide = guides[global_i]
        n = len(guide)
        guide_mask = inside_mask[offset:offset + length]
        affected = np.where(guide_mask)[0]
        lo = max(1, int(affected.min()) - smooth_margin)
        hi = min(n - 2, int(affected.max()) + smooth_margin)
        if hi <= lo:
            offset += length
            continue
        for _ in range(smooth_iterations):
            smoothed = guide.copy()
            for vi in range(lo, hi + 1):
                smoothed[vi] = 0.5 * guide[vi] + 0.25 * (guide[vi - 1] + guide[vi + 1])
            guide[lo:hi + 1] = smoothed[lo:hi + 1]
        guides[global_i] = guide
        offset += length


def fix_penetrating_guides(
    guides, body_mesh_path, scalp_verts, scalp_faces, vert_labels,
    thickness=0.01, subset=None,
):
    """Detect and repair guide strands that penetrate the body mesh."""
    n_seeds = len(guides)
    if n_seeds == 0:
        return 0, 0, set()

    boundary_ids = find_boundary_root_ids(vert_labels, scalp_faces, n_seeds)
    if subset is not None:
        candidate = subset - boundary_ids
        interior_mask = np.array([i in candidate for i in range(n_seeds)])
    else:
        interior_mask = np.array([i not in boundary_ids for i in range(n_seeds)])
    n_boundary = int(sum(1 for i in (subset or range(n_seeds)) if i in boundary_ids))
    n_interior = int(interior_mask.sum())

    print(f"  {n_interior} interior guides, {n_boundary} boundary guides (skipped)")
    if n_interior == 0:
        return 0, n_boundary, boundary_ids

    body_mesh = trimesh.load(body_mesh_path, process=False, force="mesh")
    print(f"  Body mesh: {len(body_mesh.vertices)} verts, "
          f"{len(body_mesh.faces)} faces, watertight={body_mesh.is_watertight}")

    scalp_mesh = trimesh.Trimesh(vertices=scalp_verts, faces=scalp_faces, process=False)

    interior_indices = np.where(interior_mask)[0]
    lengths = [len(guides[i]) for i in interior_indices]
    all_positions = np.vstack([guides[i] for i in interior_indices])

    # Expand body mesh outward by thickness
    body_centroid = body_mesh.centroid
    expanded_verts = body_mesh.vertices.copy()
    expanded_normals = body_mesh.vertex_normals.copy()
    to_cent = expanded_verts - body_centroid
    flip_v = np.sum(expanded_normals * to_cent, axis=1) < 0
    expanded_normals[flip_v] *= -1
    expanded_verts += expanded_normals * thickness
    expanded_body = trimesh.Trimesh(
        vertices=expanded_verts, faces=body_mesh.faces.copy(), process=False,
    )

    violating_mask = expanded_body.contains(all_positions)

    # Exclude root vertices
    offset = 0
    for length in lengths:
        violating_mask[offset] = False
        offset += length

    if not np.any(violating_mask):
        print("  No vertices violating offset threshold")
        return 0, n_boundary, boundary_ids

    closest_all, _, face_ids_all = trimesh.proximity.closest_point(
        expanded_body, all_positions[violating_mask],
    )
    face_normals_all = expanded_body.face_normals[face_ids_all].copy()
    to_surface = closest_all - body_centroid
    flip = np.sum(face_normals_all * to_surface, axis=1) < 0
    face_normals_all[flip] *= -1
    projected = closest_all + face_normals_all * 1e-4

    fixed_positions = all_positions.copy()
    fixed_positions[violating_mask] = projected

    fixed_guides = set()
    offset = 0
    for local_i, global_i in enumerate(interior_indices):
        length = lengths[local_i]
        guide_mask = violating_mask[offset:offset + length]
        if np.any(guide_mask):
            guides[global_i] = fixed_positions[offset:offset + length].copy()
            fixed_guides.add(int(global_i))
        offset += length

    n_violating = int(violating_mask.sum())
    print(f"  {n_violating} vertices in {len(fixed_guides)} guides "
          f"projected onto body + {thickness} offset")

    if fixed_guides:
        _smooth_repaired_guides(guides, fixed_guides, violating_mask,
                                interior_indices, lengths, smooth_iterations=5)
        # Re-enforce
        all_positions2 = np.vstack([guides[i] for i in interior_indices])
        viol2 = expanded_body.contains(all_positions2)
        offset = 0
        for length in lengths:
            viol2[offset] = False
            offset += length
        if np.any(viol2):
            cl2, _, fid2 = trimesh.proximity.closest_point(
                expanded_body, all_positions2[viol2],
            )
            fn2 = expanded_body.face_normals[fid2].copy()
            to_s2 = cl2 - body_centroid
            fn2[np.sum(fn2 * to_s2, axis=1) < 0] *= -1
            proj2 = cl2 + fn2 * 1e-4
            fixed2 = all_positions2.copy()
            fixed2[viol2] = proj2
            offset = 0
            for local_i, global_i in enumerate(interior_indices):
                length = lengths[local_i]
                if np.any(viol2[offset:offset + length]):
                    guides[global_i] = fixed2[offset:offset + length].copy()
                offset += length
            print(f"  {int(viol2.sum())} vertices re-enforced after smoothing")

    return len(fixed_guides), n_boundary, boundary_ids


# ===================================================================
# Hair tube generation (parallel transport)
# ===================================================================


def merge_cell_boundary(polylines, root, weld_tol=1e-5):
    """Merge per-cell boundary polylines into one closed ordered polygon."""
    if not polylines or all(len(p) < 2 for p in polylines):
        return None
    segs = [p for p in polylines if len(p) >= 2]
    if not segs:
        return None
    if len(segs) == 1:
        poly = segs[0]
        if np.linalg.norm(poly[0] - poly[-1]) < weld_tol:
            poly = poly[:-1]
        return poly if len(poly) >= 3 else None
    used = [False] * len(segs)
    used[0] = True
    chain = list(segs[0])
    for _ in range(len(segs) - 1):
        tail = chain[-1]
        best_dist = np.inf
        best_idx = -1
        best_flip = False
        for si, seg in enumerate(segs):
            if used[si]:
                continue
            d_start = np.linalg.norm(seg[0] - tail)
            d_end = np.linalg.norm(seg[-1] - tail)
            if d_start < best_dist:
                best_dist = d_start
                best_idx = si
                best_flip = False
            if d_end < best_dist:
                best_dist = d_end
                best_idx = si
                best_flip = True
        if best_idx < 0:
            break
        used[best_idx] = True
        seg = segs[best_idx]
        if best_flip:
            seg = seg[::-1]
        if np.linalg.norm(seg[0] - tail) < weld_tol:
            seg = seg[1:]
        chain.extend(seg)
    poly = np.array(chain, dtype=np.float64)
    if len(poly) > 1 and np.linalg.norm(poly[0] - poly[-1]) < weld_tol:
        poly = poly[:-1]
    return poly if len(poly) >= 3 else None


def _rodrigues(v, axis, cos_a, sin_a):
    """Rodrigues' rotation: rotate v around axis by angle (given cos, sin)."""
    return v * cos_a + np.cross(axis, v) * sin_a + axis * np.dot(axis, v) * (1 - cos_a)


def _build_root_frame(guide, scalp_normal):
    """Build an orthonormal frame at the guide root."""
    tangent = guide[1] - guide[0]
    t_len = np.linalg.norm(tangent)
    if t_len < 1e-12:
        tangent = np.array([0.0, 0.0, 1.0])
    else:
        tangent /= t_len
    normal = scalp_normal - np.dot(scalp_normal, tangent) * tangent
    n_len = np.linalg.norm(normal)
    if n_len < 1e-12:
        if abs(tangent[0]) < 0.9:
            normal = np.cross(tangent, np.array([1.0, 0.0, 0.0]))
        else:
            normal = np.cross(tangent, np.array([0.0, 1.0, 0.0]))
        n_len = np.linalg.norm(normal)
    normal /= n_len
    bitangent = np.cross(tangent, normal)
    bitangent /= np.linalg.norm(bitangent)
    return tangent, normal, bitangent


def _parallel_transport_frames(guide, scalp_normal):
    """Compute parallel-transported frames along a guide curve."""
    L = len(guide)
    tangents = np.zeros((L, 3))
    normals = np.zeros((L, 3))
    bitangents = np.zeros((L, 3))
    t0, n0, b0 = _build_root_frame(guide, scalp_normal)
    tangents[0] = t0
    normals[0] = n0
    bitangents[0] = b0
    for i in range(1, L):
        t_new = guide[i] - guide[i - 1]
        t_len = np.linalg.norm(t_new)
        if t_len < 1e-12:
            tangents[i] = tangents[i - 1]
            normals[i] = normals[i - 1]
            bitangents[i] = bitangents[i - 1]
            continue
        t_new /= t_len
        t_prev = tangents[i - 1]
        axis = np.cross(t_prev, t_new)
        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-12:
            normals[i] = normals[i - 1]
            bitangents[i] = bitangents[i - 1]
        else:
            axis /= axis_len
            cos_a = np.clip(np.dot(t_prev, t_new), -1.0, 1.0)
            sin_a = axis_len
            normals[i] = _rodrigues(normals[i - 1], axis, cos_a, sin_a)
            bitangents[i] = _rodrigues(bitangents[i - 1], axis, cos_a, sin_a)
        tangents[i] = t_new
    return tangents, normals, bitangents


def _polygon_to_local_2d(polygon_3d, root, normal, bitangent):
    """Project 3-D boundary polygon into the root's local 2-D frame."""
    offsets = polygon_3d - root
    u = offsets @ bitangent
    v = offsets @ normal
    return np.column_stack([u, v])


def _resample_polygon(polygon, n):
    """Resample a closed polygon to n evenly spaced points."""
    closed = np.vstack([polygon, polygon[0:1]])
    diffs = np.diff(closed, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cum_lengths[-1]
    if total < 1e-12:
        return polygon[:n] if len(polygon) >= n else polygon
    target_lengths = np.linspace(0, total, n, endpoint=False)
    resampled = np.zeros((n, 3), dtype=np.float64)
    for i, tl in enumerate(target_lengths):
        idx = np.searchsorted(cum_lengths, tl, side='right') - 1
        idx = min(idx, len(polygon) - 1)
        next_idx = (idx + 1) % len(closed)
        seg_len = seg_lengths[idx]
        if seg_len < 1e-12:
            resampled[i] = closed[idx]
        else:
            t = (tl - cum_lengths[idx]) / seg_len
            resampled[i] = closed[idx] + t * (closed[next_idx] - closed[idx])
    return resampled


def _build_tube_mesh(guide, cross_section_2d, polygon_3d,
                     tangents, normals, bitangents, tip_scale=0.4):
    """Build a tube mesh by sweeping a 2-D cross-section along a guide.

    Layer 0 uses the original 3-D boundary polygon (on the scalp).
    Subsequent layers linearly interpolate the scale from 1.0 to tip_scale.
    """
    L = len(guide)
    N = len(cross_section_2d)
    verts = np.zeros((L * N + 1, 3), dtype=np.float64)

    # Layer 0: actual 3D boundary points on the scalp
    for ni in range(N):
        verts[ni] = polygon_3d[ni]

    for li in range(1, L):
        t = li / (L - 1)  # 0 at root, 1 at tip
        scale = 1.0 + t * (tip_scale - 1.0)
        center = guide[li]
        n_axis = normals[li]
        b_axis = bitangents[li]
        for ni in range(N):
            u, v = cross_section_2d[ni]
            verts[li * N + ni] = center + scale * (u * b_axis + v * n_axis)

    # Tip center
    verts[L * N] = guide[-1]

    # Faces
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

    # Tip cap
    tip_idx = L * N
    last_layer = (L - 1) * N
    for ni in range(N):
        ni_next = (ni + 1) % N
        faces.append([last_layer + ni, tip_idx, last_layer + ni_next])

    return verts, np.array(faces, dtype=np.int32)


def build_tube_meshes(guides, cell_polylines, scalp_verts, scalp_faces,
                      tip_scale=0.4, resample_n=0):
    """Build tube meshes for all guides using Voronoi cell cross-sections."""
    roots = np.array([g[0] for g in guides], dtype=np.float64)
    vertex_normals = compute_smooth_normals(scalp_verts, scalp_faces)
    tree = cKDTree(scalp_verts)
    _, nearest_vi = tree.query(roots)
    root_normals = vertex_normals[nearest_vi]

    n_guides = len(guides)
    results = [None] * n_guides
    n_built = 0
    n_skipped = 0

    for gi in range(n_guides):
        guide = guides[gi]
        if len(guide) < 2:
            n_skipped += 1
            continue
        polys = cell_polylines.get(gi, [])
        polygon = merge_cell_boundary(polys, guide[0])
        if polygon is None:
            n_skipped += 1
            continue
        if resample_n > 0 and len(polygon) != resample_n:
            polygon = _resample_polygon(polygon, resample_n)
        scalp_n = root_normals[gi]
        tangents, normals, bitangents = _parallel_transport_frames(guide, scalp_n)
        cs_2d = _polygon_to_local_2d(polygon, guide[0], normals[0], bitangents[0])
        verts, faces = _build_tube_mesh(
            guide, cs_2d, polygon, tangents, normals, bitangents,
            tip_scale=tip_scale,
        )
        results[gi] = (verts, faces)
        n_built += 1

    print(f"  Built {n_built} tube meshes, {n_skipped} skipped (no boundary)")
    return results


# ===================================================================
# Visualization
# ===================================================================


def _golden_ratio_color(index, total):
    """Generate a distinct color using golden-ratio HSV spacing."""
    hue = (index * 0.618033988749895) % 1.0
    sat = 0.5 + 0.4 * ((index % 3) / 2.0)
    val = 0.6 + 0.3 * ((index % 5) / 4.0)
    c = val * sat
    x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
    m = val - c
    h_sector = int(hue * 6.0) % 6
    rgb = [(c, x, 0.), (x, c, 0.), (0., c, x),
           (0., x, c), (x, 0., c), (c, 0., x)][h_sector]
    return (rgb[0] + m, rgb[1] + m, rgb[2] + m)


def visualize_voronoi(scalp_verts, scalp_faces, vert_labels,
                      boundary_segments, mesh_boundary,
                      roots, guides, guide_group_ids):
    """Open3D visualization of Voronoi-segmented scalp with guides."""
    import open3d as o3d

    n_seeds = len(roots)
    cell_colors = [_golden_ratio_color(i, n_seeds) for i in range(n_seeds)]
    geometries = []

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(scalp_verts)
    mesh.triangles = o3d.utility.Vector3iVector(scalp_faces)
    mesh.compute_vertex_normals()
    verts_np = np.asarray(mesh.vertices)
    norms_np = np.asarray(mesh.vertex_normals)
    centroid = verts_np.mean(axis=0)
    if (norms_np * (verts_np - centroid)).sum(axis=1).mean() < 0:
        mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles)[:, ::-1])
        mesh.compute_vertex_normals()
    vertex_colors = np.zeros((len(scalp_verts), 3))
    for vi in range(len(scalp_verts)):
        label = int(vert_labels[vi])
        vertex_colors[vi] = cell_colors[label % len(cell_colors)]
    vertex_colors *= 0.5
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    geometries.append(mesh)

    if boundary_segments:
        pts, lines = [], []
        for p1, p2 in boundary_segments:
            idx = len(pts)
            pts.append(p1); pts.append(p2)
            lines.append([idx, idx + 1])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(pts))
        ls.lines = o3d.utility.Vector2iVector(np.array(lines))
        ls.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 1.0]] * len(lines))
        geometries.append(ls)

    if mesh_boundary:
        pts, lines = [], []
        for p1, p2 in mesh_boundary:
            idx = len(pts)
            pts.append(p1); pts.append(p2)
            lines.append([idx, idx + 1])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.array(pts))
        ls.lines = o3d.utility.Vector2iVector(np.array(lines))
        ls.colors = o3d.utility.Vector3dVector([[1.0, 1.0, 0.0]] * len(lines))
        geometries.append(ls)

    bbox = mesh.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
    sphere_radius = diag * 0.003
    for si in range(n_seeds):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius, resolution=8)
        sphere.translate(roots[si])
        sphere.paint_uniform_color(list(cell_colors[si]))
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    for gi, guide in enumerate(guides):
        if len(guide) < 2:
            continue
        color = list(cell_colors[gi % len(cell_colors)])
        pts_3d = np.asarray(guide, dtype=np.float64)
        lines = [[j, j + 1] for j in range(len(pts_3d) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts_3d)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geometries.append(ls)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Voronoi Scalp Segmentation", width=1280, height=720)
    for geo in geometries:
        vis.add_geometry(geo)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.15, 0.15, 0.15])
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


def visualize_tubes(tube_meshes, guides, guide_group_ids,
                    scalp_verts, scalp_faces):
    """Open3D visualization of hair tube meshes with guide strands."""
    import open3d as o3d

    n_guides = len(guides)
    cell_colors = [_golden_ratio_color(i, n_guides) for i in range(n_guides)]
    geometries = []

    scalp = o3d.geometry.TriangleMesh()
    scalp.vertices = o3d.utility.Vector3dVector(scalp_verts)
    scalp.triangles = o3d.utility.Vector3iVector(scalp_faces)
    scalp.compute_vertex_normals()
    scalp.paint_uniform_color([0.3, 0.3, 0.3])
    geometries.append(scalp)

    for gi, tm in enumerate(tube_meshes):
        if tm is None:
            continue
        verts, faces = tm
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color(list(cell_colors[gi % len(cell_colors)]))
        geometries.append(mesh)

    for gi, guide in enumerate(guides):
        if len(guide) < 2:
            continue
        color = list(cell_colors[gi % len(cell_colors)])
        pts_3d = np.asarray(guide, dtype=np.float64)
        lines = [[j, j + 1] for j in range(len(pts_3d) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts_3d)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geometries.append(ls)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Hair Tube Meshes", width=1280, height=720)
    for geo in geometries:
        vis.add_geometry(geo)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.15, 0.15, 0.15])
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


# ===================================================================
# Main
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Standalone Voronoi scalp segmentation + hair tube generation",
    )
    parser.add_argument("--config", type=str, required=True,
                        help="Path to TOML config file")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip Open3D visualization")
    args = parser.parse_args()

    cfg = toml.load(os.path.join(PROJECT_ROOT, args.config))
    data_base = os.path.join(PROJECT_ROOT, cfg["data"]["base_path"])
    output_base = os.path.join(PROJECT_ROOT, cfg["output"]["base_path"])
    scalp_cfg = cfg["scalp"]

    # --- Load scalp mesh ---
    scalp_path = os.path.join(data_base, scalp_cfg["scalp_mesh"])
    print(f"Loading scalp mesh: {scalp_path}")
    scalp_data = load_obj_with_uv(scalp_path)
    scalp_verts = scalp_data["vertices"]
    scalp_faces = triangulate_faces(scalp_data["faces_v"])
    print(f"  {len(scalp_verts)} vertices, {len(scalp_faces)} triangles")

    # --- Load guide strands ---
    guides_obj = os.path.join(output_base, "grown_strands_segmented.obj")
    print(f"Loading guide strands: {guides_obj}")
    guides, guide_group_ids = load_guides_obj_grouped(guides_obj)
    print(f"  {len(guides)} guides in "
          f"{len(set(int(g) for g in guide_group_ids))} groups")

    # --- Extract roots ---
    roots = np.array([g[0] for g in guides], dtype=np.float64)
    print(f"  {len(roots)} seed roots")

    # --- Voronoi segmentation ---
    print("\n=== Computing Voronoi segmentation ===")
    vert_labels = compute_voronoi_labels(scalp_verts, roots)
    active_labels = set(int(l) for l in vert_labels)
    n_active = len(active_labels)
    n_empty = len(roots) - n_active
    print(f"  {len(roots)} seeds, {n_active} have vertices assigned"
          + (f" ({n_empty} seeds have no nearest vertex)" if n_empty > 0 else ""))

    # --- Penetration detection & repair ---
    print("\n=== Penetration detection & repair ===")
    body_mesh_path = os.path.join(data_base, "TSR/body_clean.obj")
    thickness = float(cfg.get("strand", {}).get("penetration_thickness", 0.01))
    n_fixed, n_boundary, boundary_ids = fix_penetrating_guides(
        guides, body_mesh_path, scalp_verts, scalp_faces, vert_labels,
        thickness=thickness,
    )
    if n_fixed > 0:
        roots = np.array([g[0] for g in guides], dtype=np.float64)
        vert_labels = compute_voronoi_labels(scalp_verts, roots)

    # --- Boundary segments ---
    print("Extracting boundary segments...")
    boundary_segments, segment_cells = extract_boundary_segments(
        scalp_verts, scalp_faces, vert_labels, roots,
    )
    print(f"  {len(boundary_segments)} boundary segments")

    # --- Mesh boundary ---
    print("Extracting mesh boundary edges...")
    mesh_boundary = extract_mesh_boundary_edges(scalp_verts, scalp_faces)
    print(f"  {len(mesh_boundary)} mesh boundary edges")

    # --- Build per-cell boundary polylines ---
    print("Building per-cell boundary polylines...")
    cell_polylines = build_cell_boundary_polylines(
        boundary_segments, segment_cells, len(roots),
        mesh_boundary=mesh_boundary,
        vert_labels=vert_labels,
        vertices=scalp_verts,
        faces=scalp_faces,
    )
    n_with_boundary = sum(1 for v in cell_polylines.values() if v)
    total_polylines = sum(len(v) for v in cell_polylines.values())
    print(f"  {n_with_boundary} cells with boundary polylines "
          f"({total_polylines} polylines total)")

    # --- Build hair tube meshes ---
    print("\n=== Hair tube generation ===")
    tube_meshes = build_tube_meshes(
        guides, cell_polylines, scalp_verts, scalp_faces,
        tip_scale=0.4,
    )

    # --- Save outputs ---
    os.makedirs(output_base, exist_ok=True)
    labels_path = os.path.join(output_base, "voronoi_vert_labels.npy")
    np.save(labels_path, vert_labels)
    print(f"\nSaved vertex labels: {labels_path}")

    polylines_path = os.path.join(output_base, "voronoi_cell_boundaries.npz")
    save_dict = {}
    for cell_id, polys in cell_polylines.items():
        for pi, poly in enumerate(polys):
            save_dict[f"cell_{cell_id}_poly_{pi}"] = poly
    np.savez_compressed(polylines_path, **save_dict)
    print(f"Saved cell boundary polylines: {polylines_path}")

    # Save tube meshes as OBJ (one group per tube)
    tubes_obj_path = os.path.join(output_base, "hair_tubes.obj")
    vert_offset = 0
    n_exported = 0
    with open(tubes_obj_path, "w") as f:
        f.write("# Hair tube meshes\n")
        for gi, tm in enumerate(tube_meshes):
            if tm is None:
                continue
            verts, faces = tm
            f.write(f"g tube_{gi}\n")
            for v in verts:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in faces:
                f.write(f"f {face[0]+1+vert_offset}"
                        f" {face[1]+1+vert_offset}"
                        f" {face[2]+1+vert_offset}\n")
            vert_offset += len(verts)
            n_exported += 1
    print(f"Saved {n_exported} tube meshes: {tubes_obj_path}")

    # --- Visualization ---
    if not args.skip_viz:
        print("\nStarting Voronoi visualization (close window to continue)...")
        visualize_voronoi(
            scalp_verts, scalp_faces, vert_labels,
            boundary_segments, mesh_boundary,
            roots, guides, guide_group_ids,
        )

        print("Starting tube visualization (close window to exit)...")
        visualize_tubes(
            tube_meshes, guides, guide_group_ids,
            scalp_verts, scalp_faces,
        )

    print("Done.")


if __name__ == "__main__":
    main()
