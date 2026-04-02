"""
CCD-Based Hair Tube Expansion.

Expands tube meshes from their centerlines outward, using edge-edge ACCD
to clamp each edge's expansion at the earliest time of impact.

See: docs/superpowers/specs/2026-03-20-ccd-tube-expansion-design.md
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch

from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd, batch_vertex_triangle_ccd


def smooth_tube_polylines(
    meshes: List[Tuple[np.ndarray, np.ndarray]],
    n_cross_list: List[int],
    n_layers_list: List[int],
    n_iters: int = 5,
    alpha: float = 0.3,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Smooth tube vertices along longitudinal polylines.

    Each tube has N polylines — one per cross-section vertex index ni,
    connecting the same ni across layers 0..L-1.  Smoothing along these
    polylines removes zigzag artifacts from CCD clamping while preserving
    cross-section shape and curvature.

    Layer 0 (pinned to scalp) and tip vertex are fixed.

    Args:
        meshes: list of (verts, faces) tuples.
        n_cross_list: per-tube cross-section vertex count N.
        n_layers_list: per-tube layer count L.
        n_iters: number of smoothing iterations.
        alpha: blending weight (0=no smooth, 1=full neighbor avg).

    Returns:
        Smoothed mesh list (same structure).
    """
    if n_iters <= 0 or alpha <= 0:
        return meshes

    results = []
    for ti in range(len(meshes)):
        verts, faces = meshes[ti]
        v = verts.copy().astype(np.float64)
        L = n_layers_list[ti]
        N = n_cross_list[ti]

        for _ in range(n_iters):
            v_new = v.copy()
            # Smooth each of the N polylines independently
            for ni in range(N):
                for li in range(1, L):  # skip layer 0 (pinned)
                    idx = li * N + ni
                    if li == L - 1:
                        # Last ring layer: one neighbor below + tip above
                        prev = (li - 1) * N + ni
                        tip = L * N
                        avg = 0.5 * (v[prev] + v[tip])
                    else:
                        prev = (li - 1) * N + ni
                        nxt = (li + 1) * N + ni
                        avg = 0.5 * (v[prev] + v[nxt])
                    v_new[idx] = (1.0 - alpha) * v[idx] + alpha * avg
            # Tip: average of last-layer ring (keep centered)
            tip_idx = L * N
            last_ring = v_new[(L - 1) * N:L * N]
            v_new[tip_idx] = last_ring.mean(axis=0)
            v = v_new

        results.append((v.astype(verts.dtype), faces))
    return results


@dataclass
class TubeTopology:
    """Parsed topology for a single tube."""
    n_layers: int          # L
    n_cross: int           # N
    n_verts: int           # L*N + 1

    # Edge arrays: (M, 2) int, indices into the tube's own vertex array
    connection_edges: np.ndarray   # (L-1)*N edges: (li,ni) -- (li+1,ni)
    ring_edges: np.ndarray         # L*N edges: (li,ni) -- (li,(ni+1)%N)
    tip_edges: np.ndarray          # N edges: (L-1,ni) -- tip

    # Per-vertex layer index: shape (n_verts,)
    layer_of_vertex: np.ndarray

    # Which vertices are pinned (layer 0 + tip)
    pinned_mask: np.ndarray        # bool, shape (n_verts,)


@dataclass
class ExpansionData:
    """Expansion start/end positions for all tubes combined."""
    v_start: torch.Tensor      # (total_verts, 3) current positions
    v_end: torch.Tensor        # (total_verts, 3) target positions
    pinned: torch.Tensor       # (total_verts,) bool — pinned vertices

    # All edges across all tubes, with global vertex indices: (total_edges, 2)
    all_edges: torch.Tensor    # int64

    # Per-edge tube ownership: (total_edges,) int — which tube each edge belongs to
    edge_tube_id: torch.Tensor

    # Offsets for reconstructing per-tube vertices
    vert_offsets: List[int]    # length n_tubes+1
    edge_offsets: List[int]    # length n_tubes+1


class TubeExpander:
    """Parses tube topology and computes expansion vectors."""

    def __init__(
        self,
        tube_meshes: List[Tuple[np.ndarray, np.ndarray]],
        guides: List[np.ndarray],
        target_scale: float = 0.8,
        current_scale: float = 0.1,
        device: str = "cuda:0",
    ):
        """
        Args:
            tube_meshes: list of (verts, faces) from build_tube_meshes.
                         None entries are skipped.
            guides: list of guide polylines, same length as tube_meshes.
            target_scale: desired tip_scale after expansion.
            current_scale: the tip_scale used during generation.
            device: torch device.
        """
        self.target_scale = target_scale
        self.current_scale = current_scale
        self.device = device

        # Filter out None entries and pair with guides
        self.tube_indices: List[int] = []
        self.meshes: List[Tuple[np.ndarray, np.ndarray]] = []
        self.guides: List[np.ndarray] = []
        for i, mesh in enumerate(tube_meshes):
            if mesh is not None:
                self.tube_indices.append(i)
                self.meshes.append(mesh)
                self.guides.append(guides[i])

        self.n_tubes = len(self.meshes)
        self.topologies: List[TubeTopology] = []
        self.expansion: Optional[ExpansionData] = None

    def auto_detect_n_cross(self, verts: np.ndarray, faces: np.ndarray) -> int:
        """Detect N (cross-section vertex count) from mesh topology.

        Formula: N = 2*(n_verts - 1) - n_faces
        Derived from: n_faces = 2*(L-1)*N + N (quads between layers + tip fan)
                      n_verts = L*N + 1
        """
        n_verts = verts.shape[0]
        n_faces = faces.shape[0]
        N = 2 * (n_verts - 1) - n_faces
        return N

    def parse_topology(self) -> List[TubeTopology]:
        """Extract edge classification and layer mapping for all tubes."""
        self.topologies = []
        for verts, faces in self.meshes:
            N = self.auto_detect_n_cross(verts, faces)
            n_verts = verts.shape[0]
            L = (n_verts - 1) // N
            assert L * N + 1 == n_verts, \
                f"Vertex count {n_verts} inconsistent with L={L}, N={N}"

            topo = self._parse_single_tube(L, N)
            self.topologies.append(topo)
        return self.topologies

    def compute_expansion(self) -> ExpansionData:
        """Compute start/end positions for all tubes.

        For each vertex at layer li:
            current_s = 1.0 + (li/(L-1)) * (current_scale - 1.0)
            target_s  = 1.0 + (li/(L-1)) * (target_scale  - 1.0)
            v_boundary = guide[li] + (v_current - guide[li]) / current_s
            v_end      = guide[li] + target_s * (v_boundary - guide[li])

        Layer 0 (current_s=1.0) and tip vertex are pinned.
        """
        if not self.topologies:
            self.parse_topology()

        all_v_start = []
        all_v_end = []
        all_pinned = []
        all_edges = []
        all_edge_tube_id = []
        vert_offsets = [0]
        edge_offsets = [0]

        global_vert_offset = 0
        global_edge_offset = 0

        for ti in range(self.n_tubes):
            verts, _ = self.meshes[ti]
            guide = self.guides[ti]
            topo = self.topologies[ti]
            L, N = topo.n_layers, topo.n_cross

            v_start = verts.copy().astype(np.float32)
            v_end = verts.copy().astype(np.float32)

            for li in range(1, L):
                t = li / (L - 1)
                current_s = 1.0 + t * (self.current_scale - 1.0)
                target_s = 1.0 + t * (self.target_scale - 1.0)

                center = guide[li].astype(np.float32)

                for ni in range(N):
                    idx = li * N + ni
                    offset_from_center = v_start[idx] - center
                    # Recover full-scale boundary position
                    v_boundary = center + offset_from_center / current_s
                    v_end[idx] = center + target_s * (v_boundary - center)

            # Tip vertex: pinned at guide[-1]
            # (v_start and v_end already equal for tip)

            all_v_start.append(v_start)
            all_v_end.append(v_end)
            all_pinned.append(topo.pinned_mask)

            # Combine all edges with global offset
            edges_local = np.concatenate([
                topo.connection_edges,
                topo.ring_edges,
                topo.tip_edges,
            ], axis=0)
            edges_global = edges_local + global_vert_offset
            all_edges.append(edges_global)

            n_edges = edges_global.shape[0]
            all_edge_tube_id.append(np.full(n_edges, ti, dtype=np.int32))

            global_vert_offset += topo.n_verts
            global_edge_offset += n_edges
            vert_offsets.append(global_vert_offset)
            edge_offsets.append(global_edge_offset)

        # Stack and convert to torch
        self.expansion = ExpansionData(
            v_start=torch.from_numpy(np.concatenate(all_v_start, axis=0)).to(self.device),
            v_end=torch.from_numpy(np.concatenate(all_v_end, axis=0)).to(self.device),
            pinned=torch.from_numpy(np.concatenate(all_pinned, axis=0)).to(self.device),
            all_edges=torch.from_numpy(np.concatenate(all_edges, axis=0)).long().to(self.device),
            edge_tube_id=torch.from_numpy(np.concatenate(all_edge_tube_id, axis=0)).to(self.device),
            vert_offsets=vert_offsets,
            edge_offsets=edge_offsets,
        )
        return self.expansion

    def get_expanded_meshes(
        self, t_vertex: torch.Tensor,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Apply per-vertex expansion factors to produce final meshes.

        Args:
            t_vertex: (total_verts,) float in [0, 1], expansion factor per vertex.

        Returns:
            List of (verts, faces) with updated vertex positions.
        """
        exp = self.expansion
        v_final = exp.v_start + t_vertex.unsqueeze(1) * (exp.v_end - exp.v_start)
        v_final_np = v_final.cpu().numpy()

        results = []
        for ti in range(self.n_tubes):
            off = exp.vert_offsets[ti]
            n_v = self.topologies[ti].n_verts
            new_verts = v_final_np[off:off + n_v]
            _, faces = self.meshes[ti]
            results.append((new_verts, faces))
        return results

    def resample_centerlines(
        self, v_final: torch.Tensor,
    ) -> List[np.ndarray]:
        """Resample guide centerlines as per-layer centroids.

        Args:
            v_final: (total_verts, 3) final vertex positions.

        Returns:
            List of (L, 3) arrays — one new guide per tube.
        """
        v_np = v_final.cpu().numpy()
        new_guides = []
        for ti in range(self.n_tubes):
            topo = self.topologies[ti]
            L, N = topo.n_layers, topo.n_cross
            off = self.expansion.vert_offsets[ti]
            centers = np.zeros((L, 3), dtype=np.float32)
            for li in range(L):
                ring_start = off + li * N
                ring_end = ring_start + N
                centers[li] = v_np[ring_start:ring_end].mean(axis=0)
            new_guides.append(centers)
        return new_guides

    def update_positions(
        self,
        start_meshes: List[Tuple[np.ndarray, np.ndarray]],
        end_meshes: List[Tuple[np.ndarray, np.ndarray]],
    ):
        """Update v_start/v_end in the cached expansion data without rebuilding topology.

        Args:
            start_meshes: list of (verts, faces) — current positions.
            end_meshes: list of (verts, faces) — target positions.
        """
        exp = self.expansion
        all_v_start = []
        all_v_end = []
        for ti in range(self.n_tubes):
            v_s, _ = start_meshes[ti]
            v_e, _ = end_meshes[ti]
            all_v_start.append(v_s.astype(np.float32))
            all_v_end.append(v_e.astype(np.float32))

        exp.v_start = torch.from_numpy(
            np.concatenate(all_v_start, axis=0)).to(self.device)
        exp.v_end = torch.from_numpy(
            np.concatenate(all_v_end, axis=0)).to(self.device)

    def build_swept_triangles(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode each swept edge as 2 triangles for BVH broad-phase.

        Each edge sweeps from (a_start, b_start) to (a_end, b_end), tracing a
        quad.  We split that quad into two triangles:
            tri_0 = (a_start, b_start, a_end)
            tri_1 = (b_start, b_end,   a_end)

        Returns:
            swept_tris: (2*E, 3, 3) float32 triangle vertex coords
            tri_to_edge: (2*E,) int64 mapping each triangle back to its edge index
        """
        exp = self.expansion
        edges = exp.all_edges  # (E, 2)
        E = edges.shape[0]

        a_idx = edges[:, 0]  # (E,)
        b_idx = edges[:, 1]  # (E,)

        a_start = exp.v_start[a_idx]  # (E, 3)
        b_start = exp.v_start[b_idx]
        a_end = exp.v_end[a_idx]
        b_end = exp.v_end[b_idx]

        # tri_0: (a_start, b_start, a_end)  -- shape (E, 3, 3)
        tri_0 = torch.stack([a_start, b_start, a_end], dim=1)
        # tri_1: (b_start, b_end, a_end)
        tri_1 = torch.stack([b_start, b_end, a_end], dim=1)

        # Interleave: [tri_0[0], tri_1[0], tri_0[1], tri_1[1], ...]
        swept_tris = torch.stack([tri_0, tri_1], dim=1).reshape(2 * E, 3, 3)

        # Mapping: triangle 2*i and 2*i+1 both belong to edge i
        tri_to_edge = torch.arange(E, device=self.device).repeat_interleave(2)

        return swept_tris, tri_to_edge

    def build_face_triangles(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build mesh face triangles for BVH broad-phase VT CCD.

        Each tube face is encoded as a triangle using start positions (for
        broad-phase overlap with swept-edge quads). The actual CCD uses
        start/end positions from v_start/v_end.

        Returns:
            face_tris: (F_total, 3, 3) float32 — triangle vertex coords (start pos)
            face_verts_idx: (F_total, 3) int64 — global vertex indices per face
            face_tube_id: (F_total,) int32 — tube ownership per face
        """
        exp = self.expansion
        all_tris = []
        all_vidx = []
        all_tube_id = []

        for ti in range(self.n_tubes):
            _, faces = self.meshes[ti]
            v_off = exp.vert_offsets[ti]
            global_faces = faces.astype(np.int64) + v_off  # (F, 3)
            global_faces_t = torch.from_numpy(global_faces).to(self.device)

            # Triangle coords from start positions
            tris = exp.v_start[global_faces_t]  # (F, 3, 3)
            all_tris.append(tris)
            all_vidx.append(global_faces_t)
            all_tube_id.append(torch.full(
                (faces.shape[0],), ti, dtype=torch.int32, device=self.device))

        face_tris = torch.cat(all_tris, dim=0)
        face_verts_idx = torch.cat(all_vidx, dim=0)
        face_tube_id = torch.cat(all_tube_id, dim=0)
        return face_tris, face_verts_idx, face_tube_id

    def smooth_t_values(
        self,
        t_vertex: torch.Tensor,
        n_iters: int = 5,
        alpha: float = 0.5,
    ) -> torch.Tensor:
        """Laplacian smoothing of per-vertex t along tube mesh edges.

        Only smooths downward: a vertex's t can decrease from neighbors
        but never exceed its own CCD-computed maximum.

        Args:
            t_vertex: (total_verts,) per-vertex expansion factor from CCD.
            n_iters: number of smoothing iterations.
            alpha: blending weight per iteration (0=no smooth, 1=full neighbor avg).

        Returns:
            Smoothed t_vertex (same shape).
        """
        if n_iters <= 0 or alpha <= 0:
            return t_vertex

        exp = self.expansion
        edges = exp.all_edges  # (E, 2)
        t_max = t_vertex.clone()  # per-vertex ceiling from CCD
        t = t_vertex.clone()

        for _ in range(n_iters):
            # Accumulate neighbor sum and count via scatter
            neighbor_sum = torch.zeros_like(t)
            neighbor_cnt = torch.zeros_like(t)
            ones = torch.ones(edges.shape[0], device=self.device)

            # Each edge contributes both directions
            neighbor_sum.scatter_add_(0, edges[:, 0], t[edges[:, 1]])
            neighbor_sum.scatter_add_(0, edges[:, 1], t[edges[:, 0]])
            neighbor_cnt.scatter_add_(0, edges[:, 0], ones)
            neighbor_cnt.scatter_add_(0, edges[:, 1], ones)

            # Avoid division by zero for isolated vertices
            has_neighbors = neighbor_cnt > 0
            neighbor_avg = torch.where(
                has_neighbors, neighbor_sum / neighbor_cnt, t)

            # Blend: t_new = (1-alpha)*t + alpha*neighbor_avg
            t_new = (1.0 - alpha) * t + alpha * neighbor_avg

            # Clamp: never exceed each vertex's own CCD t
            t = torch.min(t_new, t_max)

        # Pinned vertices stay at t=1
        t[exp.pinned] = 1.0

        return t

    def run_ccd(
        self,
        max_collisions: int = 128,
        xi: float = 1e-3,
        body_mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        wrapper_mesh: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        smooth_iters: int = 20,
        smooth_alpha: float = 0.5,
    ) -> torch.Tensor:
        """Run BVH broad-phase + ACCD narrow-phase.

        Uses two separate BVH passes to avoid exceeding GPU memory:
          Pass 1 — tube-only: swept tris + face tris → EE + VT CCD
          Pass 2 — boundary:  swept tris + boundary tris → EE + VT CCD

        Args:
            max_collisions: BVH per-triangle collision buffer size.
            xi: ACCD thickness / minimum separation.
            body_mesh: (verts, faces) of body mesh (stay outside). None to skip.
            wrapper_mesh: (verts, faces) of wrapper mesh (stay inside). None to skip.
            smooth_iters: Laplacian smoothing iterations for t values (0=disable).
            smooth_alpha: smoothing blend weight (0=none, 1=full neighbor avg).

        Returns:
            t_vertex: (total_verts,) float in [0, 1], expansion factor per vertex.
        """
        from mesh_intersection.bvh_search_tree import BVH

        exp = self.expansion

        # Build swept triangles for tube edges
        swept_tris, tri_to_edge = self.build_swept_triangles()
        n_swept = swept_tris.shape[0]

        # Build face triangles for vertex-triangle CCD
        face_tris, face_verts_idx, face_tube_id = self.build_face_triangles()
        n_face = face_tris.shape[0]

        # Initialize per-edge and per-vertex TOI accumulators
        n_edges = exp.all_edges.shape[0]
        n_verts = exp.v_start.shape[0]
        edge_toi = torch.ones(n_edges, device=self.device)
        t_vertex = torch.ones(n_verts, device=self.device)

        # ============================================================
        # Pass 1: Tube-only BVH (swept + face tris) → EE + VT CCD
        # ============================================================
        tube_tris = torch.cat([swept_tris, face_tris], dim=0)
        print(f"  BVH pass 1 (tube): {tube_tris.shape[0]} triangles "
              f"({n_swept} swept + {n_face} faces)")

        search_tree = BVH(max_collisions=max_collisions)
        with torch.no_grad():
            raw = search_tree(tube_tris.unsqueeze(0)).squeeze(0)
            valid_mask = raw[:, 0] >= 0
            tri_pairs = raw[valid_mask]

        if tri_pairs.shape[0] > 0:
            is_swept_a = tri_pairs[:, 0] < n_swept
            is_swept_b = tri_pairs[:, 1] < n_swept

            # Swept-vs-swept → edge-edge CCD
            tt_mask = is_swept_a & is_swept_b
            tt_pairs = tri_pairs[tt_mask]
            if tt_pairs.shape[0] > 0:
                edge_toi, t_vertex = self._process_tube_tube(
                    tt_pairs, tri_to_edge, edge_toi, t_vertex, xi)

            # Swept-vs-face → vertex-triangle CCD
            sf_mask = is_swept_a != is_swept_b  # exactly one is swept
            sf_pairs = tri_pairs[sf_mask]
            if sf_pairs.shape[0] > 0:
                edge_toi, t_vertex = self._process_tube_swept_vs_face(
                    sf_pairs, tri_to_edge,
                    torch.arange(n_face, dtype=torch.long, device=self.device),
                    face_verts_idx, face_tube_id, n_swept,
                    edge_toi, t_vertex, xi)

        # Free tube-only BVH memory
        del tube_tris, raw, tri_pairs
        torch.cuda.empty_cache()

        # ============================================================
        # Pass 2: Boundary BVH (swept + boundary tris) → EE + VT CCD
        # ============================================================
        has_boundary = body_mesh is not None or wrapper_mesh is not None
        if has_boundary:
            # tri_source for this pass: 0=swept, 1=body, 2=wrapper
            bnd_source = torch.zeros(
                n_swept, dtype=torch.int32, device=self.device)
            bnd_face_idx = torch.full(
                (n_swept,), -1, dtype=torch.long, device=self.device)
            bnd_tris_list = [swept_tris]
            boundary_data = {}

            for source_id, mesh_data, name in [
                (1, body_mesh, "body"), (2, wrapper_mesh, "wrapper")
            ]:
                if mesh_data is None:
                    continue
                bverts, bfaces = mesh_data
                bverts_t = torch.from_numpy(
                    bverts.astype(np.float32)).to(self.device)
                bfaces_t = torch.from_numpy(
                    bfaces.astype(np.int64)).to(self.device)
                btris = bverts_t[bfaces_t]  # (n_b, 3, 3)
                n_b = btris.shape[0]

                # Inflate boundary triangles for BVH broadphase by xi:
                # expand each vertex outward from centroid so the AABB grows
                # by ~xi in all directions. CCD still uses original positions.
                centroid = btris.mean(dim=1, keepdim=True)  # (n_b, 1, 3)
                direction = btris - centroid  # (n_b, 3, 3)
                norms = direction.norm(dim=2, keepdim=True).clamp(min=1e-12)
                btris_inflated = btris + direction / norms * xi

                bnd_source = torch.cat([
                    bnd_source,
                    torch.full((n_b,), source_id, dtype=torch.int32,
                               device=self.device),
                ])
                bnd_face_idx = torch.cat([
                    bnd_face_idx,
                    torch.arange(n_b, dtype=torch.long, device=self.device),
                ])
                bnd_tris_list.append(btris_inflated)
                boundary_data[source_id] = (bverts_t, bfaces_t)
                print(f"  {name} mesh: {n_b} triangles")

            bnd_all_tris = torch.cat(bnd_tris_list, dim=0)
            n_bnd = bnd_all_tris.shape[0] - n_swept
            print(f"  BVH pass 2 (boundary): {bnd_all_tris.shape[0]} triangles "
                  f"({n_swept} swept + {n_bnd} boundary)")

            search_tree2 = BVH(max_collisions=max_collisions)
            with torch.no_grad():
                raw2 = search_tree2(bnd_all_tris.unsqueeze(0)).squeeze(0)
                valid2 = raw2[:, 0] >= 0
                tri_pairs2 = raw2[valid2]

            if tri_pairs2.shape[0] > 0:
                src_a2 = bnd_source[tri_pairs2[:, 0]]
                src_b2 = bnd_source[tri_pairs2[:, 1]]

                # Tube-vs-boundary pairs (one is 0, other is 1 or 2)
                tb_mask = ((src_a2 == 0) & (src_b2 > 0)) | \
                          ((src_a2 > 0) & (src_b2 == 0))
                tb_pairs_raw = tri_pairs2[tb_mask]

                if tb_pairs_raw.shape[0] > 0:
                    tb_src_a = src_a2[tb_mask]
                    swap = tb_src_a > 0
                    tb_tube_tri = torch.where(
                        swap, tb_pairs_raw[:, 1], tb_pairs_raw[:, 0])
                    tb_bnd_tri = torch.where(
                        swap, tb_pairs_raw[:, 0], tb_pairs_raw[:, 1])

                    edge_toi, t_vertex = self._process_tube_boundary(
                        tb_tube_tri, tb_bnd_tri, tri_to_edge,
                        bnd_source, bnd_face_idx,
                        boundary_data, edge_toi, t_vertex, xi)

            # --- Direct vertex-vs-boundary VT pass ---
            # The edge-based BVH may miss VT pairs where a vertex
            # approaches a boundary triangle that doesn't overlap
            # any of the vertex's edges' swept triangles.
            # Check all tube vertices against boundary triangles directly.
            t_vertex = self._direct_vertex_boundary_ccd(
                boundary_data, t_vertex, xi)

        # Propagate edge TOI to vertices
        edges = exp.all_edges
        t_vertex.scatter_reduce_(
            0, edges[:, 0], edge_toi, reduce="amin", include_self=True)
        t_vertex.scatter_reduce_(
            0, edges[:, 1], edge_toi, reduce="amin", include_self=True)

        # Safety margin
        EPS = 1e-6
        clamped = t_vertex < 1.0
        t_vertex[clamped] = t_vertex[clamped] * (1.0 - EPS)

        # Pinned vertices stay at t=1
        t_vertex[exp.pinned] = 1.0

        # Laplacian smoothing (downward only)
        if smooth_iters > 0:
            t_vertex = self.smooth_t_values(
                t_vertex, n_iters=smooth_iters, alpha=smooth_alpha)

        return t_vertex

    def _process_tube_tube(
        self,
        tt_pairs: torch.Tensor,
        tri_to_edge: torch.Tensor,
        edge_toi: torch.Tensor,
        t_vertex: torch.Tensor,
        xi: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process tube-vs-tube collision pairs: edge-edge CCD."""
        exp = self.expansion

        edge_i = tri_to_edge[tt_pairs[:, 0]]
        edge_j = tri_to_edge[tt_pairs[:, 1]]

        edge_pair = torch.stack([
            torch.min(edge_i, edge_j),
            torch.max(edge_i, edge_j),
        ], dim=1)

        diff_mask = edge_pair[:, 0] != edge_pair[:, 1]
        edge_pair = edge_pair[diff_mask]

        if edge_pair.shape[0] == 0:
            return edge_toi, t_vertex

        edge_pair = torch.unique(edge_pair, dim=0)

        tube_i = exp.edge_tube_id[edge_pair[:, 0]]
        tube_j = exp.edge_tube_id[edge_pair[:, 1]]
        inter_mask = tube_i != tube_j
        edge_pair = edge_pair[inter_mask]

        if edge_pair.shape[0] == 0:
            return edge_toi, t_vertex

        ei = edge_pair[:, 0]
        ej = edge_pair[:, 1]
        edges = exp.all_edges

        ai0 = edges[ei, 0]
        ai1 = edges[ei, 1]
        aj0 = edges[ej, 0]
        aj1 = edges[ej, 1]

        toi = batch_edge_edge_ccd(
            a0_start=exp.v_start[ai0], a1_start=exp.v_start[ai1],
            b0_start=exp.v_start[aj0], b1_start=exp.v_start[aj1],
            a0_end=exp.v_end[ai0],     a1_end=exp.v_end[ai1],
            b0_end=exp.v_end[aj0],     b1_end=exp.v_end[aj1],
            xi=xi,
        )

        edge_toi.scatter_reduce_(0, ei, toi, reduce="amin", include_self=True)
        edge_toi.scatter_reduce_(0, ej, toi, reduce="amin", include_self=True)

        n_clamped = (toi < 1.0).sum().item()
        print(f"  tube-tube EE: {edge_pair.shape[0]} pairs, clamped={n_clamped}")

        return edge_toi, t_vertex

    def _process_tube_swept_vs_face(
        self,
        sf_pairs: torch.Tensor,
        tri_to_edge: torch.Tensor,
        tri_face_idx: torch.Tensor,
        face_verts_idx: torch.Tensor,
        face_tube_id: torch.Tensor,
        n_swept: int,
        edge_toi: torch.Tensor,
        t_vertex: torch.Tensor,
        xi: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process tube swept-edge vs tube face pairs: vertex-triangle CCD.

        For each (swept_edge, mesh_face) pair from different tubes, test
        the 2 edge endpoints against the moving face triangle.

        Args:
            sf_pairs: (P, 2) BVH pairs where one is swept tri, other is face tri.
            tri_to_edge: (2*E,) mapping swept tri index → edge index.
            tri_face_idx: (n_face_tris,) mapping face tri BVH index → local face index.
            face_verts_idx: (F_total, 3) global vertex indices per face.
            face_tube_id: (F_total,) tube ownership per face.
            n_swept: number of swept triangles (face tris start at this offset).
        """
        exp = self.expansion

        # Identify which column is swept vs face
        is_swept_a = sf_pairs[:, 0] < n_swept
        swept_tri = torch.where(is_swept_a, sf_pairs[:, 0], sf_pairs[:, 1])
        face_tri = torch.where(is_swept_a, sf_pairs[:, 1], sf_pairs[:, 0])

        # Map to edge and face indices
        edge_idx = tri_to_edge[swept_tri]
        fidx = tri_face_idx[face_tri - n_swept]

        # Filter inter-tube only
        edge_tube = exp.edge_tube_id[edge_idx]
        face_tube = face_tube_id[fidx]
        inter = edge_tube != face_tube
        edge_idx = edge_idx[inter]
        fidx = fidx[inter]

        if edge_idx.shape[0] == 0:
            return edge_toi, t_vertex

        # Deduplicate (edge, face) pairs
        keys = torch.stack([edge_idx, fidx], dim=1)
        keys = torch.unique(keys, dim=0)
        edge_idx = keys[:, 0]
        fidx = keys[:, 1]

        M = edge_idx.shape[0]
        edges = exp.all_edges

        ev0 = edges[edge_idx, 0]
        ev1 = edges[edge_idx, 1]

        fv = face_verts_idx[fidx]  # (M, 3)
        fv0, fv1, fv2 = fv[:, 0], fv[:, 1], fv[:, 2]

        # VT CCD: edge endpoint 0 vs face
        vt_toi_0 = batch_vertex_triangle_ccd(
            p_start=exp.v_start[ev0],
            t0_start=exp.v_start[fv0], t1_start=exp.v_start[fv1],
            t2_start=exp.v_start[fv2],
            p_end=exp.v_end[ev0],
            t0_end=exp.v_end[fv0], t1_end=exp.v_end[fv1],
            t2_end=exp.v_end[fv2],
            xi=xi,
        )

        # VT CCD: edge endpoint 1 vs face
        vt_toi_1 = batch_vertex_triangle_ccd(
            p_start=exp.v_start[ev1],
            t0_start=exp.v_start[fv0], t1_start=exp.v_start[fv1],
            t2_start=exp.v_start[fv2],
            p_end=exp.v_end[ev1],
            t0_end=exp.v_end[fv0], t1_end=exp.v_end[fv1],
            t2_end=exp.v_end[fv2],
            xi=xi,
        )

        # Scatter to per-vertex TOI
        t_vertex.scatter_reduce_(
            0, ev0, vt_toi_0, reduce="amin", include_self=True)
        t_vertex.scatter_reduce_(
            0, ev1, vt_toi_1, reduce="amin", include_self=True)

        # Also update edge TOI
        vt_min = torch.min(vt_toi_0, vt_toi_1)
        edge_toi.scatter_reduce_(
            0, edge_idx, vt_min, reduce="amin", include_self=True)

        n_clamped = ((vt_toi_0 < 1.0) | (vt_toi_1 < 1.0)).sum().item()
        print(f"  tube swept-vs-face VT: {M} pairs, clamped={n_clamped}")

        return edge_toi, t_vertex

    def _process_tube_boundary(
        self,
        tb_tube_tri: torch.Tensor,
        tb_bnd_tri: torch.Tensor,
        tri_to_edge: torch.Tensor,
        tri_source: torch.Tensor,
        tri_boundary_face_idx: torch.Tensor,
        boundary_data: dict,
        edge_toi: torch.Tensor,
        t_vertex: torch.Tensor,
        xi: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process tube-vs-boundary collision pairs.

        For each (tube_swept_tri, boundary_tri) pair:
          - Edge-edge CCD: tube edge vs 3 boundary triangle edges (static)
          - Vertex-triangle CCD: 2 tube edge endpoints vs boundary triangle (static)
        """
        exp = self.expansion
        edges = exp.all_edges

        # Map tube triangles back to tube edges
        tube_edge_idx = tri_to_edge[tb_tube_tri]  # (P,)

        # Get boundary triangle source and face index
        bnd_source = tri_source[tb_bnd_tri]  # (P,) values in {1, 2}
        bnd_face = tri_boundary_face_idx[tb_bnd_tri]  # (P,)

        # Deduplicate (tube_edge, boundary_face, source) triples
        keys = torch.stack([tube_edge_idx, bnd_face, bnd_source.long()], dim=1)
        keys_unique = torch.unique(keys, dim=0)
        tube_edge_idx = keys_unique[:, 0]
        bnd_face = keys_unique[:, 1]
        bnd_source = keys_unique[:, 2].int()

        P = tube_edge_idx.shape[0]
        if P == 0:
            return edge_toi, t_vertex

        # Get tube edge endpoint positions
        te_v0_idx = edges[tube_edge_idx, 0]  # (P,)
        te_v1_idx = edges[tube_edge_idx, 1]  # (P,)

        te_v0_start = exp.v_start[te_v0_idx]  # (P, 3)
        te_v1_start = exp.v_start[te_v1_idx]
        te_v0_end = exp.v_end[te_v0_idx]
        te_v1_end = exp.v_end[te_v1_idx]

        # Get boundary triangle vertex positions (static: start == end)
        # We need to gather from the correct boundary mesh based on source
        # Process each boundary source separately
        for src_id in boundary_data:
            src_mask = bnd_source == src_id
            if src_mask.sum() == 0:
                continue

            bverts, bfaces = boundary_data[src_id]
            src_face = bnd_face[src_mask]
            src_tube_edge = tube_edge_idx[src_mask]
            src_te_v0_idx = te_v0_idx[src_mask]
            src_te_v1_idx = te_v1_idx[src_mask]
            src_te_v0_start = te_v0_start[src_mask]
            src_te_v1_start = te_v1_start[src_mask]
            src_te_v0_end = te_v0_end[src_mask]
            src_te_v1_end = te_v1_end[src_mask]

            # Boundary triangle vertices (static)
            bt0 = bverts[bfaces[src_face, 0]]  # (M, 3)
            bt1 = bverts[bfaces[src_face, 1]]
            bt2 = bverts[bfaces[src_face, 2]]

            M = src_face.shape[0]

            # --- Edge-edge CCD: tube edge vs 3 boundary edges ---
            # Boundary edges: (bt0,bt1), (bt1,bt2), (bt2,bt0)
            # Repeat tube edge 3x, boundary edges cycle through 3 edges
            te_a0s = src_te_v0_start.repeat(3, 1)  # (3M, 3)
            te_a1s = src_te_v1_start.repeat(3, 1)
            te_a0e = src_te_v0_end.repeat(3, 1)
            te_a1e = src_te_v1_end.repeat(3, 1)

            be_b0 = torch.cat([bt0, bt1, bt2], dim=0)  # (3M, 3)
            be_b1 = torch.cat([bt1, bt2, bt0], dim=0)

            ee_toi = batch_edge_edge_ccd(
                a0_start=te_a0s, a1_start=te_a1s,
                b0_start=be_b0,  b1_start=be_b1,
                a0_end=te_a0e,   a1_end=te_a1e,
                b0_end=be_b0,    b1_end=be_b1,  # static
                xi=xi,
            )  # (3M,)

            # Reduce: min across 3 boundary edges per pair
            ee_toi_3 = ee_toi.view(3, M).min(dim=0).values  # (M,)
            edge_toi.scatter_reduce_(
                0, src_tube_edge, ee_toi_3, reduce="amin", include_self=True)

            # --- Vertex-triangle CCD: 2 tube endpoints vs boundary tri ---
            # Endpoint 0
            vt_toi_0 = batch_vertex_triangle_ccd(
                p_start=src_te_v0_start,
                t0_start=bt0, t1_start=bt1, t2_start=bt2,
                p_end=src_te_v0_end,
                t0_end=bt0, t1_end=bt1, t2_end=bt2,  # static
                xi=xi,
            )  # (M,)
            # Endpoint 1
            vt_toi_1 = batch_vertex_triangle_ccd(
                p_start=src_te_v1_start,
                t0_start=bt0, t1_start=bt1, t2_start=bt2,
                p_end=src_te_v1_end,
                t0_end=bt0, t1_end=bt1, t2_end=bt2,  # static
                xi=xi,
            )  # (M,)

            # Scatter to per-vertex TOI directly
            t_vertex.scatter_reduce_(
                0, src_te_v0_idx, vt_toi_0, reduce="amin", include_self=True)
            t_vertex.scatter_reduce_(
                0, src_te_v1_idx, vt_toi_1, reduce="amin", include_self=True)

            # Also update edge TOI with VT results
            vt_min = torch.min(vt_toi_0, vt_toi_1)
            edge_toi.scatter_reduce_(
                0, src_tube_edge, vt_min, reduce="amin", include_self=True)

            src_name = "body" if src_id == 1 else "wrapper"
            n_clamped_ee = (ee_toi_3 < 1.0).sum().item()
            n_clamped_vt = ((vt_toi_0 < 1.0) | (vt_toi_1 < 1.0)).sum().item()
            print(f"  {src_name}: {M} pairs, "
                  f"EE clamped={n_clamped_ee}, VT clamped={n_clamped_vt}")

        return edge_toi, t_vertex

    def _direct_vertex_boundary_ccd(
        self,
        boundary_data: dict,
        t_vertex: torch.Tensor,
        xi: float,
        max_collisions: int = 64,
    ) -> torch.Tensor:
        """Direct vertex-vs-boundary-triangle CCD for all tube vertices.

        Complements the edge-based BVH pass by catching VT pairs where a
        vertex trajectory doesn't produce an edge swept-triangle that overlaps
        the boundary triangle AABB.

        Uses cached AABB filtering with chunked processing.
        """
        exp = self.expansion
        n_verts = exp.v_start.shape[0]

        for src_id, (bverts, bfaces) in boundary_data.items():
            btris = bverts[bfaces]  # (n_b, 3, 3)
            btri_min = btris.min(dim=1).values  # (n_b, 3)
            btri_max = btris.max(dim=1).values
            n_b = btris.shape[0]

            # Vertex trajectory AABBs: min/max of start and end, inflated by xi
            v_min = torch.min(exp.v_start, exp.v_end) - xi  # (V, 3)
            v_max = torch.max(exp.v_start, exp.v_end) + xi  # (V, 3)

            # Find overlapping pairs via chunked broadcasting
            chunk_size = max(1, min(4096, 2**28 // (n_verts * 3)))

            all_vert_idx = []
            all_tri_idx = []

            for b_start in range(0, n_b, chunk_size):
                b_end = min(b_start + chunk_size, n_b)

                # (V, 1, 3) vs (1, chunk, 3) → (V, chunk)
                overlap = (
                    (v_min[:, None, :] <= btri_max[None, b_start:b_end, :]) &
                    (v_max[:, None, :] >= btri_min[None, b_start:b_end, :])
                ).all(dim=2)  # (V, chunk)

                vi, bi = torch.where(overlap)
                if vi.shape[0] > 0:
                    all_vert_idx.append(vi)
                    all_tri_idx.append(bi + b_start)

            if not all_vert_idx:
                continue

            vert_idx = torch.cat(all_vert_idx)
            tri_idx = torch.cat(all_tri_idx)

            # Deduplicate
            keys = torch.stack([vert_idx, tri_idx], dim=1)
            keys = torch.unique(keys, dim=0)
            vert_idx = keys[:, 0]
            tri_idx = keys[:, 1]

            P = vert_idx.shape[0]
            if P == 0:
                continue

            # Run VT CCD
            bt0 = bverts[bfaces[tri_idx, 0]]
            bt1 = bverts[bfaces[tri_idx, 1]]
            bt2 = bverts[bfaces[tri_idx, 2]]

            vt_toi = batch_vertex_triangle_ccd(
                p_start=exp.v_start[vert_idx],
                t0_start=bt0, t1_start=bt1, t2_start=bt2,
                p_end=exp.v_end[vert_idx],
                t0_end=bt0, t1_end=bt1, t2_end=bt2,
                xi=xi,
            )

            # Scatter min TOI to t_vertex
            t_vertex.scatter_reduce_(
                0, vert_idx, vt_toi, reduce="amin", include_self=True)

            n_clamped = (vt_toi < 1.0).sum().item()
            src_name = "body" if src_id == 1 else "wrapper"
            print(f"  direct VT {src_name}: {P} pairs, clamped={n_clamped}")

        return t_vertex

    def expand(self, max_collisions: int = 128) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray]]:
        """Full expansion pipeline: parse, compute, CCD, apply.

        Returns:
            expanded_meshes: list of (verts, faces)
            new_guides: list of (L, 3) resampled centerlines
        """
        self.compute_expansion()
        t_vertex = self.run_ccd(max_collisions=max_collisions)

        expanded_meshes = self.get_expanded_meshes(t_vertex)

        exp = self.expansion
        v_final = exp.v_start + t_vertex.unsqueeze(1) * (exp.v_end - exp.v_start)
        new_guides = self.resample_centerlines(v_final)

        return expanded_meshes, new_guides

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_single_tube(L: int, N: int) -> TubeTopology:
        """Build edge arrays and layer mapping for one tube."""
        n_verts = L * N + 1

        # Connection edges: (li, ni) -- (li+1, ni)
        conn = []
        for li in range(L - 1):
            for ni in range(N):
                a = li * N + ni
                b = (li + 1) * N + ni
                conn.append([a, b])
        conn = np.array(conn, dtype=np.int64) if conn else np.empty((0, 2), dtype=np.int64)

        # Ring edges: (li, ni) -- (li, (ni+1)%N)
        ring = []
        for li in range(L):
            for ni in range(N):
                a = li * N + ni
                b = li * N + (ni + 1) % N
                ring.append([a, b])
        ring = np.array(ring, dtype=np.int64) if ring else np.empty((0, 2), dtype=np.int64)

        # Tip edges: last layer ring -- tip vertex
        tip = []
        tip_idx = L * N
        for ni in range(N):
            a = (L - 1) * N + ni
            tip.append([a, tip_idx])
        tip = np.array(tip, dtype=np.int64) if tip else np.empty((0, 2), dtype=np.int64)

        # Layer-of-vertex mapping
        layer_of = np.zeros(n_verts, dtype=np.int32)
        for li in range(L):
            layer_of[li * N: (li + 1) * N] = li
        layer_of[tip_idx] = L - 1  # tip treated as last layer

        # Pinned: layer 0 + tip
        pinned = np.zeros(n_verts, dtype=bool)
        pinned[:N] = True       # layer 0
        pinned[tip_idx] = True  # tip

        return TubeTopology(
            n_layers=L,
            n_cross=N,
            n_verts=n_verts,
            connection_edges=conn,
            ring_edges=ring,
            tip_edges=tip,
            layer_of_vertex=layer_of,
            pinned_mask=pinned,
        )


class GuideDeformer:
    """Deforms guide centerlines in response to tube collision clamping.

    When CCD clamps tube expansion (t < 1), the blocked displacement is
    transferred to the guide as a per-layer displacement. This displacement
    is then smoothed along the guide via Laplacian diffusion (blended
    through the elastic rod material) so that guides move smoothly.
    """

    def __init__(
        self,
        smooth_iters: int = 20,
        smooth_alpha: float = 0.5,
    ):
        """
        Args:
            smooth_iters: Laplacian smoothing iterations for guide displacement.
            smooth_alpha: blending weight per iteration (0=no smooth, 1=full neighbor avg).
        """
        self.smooth_iters = smooth_iters
        self.smooth_alpha = smooth_alpha

    def compute_layer_displacements(
        self,
        guides: List[np.ndarray],
        clamped_meshes: List[Tuple[np.ndarray, np.ndarray]],
        target_meshes: List[Tuple[np.ndarray, np.ndarray]],
        t_vertex: torch.Tensor,
        vert_offsets: List[int],
        n_cross_list: List[int],
        n_layers_list: List[int],
    ) -> List[np.ndarray]:
        """Compute per-layer guide displacement from tube CCD clamping.

        For each clamped vertex (t < 1), the blocked displacement is
        (v_target - v_clamped) — the distance the vertex wanted to go
        but couldn't due to collision. The NEGATED average over each ring
        gives the guide displacement: push the guide AWAY from the
        collision side so the tube has room next step.

        Returns:
            List of (L, 3) displacement arrays, one per tube.
        """
        t_np = t_vertex.cpu().numpy()
        displacements = []

        for ti in range(len(guides)):
            L = n_layers_list[ti]
            N = n_cross_list[ti]
            v_off = vert_offsets[ti]
            v_clamp = clamped_meshes[ti][0].astype(np.float64)
            v_tgt = target_meshes[ti][0].astype(np.float64)

            # Reshape ring vertices to (L, N, 3), skip tip vertex
            clamp_rings = v_clamp[:L * N].reshape(L, N, 3)
            tgt_rings = v_tgt[:L * N].reshape(L, N, 3)

            # Blocked displacement per vertex: (L, N, 3)
            blocked = tgt_rings - clamp_rings

            # Sum over ring, negate, average — layers 1..L-1
            ring_sum = blocked.sum(axis=1)  # (L, 3)
            layer_disp = -ring_sum / N
            layer_disp[0] = 0.0  # pin root

            displacements.append(layer_disp)
        return displacements

    def deform(
        self,
        guides: List[np.ndarray],
        displacements: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Apply per-layer displacements to guides, blended through material.

        The raw per-layer displacement is smoothed along the guide via
        Laplacian diffusion: each iteration blends each layer's displacement
        with its neighbors. This acts like propagation through an elastic
        rod — collision forces at one layer smoothly affect nearby layers.

        Root point (layer 0) is always pinned (zero displacement).

        Returns:
            List of (L, 3) new guide arrays.
        """
        new_guides = []
        alpha = self.smooth_alpha

        for ti in range(len(guides)):
            x = guides[ti].astype(np.float64)
            L = x.shape[0]
            d = displacements[ti].astype(np.float64)

            if L < 2:
                new_guides.append(x.copy())
                continue

            # Pin root
            d[0] = 0.0

            # Vectorized Laplacian smoothing of displacement along the guide
            for _ in range(self.smooth_iters):
                # Neighbor average: interior uses (left + right)/2, tip uses left only
                avg = np.empty_like(d)
                avg[0] = 0.0
                avg[1:-1] = 0.5 * (d[:-2] + d[2:])  # interior
                avg[-1] = d[-2]  # tip: only one neighbor
                d[1:] = (1.0 - alpha) * d[1:] + alpha * avg[1:]
                d[0] = 0.0

            new_guides.append((x + d).astype(np.float32))

        return new_guides

    @staticmethod
    def compute_penalty_displacements(
        guides: List[np.ndarray],
        d_threshold: float,
        strength: float = 1.0,
    ) -> List[np.ndarray]:
        """Compute repulsive displacements between guide points that are too close.

        For each pair of guide points from different guides, if their distance
        is less than d_threshold, a repulsive displacement pushes them apart.
        The magnitude is proportional to (d_threshold - distance).

        Args:
            guides: list of (L_i, 3) guide arrays.
            d_threshold: distance below which penalty activates.
            strength: multiplier on the penalty displacement.

        Returns:
            List of (L_i, 3) penalty displacement arrays, one per guide.
        """
        n_guides = len(guides)
        disps = [np.zeros_like(g, dtype=np.float64) for g in guides]

        for ti in range(n_guides):
            gi = guides[ti].astype(np.float64)
            Li = gi.shape[0]
            for tj in range(ti + 1, n_guides):
                gj = guides[tj].astype(np.float64)
                Lj = gj.shape[0]
                # Vectorized pairwise distances: (Li, Lj, 3)
                diff = gi[:, None, :] - gj[None, :, :]  # (Li, Lj, 3)
                dist = np.linalg.norm(diff, axis=2)      # (Li, Lj)
                mask = dist < d_threshold
                if not mask.any():
                    continue
                # For each close pair, compute repulsive displacement
                rows, cols = np.where(mask)
                for r, c in zip(rows, cols):
                    d = dist[r, c]
                    if d < 1e-12:
                        continue
                    direction = diff[r, c] / d  # from gj toward gi
                    magnitude = strength * (d_threshold - d)
                    # Skip root points (layer 0)
                    if r > 0:
                        disps[ti][r] += magnitude * direction
                    if c > 0:
                        disps[tj][c] -= magnitude * direction

        return disps

    @staticmethod
    def rematch_guides_to_tubes(
        meshes: List[Tuple[np.ndarray, np.ndarray]],
        n_cross_list: List[int],
        n_layers_list: List[int],
        layer0_root: Optional[List[np.ndarray]] = None,
    ) -> List[np.ndarray]:
        """Recompute guide centerlines as ring centroids.

        Layer 0 is always pinned to the original root position.

        Returns:
            List of (L, 3) guide arrays.
        """
        new_guides = []
        for ti in range(len(meshes)):
            verts, _ = meshes[ti]
            L = n_layers_list[ti]
            N = n_cross_list[ti]

            # Reshape ring vertices to (L, N, 3)
            rings = verts[:L * N].astype(np.float64).reshape(L, N, 3)
            centers = rings.mean(axis=1)  # (L, 3)

            if layer0_root is not None:
                centers[0] = layer0_root[ti]

            new_guides.append(centers.astype(np.float32))
        return new_guides

    @staticmethod
    def update_tubes_from_guides(
        current_meshes: List[Tuple[np.ndarray, np.ndarray]],
        old_guides: List[np.ndarray],
        new_guides: List[np.ndarray],
        n_cross_list: List[int],
        n_layers_list: List[int],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Translate tube vertices to follow deformed guides.

        Each layer's vertices are rigidly translated by the guide point's
        displacement, preserving cross-section shape.

        Returns:
            Updated list of (verts, faces).
        """
        updated = []
        for ti in range(len(current_meshes)):
            verts, faces = current_meshes[ti]
            L = n_layers_list[ti]
            N = n_cross_list[ti]
            old_g = old_guides[ti].astype(np.float64)
            new_g = new_guides[ti].astype(np.float64)

            v_new = verts.copy().astype(np.float64)

            # Per-layer delta: (L, 3)
            delta = new_g - old_g
            delta[0] = 0.0  # pin root

            # Broadcast delta to ring vertices: (L, N, 3)
            ring_delta = np.repeat(delta[:L], N, axis=0).reshape(L, N, 3)
            v_new[:L * N] = verts[:L * N].astype(np.float64) + ring_delta.reshape(L * N, 3)

            # Tip vertex follows last layer
            tip_idx = L * N
            v_new[tip_idx] = verts[tip_idx].astype(np.float64) + delta[L - 1]

            updated.append((v_new.astype(np.float32), faces))
        return updated

    @staticmethod
    def guide_strand_ccd(
        old_guides: List[np.ndarray],
        new_guides: List[np.ndarray],
        device: str = "cuda:0",
        max_collisions: int = 128,
    ) -> List[np.ndarray]:
        """CCD-clamp guide deformation using edge-edge CCD on guide segments.

        Builds swept triangles from guide polyline edges (old→new), runs BVH
        broad-phase + edge-edge ACCD, then clamps per-layer by min TOI.

        Args:
            old_guides: list of (L, 3) — current guide positions.
            new_guides: list of (L, 3) — proposed guide positions.
            device: torch device.
            max_collisions: BVH collision buffer size.

        Returns:
            Clamped guide positions (list of (L, 3) arrays).
        """
        from mesh_intersection.bvh_search_tree import BVH

        # Build guide edges: each guide has L-1 segments
        all_a_start = []
        all_b_start = []
        all_a_end = []
        all_b_end = []
        edge_guide_id = []  # which guide each edge belongs to
        edge_layer_id = []  # which layer (lower endpoint) each edge belongs to

        for ti in range(len(old_guides)):
            old_g = old_guides[ti].astype(np.float32)
            new_g = new_guides[ti].astype(np.float32)
            L = old_g.shape[0]
            for li in range(L - 1):
                all_a_start.append(old_g[li])
                all_b_start.append(old_g[li + 1])
                all_a_end.append(new_g[li])
                all_b_end.append(new_g[li + 1])
                edge_guide_id.append(ti)
                edge_layer_id.append(li)

        n_edges = len(all_a_start)
        if n_edges == 0:
            return [g.copy() for g in new_guides]

        a_start = torch.tensor(np.array(all_a_start), device=device)
        b_start = torch.tensor(np.array(all_b_start), device=device)
        a_end = torch.tensor(np.array(all_a_end), device=device)
        b_end = torch.tensor(np.array(all_b_end), device=device)
        guide_ids = np.array(edge_guide_id)
        layer_ids = np.array(edge_layer_id)

        # Build swept triangles for BVH: each edge → 2 triangles
        tri_0 = torch.stack([a_start, b_start, a_end], dim=1)  # (E, 3, 3)
        tri_1 = torch.stack([b_start, b_end, a_end], dim=1)
        swept = torch.stack([tri_0, tri_1], dim=1).reshape(2 * n_edges, 3, 3)
        tri_to_edge = torch.arange(n_edges, device=device).repeat_interleave(2)

        # BVH broad-phase
        search_tree = BVH(max_collisions=max_collisions)
        with torch.no_grad():
            raw = search_tree(swept.unsqueeze(0)).squeeze(0)
            valid = raw[:, 0] >= 0
            tri_pairs = raw[valid]

        # Initialize per-edge TOI
        edge_toi = torch.ones(n_edges, device=device)

        if tri_pairs.shape[0] > 0:
            ei = tri_to_edge[tri_pairs[:, 0]]
            ej = tri_to_edge[tri_pairs[:, 1]]

            # Canonical order + dedup
            edge_pair = torch.stack([
                torch.min(ei, ej), torch.max(ei, ej)], dim=1)
            diff = edge_pair[:, 0] != edge_pair[:, 1]
            edge_pair = edge_pair[diff]

            if edge_pair.shape[0] > 0:
                edge_pair = torch.unique(edge_pair, dim=0)

                # Filter inter-guide only
                gi = guide_ids[edge_pair[:, 0].cpu().numpy()]
                gj = guide_ids[edge_pair[:, 1].cpu().numpy()]
                inter = gi != gj
                edge_pair = edge_pair[torch.from_numpy(inter).to(device)]

            if edge_pair.shape[0] > 0:
                ei = edge_pair[:, 0]
                ej = edge_pair[:, 1]

                toi = batch_edge_edge_ccd(
                    a0_start=a_start[ei], a1_start=b_start[ei],
                    b0_start=a_start[ej], b1_start=b_start[ej],
                    a0_end=a_end[ei],     a1_end=b_end[ei],
                    b0_end=a_end[ej],     b1_end=b_end[ej],
                    xi=0.0,  # no thickness — only block actual crossings
                )

                edge_toi.scatter_reduce_(
                    0, ei, toi, reduce="amin", include_self=True)
                edge_toi.scatter_reduce_(
                    0, ej, toi, reduce="amin", include_self=True)

                n_clamped = (toi < 1.0).sum().item()
                print(f"  Guide strand CCD: {edge_pair.shape[0]} pairs, "
                      f"clamped={n_clamped}")

        # Convert per-edge TOI → per-layer TOI (min of adjacent edges)
        edge_toi_np = edge_toi.cpu().numpy()
        clamped = []
        idx = 0
        for ti in range(len(old_guides)):
            old_g = old_guides[ti]
            new_g = new_guides[ti]
            L = old_g.shape[0]
            layer_t = np.ones(L, dtype=np.float64)
            # Layer 0: pinned
            for li in range(L - 1):
                # Edge li connects layer li and li+1
                t = float(edge_toi_np[idx])
                layer_t[li] = min(layer_t[li], t)
                layer_t[li + 1] = min(layer_t[li + 1], t)
                idx += 1

            # Propagate: each layer can't exceed its own t or parent's t
            for li in range(1, L):
                layer_t[li] = min(layer_t[li], layer_t[li - 1])

            cg = old_g.copy().astype(np.float64)
            for li in range(1, L):
                delta = new_g[li] - old_g[li]
                cg[li] = old_g[li] + layer_t[li] * delta
            clamped.append(cg.astype(np.float32))

        return clamped

    @staticmethod
    def guide_boundary_ccd(
        old_guides: List[np.ndarray],
        new_guides: List[np.ndarray],
        boundary_verts: np.ndarray,
        boundary_faces: np.ndarray,
        device: str = "cuda:0",
        xi: float = 0.001,
    ) -> List[np.ndarray]:
        """CCD-clamp guide deformation against a boundary triangle mesh.

        Each guide vertex trajectory (old→new) is tested against boundary
        triangles using VT CCD.  Per-layer TOI is propagated root→tip.

        Args:
            old_guides: list of (L, 3) — current guide positions.
            new_guides: list of (L, 3) — proposed guide positions.
            boundary_verts: (V_b, 3) boundary mesh vertices.
            boundary_faces: (F_b, 3) boundary mesh face indices.
            device: torch device.
            xi: CCD thickness.

        Returns:
            Clamped guide positions (list of (L, 3) arrays).
        """
        # Flatten all guide vertices
        all_old = []
        all_new = []
        vert_guide_id = []
        vert_layer_id = []
        for ti in range(len(old_guides)):
            old_g = old_guides[ti].astype(np.float32)
            new_g = new_guides[ti].astype(np.float32)
            L = old_g.shape[0]
            for li in range(L):
                all_old.append(old_g[li])
                all_new.append(new_g[li])
                vert_guide_id.append(ti)
                vert_layer_id.append(li)

        n_verts = len(all_old)
        if n_verts == 0:
            return [g.copy() for g in new_guides]

        v_start = torch.tensor(np.array(all_old), device=device)  # (V, 3)
        v_end = torch.tensor(np.array(all_new), device=device)    # (V, 3)

        bverts = torch.tensor(boundary_verts.astype(np.float32), device=device)
        bfaces = torch.tensor(boundary_faces.astype(np.int64), device=device)
        btris = bverts[bfaces]  # (F_b, 3, 3)
        n_b = btris.shape[0]

        # AABB overlap: vertex trajectory vs boundary triangles
        v_min = torch.min(v_start, v_end) - xi
        v_max = torch.max(v_start, v_end) + xi
        btri_min = btris.min(dim=1).values
        btri_max = btris.max(dim=1).values

        chunk_size = max(1, min(4096, 2**28 // max(n_verts * 3, 1)))
        all_vert_idx = []
        all_tri_idx = []

        for b_s in range(0, n_b, chunk_size):
            b_e = min(b_s + chunk_size, n_b)
            overlap = (
                (v_min[:, None, :] <= btri_max[None, b_s:b_e, :]) &
                (v_max[:, None, :] >= btri_min[None, b_s:b_e, :])
            ).all(dim=2)
            vi, bi = torch.where(overlap)
            if vi.shape[0] > 0:
                all_vert_idx.append(vi)
                all_tri_idx.append(bi + b_s)

        # Initialize per-vertex TOI
        vert_toi = torch.ones(n_verts, device=device)

        if all_vert_idx:
            vert_idx = torch.cat(all_vert_idx)
            tri_idx = torch.cat(all_tri_idx)

            # Deduplicate
            keys = torch.unique(torch.stack([vert_idx, tri_idx], dim=1), dim=0)
            vert_idx = keys[:, 0]
            tri_idx = keys[:, 1]

            P = vert_idx.shape[0]
            if P > 0:
                bt0 = bverts[bfaces[tri_idx, 0]]
                bt1 = bverts[bfaces[tri_idx, 1]]
                bt2 = bverts[bfaces[tri_idx, 2]]

                vt_toi = batch_vertex_triangle_ccd(
                    p_start=v_start[vert_idx],
                    t0_start=bt0, t1_start=bt1, t2_start=bt2,
                    p_end=v_end[vert_idx],
                    t0_end=bt0, t1_end=bt1, t2_end=bt2,
                    xi=xi,
                )

                vert_toi.scatter_reduce_(
                    0, vert_idx, vt_toi, reduce="amin", include_self=True)

                n_clamped = (vt_toi < 1.0).sum().item()
                print(f"  Guide boundary CCD: {P} pairs, clamped={n_clamped}")

        # Convert per-vertex TOI → per-guide clamped positions
        vert_toi_np = vert_toi.cpu().numpy()
        guide_id_np = np.array(vert_guide_id)
        layer_id_np = np.array(vert_layer_id)

        clamped = []
        for ti in range(len(old_guides)):
            old_g = old_guides[ti]
            new_g = new_guides[ti]
            L = old_g.shape[0]
            mask = guide_id_np == ti
            layers = layer_id_np[mask]
            tois = vert_toi_np[mask]

            layer_t = np.ones(L, dtype=np.float64)
            for li, t in zip(layers, tois):
                layer_t[li] = min(layer_t[li], float(t))

            # Propagate root→tip
            for li in range(1, L):
                layer_t[li] = min(layer_t[li], layer_t[li - 1])

            cg = old_g.copy().astype(np.float64)
            for li in range(1, L):
                delta = new_g[li] - old_g[li]
                cg[li] = old_g[li] + layer_t[li] * delta
            clamped.append(cg.astype(np.float32))

        return clamped

    @staticmethod
    def compute_guide_following_positions(
        current_meshes: List[Tuple[np.ndarray, np.ndarray]],
        old_guides: List[np.ndarray],
        new_guides: List[np.ndarray],
        n_cross_list: List[int],
        n_layers_list: List[int],
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Compute target tube positions that follow new guide positions.

        Each layer's vertices are rigidly translated by the guide point's
        displacement. This is the "desired" position — CCD will clamp it.

        Returns:
            List of (verts, faces) — target positions.
        """
        return GuideDeformer.update_tubes_from_guides(
            current_meshes, old_guides, new_guides,
            n_cross_list, n_layers_list,
        )

    @staticmethod
    def smooth_guides_spline(
        guides: List[np.ndarray],
        knot_spacing_factor: float = 4.0,
        pin_root: bool = True,
    ) -> List[np.ndarray]:
        """Smooth guide curves via least-squares cubic B-spline fitting (parallel).

        Knot count is determined by arc length: n_interior = arc_length / (ref_edge * knot_spacing_factor).
        The reference edge length is the global median across all guides, so short
        curves get fewer knots (more smoothing) and long curves get more.

        Args:
            guides: list of (L, 3) guide arrays.
            knot_spacing_factor: multiplier on global median edge length to set
                                 knot spacing. Higher = fewer knots = more smoothing.
            pin_root: if True, pin layer 0 to its original position.

        Returns:
            List of (L, 3) smoothed guide arrays.
        """
        from concurrent.futures import ThreadPoolExecutor
        from scipy.interpolate import make_lsq_spline

        k = 3  # cubic

        # Compute global median edge length across all guides
        all_seg_lens = []
        for g in guides:
            if g.shape[0] < 2:
                continue
            segs = np.linalg.norm(np.diff(g.astype(np.float64), axis=0), axis=1)
            all_seg_lens.append(segs)
        if not all_seg_lens:
            return [g.copy() for g in guides]
        global_median_seg = float(np.median(np.concatenate(all_seg_lens)))
        if global_median_seg < 1e-12:
            return [g.copy() for g in guides]
        knot_spacing = global_median_seg * knot_spacing_factor

        def _smooth_one(guide: np.ndarray) -> np.ndarray:
            pts = guide.astype(np.float64)
            L = pts.shape[0]
            if L < k + 2 or np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
                return guide.copy()

            # Cumulative chord-length parameterization
            diffs = np.diff(pts, axis=0)
            seg_lens = np.linalg.norm(diffs, axis=1)
            u = np.zeros(L, dtype=np.float64)
            u[1:] = np.cumsum(seg_lens)
            total_len = u[-1]
            if total_len < 1e-12:
                return guide.copy()

            u /= total_len  # normalize to [0, 1]

            # make_lsq_spline requires strictly increasing u.
            for i in range(1, L):
                if u[i] <= u[i - 1]:
                    u[i] = u[i - 1] + 1e-14

            # Interior knots: one knot per knot_spacing of arc length.
            # Short curves get fewer knots, long curves get more.
            n_interior = max(int(total_len / knot_spacing), 1)
            n_interior = min(n_interior, L - k - 1)
            if n_interior < 1:
                return guide.copy()

            t_interior = np.linspace(u[0], u[-1], n_interior + 2)[1:-1]
            knots = np.concatenate([
                np.full(k + 1, u[0]),
                t_interior,
                np.full(k + 1, u[-1]),
            ])

            try:
                u_uniform = np.linspace(0.0, 1.0, L)
                result = np.empty_like(pts)
                for axis in range(3):
                    spl = make_lsq_spline(u, pts[:, axis], knots, k=k)
                    result[:, axis] = spl(u_uniform)
                if not np.all(np.isfinite(result)):
                    return guide.copy()
                max_dev = np.linalg.norm(result - pts, axis=1).max()
                if max_dev > total_len * 0.1:
                    return guide.copy()
                result = result.astype(np.float32)
            except Exception:
                return guide.copy()

            if pin_root:
                result[0] = guide[0]

            return result

        with ThreadPoolExecutor() as executor:
            results = list(executor.map(_smooth_one, guides))

        return results
