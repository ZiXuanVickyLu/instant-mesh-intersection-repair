"""Multi-object tube relaxation using per-tube Laplacian diffusion.

Each tube gets its own parameterization matrix M_i = (1-alpha)*I + alpha*L_i.
All tubes are combined into a single mesh for BVH collision detection.
Only inter-tube collisions drive the repulsive energy.
Gradients are diffused independently per tube via separate Cholesky solvers.
Root vertices (layer 0 on the scalp) are pinned with a strong positional constraint.

Usage:
    from hair_tube_relaxation.tube_relaxation import MultiTubeRelaxation
    relaxer = MultiTubeRelaxation(tube_meshes, n_cross_section, device='cuda')
    relaxed = relaxer.run(n_iters=60, lr=1.0, alpha=0.99)
"""

import numpy as np
import torch
from dataclasses import dataclass, field

from largesteps.geometry import compute_matrix_from_lap
from largesteps.parameterize import to_differential, from_differential
from mesh_intersection.bvh_search_tree import BVH
import mutils


@dataclass
class TubeData:
    """Per-tube bookkeeping for the optimization."""
    # Original numpy arrays
    np_verts: np.ndarray
    np_faces: np.ndarray
    # Torch tensors (on device)
    verts: torch.Tensor = None
    faces: torch.Tensor = None
    # Laplacian and parameterization
    lap: torch.Tensor = None
    M: torch.Tensor = None
    u: torch.Tensor = None  # differential coordinates (optimized)
    # Offsets in the combined mesh
    vert_offset: int = 0
    face_offset: int = 0
    n_verts: int = 0
    n_faces: int = 0
    # Root vertex indices (layer 0) in local indexing
    root_indices: np.ndarray = None
    # Initial Laplacian coordinates for curvature constraint
    L0: torch.Tensor = None
    # Initial root positions for pinning
    root_pos0: torch.Tensor = None


class MultiTubeRelaxation:
    """Optimize multiple tube meshes to remove inter-tube intersections.

    Parameters
    ----------
    tube_meshes : list of (np.ndarray, np.ndarray)
        Each element is (vertices, faces) for one tube mesh, as returned by
        standalone_voronoi_tube.build_tube_meshes().
    n_cross_section : int
        Number of vertices per cross-section ring. Used to identify root
        vertices (indices 0..n_cross_section-1 are the scalp boundary).
    device : str
        Torch device, default 'cuda'.
    """

    def __init__(self, tube_meshes, n_cross_section, device='cuda',
                 max_tubes=None):
        self.device = torch.device(device)
        self.n_cross = n_cross_section
        self.tubes: list[TubeData] = []
        if max_tubes is not None and len(tube_meshes) > max_tubes:
            print(f"  Limiting to {max_tubes} tubes (of {len(tube_meshes)})")
            tube_meshes = tube_meshes[:max_tubes]
        self._build_tubes(tube_meshes)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _build_tubes(self, tube_meshes):
        """Initialize per-tube data structures."""
        vert_offset = 0
        face_offset = 0

        for np_v, np_f in tube_meshes:
            td = TubeData(np_verts=np_v, np_faces=np_f)
            td.n_verts = np_v.shape[0]
            td.n_faces = np_f.shape[0]
            td.vert_offset = vert_offset
            td.face_offset = face_offset

            # Root vertices are the first n_cross vertices (layer 0 on scalp)
            td.root_indices = np.arange(min(self.n_cross, td.n_verts))

            vert_offset += td.n_verts
            face_offset += td.n_faces
            self.tubes.append(td)

        self.total_verts = vert_offset
        self.total_faces = face_offset

        # Build face-to-tube ownership array (used for inter-tube filtering)
        self.face_owner = np.empty(self.total_faces, dtype=np.int32)
        for i, td in enumerate(self.tubes):
            self.face_owner[td.face_offset:td.face_offset + td.n_faces] = i
        self.face_owner_t = torch.tensor(
            self.face_owner, dtype=torch.int32, device=self.device
        )

    def _init_parameterization(self, alpha):
        """Compute per-tube Laplacians, M matrices, and differential coords."""
        for td in self.tubes:
            td.verts = torch.tensor(
                td.np_verts, dtype=torch.float32, device=self.device
            )
            td.faces = torch.tensor(
                td.np_faces, dtype=torch.long, device=self.device
            )
            # Cotangent Laplacian (via scipy -> torch sparse)
            td.lap = mutils.cotan_laplacian(
                td.np_verts, td.np_faces
            ).float().to(self.device)

            # M_i = (1-alpha)*I + alpha*L_i
            td.M = compute_matrix_from_lap(
                td.lap, td.verts, lambda_=0, alpha=alpha
            )

            # Differential coordinates
            td.u = to_differential(td.M, td.verts)
            td.u.requires_grad = True

            # Cache initial Laplacian coordinates for curvature constraint
            with torch.no_grad():
                td.L0 = (td.lap @ td.verts).clone()
                td.root_pos0 = td.verts[td.root_indices].clone()

    # ------------------------------------------------------------------
    # Combined mesh assembly
    # ------------------------------------------------------------------

    def _assemble_combined(self):
        """Decode all tubes and assemble into a single (verts, faces) pair.

        Returns
        -------
        combined_verts : torch.Tensor, shape (V_total, 3)
        combined_faces : torch.Tensor, shape (F_total, 3)
        per_tube_verts : list of torch.Tensor  (each tube's decoded verts)
        """
        per_tube_verts = []
        all_verts = []
        all_faces = []

        for td in self.tubes:
            v = from_differential(td.M, td.u, 'Cholesky')
            per_tube_verts.append(v)
            all_verts.append(v)
            # Offset face indices
            all_faces.append(td.faces + td.vert_offset)

        combined_verts = torch.cat(all_verts, dim=0)
        combined_faces = torch.cat(all_faces, dim=0)
        return combined_verts, combined_faces, per_tube_verts

    # ------------------------------------------------------------------
    # Inter-tube collision detection and energy
    # ------------------------------------------------------------------

    def _detect_inter_tube_collisions(self, combined_verts, combined_faces,
                                       search_tree):
        """Run BVH and filter to inter-tube pairs only.

        Returns
        -------
        collision_idxs : torch.Tensor, shape (C, 2), face indices in combined mesh
        num_all : int, total collisions before filtering
        num_inter : int, inter-tube collisions after filtering
        """
        triangles = combined_verts[combined_faces]
        with torch.no_grad():
            raw = search_tree(triangles.unsqueeze(0)).squeeze(0)
            raw = raw[raw[:, 0] >= 0, :]
            num_all = raw.shape[0]

            if num_all == 0:
                return raw, 0, 0

            # Filter: keep only pairs from different tubes
            owner_a = self.face_owner_t[raw[:, 0]]
            owner_b = self.face_owner_t[raw[:, 1]]
            inter_mask = owner_a != owner_b
            collision_idxs = raw[inter_mask]
            num_inter = collision_idxs.shape[0]

        return collision_idxs, num_all, num_inter

    def _compute_repulsive_energy(self, combined_verts, combined_faces,
                                   collision_idxs, p=1):
        """Signed TPE (vertex-to-center) over inter-tube collision pairs.

        This is equivalent to signed_TPE_verts but uses precomputed collision
        indices (already filtered to inter-tube only).
        """
        if collision_idxs.shape[0] == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        triangles = combined_verts[combined_faces]

        with torch.no_grad():
            face_normals = torch.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            )
            face_areas = 0.5 * torch.norm(face_normals, dim=1)
            face_normals = torch.nn.functional.normalize(face_normals, dim=1)

        face_centers = torch.sum(triangles, dim=1) / 3.0

        # Vertices of face_a against center of face_b
        verts_idx = combined_faces[collision_idxs[:, 0]]  # (C, 3)
        X = combined_verts[verts_idx].reshape(-1, 3)       # (3C, 3)
        Y = face_centers[collision_idxs[:, 1]].repeat(3, 1)

        dist = torch.linalg.vector_norm(Y - X, dim=1)
        Pd = torch.sum(
            (Y - X) * face_normals[collision_idxs[:, 1]].repeat(3, 1), dim=1
        )
        r = (Pd / dist).pow(p)
        TPE1 = face_areas[collision_idxs[:, 1]].repeat(3) * r

        # Symmetric: vertices of face_b against center of face_a
        verts_idx = combined_faces[collision_idxs[:, 1]]
        X = combined_verts[verts_idx].reshape(-1, 3)
        Y = face_centers[collision_idxs[:, 0]].repeat(3, 1)

        dist = torch.linalg.vector_norm(Y - X, dim=1)
        Pd = torch.sum(
            (Y - X) * face_normals[collision_idxs[:, 0]].repeat(3, 1), dim=1
        )
        r = (Pd / dist).pow(p)
        TPE2 = face_areas[collision_idxs[:, 0]].repeat(3) * r

        return torch.sum(torch.cat([TPE1, TPE2], dim=0))

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def _compute_constraints(self, per_tube_verts, w_curvature=1e6,
                              w_root_pin=1e7):
        """Per-tube curvature preservation + root pinning.

        Parameters
        ----------
        per_tube_verts : list of torch.Tensor
        w_curvature : float
            Weight for Laplacian coordinate preservation.
        w_root_pin : float
            Weight for pinning root (scalp) vertices.

        Returns
        -------
        reg_loss : torch.Tensor  (scalar)
        """
        reg = torch.tensor(0.0, device=self.device)

        for td, v in zip(self.tubes, per_tube_verts):
            # Curvature preservation: ||L_i @ v_i - L0_i||^2
            reg = reg + w_curvature * torch.nn.functional.mse_loss(
                td.lap @ v, td.L0
            )
            # Root pinning: ||v_root - v_root_0||^2
            reg = reg + w_root_pin * torch.nn.functional.mse_loss(
                v[td.root_indices], td.root_pos0
            )

        return reg

    # ------------------------------------------------------------------
    # Main optimization loop
    # ------------------------------------------------------------------

    def run(self, n_iters=60, lr=1.0, alpha=0.99, p=1,
            max_collisions=64, w_curvature=1e6, w_root_pin=1e7,
            verbose=True):
        """Run the multi-tube relaxation optimization.

        Parameters
        ----------
        n_iters : int
            Number of optimization iterations.
        lr : float
            Learning rate for gradient descent.
        alpha : float
            Laplacian parameterization weight (0.99 = highly non-local).
        p : int
            Power for signed TPE energy.
        max_collisions : int
            BVH max collisions per triangle.
        w_curvature : float
            Curvature preservation weight.
        w_root_pin : float
            Root vertex pinning weight.
        verbose : bool
            Print progress.

        Returns
        -------
        list of (np.ndarray, np.ndarray)
            Optimized (vertices, faces) for each tube.
        """
        self._init_parameterization(alpha)

        search_tree = BVH(max_collisions=max_collisions)

        # Simple gradient descent on all u_i simultaneously
        params = [td.u for td in self.tubes]
        optimizer = torch.optim.Adam(params, lr=lr)

        best_inter_col = float('inf')
        best_tube_verts = None
        best_iter = 0

        for i in range(n_iters):
            optimizer.zero_grad()

            # Decode all tubes and assemble combined mesh
            combined_v, combined_f, per_tube_v = self._assemble_combined()

            # Detect inter-tube collisions
            col_idxs, num_all, num_inter = self._detect_inter_tube_collisions(
                combined_v, combined_f, search_tree
            )

            # Repulsive energy (only inter-tube)
            pen_loss = self._compute_repulsive_energy(
                combined_v, combined_f, col_idxs, p=p
            )

            # Per-tube constraints
            reg_loss = self._compute_constraints(
                per_tube_v, w_curvature=w_curvature, w_root_pin=w_root_pin
            )

            loss = pen_loss + reg_loss
            loss.backward()
            optimizer.step()

            # Track best solution
            with torch.no_grad():
                if num_inter < best_inter_col:
                    best_inter_col = num_inter
                    best_tube_verts = [v.detach().clone() for v in per_tube_v]
                    best_iter = i

            if verbose and ((i + 1) % 5 == 0 or i == 0):
                print(
                    f"[{i+1:3d}/{n_iters}] "
                    f"pen={pen_loss.item():.4f}  "
                    f"reg={reg_loss.item():.4f}  "
                    f"col_all={num_all}  col_inter={num_inter}"
                )

        if verbose:
            print(
                f"\nDone. Best at iter {best_iter} "
                f"with {best_inter_col} inter-tube collisions."
            )

        # Return best per-tube meshes as numpy
        results = []
        for td, v in zip(self.tubes, best_tube_verts):
            results.append((v.cpu().numpy(), td.np_faces))
        return results

    # ------------------------------------------------------------------
    # Polyscope interactive visualization
    # ------------------------------------------------------------------

    def run_vis(self, n_iters=60, lr=1.0, alpha=0.99, p=1,
                max_collisions=64, w_curvature=1e6, w_root_pin=1e7):
        """Run relaxation with interactive Polyscope visualization.

        Shows the combined tube mesh with per-tube colors and a collision
        heatmap. Provides Step / Run / Pause buttons.
        """
        import polyscope as ps
        import polyscope.imgui as psim

        self._init_parameterization(alpha)
        search_tree = BVH(max_collisions=max_collisions)

        params = [td.u for td in self.tubes]
        optimizer = torch.optim.Adam(params, lr=lr)

        # Build combined faces (numpy, for polyscope registration)
        all_np_faces = []
        for td in self.tubes:
            all_np_faces.append(td.np_faces + td.vert_offset)
        combined_np_faces = np.concatenate(all_np_faces, axis=0)

        # Per-tube face colors (fixed)
        n_tubes = len(self.tubes)
        tube_colors_np = np.zeros((self.total_faces, 3), dtype=np.float32)
        for i, td in enumerate(self.tubes):
            hue = (i * 0.618033988749895) % 1.0
            # HSV to RGB (s=0.7, v=0.9)
            c = 0.9 * 0.7
            x = c * (1.0 - abs((hue * 6.0) % 2.0 - 1.0))
            m = 0.9 - c
            h_s = int(hue * 6.0) % 6
            rgb = [(c+m, x+m, m), (x+m, c+m, m), (m, c+m, x+m),
                   (m, x+m, c+m), (x+m, m, c+m), (c+m, m, x+m)][h_s]
            tube_colors_np[td.face_offset:td.face_offset + td.n_faces] = rgb

        # Initial decode
        with torch.no_grad():
            combined_v, combined_f, _ = self._assemble_combined()
            init_verts_np = combined_v.detach().cpu().numpy()
            col_idxs, num_all, num_inter = self._detect_inter_tube_collisions(
                combined_v, combined_f, search_tree
            )

        # Polyscope init
        ps.init()
        ps.set_ground_plane_mode("none")
        ps_mesh = ps.register_surface_mesh(
            "tubes", init_verts_np, combined_np_faces,
            smooth_shade=True, enabled=True,
        )
        ps_mesh.add_color_quantity("tube_id", tube_colors_np,
                                   defined_on='faces', enabled=True)

        def _collision_mask(col_idxs_t):
            mask = np.zeros(self.total_faces, dtype=np.float32)
            if col_idxs_t.shape[0] > 0:
                cf = col_idxs_t.cpu().numpy().flatten()
                cf = cf[cf >= 0]
                np.add.at(mask, cf, 1.0)
            return mask

        ps_mesh.add_scalar_quantity(
            "collisions", _collision_mask(col_idxs),
            defined_on='faces', enabled=False, cmap='reds',
        )

        state = {
            'step': 0,
            'running': False,
            'done': False,
            'pen_loss': 0.0,
            'reg_loss': 0.0,
            'num_all': num_all,
            'num_inter': num_inter,
            'best_inter': num_inter,
            'best_iter': 0,
            'best_verts': [None],
            'latest_verts': [None],
        }

        def do_step():
            if state['done']:
                return

            optimizer.zero_grad()
            combined_v, combined_f, per_tube_v = self._assemble_combined()

            col_idxs, num_all, num_inter = self._detect_inter_tube_collisions(
                combined_v, combined_f, search_tree
            )

            pen_loss = self._compute_repulsive_energy(
                combined_v, combined_f, col_idxs, p=p
            )
            reg_loss = self._compute_constraints(
                per_tube_v, w_curvature=w_curvature, w_root_pin=w_root_pin
            )

            loss = pen_loss + reg_loss
            loss.backward()
            optimizer.step()

            # Re-decode after step for display
            with torch.no_grad():
                combined_v_new, _, per_tube_v_new = self._assemble_combined()
                col_new, num_all_new, num_inter_new = \
                    self._detect_inter_tube_collisions(
                        combined_v_new,
                        torch.cat([td.faces + td.vert_offset
                                   for td in self.tubes], dim=0),
                        search_tree,
                    )

                if num_inter_new < state['best_inter']:
                    state['best_inter'] = num_inter_new
                    state['best_iter'] = state['step']
                    state['best_verts'] = [
                        v.detach().clone() for v in per_tube_v_new
                    ]

            state['step'] += 1
            state['pen_loss'] = pen_loss.item()
            state['reg_loss'] = reg_loss.item()
            state['num_all'] = num_all_new
            state['num_inter'] = num_inter_new
            state['latest_verts'] = per_tube_v_new

            # Update polyscope
            verts_np = combined_v_new.detach().cpu().numpy()
            ps_mesh.update_vertex_positions(verts_np)
            ps_mesh.add_scalar_quantity(
                "collisions", _collision_mask(col_new),
                defined_on='faces', enabled=False, cmap='reds',
            )

            print(f"[{state['step']:3d}/{n_iters}] "
                  f"pen={state['pen_loss']:.4f}  "
                  f"reg={state['reg_loss']:.4f}  "
                  f"col_inter={num_inter_new}")

            if state['step'] >= n_iters:
                state['done'] = True
                state['running'] = False
                print(f"\nDone. Best at iter {state['best_iter']} "
                      f"with {state['best_inter']} inter-tube collisions.")

        def callback():
            psim.TextUnformatted(
                f"Step       : {state['step']} / {n_iters}")
            psim.TextUnformatted(
                f"Inter-col  : {state['num_inter']}")
            psim.TextUnformatted(
                f"All col    : {state['num_all']}")
            psim.TextUnformatted(
                f"Pen loss   : {state['pen_loss']:.6f}")
            psim.TextUnformatted(
                f"Reg loss   : {state['reg_loss']:.6f}")
            psim.TextUnformatted(
                f"Best       : step={state['best_iter']}, "
                f"col={state['best_inter']}")
            psim.Separator()

            if state['done']:
                psim.TextUnformatted("Optimization complete!")
                return

            if psim.Button("Step"):
                do_step()
            psim.SameLine()
            if state['running']:
                if psim.Button("Pause"):
                    state['running'] = False
            else:
                if psim.Button("Run"):
                    state['running'] = True

            if state['running']:
                do_step()

        ps.set_user_callback(callback)
        ps.show()

        # Return best results
        if state['best_verts'][0] is not None:
            # best_verts was set
            results = []
            for td, v in zip(self.tubes, state['best_verts']):
                results.append((v.cpu().numpy(), td.np_faces))
            return results
        else:
            # No improvement found, return initial
            results = []
            for td in self.tubes:
                results.append((td.np_verts, td.np_faces))
            return results

    # ------------------------------------------------------------------
    # Convenience: scale tubes before relaxation
    # ------------------------------------------------------------------

    @staticmethod
    def rescale_tubes(tube_meshes, n_cross_section, target_scale=1.0,
                      current_scale=0.1):
        """Rescale tube cross-sections from current_scale to target_scale.

        Only rescales layers 1+ (not layer 0 which sits on the scalp).
        The tip center vertex (last vertex) is also rescaled.

        Parameters
        ----------
        tube_meshes : list of (np.ndarray, np.ndarray)
        n_cross_section : int
        target_scale : float
        current_scale : float

        Returns
        -------
        list of (np.ndarray, np.ndarray)
            Rescaled tube meshes.
        """
        scale_ratio = target_scale / current_scale
        rescaled = []

        for np_v, np_f in tube_meshes:
            v = np_v.copy()
            N = n_cross_section
            n_verts = v.shape[0]
            # tip center is the last vertex
            tip_idx = n_verts - 1
            # Number of layers = (n_verts - 1) / N
            n_layers = (n_verts - 1) // N

            # For each layer > 0, scale offset from guide center
            # We approximate the guide center as the mean of the ring vertices
            for li in range(1, n_layers):
                ring_start = li * N
                ring_end = ring_start + N
                ring = v[ring_start:ring_end]
                center = ring.mean(axis=0)
                v[ring_start:ring_end] = center + scale_ratio * (ring - center)

            # Scale the tip vertex relative to the last ring center
            if n_layers > 0:
                last_ring = v[(n_layers - 1) * N: n_layers * N]
                last_center = last_ring.mean(axis=0)
                v[tip_idx] = last_center + scale_ratio * (v[tip_idx] - last_center)

            rescaled.append((v, np_f))

        return rescaled
