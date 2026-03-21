"""Generate hair tubes and run multi-tube relaxation.

Usage:
    # Generate tubes from voronoi config + relax:
    python -m hair_tube_relaxation.run_tube_relaxation \
        --config hair_tube_relaxation/config/conf_00352.toml \
        --output-dir output/relaxed/ \
        --target-scale 0.8 \
        --save-combined

    # Or relax pre-existing tube OBJs:
    python -m hair_tube_relaxation.run_tube_relaxation \
        --tube-dir path/to/tube_objs/ \
        --output-dir output/relaxed/ \
        --skip-rescale
"""

import os
import argparse
import glob

import numpy as np
import trimesh


def load_tube_objs(tube_dir):
    """Load all OBJ files in a directory as (verts, faces) pairs."""
    pattern = os.path.join(tube_dir, "*.obj")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No .obj files found in {tube_dir}")
    meshes = []
    for p in paths:
        m = trimesh.load(p, process=False)
        meshes.append((np.array(m.vertices, dtype=np.float64),
                        np.array(m.faces, dtype=np.int32)))
    print(f"Loaded {len(meshes)} tube meshes from {tube_dir}")
    return meshes


def generate_tubes_from_config(config_path, tip_scale=0.1):
    """Run the full voronoi tube generation pipeline from a config file.

    Returns (tube_meshes, n_cross) where tube_meshes is a list of
    (verts, faces) and n_cross is auto-detected cross-section vertex count.
    """
    import toml
    from hair_tube_relaxation.standalone_voronoi_tube import (
        load_obj_with_uv,
        load_guides_obj_grouped,
        triangulate_faces,
        compute_voronoi_labels,
        extract_boundary_segments,
        extract_mesh_boundary_edges,
        build_cell_boundary_polylines,
        fix_penetrating_guides,
        build_tube_meshes,
    )

    project_root = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    cfg = toml.load(os.path.join(project_root, config_path))
    data_base = os.path.join(project_root, cfg["data"]["base_path"])
    output_base = os.path.join(project_root, cfg["output"]["base_path"])

    # Load scalp
    scalp_path = os.path.join(data_base, cfg["scalp"]["scalp_mesh"])
    print(f"Loading scalp mesh: {scalp_path}")
    scalp_data = load_obj_with_uv(scalp_path)
    scalp_verts = scalp_data["vertices"]
    scalp_faces = triangulate_faces(scalp_data["faces_v"])
    print(f"  {len(scalp_verts)} vertices, {len(scalp_faces)} triangles")

    # Load guides
    guides_obj = os.path.join(output_base, "grown_strands_segmented.obj")
    print(f"Loading guide strands: {guides_obj}")
    guides, guide_group_ids = load_guides_obj_grouped(guides_obj)
    print(f"  {len(guides)} guides")

    roots = np.array([g[0] for g in guides], dtype=np.float64)

    # Voronoi segmentation
    print("\nComputing Voronoi segmentation...")
    vert_labels = compute_voronoi_labels(scalp_verts, roots)

    # Penetration repair
    print("Penetration detection & repair...")
    body_mesh_path = os.path.join(data_base, "TSR/body_clean.obj")
    thickness = float(cfg.get("strand", {}).get("penetration_thickness", 0.01))
    fix_penetrating_guides(
        guides, body_mesh_path, scalp_verts, scalp_faces, vert_labels,
        thickness=thickness,
    )
    roots = np.array([g[0] for g in guides], dtype=np.float64)
    vert_labels = compute_voronoi_labels(scalp_verts, roots)

    # Boundary extraction
    print("Extracting boundaries...")
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
    print("\nBuilding tube meshes...")
    results = build_tube_meshes(
        guides, cell_polylines, scalp_verts, scalp_faces,
        tip_scale=tip_scale,
    )
    tube_meshes = [r for r in results if r is not None]
    print(f"Generated {len(tube_meshes)} tube meshes")

    return tube_meshes


def save_tube_objs(tube_meshes, output_dir, prefix="tube"):
    """Save each tube mesh as a separate OBJ file."""
    os.makedirs(output_dir, exist_ok=True)
    for i, (v, f) in enumerate(tube_meshes):
        path = os.path.join(output_dir, f"{prefix}_{i:04d}.obj")
        m = trimesh.Trimesh(vertices=v, faces=f, process=False)
        m.export(path)
    print(f"Saved {len(tube_meshes)} tubes to {output_dir}")


def save_combined_obj(tube_meshes, path):
    """Save all tubes as a single combined OBJ."""
    all_verts, all_faces = [], []
    offset = 0
    for v, f in tube_meshes:
        all_verts.append(v)
        all_faces.append(f + offset)
        offset += v.shape[0]
    combined = trimesh.Trimesh(
        vertices=np.concatenate(all_verts, axis=0),
        faces=np.concatenate(all_faces, axis=0),
        process=False,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    combined.export(path)
    print(f"Saved combined mesh: {path}")


def auto_detect_n_cross(tube_meshes):
    """Detect cross-section vertex count from tube topology.

    For a tube with L layers, N cross-section verts:
      n_verts = L*N + 1,  n_faces = 2*N*(L-1) + N = N*(2L-1)
      => N = 2*(n_verts-1) - n_faces
    """
    v, f = tube_meshes[0]
    n_cross = 2 * (v.shape[0] - 1) - f.shape[0]
    return n_cross


def main():
    parser = argparse.ArgumentParser(
        description="Hair tube generation + multi-tube relaxation"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--config",
                     help="Voronoi tube config .toml (generates tubes)")
    src.add_argument("--tube-dir",
                     help="Directory with pre-existing tube OBJ files")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for relaxed tubes")

    # Scale
    parser.add_argument("--target-scale", type=float, default=0.8,
                        help="Target cross-section scale (default 0.8)")
    parser.add_argument("--current-scale", type=float, default=0.1,
                        help="Current cross-section scale (default 0.1)")
    parser.add_argument("--skip-rescale", action="store_true",
                        help="Skip rescaling")

    # Optimization
    parser.add_argument("--n-cross", type=int, default=None,
                        help="Cross-section vertex count (auto if omitted)")
    parser.add_argument("--n-iters", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.99)
    parser.add_argument("--max-collisions", type=int, default=64)
    parser.add_argument("--w-curvature", type=float, default=1e6)
    parser.add_argument("--w-root-pin", type=float, default=1e7)
    parser.add_argument("--max-tubes", type=int, default=None,
                        help="Limit number of tubes (for testing/memory)")

    # Output
    parser.add_argument("--save-combined", action="store_true",
                        help="Also save combined OBJ files")
    parser.add_argument("--skip-relax", action="store_true",
                        help="Only generate tubes, skip relaxation")
    parser.add_argument("--vis", action="store_true",
                        help="Interactive Polyscope visualization with Step/Run")
    args = parser.parse_args()

    # ---- Load or generate ----
    if args.tube_dir:
        tube_meshes = load_tube_objs(args.tube_dir)
    else:
        tube_meshes = generate_tubes_from_config(
            args.config, tip_scale=args.current_scale,
        )

    # ---- Auto-detect n_cross ----
    n_cross = args.n_cross or auto_detect_n_cross(tube_meshes)
    print(f"Cross-section vertices: {n_cross}")

    # ---- Rescale ----
    from hair_tube_relaxation.tube_relaxation import MultiTubeRelaxation

    if not args.skip_rescale:
        print(f"Rescaling tubes: {args.current_scale} -> {args.target_scale}")
        tube_meshes = MultiTubeRelaxation.rescale_tubes(
            tube_meshes, n_cross,
            target_scale=args.target_scale,
            current_scale=args.current_scale,
        )

    # ---- Save pre-relaxation ----
    os.makedirs(args.output_dir, exist_ok=True)
    if args.save_combined:
        save_combined_obj(
            tube_meshes,
            os.path.join(args.output_dir, "combined_pre_relax.obj"),
        )

    if args.skip_relax:
        save_tube_objs(tube_meshes, args.output_dir, prefix="tube_scaled")
        if args.save_combined:
            save_combined_obj(
                tube_meshes,
                os.path.join(args.output_dir, "combined_scaled.obj"),
            )
        print("Done (relaxation skipped).")
        return

    # ---- Relax ----
    print(f"\nRelaxing {len(tube_meshes)} tubes: "
          f"{args.n_iters} iters, lr={args.lr}, alpha={args.alpha}")

    relaxer = MultiTubeRelaxation(tube_meshes, n_cross, device='cuda',
                                   max_tubes=args.max_tubes)
    run_fn = relaxer.run_vis if args.vis else relaxer.run
    relaxed = run_fn(
        n_iters=args.n_iters,
        lr=args.lr,
        alpha=args.alpha,
        max_collisions=args.max_collisions,
        w_curvature=args.w_curvature,
        w_root_pin=args.w_root_pin,
    )

    # ---- Save results ----
    save_tube_objs(relaxed, args.output_dir, prefix="tube_relaxed")
    if args.save_combined:
        save_combined_obj(
            relaxed,
            os.path.join(args.output_dir, "combined_relaxed.obj"),
        )
    print("Done.")


if __name__ == "__main__":
    main()
