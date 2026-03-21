# Algorithm Analysis: Instant Self-Intersection Repair for 3D Meshes

This document provides a detailed analysis of the optimization-based mesh repair algorithm implemented in this repository.

## Overview

The system removes self-intersections from 3D triangle meshes by formulating the problem as an energy minimization in **differential coordinate space**. Rather than directly moving vertices, the algorithm optimizes Laplacian-parameterized coordinates and recovers vertex positions via a Cholesky solve at each step. This enables large, smooth deformations while preserving local geometry.

## Pipeline

```
Input mesh (.obj)
    |
    v
[1] Normalize to [0, 16.5] bounding range
    |
    v
[2] Compute cotangent Laplacian L
    Build parameterization matrix M = (1-a)I + aL,  a = 0.99
    |
    v
[3] Convert vertices to differential coords: u = M @ v
    Cache initial constraints (V0, A0, L0)
    |
    v
[4] Optimization loop (60 iterations)
    |   Decode:   v = M^{-1} u   (Cholesky solve)
    |   Detect:   collision pairs via BVH
    |   Compute:  penetration energy + regularization
    |   Update:   u <- u - lr * grad
    |
    v
[5] Output best mesh (fewest collisions)
```

## 1. Differential Coordinate Parameterization

**Files:** `largesteps/geometry.py`, `largesteps/parameterize.py`, `largesteps/solvers.py`

### Motivation

Directly optimizing vertex positions leads to noisy, local deformations. Instead, the algorithm works in a smoothed coordinate space defined by the Laplacian.

### Parameterization Matrix

```
M = (1 - alpha) * I + alpha * L
```

where `L` is the cotangent Laplacian and `alpha = 0.99`. This heavily weights the Laplacian term, making the parameterization non-local: a change to one differential coordinate affects a neighborhood of vertices.

### Forward and Inverse Maps

| Direction | Operation | Implementation |
|---|---|---|
| Vertices -> Differential | `u = M @ v` | Sparse matrix-vector multiply |
| Differential -> Vertices | `v = M^{-1} u` | Cholesky decomposition (cached) |

The Cholesky factorization of `M` is computed once and reused across iterations. A weak-reference cache (`parameterize.py`) automatically cleans up when `M` is garbage-collected.

### Cotangent Laplacian

The cotangent Laplacian (`geometry.py:laplacian_cot`) uses angle-dependent edge weights:

```
L[v_j, v_k] = cot(theta_opposite) / 4
```

computed via Heron's formula for numerical stability. The diagonal is set so each row sums to zero: `L[v_i, v_i] = -sum_j L[v_i, v_j]`.

## 2. Collision Detection (BVH)

**Files:** `externals/torch-mesh-isect/mesh_intersection/bvh_search_tree.py`, `externals/torch-mesh-isect/src/bvh_cuda_op.cu`

### Bounding Volume Hierarchy

The BVH tree partitions triangles into axis-aligned bounding boxes for O(n log n) broad-phase collision detection:

1. **Build:** Construct binary tree of AABBs over all triangles (GPU parallel)
2. **Query:** Traverse tree to find overlapping AABB pairs
3. **Narrow-phase:** For each AABB pair, test exact triangle-triangle intersection

### Interface

```python
search_tree = BVH(max_collisions=8)
triangles = vertices[faces]                              # (F, 3, 3)
collision_idxs = search_tree(triangles.unsqueeze(0))     # (1, F*max_col, 2)
collision_idxs = collision_idxs.squeeze(0)
collision_idxs = collision_idxs[collision_idxs[:, 0] >= 0, :]  # filter invalid
```

**Output:** `(C, 2)` tensor of colliding face-index pairs. Invalid entries are marked with `-1`.

### Non-Differentiability

BVH collision detection is combinatorial and non-differentiable. The algorithm bridges this by:
- Treating collision indices as constants (computed in `torch.no_grad()`)
- Computing differentiable energy only over detected collision pairs
- Re-detecting collisions each iteration to track changes

## 3. Penetration Energies

**Files:** `energies/TPE.py`, `energies/distance.py`, `energies/conical.py`

All energies compute a scalar loss over the set of colliding face pairs `{(f_i, f_j)}`.

### 3.1 Triangle Proximity Energy (TPE)

For each colliding pair `(f1, f2)` with centers `X, Y`, normals `n1, n2`, and areas `a1, a2`:

```
d     = ||Y - X||                          (Euclidean distance)
Pd_x  = |dot(X - Y, n1)|                   (projected distance onto f1 normal)
Pd_y  = |dot(Y - X, n2)|                   (projected distance onto f2 normal)

TPE += sqrt(a1 * a2) * ((Pd_x / d)^3 + (Pd_y / d)^3)
```

The cubed ratio `(Pd / d)^3` penalizes faces that are close in normal direction relative to their separation. Area weighting prioritizes larger collisions.

### 3.2 Signed TPE

Same structure as TPE but with **signed** projected distances (no absolute value) and configurable power `p`:

```
TPE += a1 * a2 * ((Pd_x / d)^p + (Pd_y / d)^p)
```

Signed distances provide directional gradients that guide faces to move apart along their normals.

### 3.3 Signed TPE Verts (default)

Instead of center-to-center, computes **vertex-to-center** distances for finer granularity:

```
For each vertex v_i in f1:
    d = ||v_i - center(f2)||
    Pd = dot(center(f2) - v_i, n2)
    TPE += a2 * (Pd / d)^p

Symmetrically for vertices of f2 against f1.
```

This provides 6 gradient sources per collision pair (3 vertices x 2 faces) instead of 2, giving stronger optimization signal.

### 3.4 Point-to-Plane (p2plane)

Uses a fixed reference normal (z-axis) for layered/planar structures:

```
loss = point_to_plane_distance(points, plane_points, plane_normals)
```

### 3.5 Conical

Delegates to the `mesh_intersection` library's `DistanceFieldPenetrationLoss` with sigma=0.5 smoothing. Uses a cone-shaped penalty field around collision surfaces.

### Energy Comparison

| Energy | Granularity | Signed | Best for |
|---|---|---|---|
| `TPE` | Center-center | No | General symmetric |
| `signed_TPE` | Center-center | Yes | Directional untangling |
| `signed_TPE_verts` | Vertex-center | Yes | Complex knots (default) |
| `p2plane` | Point-plane | Yes | Layered/planar meshes |
| `conical` | Distance field | Yes | Smooth distance penalties |

## 4. Geometric Constraints

**Files:** `constraints/volume.py`, `constraints/area.py`, `constraints/curvature.py`

Constraints prevent the mesh from degenerating during optimization. They are applied as soft penalties in the loss function.

### 4.1 Volume Preservation

```
V = sum_f dot(v0, cross(v1 - v0, v2 - v0)) / 6

reg += L1_loss(V_current, V_original)
```

Based on the divergence theorem: each triangle contributes a signed tetrahedron volume with the origin.

### 4.2 Area Preservation

```
A = sum_f 0.5 * ||cross(v1 - v0, v2 - v0)||

reg += L1_loss(A_current, A_original)
```

### 4.3 Curvature Preservation

```
L0 = L @ v_original        (initial Laplacian coordinates)

reg += 1e6 * MSE_loss(L @ v_current, L0)
```

Laplacian coordinates encode discrete mean curvature. The MSE penalty with weight **1e6** makes this the dominant constraint, effectively locking the local shape while allowing global repositioning.

### Constraint Weights

| Constraint | Weight | Effect |
|---|---|---|
| Volume | 1.0 | Loose -- allows moderate deformation |
| Area | 1.0 | Loose -- allows moderate deformation |
| Curvature | 1e6 | Dominant -- severely penalizes shape change |

The large curvature weight creates a stiff system where vertices can shift to resolve intersections but cannot significantly alter the surface's local shape.

## 5. Optimizers

**Files:** `optimizers/GD.py`, `optimizers/MomentumBrake.py`, `largesteps/optimize.py`

### 5.1 Gradient Descent (GD)

```
u <- u - lr * grad(loss, u)
```

Simple and stable. No momentum, no adaptive rates. Used as default in configs.

### 5.2 MomentumBrake

Applies momentum only when gradients are statistically consistent:

```
if step <= 10:  (warmup)
    g1 = beta1 * g1 + (1 - beta1) * grad       (standard momentum)
else:
    mean = g1 / (1 - beta1^step)
    std  = sqrt(g2 / (1 - beta2^step) - mean^2)
    if grad within [mean - 3*std, mean + 3*std]:
        g1 = beta1 * g1 + (1 - beta1) * grad    (apply momentum)
    else:
        g1 = grad                                 (reset to raw gradient)

u <- u - lr * 2 * bias_corrected(g1)
```

The 3-sigma band test suppresses momentum when gradients become erratic (e.g., near collision topology changes), preventing oscillation.

### 5.3 Adam / AdamUniform

Standard PyTorch Adam is available. AdamUniform is a variant that uses momentum without per-coordinate adaptive scaling.

## 6. Best Solution Tracking

The optimization tracks the **minimum collision count** rather than minimum loss:

```python
if num_collisions < best_col:
    best_col = num_collisions
    best_vertices = vertices.detach().clone()
    best_iter = i
```

This is important because the loss landscape is non-convex -- a lower loss does not guarantee fewer collisions. The final output is the mesh with the fewest detected intersections across all 60 iterations.

## 7. Key Design Decisions

### Why differential coordinates?

Direct vertex optimization produces local, noisy deformations. The Laplacian parameterization spreads each gradient update across a neighborhood, enabling smooth, large-scale untangling motions.

### Why alpha = 0.99?

The matrix `M = 0.01*I + 0.99*L` gives 99% weight to the Laplacian. This creates a highly non-local parameterization where small changes in `u` produce smooth, distributed vertex motions. The 1% identity component ensures `M` is strictly positive definite (invertible).

### Why is curvature weighted 1e6?

The mesh should maintain its shape while resolving intersections. The dominant curvature penalty forces the optimizer to find solutions that untangle without distorting the surface. Volume and area constraints are softer, allowing global scaling/inflation if needed.

### Why 60 iterations?

Empirically sufficient for most meshes. The celtic knot example converges to 0 collisions by step 29. More complex meshes may need tuning.

## References

- Jang et al., "Instant Self-Intersection Repair for 3D Meshes", ACM Transactions on Graphics (SIGGRAPH), 2025
- Nicolet et al., "Large Steps in Inverse Rendering of Geometry", ACM Transactions on Graphics, 2021 (largesteps parameterization)
- Meyer et al., "Discrete Differential-Geometry Operators for Triangulated 2-Manifolds", 2003 (cotangent Laplacian)
