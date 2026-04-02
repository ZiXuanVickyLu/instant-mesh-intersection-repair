"""
Tests for the Warp ACCD edge-edge CCD implementation.

Run:  python -m hair_tube_relaxation.test_accd
"""

import torch
import numpy as np
import warp as wp


def test_no_collision():
    """Two edges moving apart -- should return TOI = 1.0 (no collision)."""
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Edge A: horizontal at y=1, moving up
    a0s = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    a1s = torch.tensor([[1.0, 1.0, 0.0]], device=device)
    # Edge B: horizontal at y=-1, moving down
    b0s = torch.tensor([[0.0, -1.0, 0.0]], device=device)
    b1s = torch.tensor([[1.0, -1.0, 0.0]], device=device)

    a0e = torch.tensor([[0.0, 2.0, 0.0]], device=device)
    a1e = torch.tensor([[1.0, 2.0, 0.0]], device=device)
    b0e = torch.tensor([[0.0, -2.0, 0.0]], device=device)
    b1e = torch.tensor([[1.0, -2.0, 0.0]], device=device)

    toi = batch_edge_edge_ccd(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e)
    assert toi[0].item() == 1.0, f"Expected 1.0, got {toi[0].item()}"
    print("[PASS] test_no_collision")


def test_head_on_collision():
    """Two edges moving toward each other -- should collide around t=0.5."""
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Edge A: at y=+0.5, moving to y=-0.5
    a0s = torch.tensor([[0.0, 0.5, 0.0]], device=device)
    a1s = torch.tensor([[1.0, 0.5, 0.0]], device=device)
    a0e = torch.tensor([[0.0, -0.5, 0.0]], device=device)
    a1e = torch.tensor([[1.0, -0.5, 0.0]], device=device)

    # Edge B: at y=-0.5, moving to y=+0.5
    b0s = torch.tensor([[0.0, -0.5, 0.0]], device=device)
    b1s = torch.tensor([[1.0, -0.5, 0.0]], device=device)
    b0e = torch.tensor([[0.0, 0.5, 0.0]], device=device)
    b1e = torch.tensor([[1.0, 0.5, 0.0]], device=device)

    toi = batch_edge_edge_ccd(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e)
    # They meet at t~0.5 (distance=1.0, closing speed=2.0, xi=1e-3)
    val = toi[0].item()
    # ACCD is conservative -- it clamps before exact contact
    assert 0.40 < val < 0.51, f"Expected ~0.45-0.50, got {val}"
    print(f"[PASS] test_head_on_collision (toi={val:.4f})")


def test_parallel_no_overlap():
    """Two parallel edges that never overlap in projection -- no collision."""
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Edge A along x at z=0, Edge B along x at z=2, both moving toward y=0
    a0s = torch.tensor([[0.0, 0.1, 0.0]], device=device)
    a1s = torch.tensor([[1.0, 0.1, 0.0]], device=device)
    b0s = torch.tensor([[0.0, 0.1, 2.0]], device=device)
    b1s = torch.tensor([[1.0, 0.1, 2.0]], device=device)

    a0e = torch.tensor([[0.0, -0.1, 0.0]], device=device)
    a1e = torch.tensor([[1.0, -0.1, 0.0]], device=device)
    b0e = torch.tensor([[0.0, -0.1, 2.0]], device=device)
    b1e = torch.tensor([[1.0, -0.1, 2.0]], device=device)

    toi = batch_edge_edge_ccd(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e)
    assert toi[0].item() == 1.0, f"Expected 1.0, got {toi[0].item()}"
    print("[PASS] test_parallel_no_overlap")


def test_crossing_edges():
    """Two skew edges that cross mid-sweep."""
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Edge A: along x, at z=+0.05. Moves to z=-0.05
    a0s = torch.tensor([[-1.0, 0.0, 0.05]], device=device)
    a1s = torch.tensor([[1.0, 0.0, 0.05]], device=device)
    a0e = torch.tensor([[-1.0, 0.0, -0.05]], device=device)
    a1e = torch.tensor([[1.0, 0.0, -0.05]], device=device)

    # Edge B: along y, at z=-0.05. Moves to z=+0.05
    b0s = torch.tensor([[0.0, -1.0, -0.05]], device=device)
    b1s = torch.tensor([[0.0, 1.0, -0.05]], device=device)
    b0e = torch.tensor([[0.0, -1.0, 0.05]], device=device)
    b1e = torch.tensor([[0.0, 1.0, 0.05]], device=device)

    toi = batch_edge_edge_ccd(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e)
    val = toi[0].item()
    # Initial separation = 0.1, closing speed = 0.2, meet at t~0.5
    assert 0.0 < val < 0.55, f"Expected collision at t<0.55, got {val}"
    print(f"[PASS] test_crossing_edges (toi={val:.4f})")


def test_static_edges():
    """Both edges static -- no collision."""
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    a0s = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    a1s = torch.tensor([[1.0, 0.0, 0.0]], device=device)
    b0s = torch.tensor([[0.0, 1.0, 0.0]], device=device)
    b1s = torch.tensor([[1.0, 1.0, 0.0]], device=device)

    toi = batch_edge_edge_ccd(a0s, a1s, b0s, b1s, a0s, a1s, b0s, b1s)
    assert toi[0].item() == 1.0, f"Expected 1.0, got {toi[0].item()}"
    print("[PASS] test_static_edges")


def test_batch():
    """Multiple edge pairs in a single batch call."""
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    n = 1024
    # Random edges well separated -- most should be no-collision
    a0s = torch.randn(n, 3, device=device) + torch.tensor([0, 5, 0], device=device)
    a1s = a0s + torch.randn(n, 3, device=device) * 0.1
    b0s = torch.randn(n, 3, device=device) + torch.tensor([0, -5, 0], device=device)
    b1s = b0s + torch.randn(n, 3, device=device) * 0.1

    # Small random motion
    a0e = a0s + torch.randn(n, 3, device=device) * 0.01
    a1e = a1s + torch.randn(n, 3, device=device) * 0.01
    b0e = b0s + torch.randn(n, 3, device=device) * 0.01
    b1e = b1s + torch.randn(n, 3, device=device) * 0.01

    toi = batch_edge_edge_ccd(a0s, a1s, b0s, b1s, a0e, a1e, b0e, b1e)
    assert toi.shape == (n,), f"Shape mismatch: {toi.shape}"
    # Most should be 1.0 (no collision) since edges are 10 units apart
    n_collisions = (toi < 1.0).sum().item()
    print(f"[PASS] test_batch ({n} pairs, {n_collisions} collisions)")


def test_empty():
    """Empty input."""
    from hair_tube_relaxation.accd_warp import batch_edge_edge_ccd

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    empty = torch.zeros(0, 3, device=device)
    toi = batch_edge_edge_ccd(empty, empty, empty, empty, empty, empty, empty, empty)
    assert toi.shape == (0,)
    print("[PASS] test_empty")


if __name__ == "__main__":
    wp.init()
    test_empty()
    test_static_edges()
    test_no_collision()
    test_parallel_no_overlap()
    test_head_on_collision()
    test_crossing_edges()
    test_batch()
    print("\nAll tests passed.")
