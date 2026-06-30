#!/usr/bin/env python3
"""Unit tests for metrics_utils.py — verify label normalization and metric computation."""

import sys
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from metrics_utils import ensure_1d_label, clustering_metrics, check_metric_inputs


def test_ensure_1d_label():
    """Test all acceptable label shapes."""
    y1 = np.array([[1], [2], [1], [2]])     # [n, 1]
    y2 = np.array([[1, 2, 1, 2]])           # [1, n]
    y3 = np.array([1, 2, 1, 2])             # [n]

    r1 = ensure_1d_label(y1, "y1")
    r2 = ensure_1d_label(y2, "y2")
    r3 = ensure_1d_label(y3, "y3")

    assert r1.shape == (4,), f"y1 shape mismatch: {r1.shape}"
    assert r2.shape == (4,), f"y2 shape mismatch: {r2.shape}"
    assert r3.shape == (4,), f"y3 shape mismatch: {r3.shape}"
    assert np.array_equal(r1, r3), f"y1 and y3 should be equal after squeeze"

    print(f"  ensure_1d_label: [n,1]→{r1.shape}, [1,n]→{r2.shape}, [n]→{r3.shape}  ✅")


def test_invalid_shape():
    """Test that invalid 2D shapes raise error."""
    y_bad = np.array([[1, 2], [3, 4]])  # [n, 2] — should fail
    try:
        ensure_1d_label(y_bad, "bad")
        assert False, "Should have raised ValueError for [n,2] input"
    except ValueError as e:
        print(f"  Invalid shape rejection: {str(e)[:60]}...  ✅")


def test_clustering_metrics():
    """Test metric computation with known labels."""
    y1 = np.array([[1], [2], [1], [2]])
    pred = np.array([0, 1, 0, 1])

    acc, nmi, ari = clustering_metrics(y1, pred)

    assert 0.0 <= acc <= 1.0, f"ACC out of range: {acc}"
    assert 0.0 <= nmi <= 1.0, f"NMI out of range: {nmi}"
    assert 0.0 <= ari <= 1.0, f"ARI out of range: {ari}"

    print(f"  clustering_metrics: acc={acc:.4f}, nmi={nmi:.4f}, ari={ari:.4f}  ✅")


def test_check_metric_inputs():
    """Test runtime input check."""
    y_true = np.array([1, 2, 1, 2])
    y_pred = np.array([0, 1, 0, 1])

    yt, yp = check_metric_inputs(y_true, y_pred, dataset="test", method="DMCAG")
    assert yt.shape == (4,)
    assert yp.shape == (4,)
    print(f"  check_metric_inputs: y_true→{yt.shape}, y_pred→{yp.shape}  ✅")


def test_length_mismatch():
    """Test that length mismatch raises error."""
    y_true = np.array([1, 2, 1, 2])
    y_pred = np.array([0, 1, 0])

    try:
        check_metric_inputs(y_true, y_pred, dataset="test", method="DMCAG")
        assert False, "Should have raised ValueError for length mismatch"
    except ValueError as e:
        print(f"  Length mismatch rejection: {str(e)[:60]}...  ✅")


def main():
    print("Testing metrics_utils...")
    test_ensure_1d_label()
    test_invalid_shape()
    test_clustering_metrics()
    test_check_metric_inputs()
    test_length_mismatch()
    print()
    print("metrics_test_ok")
    print("All 5 tests passed ✅")


if __name__ == "__main__":
    main()
