"""
src/core/fusion.py
------------------
SINGLE SOURCE OF TRUTH for all scoring, fusion, and risk-level logic.

Rule: EVERY file that needs these functions MUST import from here.
      NO duplication allowed. NO local redefinitions.

PRD contracts implemented:
  - Risk fusion    (PRD §4.2): risk = clip(0.75*xgb + 0.25*gnn, 0, 1)
  - Risk levels    (FRD §5.1): LOW / MED / HIGH / CRITICAL thresholds
  - Parity check             : assert train_vec == infer_vec (hard fail)
"""

import numpy as np
from typing import Union

# ---------------------------------------------------------------------------
# FUSION WEIGHTS  (PRD §4.2)
# ---------------------------------------------------------------------------
_XGB_WEIGHT: float = 0.75
_GNN_WEIGHT: float = 0.25

# ---------------------------------------------------------------------------
# RISK LEVEL THRESHOLDS  (FRD §5.1)
# ---------------------------------------------------------------------------
_THRESH_LOW:  float = 0.35
_THRESH_MED:  float = 0.55
_THRESH_HIGH: float = 0.75


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def fuse(xgb_score: float, gnn_embedding: float) -> float:
    """
    Compute the final risk score by fusing XGBoost output and GNN embedding.

    Formula (PRD §4.2):
        risk = clip(0.75 * xgb_score + 0.25 * gnn_embedding, 0, 1)

    Parameters
    ----------
    xgb_score     : float in [0, 1] — XGBoost regression output.
    gnn_embedding : float in [0, 1] — AnalyticGNN scalar embedding.

    Returns
    -------
    risk : float in [0, 1]

    Raises
    ------
    ValueError : if either input is outside [0, 1].
    """
    if not (0.0 <= xgb_score <= 1.0):
        raise ValueError(
            f"[fusion.fuse] xgb_score={xgb_score:.6f} is outside [0, 1]. "
            f"Check that the XGBoost model output is clipped correctly."
        )
    if not (0.0 <= gnn_embedding <= 1.0):
        raise ValueError(
            f"[fusion.fuse] gnn_embedding={gnn_embedding:.6f} is outside [0, 1]. "
            f"Check AnalyticGNN.compute_single() output."
        )

    result = float(np.clip(_XGB_WEIGHT * xgb_score + _GNN_WEIGHT * gnn_embedding, 0.0, 1.0))
    return result


def get_risk_level(score: float) -> str:
    """
    Map a [0, 1] risk score to a human-readable level string.

    Thresholds (FRD §5.1):
        score < 0.35  → LOW
        score < 0.55  → MED
        score < 0.75  → HIGH
        score >= 0.75 → CRITICAL

    Parameters
    ----------
    score : float in [0, 1]

    Returns
    -------
    level : str — one of 'LOW', 'MED', 'HIGH', 'CRITICAL'
    """
    if score < _THRESH_LOW:
        return "LOW"
    if score < _THRESH_MED:
        return "MED"
    if score < _THRESH_HIGH:
        return "HIGH"
    return "CRITICAL"


def assert_parity(
    train_vec: np.ndarray,
    infer_vec: np.ndarray,
    label: str = "feature vector",
    tol: float = 1e-9,
) -> None:
    """
    Assert that a training-time feature vector is byte-identical to the
    inference-time vector produced by the same input.

    HARD FAIL — never catch this assertion in production code.

    Parameters
    ----------
    train_vec : np.ndarray produced by the training pipeline.
    infer_vec : np.ndarray produced by the inference pipeline.
    label     : descriptive label for error messages.
    tol       : absolute tolerance for np.allclose (default 1e-9 = near exact).

    Raises
    ------
    AssertionError : if shapes differ or values are not allclose.
    """
    train_flat = np.asarray(train_vec, dtype=np.float64).flatten()
    infer_flat = np.asarray(infer_vec, dtype=np.float64).flatten()

    if train_flat.shape != infer_flat.shape:
        raise AssertionError(
            f"[fusion.assert_parity] TRAINING-SERVING SKEW DETECTED in '{label}':\n"
            f"  Training shape : {train_flat.shape}\n"
            f"  Inference shape: {infer_flat.shape}\n"
            f"  This is a critical bug. Investigate immediately."
        )

    if not np.allclose(train_flat, infer_flat, atol=tol):
        diffs = np.abs(train_flat - infer_flat)
        worst_idx = int(np.argmax(diffs))
        raise AssertionError(
            f"[fusion.assert_parity] TRAINING-SERVING SKEW DETECTED in '{label}':\n"
            f"  Max deviation  : {diffs.max():.2e}  at index {worst_idx}\n"
            f"  Training value : {train_flat[worst_idx]:.10f}\n"
            f"  Inference value: {infer_flat[worst_idx]:.10f}\n"
            f"  Full train vec : {train_flat}\n"
            f"  Full infer vec : {infer_flat}\n"
            f"  This is a critical bug. Investigate immediately."
        )
