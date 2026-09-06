# Unified Preprocessing & Feature Engineering
## Cooling System ML Pipeline (V2 - Sensor-Free)

This document describes the shared preprocessing architecture used to ensure absolute consistency between Machine Learning model training (batch) and real-time inference (streaming).

### 1. Overview: The Problem of "Skew"
In many ML systems, the code used to "clean" data for training is different from the code used for live predictions. This leads to **Training-Serving Skew**, where the model receives data in a format it doesn't recognize.

Our system solves this by using a **Shared Feature Engine** located in `src/features.py`.

---

### 2. The Core Engine: `FeatureProcessor`
Both `preprocess.py` and `inference.py` use the `FeatureProcessor` class. This class acts as the "source of truth" for:

*   **Feature Ordering**: Maintains the strict order of the 15 features required by the XGBoost model.
*   **Log-Scaling (I/O)**: Uses $log(1 + x)$ to stabilize high-variance Disk and Network I/O metrics.
*   **Thermal Proxy Logic**: Specifically designed for **sensor-free** operation. It uses utilization metrics to derive `heat_norm` and `heat_delta` instead of relying on physical thermal sensors.
*   **Validation Layer**: Automatically clips values to `[0, 1]`, handles `NaN` values, and asserts the vector length.

---

### 3. Training Workflow (Pre-fitting)
During training preprocessing:
1.  `preprocess.py` loads the raw telemetry CSV.
2.  It calls `processor.fit(df)` to calculate the distribution of Disk and Network I/O using log-maximums.
3.  It saves these scaling parameters to `models/preprocessor_state.pkl`.
4.  It generates `X.csv` by feeding the raw data through the processor row-by-row.

### 4. Inference Workflow (Syncing)
During real-time inference:
1.  `inference.py` loads the `models/preprocessor_state.pkl` on startup.
2.  It initializes its own `FeatureProcessor` with these exact training-time log bounds.
3.  Each new 1-second telemetry point is fed into the processor, which updates its internal buffer and returns a feature vector ready for the model.

---

### 5. Feature Dictionary (V2)
| Column Index | Feature Name | Description |
| :--- | :--- | :--- |
| 0-2 | `cpu_norm`, `gpu_norm`, `mem_norm` | Normalized percentages [0, 1] |
| 3-4 | `disk_io_norm`, `net_io_norm` | **Log-scaled** I/O based on training bounds |
| 5 | `heat_norm` | Derived: `0.6*cpu + 0.4*gpu` |
| 6-8 | `cpu_roll_5`, `gpu_roll_5`, `heat_roll_5` | 5-step (5-second) rolling mean |
| 9-11 | `cpu_roll_10`, `gpu_roll_10`, `heat_roll_10` | 10-step (10-second) rolling mean |
| 12-14 | `cpu_delta`, `gpu_delta`, `heat_delta` | Current value minus previous value |

---

### 6. Implementation & Robustness
*   **Log Scaling**: Disk and Network rates are processed as $log(1 + rate) / max\_log\_rate$. This compresses large spikes (e.g. 50MB/s) and preserves detail in lower ranges.
*   **Buffer Warmup**: Rolling features calculate means based on the available window (e.g., at 3 seconds, it's a 3-step average) to ensure immediate prediction capability.
*   **Validation**: Every feature vector is passed through a validation layer that:
    - Replaces `NaN` with `0.0`.
    - Clips all values to `[0.0, 1.0]`.
    - Asserts the 15-feature shape.
