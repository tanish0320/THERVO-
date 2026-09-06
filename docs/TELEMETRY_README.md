# Telemetry Logger — Setup & Usage Guide
# AI-Driven Sensor-Free Predictive Cooling System

## 1. Install Dependencies

```bash
pip install psutil
```

> **GPU support** — no extra pip install needed.  
> `nvidia-smi` must be accessible in your system PATH.  
> It ships with every NVIDIA display/compute driver installation.

---

## 2. Run the Logger

### Default (1-second interval → telemetry_data.csv)
```bash
python telemetry_logger.py
```

### Custom interval (e.g., 2 seconds)
```bash
python telemetry_logger.py --interval 2
```

### Custom output file
```bash
python telemetry_logger.py --output my_server_data.csv
```

### Combine options
```bash
python telemetry_logger.py --interval 0.5 --output rack_a1_telemetry.csv
```

---

## 3. CSV Output Schema

| Column       | Type   | Description                        |
|--------------|--------|------------------------------------|
| timestamp    | string | YYYY-MM-DD HH:MM:SS.mmm            |
| cpu          | float  | CPU utilisation %                  |
| gpu          | float  | GPU utilisation % (0 if no GPU)    |
| memory       | float  | RAM utilisation %                  |
| disk_io      | float  | Disk read+write bytes/sec          |
| network_io   | float  | Network sent+recv bytes/sec        |
| gpu_temp     | float  | GPU temperature °C (0 if no GPU)   |
| cpu_temp     | float  | CPU temperature °C (0 if unavail.) |

---

## 4. CPU Temperature Notes

| Platform | Method                       | Requirements                              |
|----------|------------------------------|-------------------------------------------|
| Linux    | psutil sensors_temperatures  | Works on most Intel/AMD systems           |
| macOS    | psutil sensors_temperatures  | Limited to supported models               |
| Windows  | OpenHardwareMonitor WMI      | OHM must be running as Administrator      |

To enable CPU temps on Windows:
1. Download [OpenHardwareMonitor](https://openhardwaremonitor.org/)
2. Run as Administrator
3. Enable "Options → Remote Web Server" (or just keep it running)

---

## 5. Log Files

| File                  | Contents                                      |
|-----------------------|-----------------------------------------------|
| telemetry_data.csv    | All collected metrics (append mode)           |
| telemetry_logger.log  | Full debug + error log with timestamps        |

---

## 6. Stop the Logger

Press **Ctrl-C** at any time. Data is flushed to CSV on every tick — nothing is lost.

---

## 7. Use Collected Data for ML Training

Feed `telemetry_data.csv` directly into your GNN + XGBoost pipeline:

```python
import pandas as pd

df = pd.read_csv("telemetry_data.csv", parse_dates=["timestamp"])
features = df[["cpu", "gpu", "memory", "disk_io", "network_io"]].values
# → pass to XGBoost / GNN feature engineering pipeline
```
