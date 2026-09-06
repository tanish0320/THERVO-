import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import os

def generate_telemetry_csv(filename=os.path.join("..", "data", "raw", "telemetry_data.csv"), n_rows=100):
    """
    Generates a synthetic telemetry dataset mirroring the schema in TELEMETRY_README.md.
    """
    np.random.seed(42)
    start_time = datetime.now() - timedelta(minutes=n_rows)
    
    data = []
    for i in range(n_rows):
        timestamp = (start_time + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # CPU/GPU/Memory are percentages
        cpu = np.random.uniform(10, 90)
        gpu = np.random.uniform(5, 80)
        memory = np.random.uniform(30, 70)
        
        # Disk/Network IO in bytes/sec (arbitrary large ranges)
        disk_io = np.random.uniform(1000, 500000)
        network_io = np.random.uniform(500, 100000)
        
        # Temperatures usually lag behind utilization
        gpu_temp = 30 + (gpu * 0.5) + np.random.normal(0, 2)
        cpu_temp = 35 + (cpu * 0.4) + np.random.normal(0, 3)
        
        # Introduce some missing values and outliers for cleaning testing
        if i % 20 == 0:
            cpu = np.nan
        if i % 25 == 0:
            gpu_temp = 200  # Extreme outlier
            
        data.append([timestamp, cpu, gpu, memory, disk_io, network_io, gpu_temp, cpu_temp])

    columns = ["timestamp", "cpu", "gpu", "memory", "disk_io", "network_io", "gpu_temp", "cpu_temp"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename, index=False)
    print(f"Generated {n_rows} rows of test data in {filename}")

if __name__ == "__main__":
    generate_telemetry_csv()
