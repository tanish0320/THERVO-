import time
import subprocess
import threading
import os

# ========== CONFIG ==========
CPU_CORES = 6
PHASES = [
    ("WARMUP", 300),
    ("IDLE", 600),
    ("CPU", 600),
    ("IDLE", 300),
    ("MEMORY", 600),
    ("IDLE", 300),
    ("DISK", 600),
    ("NETWORK", 600),
    ("MIXED", 600),
    ("BURST", 300),
]

LOG_FILE = "phase_log.txt"

# ========== HELPERS ==========
def log(msg):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

# ---------- CPU ----------
def cpu_stress(duration):
    try:
        return subprocess.Popen(
            ["stress-ng", "--cpu", str(CPU_CORES), "--timeout", str(duration)]
        )
    except:
        # fallback python loop
        def burn():
            end = time.time() + duration
            while time.time() < end:
                [x**2 for x in range(10_000)]
        t = threading.Thread(target=burn)
        t.start()
        return t

# ---------- MEMORY ----------
def memory_stress(duration):
    def mem():
        try:
            a = []
            for _ in range(10):
                a.append(bytearray(100 * 1024 * 1024))  # ~100MB chunks
                time.sleep(1)
            time.sleep(duration)
        except:
            pass
    t = threading.Thread(target=mem)
    t.start()
    return t

# ---------- DISK ----------
def disk_stress(duration):
    def disk():
        end = time.time() + duration
        while time.time() < end:
            with open("temp_stress.bin", "wb") as f:
                f.write(os.urandom(100 * 1024 * 1024))  # 100MB write
            os.remove("temp_stress.bin")
    t = threading.Thread(target=disk)
    t.start()
    return t

# ---------- NETWORK ----------
def network_stress(duration):
    def net():
        end = time.time() + duration
        while time.time() < end:
            try:
                subprocess.run(
                    ["curl", "-L", "-o", "temp_net.bin", "http://speedtest.tele2.net/100MB.zip"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                os.remove("temp_net.bin")
            except:
                pass
    t = threading.Thread(target=net)
    t.start()
    return t

# ---------- MIXED ----------
def mixed_stress(duration):
    threads = []
    threads.append(cpu_stress(duration))
    threads.append(memory_stress(duration))
    threads.append(disk_stress(duration))
    threads.append(network_stress(duration))
    return threads

# ---------- BURST ----------
def burst_stress(duration):
    end = time.time() + duration
    while time.time() < end:
        log("BURST → CPU spike")
        p = cpu_stress(10)
        time.sleep(5)
        log("BURST → stop")
        time.sleep(5)

# ========== MAIN ==========
def run_phase(name, duration):
    log(f"START {name}")

    processes = []

    if name == "CPU":
        processes.append(cpu_stress(duration))

    elif name == "MEMORY":
        processes.append(memory_stress(duration))

    elif name == "DISK":
        processes.append(disk_stress(duration))

    elif name == "NETWORK":
        processes.append(network_stress(duration))

    elif name == "MIXED":
        processes.extend(mixed_stress(duration))

    elif name == "BURST":
        burst_stress(duration)
        return

    # idle / warmup just sleep
    time.sleep(duration)

    log(f"END {name}")


def main():
    log("=== WORKLOAD SCRIPT STARTED ===")

    for phase, duration in PHASES:
        run_phase(phase, duration)

    log("=== WORKLOAD SCRIPT COMPLETED ===")


if __name__ == "__main__":
    main()
