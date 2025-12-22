import multiprocessing as mp
import time
import os

# ---------------- CPU LOAD ----------------
def cpu_worker(busy_time, idle_time):
    while True:
        start = time.perf_counter()
        while (time.perf_counter() - start) < busy_time:
            pass
        time.sleep(idle_time)

# ---------------- GPU LOAD ----------------
def gpu_worker(busy_time, idle_time, matrix_size, device_index=0):
    import torch
    import time

    torch.cuda.set_device(device_index)
    device = torch.device(f"cuda:{device_index}")

    # Pre-allocate tensors ONCE
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    while True:
        start = time.perf_counter()
        while (time.perf_counter() - start) < busy_time:
            torch.matmul(a, b)

        # Make sure GPU work is actually finished
        torch.cuda.synchronize(device)
        time.sleep(idle_time)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # ===== CONFIG =====
    # CPU
    total_cpu_cores = 32           # Ryzen 9 9950X physical cores
    cpu_target_load = 0.30         # 60–70%
    cpu_cycle = 0.1

    # GPU
    gpu_target_load = 0.30         # 60–70%
    gpu_cycle = 0.1
    gpu_matrix_size = 4096         # adjust if needed
    gpu_workers = 1                # usually 1 saturates GPU well

    # ==================
    cpu_busy = cpu_cycle * cpu_target_load
    cpu_idle = cpu_cycle * (1 - cpu_target_load)

    gpu_busy = gpu_cycle * gpu_target_load
    gpu_idle = gpu_cycle * (1 - gpu_target_load)

    print(f"PID {os.getpid()}")
    print(f"CPU load target: {int(cpu_target_load*100)}%")
    print(f"GPU load target: {int(gpu_target_load*100)}%")

    # Start CPU workers
    cpu_processes = []
    for _ in range(total_cpu_cores):
        p = mp.Process(target=cpu_worker, args=(cpu_busy, cpu_idle))
        p.daemon = True
        p.start()
        cpu_processes.append(p)

    # Start GPU workers
    gpu_processes = []
    for i in range(gpu_workers):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_busy, gpu_idle, gpu_matrix_size, 0)  # GPU 0
        )
        p.daemon = True
        p.start()
        gpu_processes.append(p)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping load generators...")
