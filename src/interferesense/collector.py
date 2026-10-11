import torch
import multiprocessing as mp
from pathlib import Path
import time
import psutil
from torch import nn
import pynvml


def kernel_worker(spec, config, ready, start, stop, count, memory):
    torch.set_num_threads(config["threads"])
    device = torch.device(
        config["device"]
        if config["device"] != "auto"
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    torch.manual_seed(config["seed"])
    size = spec["size"]
    value = torch.randn(size, size, device=device)
    other = torch.randn_like(value)
    model = nn.Sequential(nn.Linear(size, size), nn.ReLU(), nn.Linear(size, size)).to(
        device
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    location = Path(config["output"]) / "scratch" / f"{mp.current_process().pid}.bin"
    location.parent.mkdir(parents=True, exist_ok=True)

    def execute():
        if spec["kernel"] == "matmul":
            torch.mm(value, other)
        elif spec["kernel"] == "memory":
            torch.tanh(value) + other
        elif spec["kernel"] == "mlp":
            optimizer.zero_grad(set_to_none=True)
            model(value).square().mean().backward()
            optimizer.step()
        elif spec["kernel"] == "io":
            location.write_bytes(value.detach().cpu().numpy().tobytes())
        else:
            raise ValueError(f"Unknown measured kernel: {spec['kernel']}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            memory.value = torch.cuda.max_memory_allocated(device) / 1e9

    execute()
    ready.put(mp.current_process().pid)
    start.wait()
    while not stop.is_set():
        execute()
        count.value += 1
    location.unlink(missing_ok=True)


class Telemetry:
    def __init__(self, config):
        self.config = config
        self.gpu = None
        self.previous = None
        self.nvml = False
        if config["device"] == "cpu":
            return
        try:
            pynvml.nvmlInit()
            self.nvml = True
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(config["gpu_index"])
        except (
            pynvml.NVMLError_LibraryNotFound,
            pynvml.NVMLError_DriverNotLoaded,
            pynvml.NVMLError_NoPermission,
        ):
            self.gpu = None

    def read(self, pids):
        now = time.monotonic()
        cpu = min(1.0, psutil.cpu_percent() / 100)
        disk, net = psutil.disk_io_counters(), psutil.net_io_counters()
        storage = network = None
        if self.previous is not None and disk and net:
            before, old_disk, old_net = self.previous
            dt = max(now - before, 1e-8)
            storage = min(
                1.0,
                (
                    (disk.read_bytes + disk.write_bytes)
                    - (old_disk.read_bytes + old_disk.write_bytes)
                )
                / dt
                / self.config["storage_ceiling_bytes_s"],
            )
            network = min(
                1.0,
                (
                    (net.bytes_sent + net.bytes_recv)
                    - (old_net.bytes_sent + old_net.bytes_recv)
                )
                / dt
                / self.config["network_ceiling_bytes_s"],
            )
        self.previous = (now, disk, net)
        groups = [None, None, None, None, None, cpu, storage, network]
        memory = 0.0
        if self.gpu is not None:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(self.gpu)
            groups[0], groups[2] = utilization.gpu / 100, utilization.memory / 100
            try:
                tx = pynvml.nvmlDeviceGetPcieThroughput(
                    self.gpu, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                rx = pynvml.nvmlDeviceGetPcieThroughput(
                    self.gpu, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
                groups[4] = min(
                    1.0, (tx + rx) * 1024 / self.config["pcie_ceiling_bytes_s"]
                )
            except pynvml.NVMLError_NotSupported:
                groups[4] = None
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(self.gpu)
            memory = (
                sum(
                    process.usedGpuMemory
                    for process in processes
                    if process.pid in pids
                    and process.usedGpuMemory is not None
                    and process.usedGpuMemory < 2**60
                )
                / 1e9
            )
        return {"timestamp": now, "groups": groups, "memory_gb": memory}

    def close(self):
        if self.nvml:
            pynvml.nvmlShutdown()


def measure(specs, config):
    context = mp.get_context("spawn")
    ready = context.Queue()
    start, stop = context.Event(), context.Event()
    counters = [context.RawValue("q", 0) for _ in specs]
    memories = [context.RawValue("d", 0) for _ in specs]
    processes = [
        context.Process(
            target=kernel_worker, args=(spec, config, ready, start, stop, count, memory)
        )
        for spec, count, memory in zip(specs, counters, memories)
    ]
    for process in processes:
        process.start()
    collector = Telemetry(config)
    telemetry = []
    try:
        pids = [ready.get(timeout=config["startup_timeout_s"]) for _ in specs]
        start.set()
        time.sleep(config["warmup_s"])
        initial = [counter.value for counter in counters]
        began = time.monotonic()
        while time.monotonic() - began < config["duration_s"]:
            sample = collector.read(pids)
            sample["units"] = [counter.value for counter in counters]
            telemetry.append(sample)
            time.sleep(config["sampling_s"])
        seconds = time.monotonic() - began
        units = [counter.value - before for counter, before in zip(counters, initial)]
        if not all(value > 0 for value in units):
            raise RuntimeError(
                "No useful work was observed; increase warmup or measurement duration"
            )
        return {
            "jobs": [spec["id"] for spec in specs],
            "units": units,
            "seconds": seconds,
            "telemetry": telemetry,
            "memory_gb": [memory.value for memory in memories],
            "truth_boundaries_s": [],
        }
    finally:
        stop.set()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join()
        collector.close()
