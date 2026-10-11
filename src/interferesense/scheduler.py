import numpy as np
from scipy.optimize import linear_sum_assignment

VARIANTS = [
    "interferesense",
    "fcfs",
    "memory",
    "utilization",
    "random",
    "static",
    "no_phase",
    "symmetric",
    "gpu_only",
    "no_correction",
    "no_debt",
]


def activate_phase(job):
    phase = job["phases"][job["phase_index"]]
    for key in ["fingerprint", "solo_rate", "memory_gb"]:
        job[key] = phase.get(key, job[key])
    job["phase_remaining"] = phase["work"]


def measured_slowdown(first, second, observed):
    key = (
        first["profile"],
        first["phase_index"],
        second["profile"],
        second["phase_index"],
    )
    if key in observed:
        return observed[key]
    if len(first["phases"]) == len(second["phases"]) == 1:
        key = (first["profile"], second["profile"])
        if key in observed:
            return observed[key]
    raise ValueError(f"Missing measured ordered slowdown for {key}")


def simulate(jobs, observed, predictor, config, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown scheduling variant: {variant}")
    jobs = [
        dict(
            job,
            remaining=job["work"],
            waited_events=0,
            debt=0.0,
            started=None,
            finished=None,
        )
        for job in jobs
    ]
    if not jobs or config["gpus"] < 1 or config["tick_s"] <= 0:
        raise ValueError(
            "Scheduling needs jobs, a positive device count and a positive tick"
        )
    for job in jobs:
        job["phases"] = job.get("phases", [{"work": job["work"]}])
        if any(phase["work"] <= 0 for phase in job["phases"]) or not np.isclose(
            sum(phase["work"] for phase in job["phases"]), job["work"]
        ):
            raise ValueError(
                "Positive phase work budgets must sum to the job work budget"
            )
        if any(
            phase.get("memory_gb", job["memory_gb"]) > config["memory_gb"] * 0.92
            for phase in job["phases"]
        ):
            raise ValueError(
                f"Job {job['id']} cannot fit within the configured device memory reserve"
            )
        job["phase_index"] = 0
        activate_phase(job)
    running = [[] for _ in range(config["gpus"])]
    rng = np.random.default_rng(config["seed"])
    time = 0.0
    ticks = 0
    utilization = 0.0
    max_slowdown = 1.0
    alarms = violations = 0
    residuals = {}
    while any(job["finished"] is None for job in jobs):
        if time > config["simulation_timeout_s"]:
            raise RuntimeError(
                "Scheduler did not complete the trace within the configured time bound"
            )
        for g, occupants in enumerate(running):
            if (
                sum(jobs[i]["memory_gb"] for i in occupants)
                > config["memory_gb"] * 0.92
            ):
                running[g] = occupants[:1]
        ready = [
            i
            for i, job in enumerate(jobs)
            if job["arrival"] <= time
            and job["finished"] is None
            and all(i not in slot for slot in running)
        ]
        ready.sort(
            key=lambda i: (
                jobs[i]["arrival"] - (0 if variant == "no_debt" else jobs[i]["debt"])
            )
        )
        for _ in range(2):
            slots = [g for g, occupants in enumerate(running) if len(occupants) < 2]
            candidates = [i for i in ready if all(i not in slot for slot in running)]
            if not candidates or not slots:
                break
            cost = np.full((len(candidates), len(slots)), 1e12)
            for row, i in enumerate(candidates):
                job = jobs[i]
                for column, g in enumerate(slots):
                    occupants = running[g]
                    memory = job["memory_gb"] + sum(
                        jobs[j]["memory_gb"] for j in occupants
                    )
                    if memory > config["memory_gb"] * 0.92:
                        continue
                    if (variant == "fcfs" or job["waited_events"] >= 5) and occupants:
                        continue
                    first = second = 1.0
                    if occupants:
                        other = jobs[occupants[0]]
                        headroom = [max(0.0, 1 - memory / config["memory_gb"])] + [
                            1.0
                        ] * 8
                        if variant == "static":
                            first = measured_slowdown(job, other, observed)
                            second = measured_slowdown(other, job, observed)
                        elif variant not in ["memory", "utilization", "random"]:
                            a = (
                                job["whole_fingerprint"]
                                if variant == "no_phase"
                                else job["fingerprint"]
                            )
                            b = (
                                other["whole_fingerprint"]
                                if variant == "no_phase"
                                else other["fingerprint"]
                            )
                            first = predictor.predict(a, b, headroom)
                            second = predictor.predict(b, a, headroom)
                        if (
                            variant not in ["memory", "utilization", "random"]
                            and max(first, second) > config["slowdown_cap"]
                        ):
                            continue
                    completion = job["remaining"] / job["solo_rate"] * first
                    debt = job["debt"] + sum(jobs[j]["debt"] for j in occupants)
                    if variant == "memory":
                        score = -memory
                    elif variant == "utilization":
                        pressure = job["fingerprint"][0] or 0
                        score = abs(
                            1
                            - pressure
                            - sum(jobs[j]["fingerprint"][0] or 0 for j in occupants)
                        )
                    elif variant == "random":
                        score = rng.random()
                    elif variant == "fcfs":
                        score = job["arrival"]
                    else:
                        score = (
                            0.55 * completion / config["normalization_s"]
                            + 0.30 * max(first, second)
                            + (
                                0
                                if variant == "no_debt"
                                else 0.15 * debt / config["normalization_s"]
                            )
                        )
                    cost[row, column] = score
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                if cost[row, column] < 1e11:
                    i, g = candidates[row], slots[column]
                    running[g].append(i)
                    if jobs[i]["started"] is None:
                        jobs[i]["started"] = time
                    jobs[i]["waited_events"] = 0
        for i in ready:
            if all(i not in slot for slot in running):
                jobs[i]["debt"] += config["tick_s"]
                jobs[i]["waited_events"] += 1
        utilization += sum(bool(slot) for slot in running) / len(running)
        for g, occupants in enumerate(running):
            slowdowns = [1.0] * len(occupants)
            if len(occupants) == 2:
                first, second = occupants
                a, b = jobs[first], jobs[second]
                slowdowns = [
                    measured_slowdown(a, b, observed),
                    measured_slowdown(b, a, observed),
                ]
                max_slowdown = max(max_slowdown, *slowdowns)
                violations += int(max(slowdowns) > config["slowdown_cap"])
                if variant not in [
                    "fcfs",
                    "memory",
                    "utilization",
                    "random",
                    "static",
                    "no_correction",
                ]:
                    headroom = [
                        max(
                            0.0,
                            1 - (a["memory_gb"] + b["memory_gb"]) / config["memory_gb"],
                        )
                    ] + [1.0] * 8
                    af = (
                        a["whole_fingerprint"]
                        if variant == "no_phase"
                        else a["fingerprint"]
                    )
                    bf = (
                        b["whole_fingerprint"]
                        if variant == "no_phase"
                        else b["fingerprint"]
                    )
                    predicted = [
                        predictor.predict(af, bf, headroom),
                        predictor.predict(bf, af, headroom),
                    ]
                    key = (first, a["phase_index"], second, b["phase_index"])
                    value, consecutive = residuals.get(key, (0.0, 0))
                    error = max(
                        abs(actual - estimate) / max(estimate, 1e-8)
                        for actual, estimate in zip(slowdowns, predicted)
                    )
                    value = 0.3 * error + 0.7 * value
                    consecutive = consecutive + 1 if value > 0.12 else 0
                    residuals[key] = (value, consecutive)
                    if (
                        consecutive >= 3
                        and max(p * (1 + value) for p in predicted)
                        > config["slowdown_cap"]
                    ):
                        running[g] = [first]
                        occupants = running[g]
                        slowdowns = [1.0]
                        alarms += 1
                        residuals.pop(key)
            finished = []
            for i, slowdown in zip(occupants, slowdowns):
                job = jobs[i]
                completed = min(
                    job["phase_remaining"],
                    config["tick_s"] * job["solo_rate"] / slowdown,
                )
                job["remaining"] -= completed
                job["phase_remaining"] -= completed
                job["debt"] += config["tick_s"] * max(0.0, slowdown - 1)
                if job["remaining"] <= 1e-9:
                    job["finished"] = time + config["tick_s"]
                    finished.append(i)
                elif job["phase_remaining"] <= 1e-9:
                    job["phase_index"] += 1
                    activate_phase(job)
            running[g] = [i for i in occupants if i not in finished]
        time += config["tick_s"]
        ticks += 1
    completion = [job["finished"] - job["arrival"] for job in jobs]
    shares = np.asarray(
        [
            job["work"] / job["solo_rate"] / (job["finished"] - job["arrival"])
            for job in jobs
        ]
    )
    summary = {
        "variant": variant,
        "measurement": "measured_pair_trace_simulation",
        "mean_jct_s": float(np.mean(completion)),
        "p95_jct_s": float(np.quantile(completion, 0.95)),
        "makespan_s": time,
        "throughput": len(jobs) / time,
        "gpu_utilization": utilization / ticks,
        "worst_slowdown": max_slowdown,
        "jain_fairness": float(
            shares.sum() ** 2 / (len(shares) * np.square(shares).sum())
        ),
        "alarms": alarms,
        "cap_violation_ticks": violations,
    }
    return summary, [
        {
            "job": job["id"],
            "arrival": job["arrival"],
            "start": job["started"],
            "finish": job["finished"],
            "jct_s": job["finished"] - job["arrival"],
        }
        for job in jobs
    ]
