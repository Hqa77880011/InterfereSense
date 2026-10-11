from .io import configure, json_write, csv_write, plot
from pathlib import Path
import json
import pickle
import numpy as np

from .collector import measure
from .model import segment, fingerprint, boundary_metrics, Predictor
from .scheduler import simulate


def collect(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    specs = config["specs"]
    solo, pairs = {}, []
    for spec in specs:
        solo[spec["id"]] = measure([spec], config)
        json_write(output / "raw.json", {"solo": solo, "pairs": pairs})
    for i, first in enumerate(specs):
        for second in specs[i:]:
            pairs.append(measure([first, second], config))
            json_write(output / "raw.json", {"solo": solo, "pairs": pairs})


def prepare(config):
    raw = json.loads(Path(config["raw"]).read_text())
    profiles = {}
    phase_rows = []
    boundary_rows = []
    for identifier, record in raw["solo"].items():
        values = np.asarray([row["groups"] for row in record["telemetry"]], dtype=float)
        boundaries = segment(
            values, config["min_phase_samples"], config["pelt_penalty"]
        )
        if "truth_boundaries_s" in record:
            timestamps = [row["timestamp"] for row in record["telemetry"]]
            detected = [timestamps[index] - timestamps[0] for index in boundaries[:-1]]
            boundary_rows.append(
                {
                    "job": identifier,
                    **boundary_metrics(
                        detected,
                        record["truth_boundaries_s"],
                        config["boundary_tolerance_s"],
                    ),
                }
            )
        start = 0
        phases = []
        for boundary in boundaries:
            phase_rows.append(
                {"job": identifier, "start_sample": start, "end_sample": boundary}
            )
            phase_duration = record["seconds"] * (boundary - start) / len(values)
            phases.append(
                {
                    "fingerprint": fingerprint(
                        values[start:boundary],
                        phase_duration,
                        config["duration_s"],
                        boundary / len(values),
                    ),
                    "start_sample": start,
                    "end_sample": boundary,
                }
            )
            start = boundary
        profiles[identifier] = {
            "fingerprint": fingerprint(values, record["seconds"], config["duration_s"]),
            "solo_rate": record["units"][0] / record["seconds"],
            "memory_gb": max(
                record.get("memory_gb", [0])[0],
                max([row["memory_gb"] for row in record["telemetry"]] or [0]),
            ),
            "phases": phases,
        }
    records = []
    for pair in raw["pairs"]:
        for victim_index, other_index in [(0, 1), (1, 0)]:
            victim, other = pair["jobs"][victim_index], pair["jobs"][other_index]
            shared = pair["units"][victim_index] / pair["seconds"]
            if shared <= 0:
                raise ValueError("Pair throughput must be measured and positive")
            a, b = profiles[victim], profiles[other]
            records.append(
                {
                    "victim": victim,
                    "corunner": other,
                    "victim_fingerprint": a["fingerprint"],
                    "corunner_fingerprint": b["fingerprint"],
                    "whole_victim": a["fingerprint"],
                    "whole_corunner": b["fingerprint"],
                    "headroom": [
                        max(
                            0.0,
                            1 - (a["memory_gb"] + b["memory_gb"]) / config["memory_gb"],
                        )
                    ]
                    + [1.0] * 8,
                    "slowdown": a["solo_rate"] / shared,
                }
            )
    out = Path(config["data"])
    json_write(out / "pairs.json", records)
    json_write(out / "profiles.json", profiles)
    csv_write(out / "phases.csv", phase_rows)
    if boundary_rows:
        csv_write(out / "boundary_metrics.csv", boundary_rows)
    rng = np.random.default_rng(config["seed"])
    keys = list(profiles)
    order = rng.permutation(len(keys))
    if len(keys) < 3:
        raise ValueError(
            "At least three job configurations are needed for train/validation/test splits"
        )
    train_count = min(len(keys) - 2, max(1, int(len(keys) * 0.6)))
    validation_count = max(1, int(len(keys) * 0.2))
    split = {
        "train": [keys[i] for i in order[:train_count]],
        "validation": [
            keys[i] for i in order[train_count : train_count + validation_count]
        ],
        "test": [keys[i] for i in order[train_count + validation_count :]],
    }
    json_write(out / "split.json", split)
    jobs = []
    for i in range(config["job_count"]):
        key = keys[int(rng.integers(len(keys)))]
        profile = profiles[key]
        jobs.append(
            {
                "id": i,
                "profile": key,
                "arrival": float(i * config["arrival_spacing_s"]),
                "work": profile["solo_rate"] * config["job_duration_s"],
                "solo_rate": profile["solo_rate"],
                "memory_gb": profile["memory_gb"],
                "fingerprint": profile["fingerprint"],
                "whole_fingerprint": profile["fingerprint"],
            }
        )
    json_write(out / "jobs.json", jobs)


def train(config):
    root = Path(config["data"])
    records = json.loads((root / "pairs.json").read_text())
    split = json.loads((root / "split.json").read_text())
    partitions = {
        name: [
            row
            for row in records
            if row["victim"] in groups and row["corunner"] in groups
        ]
        for name, groups in split.items()
    }
    if not all(partitions.values()):
        raise ValueError(
            "Each split needs measured within-group pairs; include self-pairs when collecting"
        )
    predictor = Predictor(config, config["predictor_variant"])
    predictor.fit(partitions["train"])
    summaries, predictions = [], []
    for name in ["validation", "test"]:
        summary, values = predictor.metrics(partitions[name])
        summaries.append(
            {"variant": config["predictor_variant"], "split": name, **summary}
        )
        predictions.extend(
            [
                dict(row, prediction=float(value), split=name)
                for row, value in zip(partitions[name], values)
            ]
        )
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "predictor.pkl").open("wb") as stream:
        pickle.dump(predictor, stream)
    csv_write(output / "prediction_metrics.csv", summaries)
    csv_write(
        output / "prediction_pairs.csv",
        [
            {
                key: row[key]
                for key in ["victim", "corunner", "slowdown", "prediction", "split"]
            }
            for row in predictions
        ],
    )
    json_write(output / "config.json", config)
    print(json.dumps(summaries), flush=True)
    return str(output / "predictor.pkl")


def schedule(config):
    root = Path(config["data"])
    records = json.loads((root / "pairs.json").read_text())
    observed = {}
    for row in records:
        key = (
            (row["victim"], row["victim_phase"], row["corunner"], row["corunner_phase"])
            if "victim_phase" in row and "corunner_phase" in row
            else (row["victim"], row["corunner"])
        )
        observed.setdefault(key, []).append(row["slowdown"])
    observed = {key: float(np.mean(values)) for key, values in observed.items()}
    with Path(config["checkpoint"]).open("rb") as stream:
        predictor = pickle.load(stream)
    jobs = json.loads((root / "jobs.json").read_text())
    summary, rows = simulate(jobs, observed, predictor, config, config["variant"])
    csv_write(Path(config["output"]) / "jobs.csv", rows)
    json_write(Path(config["output"]) / "scheduler_metrics.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def compare(config):
    rows = []
    for variant in config["variants"]:
        current = dict(
            config, variant=variant, output=str(Path(config["output"]) / variant)
        )
        current["predictor_variant"] = (
            variant
            if variant in ["no_phase", "symmetric", "gpu_only"]
            else "interferesense"
        )
        current["checkpoint"] = train(current)
        rows.append(schedule(current))
    csv_write(Path(config["output"]) / "comparison.csv", rows)


def main():
    command, config = configure(
        ["collect", "prepare", "train", "schedule", "compare", "plot"]
    )
    if command == "collect":
        collect(config)
    elif command == "prepare":
        prepare(config)
    elif command == "train":
        train(config)
    elif command == "schedule":
        schedule(config)
    elif command == "compare":
        compare(config)
    else:
        root = Path(config["output"])
        plot(
            root / "comparison.csv",
            root / "jct_fairness.pdf",
            "jain_fairness",
            "mean_jct_s",
        )
