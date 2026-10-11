import numpy as np
import ruptures as rpt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

GROUPS = ["sm", "tensor", "dram", "l2", "pcie", "cpu", "storage", "network"]


def segment(values, minimum=20, penalty=2.0):
    values = np.asarray(values, dtype=float)
    if len(values) < minimum:
        return [len(values)]
    available = np.isfinite(values).any(0)
    filled = np.where(np.isfinite(values), values, 0.0)
    if not available.any():
        return [len(values)]
    signal = filled[:, available]
    return (
        rpt.Pelt(model="l2", min_size=minimum, jump=1)
        .fit(signal)
        .predict(pen=penalty * signal.shape[1] * np.log(len(values)))
    )


def fingerprint(values, duration, horizon, elapsed_share=1.0):
    values = np.asarray(values, dtype=float)
    statistics, masks = [], []
    for column in values.T:
        selected = column[np.isfinite(column)]
        masks.append(float(len(selected) > 0))
        statistics.extend(
            [
                float(selected.mean()),
                float(selected.std()),
                float(np.median(selected)),
                float(np.quantile(selected, 0.95)),
            ]
            if len(selected)
            else [None] * 4
        )
    return statistics + [duration / max(horizon, 1e-8), elapsed_share] + masks


def boundary_metrics(detected, truth, tolerance):
    remaining = list(truth)
    matched = 0
    for boundary in sorted(detected):
        if remaining:
            index = min(
                range(len(remaining)), key=lambda i: abs(remaining[i] - boundary)
            )
            if abs(remaining[index] - boundary) <= tolerance:
                matched += 1
                remaining.pop(index)
    precision = matched / len(detected) if detected else float(not truth)
    recall = matched / len(truth) if truth else float(not detected)
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def pair_features(victim, corunner, headroom, variant):
    first, second = np.asarray(victim, dtype=float), np.asarray(corunner, dtype=float)
    if variant == "gpu_only":
        first[20:32] = np.nan
        second[20:32] = np.nan
        first[39:42] = 0
        second[39:42] = 0
    if variant == "symmetric":
        return np.concatenate(
            [(first + second) / 2, first * second, np.abs(first - second), headroom]
        )
    return np.concatenate(
        [first, second, first * second, np.abs(first - second), headroom]
    )


class Predictor:
    def __init__(self, config, variant="interferesense"):
        self.variant = variant
        self.model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            GradientBoostingRegressor(
                n_estimators=config["trees"],
                max_depth=config["depth"],
                learning_rate=config["learning_rate"],
                random_state=config["seed"],
            ),
        )

    def rows(self, records):
        return np.stack(
            [
                pair_features(
                    row["whole_victim"]
                    if self.variant == "no_phase"
                    else row["victim_fingerprint"],
                    row["whole_corunner"]
                    if self.variant == "no_phase"
                    else row["corunner_fingerprint"],
                    row["headroom"],
                    self.variant,
                )
                for row in records
            ]
        )

    def fit(self, records):
        if len(records) < 2:
            raise ValueError(
                "At least two measured pair records are needed for fitting"
            )
        self.model.fit(self.rows(records), np.log([row["slowdown"] for row in records]))

    def predict(self, victim, corunner, headroom):
        value = self.model.predict(
            pair_features(victim, corunner, headroom, self.variant)[None]
        )[0]
        return float(np.exp(np.clip(value, -10, 10)))

    def metrics(self, records):
        true = np.asarray([row["slowdown"] for row in records])
        predicted = np.exp(self.model.predict(self.rows(records)))
        return {
            "mape": float(np.mean(np.abs(predicted - true) / true) * 100),
            "rmse": float(np.sqrt(mean_squared_error(true, predicted))),
            "r2": float(r2_score(true, predicted)) if len(true) > 1 else None,
        }, predicted
