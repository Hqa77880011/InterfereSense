# InterfereSense

Implementation of **InterfereSense: Phase-Aware Contention Fingerprinting for Co-Scheduling Heterogeneous AI Jobs**.

## Installation

Use Python `>=3.10`. Run commands from the repository root. Create the environment, activate it with `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (PowerShell), then install the package.

```bash
python -m venv .venv
```

```bash
python -m pip install -e .
```

## Data preparation

The local collector measures the kernels in `specs` and saves telemetry and ordered pair slowdowns to `raw.json`. `prepare` segments these measured traces and fixes job-configuration partitions. CPU runs use psutil; CUDA telemetry additionally uses NVML. These are local kernel measurements, not supplied production traces.

```bash
python -m interferesense collect --config configs/reproduce.json --output outputs/reproduce_measure
python -m interferesense prepare --config configs/reproduce.json
```

## Training and execution

`configs/reproduce.json` fixes seed `17`. Use `--seed` for a new run and keep its data/config paired. Paths are configurable with the CLI or JSON. The predictor fits measured pair slowdowns; scheduling replays the prepared jobs.

```bash
python -m interferesense train --config configs/reproduce.json --output outputs/predictor
```

## Evaluation

```bash
python -m interferesense schedule --config configs/reproduce.json --checkpoint outputs/predictor/predictor.pkl --output outputs/schedule
```

## Comparisons and ablations

`compare` runs the configured variants; it includes model fitting where required. Keep comparison outputs separate from the main run.

```bash
python -m interferesense compare --config configs/reproduce.json --output outputs/comparison
python -m interferesense plot --config configs/reproduce.json --output outputs/comparison
```

## Outputs

Run directories contain `prediction_metrics.csv`, `scheduler_metrics.json` as produced by the corresponding steps. `config.json` records effective settings. Comparison results are in `outputs/comparison/comparison.csv`; plotting reads those computed files.

## Verification

Source checks used Python 3.13.5. Installed dependency metadata: numpy 2.1.3, torch 2.7.1+cu118, torchvision 0.22.1, scipy 1.15.3, scikit-learn 1.6.1, matplotlib 3.10.0. Python AST, configuration files, CLI parser definitions and source input/output paths were checked without invoking command handlers. Training, full evaluation and CI were not run for this review.
