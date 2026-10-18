# InterfereSense

Implementation of **InterfereSense: Phase-Aware Contention Fingerprinting for Co-Scheduling Heterogeneous AI Jobs**. External resource traces are segmented into stable phases, summarized as 42-component fingerprints and paired in victim/co-runner order to predict directional slowdown.

## Installation

Use Python 3.10 or newer. Install matching PyTorch and torchvision builds for your device, then install the package:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

On Windows, activate with `.venv\Scripts\Activate.ps1`. For NVIDIA GPUs, choose a CUDA build from the [PyTorch installer](https://pytorch.org/get-started/locally/) in place of the CPU command. The supported dependency range is PyTorch 2.4–2.7 with its matching torchvision release.

## Collect measurements

The local collector runs actual matrix multiplication, memory and MLP-training kernels at two configured sizes. Each configuration is measured alone and in unordered pairs, including self-pairs; each pair supplies both ordered throughput ratios.

```bash
python -m interferesense collect --config configs/local.json --output outputs/measure
python -m interferesense prepare --config configs/local.json
python -m interferesense train --config configs/local.json --output outputs/predictor
python -m interferesense schedule --config configs/local.json --checkpoint outputs/predictor/predictor.pkl --output outputs/schedule
python -m interferesense compare --config configs/local.json --output outputs/comparison
python -m interferesense plot --config configs/local.json --output outputs/comparison
```

`raw` in the configuration points to `outputs/measure/raw.json`. Preparation segments solo traces, derives fingerprints and computes slowdown as measured solo useful work per second divided by shared useful work per second. Counts come from completed kernel iterations. No predictor output is used as the slowdown label.

`collect` can take several minutes because it measures all pairs in `specs`. The duration, warm-up and sampling interval are configurable. GPU runs use the selected device; CPU-only runs are available with `--device cpu`. Telemetry uses NVML and psutil. SM and DRAM pressure use NVML utilization proxies; Tensor Core and L2 counters are unavailable in this collector and have explicit missing masks. CPU, disk and network counters are host-level. PCIe normalization uses the configured bandwidth ceiling. More detailed DCGM traces can use the prepared schema described below.

## Training and trace scheduling

Job configurations are partitioned into 60% training, 20% validation and the remaining test configurations. Only pairs with both configurations inside a partition are used for its predictor evaluation, preventing either job configuration from crossing splits. Self-pairs support small held-out groups. Cross-partition pairs remain available as measured environment inputs for scheduling simulations.

The regressor learns log slowdown with 300 depth-six gradient-boosted trees, learning rate 0.05 and a training-only median imputer. The implementation uses scikit-learn gradient boosting rather than XGBoost. Pair features concatenate victim, co-runner, their elementwise product, absolute difference and device headroom; the symmetric ablation removes ordering.

`schedule` is a discrete-time cluster trace simulation driven by the measured ordered slowdown matrix. Model predictions decide placement, while measured throughput ratios determine progress and residual alarms. Its output is tagged `measured_pair_trace_simulation`. `gpus` and `memory_gb` describe the simulated cluster. It does not launch Kubernetes jobs or claim measured cluster throughput.

| Variant | Policy |
| --- | --- |
| `fcfs` | One job per device in arrival order |
| `memory` | Memory-feasible packing |
| `utilization` | Average GPU-pressure packing |
| `random` | Random feasible packing |
| `static` | Measured static interference matrix |
| `interferesense` | Directional prediction, constrained matching, debt and correction |
| `no_phase`, `symmetric`, `gpu_only` | Predictor feature ablations |
| `no_correction`, `no_debt` | Scheduler-control ablations |

## Input schemas and parameters

Commands read the JSON configuration. `--data`, `--output`, `--device`, `--variant`, `--seed` and `--checkpoint` override run settings. Other fields use `--set KEY=JSON`, for example `--set duration_s=3.0` or `--set 'predictor_variant="no_phase"'`. String values need JSON quotes. Use separate output directories for policies.

`pairs.json` stores measured `slowdown`, two 42-value fingerprints, whole-run fingerprints, job configuration identifiers and nine headroom values. `profiles.json` stores solo rate, memory size and detected segment fingerprints. `split.json` fixes the configuration partition. `jobs.json` specifies arrivals, useful-work budgets, solo rates and fingerprints. For multi-phase inputs, pair records also contain `victim_phase` and `corunner_phase` indices. Each job has a `phases` array with positive `work` budgets summing to its total `work`; each phase supplies its `fingerprint`, `solo_rate` and `memory_gb`. The simulator switches these values when that phase completes. Every ordered phase pair used by a schedule needs a measured slowdown label. Prepared JSON files can be supplied directly for externally collected workloads.

Fingerprints contain mean/std/median/P95 for eight resource groups, normalized phase duration, elapsed segment share and eight availability bits. Missing statistics are `null` and are imputed using training data only. PELT uses squared-error cost, a minimum of 20 samples and penalty `2*d*log(n)`, where d is the number of observed groups. At a 50 ms interval the minimum duration is one second.

Memory feasibility reserves 8% of device capacity. The default predicted slowdown cap is 1.5. Minimum-cost bipartite matching uses completion, worst-direction slowdown and accumulated debt, with weights 0.55/0.30/0.15. A waiting job gets a standalone preference after five scheduling ticks. The tick-driven implementation uses `tick_s` as its scheduling interval. Residual EWMA has weight 0.3; three successive values above 0.12 trigger pair reassessment. The simulation models separation by pausing the second job and returning it to the queue.

## Results and scope

`phases.csv` records detected boundaries. When solo raw records include `truth_boundaries_s`, `boundary_metrics.csv` reports one-to-one matched precision, recall and F1 within `boundary_tolerance_s` (0.1 seconds by default). The stable local kernels have no true internal boundaries; this check does not measure detection of multi-phase workloads. `prediction_metrics.csv` reports held-out MAPE, RMSE and R². `prediction_pairs.csv` preserves measured and predicted ordered ratios. `jobs.csv` records simulated arrival, start and finish times. `scheduler_metrics.json` and `comparison.csv` contain mean/P95 JCT, makespan, throughput, utilization, worst slowdown, cap-violation ticks and Jain fairness. The plot compares mean JCT with fairness.

The supplied local kernels each expose a stable workload phase, so their phase and whole-run fingerprints coincide. Multi-phase prepared inputs supply phase work budgets, different phase and whole-run descriptors, and ordered phase-pair measurements; phase feature ablations are meaningful on those inputs. The local collector is a controlled measurement source rather than the paper's eight model/system jobs. Scheduling scores describe replay of those measurements, with discretization controlled by `tick_s`. Trusted predictor files use Python pickle and should be loaded only from your own runs.

Data, weights and generated outputs are ignored by Git. Run directories store the effective `config.json`, CSV measurements and JSON metrics. Plots read the generated measurement files.

## Registered compact reproduction setting

Six actual CPU kernel configurations collected solo and in all pairs; 100 depth-three trees; 48-job measured-pair scheduling replay. Stable kernels are single-phase, so phase removal is omitted from this compact comparison; multi-phase inputs remain supported by the formal schema.

Run from the repository root after installation. The following commands use the registered setting; they are provided for execution and have not been run as part of the source review.

```bash
python -m interferesense collect --config configs/reproduce.json --output outputs/reproduce_measure
python -m interferesense prepare --config configs/reproduce.json
python -m interferesense compare --config configs/reproduce.json --output outputs/reproduce_comparison
```

This setting needs no external dataset or pretrained weights. Generated inputs and declared fault/shift laws define the finite task; reported quantities retain that provenance.
