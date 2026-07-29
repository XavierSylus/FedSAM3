# FedSAM3-Cream

FedSAM3-Cream is a BraTS heterogeneous federated medical segmentation research
project built around the original SAM3 encoder. The current experiment protocol
is a strict `parameterwise routing x FedProx` comparison with three client
modalities:

| Client | Modality | Local task objective |
|---|---|---|
| `client_1` | `text_only` | prototype-logistic text loss |
| `client_2` | `image_only` | segmentation loss |
| `client_3` | `multimodal` | segmentation plus CREAM loss |

The authoritative training entry point is `main.py`. The authoritative protocol
and experiment manifest are:

- `docs/FEDSAM3_EXPERIMENT_PROTOCOL.md`
- `docs/TRAINING_READINESS_WORKFLOW_SPEC.md`
- `configs/fedsam3_experiment_manifest.json`

Files under `core_projects/` are read-only dependencies. Do not modify the
original SAM3 encoder or its geometric encoding behavior.

## Experiment matrix

The four main cells use all three clients, seed `3407`, identical data, model,
task losses, optimizer lifecycle, learning rates, local epochs, public proxies,
and FedAvg implementation.

| Cell | Routing | FedProx | Config |
|---|---|---|---|
| U-FedAvg | unrestricted | disabled | `configs/fedsam3_2x2_u_fedavg.yaml` |
| U-FedProx | unrestricted | enabled | `configs/fedsam3_2x2_u_fedprox.yaml` |
| R-FedAvg | restricted | disabled | `configs/fedsam3_2x2_r_fedavg.yaml` |
| R-FedProx | restricted | enabled | `configs/fedsam3_2x2_r_fedprox.yaml` |

The additional ratio experiment disables `client_1` and retains `client_2` and
`client_3`:

- client participation ratio relative to the declared pool: `2/3`;
- missing-modality ratio among enabled clients: `1/2`;
- config: `configs/fedsam3_ratio_2of3_r_fedprox.yaml`.

The main matrix has a `3/3` participation ratio and a `2/3` missing-modality
ratio.

## Mathematical contracts

Segmentation output channels are strictly `[WT, TC, ET]`. BraTS labels use the
unique conversion `0/1/2/4 <-> [WT, TC, ET]`, with nested-region closure when
converting predictions back to labels.

The local FedProx objective is:

```text
L_local = L_task + mu / 2 * sum(p in O_k) ||theta_k,p - theta_t,p||^2
```

`O_k` is exactly the client's named optimizer parameter set and upload set.
FedProx does not change the data, forward path, task loss, optimizer scope, or
server aggregation formula.

Unrestricted routing includes every active client's positive
`private_case_count` in every parameter denominator. A client that did not
optimize a parameter contributes a zero delta without leaving the denominator.

Restricted routing includes only clients that uploaded the parameter and whose
modality is allowed for its parameter group, then renormalizes their
`private_case_count` weights. An empty eligible set preserves the round-global
parameter and writes an audit event.

The shared parameter groups are:

- `TEXT_ADAPTER`
- `TEXT_PARAMS`
- `VISION_ADAPTER`
- `IMAGE_PARAMS`
- `FUSION_PARAMS`

Named buffers remain server-owned and are never uploaded or aggregated.

## Server storage contract

All experiment YAML files use:

```text
data_root: /autodl-fs/data/FedSAM3-Cream/datasets/federated_split
SAM3 checkpoint: /autodl-fs/data/FedSAM3-Cream/datasets/checkpoints/sam3.pt
log root: /autodl-fs/data/FedSAM3-Cream/experiments/logs
```

The runtime data loader expects the federated split beneath `data_root`, including
`train/client_1`, `train/client_2`, `train/client_3`, and the corresponding
validation client directories. Each client directory contains its declared
`private` and, for training proxy generation, `public` data.

The YAML `data_source` fields point to the external client dataset manifests.
Both the runtime split and these manifests must exist before server validation.

## Server qualification

Local training, smoke runs, and model forward/backward validation are prohibited.
Run all executable checks on the target server.

First confirm the exact candidate:

```bash
git fetch origin --prune
git checkout main
git pull --ff-only origin main
git status --short
git rev-parse HEAD
```

Run the server preflight for the S2 config:

```bash
python scripts/server_preflight.py \
  --config configs/fedsam3_s2_three_client_preflight.yaml
```

Run the S1 contract suite:

```bash
pytest -q \
  tests/test_brats_region_contract.py \
  tests/test_segmentation_contract.py \
  tests/test_metrics_contract.py \
  tests/test_text_loss_contract.py \
  tests/test_parameterwise_aggregation.py \
  tests/test_optimizer_upload_contract.py \
  tests/test_buffer_distribution_contract.py \
  tests/test_fedprox_same_loss_contract.py \
  tests/test_parameter_group_effectiveness.py \
  tests/test_update_diagnostics.py \
  tests/test_experiment_matrix_contract.py \
  tests/test_reproducibility_contract.py \
  tests/test_federated_aggregation_wiring.py \
  tests/test_fedprox_proxy_contract.py \
  tests/test_phase_b_preflight.py
```

Run the S2 three-client parameter-group preflight:

```bash
python main.py \
  --config configs/fedsam3_s2_three_client_preflight.yaml
```

Before formal training, complete and preserve evidence for S3–S6 as specified in
`docs/TRAINING_READINESS_WORKFLOW_SPEC.md`:

- single-client small-sample overfit;
- one-round three-client smoke;
- constructive U/R comparison;
- constructive FedAvg/FedProx comparison.

Do not start the formal matrix until S1–S6 pass.

## Formal training

After S1–S6 evidence has been reviewed, run the five experiments sequentially:

```bash
FEDSAM3_TRAINING_APPROVED=1 bash run_experiments.sh
```

For an unattended server process, redirect the launcher output to a persistent
server log without overriding any YAML path.

`run_experiments.sh` validates all five configs before starting the first cell,
uses each YAML's declared storage paths, never deletes existing logs, and stops
at the first failed experiment.

## Required run evidence

Each run writes its primary artifacts beneath the YAML `logging.log_dir`:

- `checkpoints/latest_checkpoint.pth`
- `checkpoints/final_model.pth`
- `checkpoints/training_history.json`
- `checkpoints/run_metadata.json`
- `parameter_group_effectiveness.jsonl`
- `parameter_group_diagnostics.jsonl`
- `parameter_group_diagnostics.csv`
- TensorBoard logs and configured segmentation masks

The evidence package must retain:

- exact Git commit, configuration hash, data manifest hash, and random seed;
- Python, PyTorch, CUDA, cuDNN, hostname, and GPU inventory;
- enabled clients, modalities, case IDs, and `private_case_count`;
- per-round upload, routing, normalized weight, and buffer audits;
- client and server parameter drift by parameter group;
- pairwise shared-update cosine, angle, and conflict rate by parameter group;
- WT/TC/ET Dice, IoU, HD95, and empty-region counts.

Do not report a completed comparison from partial runs. Label S2/S3 outputs and
unfinished formal cells as validation or preliminary evidence.

## Core implementation

| File | Responsibility |
|---|---|
| `src/config_manager.py` | YAML mapping and protocol validation |
| `src/federated_trainer.py` | federated scheduling, validation, metadata, and artifacts |
| `src/client.py` | modality task losses, optimizer-scoped upload, and FedProx |
| `src/server.py` | parameterwise U/R FedAvg, proxy generation, and audits |
| `src/parameter_groups.py` | shared parameter classifier and routing allowlists |
| `src/update_diagnostics.py` | update conflict and parameter drift diagnostics |
| `data/heterogeneous_dataset_loader.py` | heterogeneous BraTS client loaders |
