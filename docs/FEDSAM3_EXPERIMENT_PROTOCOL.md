# Federated Experiment Protocol

## Client Model Structure

The client model is `SAM3MedicalIntegrated`:

- the pretrained SAM3 foundation parameters remain frozen;
- parameter-efficient adapters and task heads are trainable;
- every client instantiates the same model skeleton;
- modality controls the optimizer scope, upload scope, and restricted aggregation route.

Trainable and uploaded parameters use one shared classifier:

| Group | Modules | Eligible modalities under restricted routing |
|---|---|---|
| `VISION_ADAPTER` | vision adapters, wrapped vision blocks, LoRA | `image_only`, `multimodal` |
| `TEXT_ADAPTER` | text adapters | `text_only`, `multimodal` |
| `IMAGE_PARAMS` | image projection, prompt/mask decoder, output and medical segmentation heads | `image_only`, `multimodal` |
| `TEXT_PARAMS` | text encoder and text projection | `text_only`, `multimodal` |
| `FUSION_PARAMS` | cross-modal text mapping and fusion gate | `multimodal` |

Every trainable or uploaded parameter must belong to exactly one group. An
unclassified parameter terminates the run.

## Upload, Aggregation, and Dispatch

For round `t`:

1. The server snapshots the trainable round-global state `w_t`.
2. `client_3` and its fixed public multimodal loader generate one image proxy
   and one text proxy from `w_t`; the same detached pair is dispatched to all clients.
3. Missing, empty, non-finite, or zero-norm proxies terminate the run.
4. Every client initializes from the same `w_t` and creates a fresh optimizer.
5. Local training updates only the modality-specific optimizer scope.
6. Each client uploads only its configured state subset and its public representation.
7. Before aggregation, all client deltas are computed against the same `w_t`.
8. Unrestricted routing uses every client that uploaded a classified key.
9. Restricted routing additionally filters uploaders by the table above.
10. FedAvg is applied independently to each parameter key; keys without an eligible
    update retain their value from `w_t`.
11. The result becomes `w_(t+1)` and is dispatched at the next round.

FedProx changes only the local objective:

`L_local = L_task + mu / 2 * sum(||w - w_t||_2^2)`

The proximal term is evaluated exactly once and only on trainable parameters
within the client's upload scope. It does not change any task-loss definition.

## Strict 2x2 Comparison

All four cells use `client_2:image_only` and `client_3:multimodal`, seed `3407`,
the same data, round-global initialization, fresh optimizer lifecycle, learning
rates, local epochs, task losses, public proxies, and FedAvg implementation.

| Cell | Restricted routing | FedProx | Config |
|---|---:|---:|---|
| U-FedAvg | No | No | `configs/fedsam3_2x2_u_fedavg.yaml` |
| U-FedProx | No | Yes | `configs/fedsam3_2x2_u_fedprox.yaml` |
| R-FedAvg | Yes | No | `configs/fedsam3_2x2_r_fedavg.yaml` |
| R-FedProx | Yes | Yes | `configs/fedsam3_2x2_r_fedprox.yaml` |

Fixed local task losses:

- `image_only`: segmentation loss;
- `multimodal`: segmentation loss plus `lambda_cream * CREAM loss`;
- `lambda_cream = 0.1`;
- `mu = 0.01` only when FedProx is enabled.

## Missing-Modality Ratio

The ratio is defined at client level:

`missing-modality ratio = clients without both image and text / all clients`

The main 2x2 setting uses `image_only + multimodal`, giving `1/2`.
The additional setting uses `text_only + image_only + multimodal`, giving `2/3`:

`configs/fedsam3_ratio_2of3_r_fedprox.yaml`

The realized ratio is derived from the enabled client list and stored in the protocol metadata.

## Parameter-Group Diagnostics

The routing and diagnostics share the same parameter classifier:

- `VISION_ADAPTER`
- `TEXT_ADAPTER`
- `IMAGE_PARAMS`
- `TEXT_PARAMS`
- `FUSION_PARAMS`

For client `i`, group `g`, and round `t`:

`Delta_(i,g,t) = w_(i,g,t) - w_(g,t)`

Recorded client drift:

- update L2 norm;
- reference L2 norm;
- relative drift;
- update RMS;
- number of scalar elements and parameter tensors.

For each client pair, conflict is computed on their shared keys within the group:

- cosine similarity;
- conflict angle in degrees;
- negative-cosine indicator;
- shared scalar and parameter counts;
- group-level conflict rate, defined as the fraction of pairs with negative cosine.

Global parameter drift is computed between the aggregated state and the pre-round global state. Results are saved under `parameter_group_diagnostics` in `training_history.json`.
Each completed round is also appended immediately to
`parameter_group_diagnostics.jsonl` and `parameter_group_diagnostics.csv`.

## Reproducibility

Each run records:

- random seed;
- full configuration and protocol hashes;
- enabled clients and missing-modality ratio;
- Python, PyTorch, CUDA, device, and hostname;
- Git commit;
- per-round losses, resource use, conflict, and drift diagnostics.
