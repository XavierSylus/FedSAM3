# ICASSP Experiment Protocol

## Client Model

The client model is `SAM3MedicalIntegrated`:

- frozen SAM3 image backbone;
- trainable vision adapters;
- image and text projection heads;
- text prompt encoder;
- mask decoder, output mapping, and medical segmentation head.

The optimizer is modality-aware:

| Client | Trainable and upload scope |
|---|---|
| `image_only` | vision adapters, image projection, decoder and segmentation heads; unrestricted experiments also upload unchanged text parameters |
| `multimodal` | all trainable parameters |
| `text_only` | text projection parameters |

## Upload, Aggregation, and Dispatch

For round `t`:

1. The server snapshots the trainable round-global state `w_t`.
2. Every selected client initializes from the same `w_t`.
3. Every client creates a fresh optimizer and performs local training.
4. The client uploads its modality-specific trainable state.
5. The server forms the union of uploaded parameter keys.
6. Unrestricted aggregation uses every client that uploaded a key.
7. Restricted aggregation additionally filters uploaders using the parameter-group routing rules.
8. FedAvg is applied independently for each parameter key.
9. Missing keys retain their value from `w_t`.
10. The aggregated state becomes `w_(t+1)` and is dispatched in the next round.

FedProx changes only the local objective:

`L_local = L_task + mu / 2 * sum(||w - w_t||_2^2)`

The proximal term is evaluated only on trainable parameters within the client's upload scope.

## Strict 2x2 Comparison

All four cells use the same clients, data, seed, initialization, optimizer lifecycle, learning rates, local epochs, task losses, and FedAvg implementation.

| Cell | Restricted routing | FedProx | Config |
|---|---:|---:|---|
| U-FedAvg | No | No | `configs/icassp_2x2_u_fedavg.yaml` |
| U-FedProx | No | Yes | `configs/icassp_2x2_u_fedprox.yaml` |
| R-FedAvg | Yes | No | `configs/icassp_2x2_r_fedavg.yaml` |
| R-FedProx | Yes | Yes | `configs/icassp_2x2_r_fedprox.yaml` |

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

`configs/icassp_ratio_2of3_r_fedprox.yaml`

The realized ratio is derived from the enabled client list and stored in the protocol metadata.

## Parameter-Group Diagnostics

The routing and diagnostics share the same parameter classifier:

- `VISION_ADAPTER`
- `TEXT_ADAPTER`
- `IMAGE_PARAMS`
- `TEXT_PARAMS`
- `COMPAT_FALLBACK`

For client `i`, group `g`, and round `t`:

`Delta_(i,g,t) = w_(i,g,t) - w_(g,t)`

Recorded client drift:

- update L2 norm;
- reference L2 norm;
- relative drift;
- number of analyzed parameters.

For each client pair, conflict is computed on their shared keys within the group:

- cosine similarity;
- conflict angle in degrees;
- negative-cosine indicator;
- group-level negative-cosine ratio.

Global parameter drift is computed between the aggregated state and the pre-round global state. Results are saved under `parameter_group_diagnostics` in `training_history.json`.

## Reproducibility

Each run records:

- random seed;
- full configuration and protocol hashes;
- enabled clients and missing-modality ratio;
- Python, PyTorch, CUDA, device, and hostname;
- Git commit;
- per-round losses, resource use, conflict, and drift diagnostics.
