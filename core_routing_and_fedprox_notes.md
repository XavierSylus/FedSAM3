# FedSAM3-Hetero 核心实现留底说明

日期：2026-04-30  
代码提交：`2967082b986d533034c3859a248a2ab702fa087d`

本文件用于论文投稿前的项目留底，说明当前代码中 restricted routing 与 FedProx baseline 的实际实现位置。压缩包中不应只包含配置、日志和绘图脚本，还应包含核心聚合实现、客户端上传边界、训练调度入口、配置和测试文件。

## 1. 建议提交给导师的文件清单

建议压缩包命名为：

```text
FedSAM3_core_implementation_evidence_20260430.zip
```

建议目录结构：

```text
FedSAM3_core_implementation_evidence_20260430/
  core_routing_and_fedprox_notes.md
  git_commit.txt
  src/
    server.py
    client.py
    federated_trainer.py
    config_manager.py
  configs/
    exp_group_c.yaml
    exp_baseline_d_fedprox.yaml
  tests/
    test_aggregation_routing.py
    test_fedprox_config.py
  README.md
  tech_project.md
```

其中 `git_commit.txt` 建议写入：

```text
Commit:
2967082b986d533034c3859a248a2ab702fa087d
```

必交文件及原因：

| 文件 | 作用 |
|---|---|
| `src/server.py` | restricted routing 的参数分组、参与客户端路由、逐参数聚合主实现 |
| `src/client.py` | 各类客户端实际上传哪些参数，以及 FedProx loss 的实际计算和加入位置 |
| `src/federated_trainer.py` | Group C 是否启用 routing、client modality 如何传给 server、FedProx 全局参考参数如何传入 client |
| `src/config_manager.py` | YAML 中 `federated.use_decoupled_agg` 与 `baseline.method/mu` 如何解析到训练配置 |
| `configs/exp_group_c.yaml` | restricted routing 实验配置，包含 `use_decoupled_agg: true` |
| `configs/exp_baseline_d_fedprox.yaml` | FedProx baseline 配置，包含 `baseline.method: fedprox` 和 `mu: 0.01` |
| `tests/test_aggregation_routing.py` | routing 规则测试，验证 IMAGE/TEXT/ADAPTER 分组参与客户端 |
| `tests/test_fedprox_config.py` | FedProx 配置解析测试 |
| `README.md`、`tech_project.md` | 项目整体说明和技术说明，可作为辅助材料 |

## 2. Restricted Routing 核心实现

restricted routing 的核心实现位于：

```text
src/server.py
```

核心类：

```python
class CreamAggregator:
```

### 2.1 参数分组常量

代码出处：`src/server.py`，约第 23-47 行。

当前代码中使用五级白名单路由。常量如下：

```python
_TEXT_ADAPTER_KEYWORDS = ('text_adapter',)
_VISION_ADAPTER_KEYWORDS = ('adapters.', 'wrapped_blocks.', 'lora')
_TEXT_KEYWORDS = ('text_encoder', 'text_proj')
_IMAGE_KEYWORDS = (
    'image_encoder', 'backbone', 'neck', 'sam3_model',
    'mask_decoder', 'segmentation_head', 'prompt_encoder',
    'image_proj', '_output_conv', 'medical_seg_head',
    'text_prompt_encoder',
)
```

分组含义：

| 参数组 | 命中关键词 | 允许参与聚合的 client 模态 |
|---|---|---|
| `TEXT_ADAPTER` | `text_adapter` | `text_only + multimodal` |
| `VISION_ADAPTER` | `adapters.`, `wrapped_blocks.`, `lora` | `image_only + multimodal` |
| `TEXT_PARAMS` | `text_encoder`, `text_proj` | `text_only + multimodal` |
| `IMAGE_PARAMS` | `image_encoder`, `backbone`, `neck`, `sam3_model`, `mask_decoder`, `segmentation_head`, `prompt_encoder`, `image_proj`, `_output_conv`, `medical_seg_head`, `text_prompt_encoder` | `image_only + multimodal` |
| `COMPAT_FALLBACK` | 未命中上述白名单 | 按实际上传者决定，不额外扩大参与范围 |

需要特别说明的是，`text_adapter` 的优先级最高。即使名称中包含 `adapter`，也优先进入 `TEXT_ADAPTER`，不会被误分到 `VISION_ADAPTER`。

### 2.2 核心路由函数

代码出处：`src/server.py`，约第 203-261 行。

函数名：

```python
def _get_participating_clients_dynamic(
    self,
    param_name: str,
    client_weights_list: List[Optional[Dict[str, torch.Tensor]]],
    client_modalities: Optional[List[str]] = None
) -> List[int]:
```

核心逻辑：

```python
uploaded_indices = [
    i for i, w in enumerate(client_weights_list)
    if w is not None and param_name in w
]

if client_modalities is None or len(client_modalities) != len(client_weights_list):
    return uploaded_indices

is_text_adapter   = any(kw in param_name for kw in self._TEXT_ADAPTER_KEYWORDS)
is_vision_adapter = any(kw in param_name for kw in self._VISION_ADAPTER_KEYWORDS)
is_text           = any(kw in param_name for kw in self._TEXT_KEYWORDS)
is_image          = any(kw in param_name for kw in self._IMAGE_KEYWORDS)

if is_text_adapter:
    allowed_modalities = {'text_only', 'multimodal'}
    route_label = 'TEXT_ADAPTER'
elif is_vision_adapter:
    allowed_modalities = {'image_only', 'multimodal'}
    route_label = 'VISION_ADAPTER'
elif is_text:
    allowed_modalities = {'text_only', 'multimodal'}
    route_label = 'TEXT_PARAMS'
elif is_image:
    allowed_modalities = {'image_only', 'multimodal'}
    route_label = 'IMAGE_PARAMS'
else:
    allowed_modalities = {client_modalities[i] for i in uploaded_indices}
    route_label = 'COMPAT_FALLBACK'

filtered_indices = [
    i for i in uploaded_indices if client_modalities[i] in allowed_modalities
]
```

该函数的关键点是：

1. 先找“实际上传了当前参数”的客户端。
2. 再按参数组对应的模态白名单过滤。
3. 如果参数未命中白名单，则进入 `COMPAT_FALLBACK`，仅在实际上传者之间聚合。
4. 若所有上传者都被白名单过滤掉，则返回空列表，调用方跳过该参数并保留全局当前值。

### 2.3 核心聚合函数

代码出处：`src/server.py`，约第 356-496 行。

函数名：

```python
def aggregate_weights(
    self,
    client_weights: List[Optional[Dict[str, torch.Tensor]]],
    client_public_reps: List[torch.Tensor],
    global_features_for_contrastive: Optional[torch.Tensor] = None,
    client_modalities: Optional[List[str]] = None
) -> Dict[str, torch.Tensor]:
```

核心片段：

```python
all_param_names = set()
for w in client_weights:
    all_param_names.update(w.keys())

for param_name in all_param_names:
    participating_indices = self._get_participating_clients_dynamic(
        param_name, client_weights, client_modalities
    )
    if not participating_indices:
        continue
```

FedAvg/加权聚合路径中，代码会只使用 `participating_indices` 中的客户端参数：

```python
param_list = []
weight_list = []
for idx in participating_indices:
    if idx < len(client_weights) and param_name in client_weights[idx]:
        p = client_weights[idx][param_name].to(self.device)
        param_list.append(p)
        weight_list.append(agg_weights[idx])

w_tensor = torch.tensor(weight_list, device=self.device)
w_sum = w_tensor.sum()
w_tensor = w_tensor / w_sum if w_sum > 1e-8 else torch.full_like(w_tensor, 1.0 / len(param_list))

stacked = torch.stack(param_list, dim=0)
aggregated_state[param_name] = torch.sum(
    stacked * w_tensor.view(-1, *([1] * (stacked.dim() - 1))), dim=0
)
```

因此，restricted routing 并不是只在配置中声明，而是在 server 聚合时对每一个参数逐项执行。

## 3. 当前 Group C 中哪些 client 能参与聚合

Group C 配置出处：

```text
configs/exp_group_c.yaml
```

关键配置：

```yaml
federated:
  clients:
  - client_id: client_2
    modality: image_only
    enabled: true
  - client_id: client_3
    modality: multimodal
    enabled: true
  use_decoupled_agg: true
```

因此当前 Group C 实际启用的客户端为：

| client | 模态 |
|---|---|
| `client_2` | `image_only` |
| `client_3` | `multimodal` |

当前 Group C 下的实际聚合关系：

| 参数组 | 允许模态 | 当前实际可参与 client |
|---|---|---|
| `IMAGE_PARAMS` | `image_only + multimodal` | `client_2 + client_3` |
| `VISION_ADAPTER` | `image_only + multimodal` | `client_2 + client_3` |
| `TEXT_PARAMS` | `text_only + multimodal` | 只有 `client_3` |
| `TEXT_ADAPTER` | `text_only + multimodal` | 只有 `client_3` |
| `COMPAT_FALLBACK` | 实际上传者 | 仅实际上传该参数的客户端 |

说明：

1. 当前 Group C 没有启用 `client_1=text_only`，所以 `TEXT_PARAMS/TEXT_ADAPTER` 当前只有 `client_3=multimodal` 能参与。
2. `client_2=image_only` 被禁止参与 `TEXT_PARAMS/TEXT_ADAPTER`。
3. `client_2=image_only` 和 `client_3=multimodal` 都可以参与 `IMAGE_PARAMS/VISION_ADAPTER`。

## 4. Client 上传参数边界

客户端上传边界位于：

```text
src/client.py
```

### 4.1 text_only client

代码出处：`src/client.py`，约第 1113-1143 行。

`TextOnlyTrainer.get_return_values()` 只返回文本相关参数：

```python
text_only_state = {
    k: v for k, v in full_state.items()
    if 'text_encoder' in k or 'text_proj' in k or 'text_adapter' in k
}
```

因此 `text_only` 只上传：

```text
text_encoder
text_proj
text_adapter
```

### 4.2 image_only client

代码出处：`src/client.py`，约第 1269-1309 行。

`ImageOnlyTrainer.get_uploadable_state()` 默认剔除文本参数：

```python
text_param_keywords = ('text_encoder', 'text_proj', 'text_adapter')
image_only_state = {
    k: v for k, v in full_state.items()
    if not any(kw in k for kw in text_param_keywords)
}
```

因此 `image_only` 默认不上传：

```text
text_encoder
text_proj
text_adapter
```

### 4.3 multimodal client

代码出处：`src/client.py`，约第 1517-1540 行。

`MultimodalTrainer.get_return_values()` 返回可训练参数，并同时返回 image/text representation：

```python
return self.get_model_state(model), local_reps, text_rep, training_stats
```

因此 `multimodal` 可以同时参与图像侧和文本侧参数聚合。

## 5. Routing 开关如何进入聚合函数

训练调度入口位于：

```text
src/federated_trainer.py
```

`client_modalities` 从已经过滤后的 `self.client_configs` 中构造，确保和当前实际参与训练的 client 对齐。代码出处：`src/federated_trainer.py`，约第 990-1022 行。

关键片段：

```python
client_ids_sorted = sorted(round_client_updates.keys())

client_modality_map = {cid: cfg['modality'] for cid, cfg in self.client_configs.items()}
client_modalities = [client_modality_map.get(cid, 'image_only') for cid in client_ids_sorted]

if not self.config.use_decoupled_agg:
    client_modalities = None

aggregated_state = self.server.aggregate_weights(
    [round_client_updates[cid] for cid in client_ids_sorted],
    [round_client_reps[cid] for cid in client_ids_sorted],
    client_modalities=client_modalities
)
```

含义：

1. `use_decoupled_agg: true` 时，`client_modalities` 会传入 `server.aggregate_weights()`，restricted routing 生效。
2. `use_decoupled_agg: false` 时，`client_modalities=None`，server 退回为按实际上传者直接聚合，不执行白名单过滤。

Group C 中 `configs/exp_group_c.yaml` 设置：

```yaml
use_decoupled_agg: true
```

因此 Group C 是实际启用 restricted routing 的实验组。

## 6. FedProx Loss 实际加入位置

FedProx baseline 配置位于：

```text
configs/exp_baseline_d_fedprox.yaml
```

关键配置：

```yaml
baseline:
  method: fedprox
  mu: 0.01
```

配置解析位于：

```text
src/config_manager.py
```

约第 390-394 行：

```python
if 'baseline' in config_dict:
    baseline = config_dict['baseline']
    flattened['baseline_method'] = str(baseline.get('method', 'none')).lower()
    flattened['fedprox_mu'] = baseline.get('mu', 0.0)
```

### 6.1 每轮保存全局参考参数

代码出处：`src/federated_trainer.py`，约第 705-712 行。

```python
fedprox_mode = str(getattr(self.config, 'baseline_method', 'none')).lower() == 'fedprox'
round_global_reference_state = None
if fedprox_mode:
    round_global_reference_state = self.get_trainable_state_dict(self.global_model)
```

含义：每一轮开始时，保存当前全局模型的可训练参数，作为 FedProx 的全局参考点。

### 6.2 每个 client 训练时传入全局参考参数

代码出处：`src/federated_trainer.py`，约第 879-887 行。

```python
updated_weights, img_rep, txt_rep, stats = trainer.run(
    model=self.global_model,
    optimizer=optimizer,
    global_reps=global_reps,
    lambda_cream=self.config.lambda_cream,
    global_reference_state=round_global_reference_state
)
```

### 6.3 FedProx loss 加入 total_loss 的位置

代码出处：`src/client.py`，约第 505-576 行。

在 `BaseClientTrainer.tra()` 的主训练循环中，先调用各子类的 `compute_loss()` 得到原始任务损失：

```python
total_loss, seg_loss, cream_loss, public_rep = self.compute_loss(
    model, private_inputs, public_inputs,
    {'text': global_text_rep, 'image': global_image_rep},
    lambda_cream
)
```

随后追加 FedProx penalty：

```python
if fedprox_param_names:
    total_loss = total_loss + self._compute_fedprox_penalty(
        model=model,
        global_reference_state=global_reference_state,
        fedprox_param_names=fedprox_param_names,
    )
```

之后才进行梯度累加缩放：

```python
scaled_loss = total_loss / self.accumulation_steps
```

因此 FedProx loss 是实际加入反向传播路径的。

### 6.4 FedProx penalty 公式实现

代码出处：`src/client.py`，约第 603-629 行。

```python
def _compute_fedprox_penalty(
    self,
    model: nn.Module,
    global_reference_state: Optional[Dict[str, torch.Tensor]],
    fedprox_param_names: set,
) -> torch.Tensor:
    if (
        self.baseline_method != "fedprox"
        or self.fedprox_mu <= 0
        or not global_reference_state
        or not fedprox_param_names
    ):
        return torch.tensor(0.0, device=self.device)

    proximal_term = torch.tensor(0.0, device=self.device)
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in fedprox_param_names:
            continue
        global_param = global_reference_state.get(name)
        if global_param is None:
            continue
        proximal_term = proximal_term + torch.sum(
            (param - global_param.to(self.device)) ** 2
        )

    return 0.5 * self.fedprox_mu * proximal_term
```

对应公式：

```text
FedProx penalty = 0.5 * mu * sum(||w_local - w_global||^2)
```

当前实现的计算范围是：

```text
当前 client 实际可上传、且 requires_grad=True 的参数
```

这与 restricted routing 的客户端上传边界一致，避免在不可上传或未训练参数上额外计算近端项。

### 6.5 需要避免混淆的一点

`src/client.py` 中 `TextOnlyTrainer.compute_loss()` 附近有一段旧式 `global_reps['global_weights']` 近端约束逻辑，但当前训练调用链没有向 `global_reps` 注入 `global_weights`。当前实际生效的 FedProx 路径是 `BaseClientTrainer.tra()` 中统一追加的：

```python
total_loss = total_loss + self._compute_fedprox_penalty(...)
```

## 7. 测试文件出处

### 7.1 Restricted routing 测试

测试文件：

```text
tests/test_aggregation_routing.py
```

核心测试用例约第 332-342 行：

```python
test_cases = [
    ('image_encoder.patch_embed.weight', {'image_only', 'multimodal'}, {'text_only'}),
    ('mask_decoder.proj.weight',         {'image_only', 'multimodal'}, {'text_only'}),
    ('adapters.0.down_proj.weight',      {'image_only', 'multimodal'}, {'text_only'}),
    ('wrapped_blocks.0.adapter.w',       {'image_only', 'multimodal'}, {'text_only'}),
    ('image_proj.0.weight',              {'image_only', 'multimodal'}, {'text_only'}),
    ('text_encoder.layer.0.weight',      {'text_only', 'multimodal'}, {'image_only'}),
    ('text_proj.0.weight',               {'text_only', 'multimodal'}, {'image_only'}),
    ('text_adapter.down_proj.weight',    {'text_only', 'multimodal'}, {'image_only'}),
]
```

该测试验证：

1. `IMAGE_PARAMS/VISION_ADAPTER` 不允许 `text_only` 参与。
2. `TEXT_PARAMS/TEXT_ADAPTER` 不允许 `image_only` 参与。
3. `text_adapter` 优先进入 `TEXT_ADAPTER`，不会被误归入视觉 adapter。

### 7.2 FedProx 配置测试

测试文件：

```text
tests/test_fedprox_config.py
```

核心测试约第 13-41 行：

```python
def test_fedprox_baseline_section_is_flattened():
    ...
    assert config.baseline_method == "fedprox"
    assert config.fedprox_mu == pytest.approx(0.01)
```

该测试验证 YAML 中：

```yaml
baseline:
  method: fedprox
  mu: 0.01
```

可以正确解析为训练配置中的：

```text
baseline_method = fedprox
fedprox_mu = 0.01
```

## 8. 给导师核查时的最短追踪路径

如果导师只想快速核查，可以按下面路径检查：

1. 看 `configs/exp_group_c.yaml`：
   - `use_decoupled_agg: true`
   - `client_2=image_only`
   - `client_3=multimodal`

2. 看 `src/federated_trainer.py`：
   - `client_modalities` 从实际启用的 `client_configs` 构造。
   - `use_decoupled_agg=True` 时传入 `server.aggregate_weights()`。

3. 看 `src/server.py`：
   - `_TEXT_ADAPTER_KEYWORDS`
   - `_VISION_ADAPTER_KEYWORDS`
   - `_TEXT_KEYWORDS`
   - `_IMAGE_KEYWORDS`
   - `_get_participating_clients_dynamic()`
   - `aggregate_weights()`

4. 看 `src/client.py`：
   - `TextOnlyTrainer.get_return_values()`
   - `ImageOnlyTrainer.get_uploadable_state()`
   - `MultimodalTrainer.get_return_values()`
   - `_compute_fedprox_penalty()`
   - `total_loss = total_loss + self._compute_fedprox_penalty(...)`

5. 看 `configs/exp_baseline_d_fedprox.yaml`：
   - `baseline.method: fedprox`
   - `baseline.mu: 0.01`

6. 看测试：
   - `tests/test_aggregation_routing.py`
   - `tests/test_fedprox_config.py`

## 9. 结论

当前项目中 restricted routing 不是论文中的概念性描述，而是在 `src/server.py` 的逐参数聚合过程中实际执行的白名单路由。其核心逻辑是：先确认哪些客户端实际上传了某个参数，再根据参数所属组限制可参与的客户端模态。

当前 Group C 中，`client_2=image_only` 与 `client_3=multimodal` 被启用，且 `use_decoupled_agg: true`。因此：

```text
IMAGE_PARAMS / VISION_ADAPTER:
  client_2 + client_3 可参与

TEXT_PARAMS / TEXT_ADAPTER:
  当前只有 client_3 可参与

COMPAT_FALLBACK:
  只在实际上传该参数的客户端之间聚合
```

FedProx baseline 的实际实现路径为：`configs/exp_baseline_d_fedprox.yaml` 启用 `baseline.method=fedprox` 和 `mu=0.01`，`src/federated_trainer.py` 每轮保存全局参考参数并传入 client，`src/client.py` 在本地训练 loss 计算后、反向传播前加入：

```text
0.5 * mu * sum(||w_local - w_global||^2)
```

因此，压缩包中应保留上述源码、配置、测试和本说明文件，作为后续抽查时的核心实现证据。
