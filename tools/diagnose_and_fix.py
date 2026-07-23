"""
FedSAM3-Cream 诊断和修复工具

问题清单:
1. Dice 异常高 (0.98) - 预期纯图像在 0.75+，但不应该是 0.98
2. 梯度余弦相似度恒为 90度 - 说明跨模态梯度没有真正对齐或计算有误
3. Group B 文本污染没有让 Dice 下降
4. Group C Dice 没有高于 Group A

Author: FedSAM3-Cream Team
Date: 2026-03-28
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class DiagnosticTool:
    """诊断工具类"""

    def __init__(self):
        self.issues = []
        self.fixes = []

    def add_issue(self, category: str, description: str, severity: str):
        """添加发现的问题"""
        self.issues.append({
            'category': category,
            'description': description,
            'severity': severity  # 'critical' | 'high' | 'medium' | 'low'
        })

    def add_fix(self, category: str, description: str, code: str = None):
        """添加修复方案"""
        self.fixes.append({
            'category': category,
            'description': description,
            'code': code
        })

    def diagnose_dice_anomaly(self):
        """诊断 Dice 异常高的问题"""
        print("=" * 80)
        print("诊断 1: Dice 异常高 (0.98) - 预期 ~0.75")
        print("=" * 80)

        # 检查 1: 验证集大小
        data_root = Path("data/federated_split")
        if data_root.exists():
            for client_dir in data_root.iterdir():
                if client_dir.is_dir():
                    dataset_file = client_dir / "dataset.json"
                    if dataset_file.exists():
                        with open(dataset_file, 'r') as f:
                            data = json.load(f)

                        # 统计验证集样本数
                        val_samples = [s for s in data.get('samples', []) if s.get('split') == 'val']
                        print(f"\n{client_dir.name}: 验证集样本数 = {len(val_samples)}")

                        if len(val_samples) < 10:
                            self.add_issue(
                                "Data Quality",
                                f"{client_dir.name} 验证集样本数过少 ({len(val_samples)}), 指标可能不稳定",
                                "high"
                            )
        else:
            self.add_issue(
                "Data Missing",
                f"数据目录不存在: {data_root}",
                "critical"
            )

        # 检查 2: 指标计算逻辑
        print("\n检查指标计算逻辑...")
        self.add_fix(
            "Dice Calculation",
            "验证 Dice 计算是否使用了正确的 volume-level 累加",
            """
# 确认 src/client.py 中的 validate() 方法使用全局累加:
global_intersection += (pred_binary * t_binary).sum().item()
global_pred_sum += pred_binary.sum().item()
global_gt_sum += t_binary.sum().item()
dice = (2.0 * global_intersection) / (global_pred_sum + global_gt_sum + 1e-8)
            """
        )

        # 检查 3: 是否在训练集上验证
        self.add_fix(
            "Validation Set",
            "确认验证是在验证集而不是训练集上进行",
            """
# 在 federated_trainer.py 中检查:
val_loader = setup_validation_loader(split='val')  # 确保是 'val' 而不是 'train'
            """
        )

    def diagnose_gradient_cosine(self):
        """诊断梯度余弦相似度问题"""
        print("\n" + "=" * 80)
        print("诊断 2: 梯度余弦相似度恒为 90度")
        print("=" * 80)

        # 检查是否实现了梯度监控
        server_file = Path("src/server.py")
        if server_file.exists():
            content = server_file.read_text(encoding='utf-8')
            if 'cosine' not in content.lower() or 'gradient' not in content.lower():
                self.add_issue(
                    "Missing Feature",
                    "server.py 中未实现梯度余弦相似度监控",
                    "critical"
                )

                # 提供修复代码
                self.add_fix(
                    "Gradient Monitoring",
                    "在 server.py 中添加梯度余弦相似度计算",
                    """
def compute_gradient_cosine_similarity(self, clients_weights: List[Dict]) -> Dict[str, float]:
    \"\"\"
    计算客户端之间的梯度余弦相似度

    Args:
        clients_weights: 客户端权重更新列表

    Returns:
        余弦相似度字典
    \"\"\"
    if len(clients_weights) < 2:
        return {}

    similarities = {}

    # 提取所有客户端的权重更新（梯度方向）
    grad_dicts = []
    for i, client_data in enumerate(clients_weights):
        weights = client_data.get('weights', {})
        if weights:
            grad_dicts.append(weights)

    if len(grad_dicts) < 2:
        return {}

    # 计算两两之间的余弦相似度
    for i in range(len(grad_dicts)):
        for j in range(i + 1, len(grad_dicts)):
            # 关注 VisionAdapter 层（最容易观察跨模态冲突）
            adapter_params_i = {k: v for k, v in grad_dicts[i].items() if 'adapter' in k.lower()}
            adapter_params_j = {k: v for k, v in grad_dicts[j].items() if 'adapter' in k.lower()}

            # 计算余弦相似度
            cos_sim = self._compute_param_cosine(adapter_params_i, adapter_params_j)

            key = f"client_{i}_vs_client_{j}_adapter_cosine"
            similarities[key] = cos_sim

            # 计算夹角（度）
            angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
            similarities[f"client_{i}_vs_client_{j}_adapter_angle"] = angle

    return similarities

def _compute_param_cosine(self, params1: Dict, params2: Dict) -> float:
    \"\"\"
    计算两个参数字典的余弦相似度
    \"\"\"
    # 找到共同的参数键
    common_keys = set(params1.keys()) & set(params2.keys())

    if not common_keys:
        return 0.0

    # 展平参数为一维向量
    vec1 = []
    vec2 = []
    for key in sorted(common_keys):
        vec1.append(params1[key].flatten())
        vec2.append(params2[key].flatten())

    vec1 = torch.cat(vec1)
    vec2 = torch.cat(vec2)

    # 计算余弦相似度
    cos_sim = F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()

    return cos_sim
                    """
                )

        # 检查调用点
        self.add_fix(
            "Gradient Logging",
            "在聚合后记录梯度相似度",
            """
# 在 CreamAggregator.aggregate() 方法中添加:
def aggregate(self, clients_data, round_idx):
    # ... 聚合逻辑 ...

    # 计算梯度余弦相似度
    grad_similarities = self.compute_gradient_cosine_similarity(clients_data)

    # 记录到日志
    self.logger.info(f"Round {round_idx} 梯度余弦相似度:")
    for key, value in grad_similarities.items():
        self.logger.info(f"  {key}: {value:.4f}")

    return aggregated_weights, grad_similarities
            """
        )

    def diagnose_group_comparison(self):
        """诊断 Group A/B/C 对比实验问题"""
        print("\n" + "=" * 80)
        print("诊断 3: Group B/C 实验效果不符合预期")
        print("=" * 80)

        # 检查配置文件差异
        config_dir = Path("configs")
        group_configs = {
            'A': config_dir / "exp_group_a.yaml",
            'B': config_dir / "exp_group_b.yaml",
            'C': config_dir / "exp_group_c.yaml"
        }

        for group, config_file in group_configs.items():
            if not config_file.exists():
                self.add_issue(
                    "Missing Config",
                    f"配置文件不存在: {config_file}",
                    "high"
                )

        # 关键差异检查
        print("\n检查关键配置差异:")
        print("Group A: image_only, use_decoupled_agg=false, aggregation_method=fedavg")
        print("Group B: image_only + multimodal, use_decoupled_agg=false, aggregation_method=fedavg")
        print("Group C: text_only + image_only + multimodal, use_decoupled_agg=true, aggregation_method=contrastive_weighted")

        # 验证解耦聚合是否真正生效
        self.add_fix(
            "Decoupled Aggregation Verification",
            "在 server.py 中添加解耦聚合的详细日志",
            """
def _get_participating_clients_dynamic(self, param_name, clients_data):
    \"\"\"五级白名单路由 + 详细日志\"\"\"
    all_clients = clients_data

    # Level 1: TEXT_ADAPTER
    if 'text_adapter' in param_name:
        eligible = [c for c in all_clients if c['modality'] in ['text_only', 'multimodal']]
        self.logger.debug(f"[WHITE_LIST] {param_name} -> TEXT_ADAPTER: {[c['client_id'] for c in eligible]}")
        return eligible

    # Level 2: VISION_ADAPTER
    elif any(kw in param_name for kw in ['adapters.', 'wrapped_blocks.', 'lora']):
        eligible = [c for c in all_clients if c['modality'] in ['image_only', 'multimodal']]
        self.logger.debug(f"[WHITE_LIST] {param_name} -> VISION_ADAPTER: {[c['client_id'] for c in eligible]}")
        return eligible

    # ... 其他级别 ...

    # 如果没有合格客户端，打印 WARNING 而不是静默失败
    if not eligible:
        self.logger.warning(f"[WHITE_LIST] {param_name} -> 无合格客户端！")

    return eligible
            """
        )

        # 检查 Group B 是否真的允许文本污染
        self.add_fix(
            "Group B Pollution Verification",
            "确认 Group B 确实没有启用解耦聚合",
            """
# exp_group_b.yaml 中必须有:
federated:
  use_decoupled_agg: false  # 关键！这会导致文本参数污染视觉参数

server:
  aggregation_method: fedavg  # 不使用对比加权
            """
        )

    def check_data_leakage(self):
        """检查数据泄漏"""
        print("\n" + "=" * 80)
        print("诊断 4: 数据泄漏检查")
        print("=" * 80)

        self.add_fix(
            "Data Leakage Prevention",
            "确保训练集和验证集完全分离",
            """
# 在数据划分脚本中添加验证:
def verify_no_overlap(train_ids, val_ids, test_ids):
    \"\"\"验证训练/验证/测试集无重叠\"\"\"
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    overlap_train_val = train_set & val_set
    overlap_train_test = train_set & test_set
    overlap_val_test = val_set & test_set

    assert len(overlap_train_val) == 0, f"训练集和验证集有重叠: {overlap_train_val}"
    assert len(overlap_train_test) == 0, f"训练集和测试集有重叠: {overlap_train_test}"
    assert len(overlap_val_test) == 0, f"验证集和测试集有重叠: {overlap_val_test}"

    print("✓ 数据集无重叠，通过验证")
            """
        )

    def generate_report(self):
        """生成诊断报告"""
        print("\n" + "=" * 80)
        print("诊断报告")
        print("=" * 80)

        print(f"\n发现 {len(self.issues)} 个问题:")
        for i, issue in enumerate(self.issues, 1):
            print(f"\n问题 {i} [{issue['severity'].upper()}]:")
            print(f"  类别: {issue['category']}")
            print(f"  描述: {issue['description']}")

        print(f"\n\n提供 {len(self.fixes)} 个修复方案:")
        for i, fix in enumerate(self.fixes, 1):
            print(f"\n修复方案 {i}:")
            print(f"  类别: {fix['category']}")
            print(f"  描述: {fix['description']}")
            if fix['code']:
                print(f"  代码:")
                print("  " + "\n  ".join(fix['code'].strip().split('\n')))

    def run_all_diagnostics(self):
        """运行所有诊断"""
        self.diagnose_dice_anomaly()
        self.diagnose_gradient_cosine()
        self.diagnose_group_comparison()
        self.check_data_leakage()
        self.generate_report()


def main():
    """主函数"""
    print("FedSAM3-Cream 诊断工具")
    print("=" * 80)
    print("目标:")
    print("1. Dice 目标: 纯图像分割准确度 ~0.75+ (当前异常高 0.98)")
    print("2. 梯度余弦相似度: 应该显示跨模态冲突 (当前恒为 90度)")
    print("3. Group B 应该因文本污染而 Dice 下降")
    print("4. Group C 应该因解耦聚合而 Dice 高于 Group A")
    print("=" * 80)

    tool = DiagnosticTool()
    tool.run_all_diagnostics()

    print("\n\n" + "=" * 80)
    print("建议的行动步骤:")
    print("=" * 80)
    print("""
1. 【立即执行】验证数据集划分:
   python scripts/validate_dataset.py --check-leakage --verbose

2. 【立即执行】添加梯度余弦相似度监控:
   - 在 src/server.py 中添加 compute_gradient_cosine_similarity() 方法
   - 在聚合时调用并记录结果
   - 在 TensorBoard 中可视化

3. 【立即执行】验证解耦聚合是否生效:
   - 在 CreamAggregator 中添加详细的路由日志
   - 运行 Group C 实验并检查日志输出

4. 【今日完成】重新运行三组实验:
   - Group A: 纯视觉基线 (预期 Dice ~0.75)
   - Group B: 文本污染对照 (预期 Dice < Group A)
   - Group C: 解耦蒸馏终极 (预期 Dice > Group A, ~0.80)

5. 【后续优化】如果 Dice 仍然异常:
   - 检查损失函数权重 (lambda_cream)
   - 检查模型是否过拟合
   - 增加验证集样本数
    """)


if __name__ == "__main__":
    main()
