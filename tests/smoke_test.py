"""
冒烟测试 (Smoke Test) - 快速验证所有核心功能是否正常工作
更新版本：适配新的配置系统和项目结构
"""
import torch
import sys
import os
from pathlib import Path
from typing import Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("FedSAM3-Cream 冒烟测试")
print("=" * 60)

# 测试计数器
tests_passed = 0
tests_failed = 0
errors = []

def test_result(test_name: str, passed: bool, error_msg: str = ""):
    """记录测试结果"""
    global tests_passed, tests_failed
    if passed:
        tests_passed += 1
        print(f"  [PASS] {test_name}")
    else:
        tests_failed += 1
        errors.append(f"[FAIL] {test_name}: {error_msg}")
        print(f"  [FAIL] {test_name}: {error_msg}")

# ==================== 测试 1: 环境检查 ====================
print("\n[1/10] 环境检查...")
try:
    import torch
    import torch.nn as nn
    import numpy as np
    import yaml
    from torch.utils.data import DataLoader, TensorDataset
    
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    
    test_result("Python 环境", True)
    test_result(f"PyTorch 版本: {torch.__version__}", True)
    test_result(f"CUDA 可用: {cuda_available}", True)
    if cuda_available:
        test_result(f"GPU: {torch.cuda.get_device_name(0)}", True)
    test_result("NumPy 可用", True)
    test_result("YAML 可用", True)
except Exception as e:
    test_result("环境检查", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 2: 配置系统 ====================
print("\n[2/10] 配置系统测试...")
try:
    from src.config import Config, load_config
    
    # 测试配置文件加载
    config_path = project_root / "configs" / "exp_baseline.yaml"
    if config_path.exists():
        config = Config.from_yaml(str(config_path))
        test_result("配置文件加载", True)
        test_result(f"数据根目录: {config.data_root}", True)
        test_result(f"训练轮数: {config.rounds}", True)
        test_result(f"批次大小: {config.batch_size}", True)
        test_result(f"学习率: {config.learning_rate}", True)
        test_result(f"Lambda Cream: {config.lambda_cream}", True)
        
        # 测试配置字典转换
        config_dict = config.to_dict()
        test_result("配置字典转换", isinstance(config_dict, dict))
    else:
        test_result("配置文件加载", False, f"配置文件不存在: {config_path}")
except Exception as e:
    test_result("配置系统测试", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 3: 模型导入和初始化 ====================
print("\n[3/10] 模型导入和初始化...")
try:
    from src.model import SAM3_Medical, DEVICE, IMG_SIZE
    
    model = SAM3_Medical(
        img_size=1024,
        embed_dim=768,
        decoder_dim=256,
        num_classes=1,
        adapter_skip=64
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.get_trainable_params())
    frozen_params = total_params - trainable_params
    
    test_result("模型导入", True)
    test_result(f"模型初始化 (设备: {DEVICE})", True)
    test_result(f"总参数: {total_params:,}", True)
    test_result(f"可训练参数: {trainable_params:,}", True)
    test_result(f"冻结参数: {frozen_params:,}", True)
    test_result("参数冻结检查", frozen_params > trainable_params)
except Exception as e:
    test_result("模型导入和初始化", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 4: 模型前向传播 ====================
print("\n[4/10] 模型前向传播...")
try:
    from src.model import SAM3_Medical, DEVICE, IMG_SIZE
    
    model = SAM3_Medical(
        img_size=1024,
        embed_dim=768,
        decoder_dim=256,
        num_classes=1,
        adapter_skip=64
    ).to(DEVICE)
    model.eval()
    
    # 测试 forward
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    expected_shape = (batch_size, 1, IMG_SIZE, IMG_SIZE)
    assert output.shape == expected_shape, f"输出形状错误: {output.shape} != {expected_shape}"
    
    # 测试 extract_features
    features = model.extract_features(dummy_input)
    expected_feat_shape = (batch_size, (IMG_SIZE // 16) ** 2, 768)
    assert features.shape == expected_feat_shape, f"特征形状错误: {features.shape} != {expected_feat_shape}"
    
    test_result("forward() 前向传播", True)
    test_result("extract_features() 特征提取", True)
    test_result(f"输出形状: {output.shape}", True)
    test_result(f"特征形状: {features.shape}", True)
except Exception as e:
    test_result("模型前向传播", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 5: 损失函数 ====================
print("\n[5/10] 损失函数测试...")
try:
    from src.cream_losses import CreamContrastiveLoss
    from src.model import SAM3_Medical, DEVICE
    
    model = SAM3_Medical(
        img_size=1024,
        embed_dim=768,
        decoder_dim=256,
        num_classes=1,
        adapter_skip=64
    ).to(DEVICE)
    loss_fn = CreamContrastiveLoss(tau=0.07)
    
    # 创建测试数据
    batch_size = 2
    dummy_img = torch.randn(batch_size, 3, 1024, 1024).to(DEVICE)
    local_features = model.extract_features(dummy_img)  # (B, N, D)
    global_text_rep = torch.randn(768).to(DEVICE)
    global_image_rep = torch.randn(768).to(DEVICE)
    
    # 计算损失
    L_inter, L_intra = loss_fn(local_features, global_text_rep, global_image_rep)
    
    assert L_inter.item() >= 0, "L_inter 应该 >= 0"
    assert L_intra.item() >= 0, "L_intra 应该 >= 0"
    
    test_result("CreamContrastiveLoss 初始化", True)
    test_result("L_inter 计算", True)
    test_result("L_intra 计算", True)
    test_result(f"L_inter = {L_inter.item():.4f}, L_intra = {L_intra.item():.4f}", True)
except Exception as e:
    test_result("损失函数测试", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 6: 客户端训练器 ====================
print("\n[6/10] 客户端训练器测试...")
try:
    # ★ Fix: 使用新架构三层 Trainer，旧版 ClientTrainer 已在 Phase 2 重构中删除
    from src.client import ImageOnlyTrainer
    from src.model import SAM3_Medical, DEVICE, IMG_SIZE
    from torch.utils.data import DataLoader, TensorDataset
    
    # 创建虚拟数据
    batch_size = 4
    num_samples = 10
    private_imgs  = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    private_masks = torch.sigmoid(torch.randn(num_samples, 1, IMG_SIZE, IMG_SIZE))
    public_imgs   = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    
    private_loader = DataLoader(TensorDataset(private_imgs, private_masks), batch_size=batch_size, shuffle=True)
    public_loader  = DataLoader(TensorDataset(public_imgs),                 batch_size=batch_size, shuffle=True)
    
    # 创建训练器（新架构：Trainer 不持有 model，model 外部传入）
    model   = SAM3_Medical(img_size=IMG_SIZE, embed_dim=768, decoder_dim=256, num_classes=1, adapter_skip=64).to(DEVICE)
    trainer = ImageOnlyTrainer(
        private_loader=private_loader,
        public_loader=public_loader,
        device=DEVICE,
        local_epochs=1
    )
    
    # 创建全局表示
    global_reps = {
        'global_text_rep': torch.randn(768).to(DEVICE),
        'global_image_rep': torch.randn(768).to(DEVICE)
    }
    
    # ★ Fix: 新接口：trainer.run(model, optimizer, global_reps)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    updated_weights, img_rep, txt_rep, stats = trainer.run(model, optimizer, global_reps)
    local_reps = img_rep if img_rep is not None else torch.zeros(768)
    
    assert isinstance(updated_weights, dict), "返回的状态应该是字典"
    assert 'avg_loss' in stats, "统计信息应该包含 avg_loss"
    
    test_result("ImageOnlyTrainer 初始化", True)
    test_result("trainer.run() 训练", True)
    test_result(f"局部表示形状: {local_reps.shape}", True)
    test_result(f"平均损失: {stats.get('avg_loss', 0):.4f}", True)
except Exception as e:
    test_result("客户端训练器测试", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 7: 服务器聚合器 ====================
print("\n[7/10] 服务器聚合器测试...")
try:
    from src.server import CreamAggregator
    from src.model import SAM3_Medical, DEVICE
    
    # 创建全局模型和聚合器
    global_model = SAM3_Medical(
        img_size=1024,
        embed_dim=768,
        decoder_dim=256,
        num_classes=1,
        adapter_skip=64
    ).to(DEVICE)
    aggregator = CreamAggregator(
        global_model, 
        device=DEVICE,
        aggregation_method='contrastive_weighted',
        global_rep_alpha=0.9
    )
    
    # 创建虚拟客户端权重和表示
    num_clients = 3
    client_weights = []
    client_public_reps = []
    
    for i in range(num_clients):
        client_model = SAM3_Medical(
            img_size=1024,
            embed_dim=768,
            decoder_dim=256,
            num_classes=1,
            adapter_skip=64
        ).to(DEVICE)
        client_weights.append(client_model.state_dict())
        client_public_reps.append(torch.randn(768).to(DEVICE))
    
    # 聚合
    aggregated_state = aggregator.aggregate_weights(client_weights, client_public_reps)
    
    # 获取全局表示
    global_reps = aggregator.get_global_reps()
    
    assert isinstance(aggregated_state, dict), "聚合状态应该是字典"
    assert 'global_text_rep' in global_reps, "应该有全局文本表示"
    assert 'global_image_rep' in global_reps, "应该有全局图像表示"
    assert global_reps['global_text_rep'].shape == (768,), "全局文本表示形状错误"
    assert global_reps['global_image_rep'].shape == (768,), "全局图像表示形状错误"
    
    test_result("CreamAggregator 初始化", True)
    test_result("aggregate_weights() 聚合", True)
    test_result("get_global_reps() 获取全局表示", True)
    test_result("全局表示形状正确", True)
except Exception as e:
    test_result("服务器聚合器测试", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 8: 数据加载器 ====================
print("\n[8/10] 数据加载器测试...")
try:
    from torch.utils.data import DataLoader, TensorDataset
    from src.model import IMG_SIZE
    
    # 创建虚拟数据
    batch_size = 4
    num_samples = 20
    images = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    masks = torch.randn(num_samples, 1, IMG_SIZE, IMG_SIZE)
    
    dataset = TensorDataset(images, masks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 测试迭代
    batch_count = 0
    for batch in loader:
        batch_count += 1
        img_batch, mask_batch = batch
        assert img_batch.shape[0] <= batch_size, "批次大小错误"
        assert img_batch.shape[1] == 3, "图像通道数错误"
        assert mask_batch.shape[1] == 1, "掩码通道数错误"
        if batch_count >= 3:  # 只测试前3个批次
            break
    
    test_result("DataLoader 创建", True)
    test_result("数据批次迭代", True)
    test_result(f"批次大小: {batch_size}", True)
    test_result(f"成功迭代 {batch_count} 个批次", True)
except Exception as e:
    test_result("数据加载器测试", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 9: 完整训练循环（1轮） ====================
print("\n[9/10] 完整训练循环测试（1轮）...")
try:
    from src.model import SAM3_Medical, DEVICE, IMG_SIZE
    from src.server import CreamAggregator
    # ★ Fix: 使用新架构三层 Trainer
    from src.client import ImageOnlyTrainer
    from torch.utils.data import DataLoader, TensorDataset
    
    # 创建全局模型
    global_model = SAM3_Medical(img_size=IMG_SIZE, embed_dim=768, decoder_dim=256, num_classes=1, adapter_skip=64).to(DEVICE)
    
    # 创建服务器
    server = CreamAggregator(global_model, device=DEVICE, aggregation_method='contrastive_weighted', global_rep_alpha=0.9)
    
    # 创建单个客户端（简化测试）
    batch_size = 4
    num_samples = 10
    private_imgs  = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    private_masks = torch.sigmoid(torch.randn(num_samples, 1, IMG_SIZE, IMG_SIZE))
    public_imgs   = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    
    private_loader = DataLoader(TensorDataset(private_imgs, private_masks), batch_size=batch_size, shuffle=True)
    public_loader  = DataLoader(TensorDataset(public_imgs),                 batch_size=batch_size, shuffle=True)
    
    client_model = SAM3_Medical(img_size=IMG_SIZE, embed_dim=768, decoder_dim=256, num_classes=1, adapter_skip=64).to(DEVICE)
    trainer = ImageOnlyTrainer(private_loader=private_loader, public_loader=public_loader, device=DEVICE, local_epochs=1)
    
    # ★ Fix: 新架构下，模型分发通过 load_state_dict 完成
    global_state = server.get_global_model().state_dict()
    client_model.load_state_dict(global_state, strict=False)
    
    # ★ Fix: trainer.run(model, optimizer, global_reps)
    global_reps = server.get_global_reps()
    optimizer = torch.optim.Adam([p for p in client_model.parameters() if p.requires_grad], lr=2e-4)
    updated_weights, img_rep, txt_rep, stats = trainer.run(client_model, optimizer, global_reps)
    local_reps = img_rep if img_rep is not None else torch.zeros(768)
    
    # 聚合
    client_weights     = [updated_weights]
    client_public_reps = [local_reps]
    aggregated_state   = server.aggregate_weights(client_weights, client_public_reps)
    
    # 更新全局模型
    server.get_global_model().load_state_dict(aggregated_state, strict=False)
    
    test_result("完整训练循环（1轮）", True)
    test_result("模型状态更新", True)
    test_result("服务器聚合", True)
    test_result("全局模型更新", True)
except Exception as e:
    test_result("完整训练循环测试", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试 10: 主入口测试 ====================
print("\n[10/10] 主入口测试...")
try:
    # 测试 main.py 是否可以导入
    import main
    test_result("main.py 导入", True)
    
    # 测试配置加载函数
    config_path = project_root / "configs" / "exp_baseline.yaml"
    if config_path.exists():
        from src.config import Config
        config = Config.from_yaml(str(config_path))
        test_result("通过 Config.from_yaml 加载配置", True)
        
        # 测试配置合并
        override_dict = {'training': {'rounds': 10}}
        config.merge_from_dict(override_dict)
        assert config.rounds == 10, "配置合并失败"
        test_result("配置合并功能", True)
    else:
        test_result("配置文件存在性检查", False, f"配置文件不存在: {config_path}")
except Exception as e:
    test_result("主入口测试", False, str(e))
    import traceback
    traceback.print_exc()

# ==================== 测试总结 ====================
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
print(f"通过: {tests_passed}")
print(f"失败: {tests_failed}")
print(f"总计: {tests_passed + tests_failed}")

if tests_failed > 0:
    print("\n失败的测试:")
    for error in errors:
        print(f"  {error}")
    print("\n[FAIL] 冒烟测试失败！请检查上述错误。")
    sys.exit(1)
else:
    print("\n[SUCCESS] 所有冒烟测试通过！系统基本功能正常。")
    print("\n建议下一步:")
    print("  1. 运行完整训练: python main.py --config configs/exp_baseline.yaml")
    print("  2. 检查数据目录是否存在: data/train, data/val, data/test")
    print("  3. 验证 GPU 内存是否足够（如果使用 CUDA）")
    sys.exit(0)
