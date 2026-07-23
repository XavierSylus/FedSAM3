"""
Main Federated Learning Loop: Simulation of FedSAM3-Cream framework.
Orchestrates server and multiple clients with heterogeneous data modalities.

★ Fix (2026-03-13): 完全迁移到新架构的三层 Trainer，消除对已删除的旧版 ClientTrainer 的依赖。
"""

# 联邦主循环 演示框架结构 支持不同数据模态

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Dict, Tuple
import copy

from src.model import SAM3_Medical, DEVICE, BATCH_SIZE, LR, IMG_SIZE, ROUNDS
# ★ Fix: 使用新架构的三层 Trainer，旧版 ClientTrainer 已在 Phase 2 重构中删除
from src.client import ImageOnlyTrainer, TextOnlyTrainer, MultimodalTrainer
from src.server import CreamAggregator


class HeterogeneousClient:
    """
    Wrapper for clients with different data modalities.
    """

    def __init__(
        self,
        client_id: str,
        modality: str,  # 'image_only', 'text_only', 'multimodal'
        trainer,        # BaseClientTrainer 子类实例
        model: nn.Module  # ★ Fix: 存储 model 引用，用于 load_model_state
    ):
        """
        Args:
            client_id: Unique identifier for the client
            modality: Data modality type
            trainer: BaseClientTrainer subclass instance
            model: The SAM3_Medical model instance this client owns
        """
        self.client_id = client_id
        self.modality = modality
        self.trainer = trainer
        self.model = model  # ★ Fix: 直接持有 model 引用

    def load_model_state(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """
        ★ Fix (2026-03-13)：将全局 state_dict 分发到此客户端模型。
        strict=False 以容忍冻结主干等不参与联邦聚合的键缺失。
        """
        missing, unexpected = self.model.load_state_dict(
            {k: v.to(DEVICE) for k, v in state_dict.items()},
            strict=False
        )

    def train(self, global_reps: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Train the client for one epoch.

        Args:
            global_reps: Global representations from server
        Returns:
            Tuple of (updated_weights, local_public_reps)
        """
        # 所有模态统一使用 optimizer，并通过 trainer.run() 返回结果
        optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=LR
        )
        weights, img_rep, txt_rep, stats = self.trainer.run(
            self.model, optimizer, global_reps
        )
        # run() 返回的 weights 已是过滤后的 state_dict
        local_reps = img_rep if img_rep is not None else (txt_rep if txt_rep is not None else torch.zeros(self.trainer.contrastive_dim))
        return weights, local_reps


def create_dummy_data(num_samples: int, has_mask: bool = True) -> tuple:
    """
    Create dummy data for testing.
    
    Args:
        num_samples: Number of samples to create
        has_mask: Whether to include segmentation masks
    Returns:
        Tuple of (images, masks) or (images,) if has_mask=False
    """
    images = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    if has_mask:
        masks = torch.randn(num_samples, 1, IMG_SIZE, IMG_SIZE)
        # Normalize masks to [0, 1] range
        masks = torch.sigmoid(masks)
        return images, masks
    else:
        return (images,)


def setup_clients() -> List[HeterogeneousClient]:
    """
    Setup 3 clients with heterogeneous data:
    - Client A: Image Only  → ImageOnlyTrainer
    - Client B: Text Only   → TextOnlyTrainer
    - Client C: Multimodal  → MultimodalTrainer

    ★ Fix (2026-03-13): 完全迁移到新架构三层 Trainer。
    每个 HeterogeneousClient 现在明确持有自己的 model 实例。

    Returns:
        List of HeterogeneousClient instances
    """
    clients = []
    num_samples = 50

    # ── Client A: Image Only ─────────────────────────────────
    print("Setting up Client A (Image Only)...")
    private_imgs_A = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    private_masks_A = torch.sigmoid(torch.randn(num_samples, 1, IMG_SIZE, IMG_SIZE))
    public_imgs_A  = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)

    private_loader_A = DataLoader(
        TensorDataset(private_imgs_A, private_masks_A), batch_size=BATCH_SIZE, shuffle=True
    )
    public_loader_A = DataLoader(
        TensorDataset(public_imgs_A), batch_size=BATCH_SIZE, shuffle=True
    )
    model_A  = SAM3_Medical().to(DEVICE)
    trainer_A = ImageOnlyTrainer(
        private_loader=private_loader_A,
        public_loader=public_loader_A,
        device=DEVICE,
    )
    client_A = HeterogeneousClient("Client_A", "image_only", trainer_A, model_A)
    clients.append(client_A)

    # ── Client B: Text Only ──────────────────────────────────
    print("Setting up Client B (Text Only)...")
    # text_feat 维度与 model.embed_dim 一致 (768)
    text_dim = 768
    private_text_B = torch.randn(num_samples, text_dim)
    public_text_B  = torch.randn(num_samples, text_dim)

    private_loader_B = DataLoader(
        TensorDataset(private_text_B), batch_size=BATCH_SIZE, shuffle=True
    )
    public_loader_B = DataLoader(
        TensorDataset(public_text_B), batch_size=BATCH_SIZE, shuffle=True
    )
    model_B  = SAM3_Medical().to(DEVICE)
    trainer_B = TextOnlyTrainer(
        private_loader=private_loader_B,
        public_loader=public_loader_B,
        device=DEVICE,
    )
    client_B = HeterogeneousClient("Client_B", "text_only", trainer_B, model_B)
    clients.append(client_B)

    # ── Client C: Multimodal ─────────────────────────────────
    print("Setting up Client C (Multimodal)...")
    private_imgs_C  = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)
    private_masks_C = torch.sigmoid(torch.randn(num_samples, 1, IMG_SIZE, IMG_SIZE))
    private_text_C  = torch.randn(num_samples, 768)
    public_imgs_C   = torch.randn(num_samples, 3, IMG_SIZE, IMG_SIZE)

    private_loader_C = DataLoader(
        TensorDataset(private_imgs_C, private_masks_C, private_text_C),
        batch_size=BATCH_SIZE, shuffle=True
    )
    public_loader_C = DataLoader(
        TensorDataset(public_imgs_C), batch_size=BATCH_SIZE, shuffle=True
    )
    model_C  = SAM3_Medical().to(DEVICE)
    trainer_C = MultimodalTrainer(
        private_loader=private_loader_C,
        public_loader=public_loader_C,
        device=DEVICE,
    )
    client_C = HeterogeneousClient("Client_C", "multimodal", trainer_C, model_C)
    clients.append(client_C)

    return clients


def main():
    """
    Main federated learning simulation loop.
    """
    print("=" * 60)
    print("FedSAM3-Cream Federated Learning Simulation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"  - CUDA Available: {torch.cuda.is_available()}")
        print(f"  - GPU Count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"  - Current GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print(f"  - Using CPU (CUDA not available)")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LR}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Rounds: {ROUNDS}")
    print("=" * 60)
    
    # Initialize global model
    print("\n[1/4] Initializing global model...")
    global_model = SAM3_Medical().to(DEVICE)
    
    # Initialize server
    print("[2/4] Initializing server...")
    server = CreamAggregator(global_model, device=DEVICE)
    
    # Setup clients
    print("[3/4] Setting up clients...")
    clients = setup_clients()
    print(f"    - {len(clients)} clients initialized")
    for client in clients:
        print(f"      * {client.client_id}: {client.modality}")
    
    # Initialize clients with global model
    print("[4/4] Distributing initial global model to clients...")
    global_state = server.get_global_model().state_dict()
    for client in clients:
        # ★ Fix (2026-03-13)：使用 HeterogeneousClient.load_model_state 代理方法，
        #   兼容新/旧 trainer 架构，避免 AttributeError: load_model_state
        client.load_model_state(global_state)
    
    print("\n" + "=" * 60)
    print("Starting Federated Learning Training...")
    print("=" * 60)
    
    # Main federated learning loop
    for round_num in range(1, ROUNDS + 1):
        print(f"\n--- Round {round_num}/{ROUNDS} ---")
        
        # Step 1: Server distributes global model + global reps
        print("  [Server] Distributing global model and representations...")
        global_reps = server.get_global_reps()
        
        # Update all clients with global model
        global_state = server.get_global_model().state_dict()
        for client in clients:
            client.load_model_state(global_state)
        
        # Step 2: Clients train locally (1 epoch)
        print("  [Clients] Local training...")
        client_weights = []
        client_public_reps = []
        
        for client in clients:
            print(f"    - Training {client.client_id} ({client.modality})...")
            updated_weights, local_reps = client.train(global_reps)
            client_weights.append(updated_weights)
            client_public_reps.append(local_reps)
        
        # Step 3: Clients upload weights + new public data representations
        print("  [Clients] Uploading weights and representations...")
        # (Already collected above)
        
        # Step 4: Server aggregates
        print("  [Server] Aggregating client updates...")
        aggregated_state = server.aggregate_weights(client_weights, client_public_reps)
        
        # Get updated global representations
        updated_global_reps = server.get_global_reps()
        
        # Logging
        if round_num % 10 == 0 or round_num == 1:
            print(f"\n  Round {round_num} Summary:")
            print(f"    - Global text rep norm: {updated_global_reps['global_text_rep'].norm().item():.4f}")
            print(f"    - Global image rep norm: {updated_global_reps['global_image_rep'].norm().item():.4f}")
            print(f"    - Number of aggregated clients: {len(client_weights)}")
    
    print("\n" + "=" * 60)
    print("Federated Learning Training Completed!")
    print("=" * 60)
    
    # Final model evaluation (placeholder)
    print("\nFinal global model statistics:")
    final_model = server.get_global_model()
    total_params = sum(p.numel() for p in final_model.parameters())
    # ★ Fix: 旧版 get_trainable_params() 方法已不存在，改用标准 PyTorch 过滤
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    print(f"  - Frozen parameters: {total_params - trainable_params:,}")



if __name__ == "__main__":
    main()

