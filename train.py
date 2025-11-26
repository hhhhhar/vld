from datetime import date
import os
import time
from IPython import embed
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import logging
import os.path as osp

# 引入你的数据集和模型
# 假设 h5_robot_dataset.py 和 vld_diffusion_system_device.py 在同一目录下
# 或者已经安装在 python path 中
from dataset.h5_robot_dataset import H5RobotDataset, VLDCollator
from model.main import VLDDiffusionSystem

# 设置 logger
log = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, cfg, filename="checkpoint.pth"):
    """保存模型权重和优化器状态"""
    save_path = os.path.join(os.getcwd(), filename)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': OmegaConf.to_container(cfg, resolve=True)
    }, save_path)
    log.info(f"Saved checkpoint to {save_path}")

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig):
    # 1. 打印配置信息
    print(OmegaConf.to_yaml(cfg))
    
    # 设置随机种子
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)
    
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 2. 准备数据
    # ------------------------------------------------------------------
    log.info("Initializing Dataset...")
    dataset = H5RobotDataset(**cfg.data)
    
    # 准备 Collator
    collator = VLDCollator(tokenizer_name=cfg.model.text_model_name)
    
    dataloader = DataLoader(
        dataset, collate_fn=collator, **cfg.dataloader)
    log.info(f"Dataset loaded. Size: {len(dataset)}, Batches: {len(dataloader)}")

    # ------------------------------------------------------------------
    # 3. 初始化模型
    # ------------------------------------------------------------------
    log.info("Initializing Model...")
    model = VLDDiffusionSystem(**cfg.model)
    model.to(device)
    
    # ------------------------------------------------------------------
    # 4. 优化器与调度器
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.training.learning_rate, 
        weight_decay=cfg.training.weight_decay
    )
    
    # 简单的 Cosine Annealing 调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=cfg.training.epochs, 
        eta_min=1e-6
    )

    # 恢复检查点 (如果需要)
    start_epoch = 0
    if cfg.checkpoint.resume and os.path.exists(cfg.checkpoint.path):
        log.info(f"Resuming from checkpoint: {cfg.checkpoint.path}")
        checkpoint = torch.load(cfg.checkpoint.path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # ------------------------------------------------------------------
    # 5. 训练循环
    # ------------------------------------------------------------------
    log.info("Start Training...")
    cp_dir = f"cps/{date.today}"
    
    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        
        # 使用 tqdm 显示进度条
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}")
        
        for step, batch in enumerate(pbar):
            # 解包数据 (根据 h5_robot_dataloader.py 的格式)
            # 假设 batch 是 list/tuple: [imgs, pcs, txts, acts]
            # 如果 txts 是被 tokenizer 处理过的 dict (input_ids, attention_mask)，需要相应处理
            imgs, pcs, txts, acts = batch
            
            # 将 Tensor 移动到 Device
            imgs = imgs.to(device)
            pcs = pcs.to(device)
            acts = acts.to(device) 
            for keys in txts:
                txts[keys] = txts[keys].to(device)
            
            # 文本通常在 Backbone 内部处理 Tokenizer 或者在这里 .to(device)
            # 如果 collator 已经把 text 变成了 tensor (input_ids 等)，则需要 .to(device)
            if isinstance(txts, torch.Tensor):
                txts = txts.to(device)
            elif isinstance(txts, dict): # 常见的 tokenizer 输出
                txts = {k: v.to(device) for k, v in txts.items()}
            # 如果 txts 是 raw list of strings，则不需要 .to(device)
            
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播 (计算 Loss)
            # VLDDiffusionSystem.forward 返回的是 Loss
            # embed()
            loss = model(imgs, pcs, txts, acts)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪 (防止梯度爆炸，对 Transformer 很重要)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录日志
            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "lr": f"{current_lr:.6f}"
            })
            
            if step % cfg.training.log_interval == 0:
                # 这里可以添加 TensorBoard logging
                pass
        
        # Epoch 结束处理
        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        log.info(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        
        # 保存 Checkpoint
        if (epoch + 1) % cfg.training.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, cfg, filename=osp.join(cp_dir,f"checkpoint_epoch_{epoch+1}.pth"))
            
    # 保存最终模型
    save_checkpoint(model, optimizer, cfg.training.epochs, cfg, filename=osp.join(cp_dir, "checkpoint_final.pth"))
    log.info("Training Finished.")

if __name__ == "__main__":
    main()