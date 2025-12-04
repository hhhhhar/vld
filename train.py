from datetime import datetime
import os
import time
from IPython import embed
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import logging
import os.path as osp

# 引入 accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed

# 引入你的数据集和模型
# 注意：你需要确保你的 dataset 模块 (h5_robot_dataset.py) 包含了 VLDCollator 并且返回了 is_terminal_chunk
from dataset.h5_robot_dataset import H5RobotDataset, VLDCollator
from model.main import VLDDiffusionSystem

# 设置 logger
log = logging.getLogger(__name__)

def save_checkpoint(accelerator, model, optimizer, epoch, cfg, filename="checkpoint.pth"):
    """
    保存模型权重和优化器状态
    注意：在多卡训练中，需要等待所有进程同步，且只在主进程保存
    """
    # 等待所有进程到达此处
    accelerator.wait_for_everyone()
    
    # 只有主进程执行保存操作
    if accelerator.is_main_process:
        save_path = os.path.join(os.getcwd(), filename)
        
        # 获取原始模型（去除 DDP 包装）
        unwrapped_model = accelerator.unwrap_model(model)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': OmegaConf.to_container(cfg, resolve=True)
        }, save_path)
        log.info(f"Saved checkpoint to {save_path}")

@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def main(cfg: DictConfig):
    # ------------------------------------------------------------------
    # 1. 初始化 Accelerator
    # ------------------------------------------------------------------
    # 创建 DDP 配置，允许模型中有未使用的参数 (Diffusion Policy 经常出现)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # 可以在这里指定混合精度，例如 mixed_precision="fp16"
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    
    # 只在主进程打印配置，避免多卡刷屏
    if accelerator.is_main_process:
        print(OmegaConf.to_yaml(cfg))
        log.info(f"Using device: {accelerator.device}")
    
    # 设置随机种子 (Accelerate 提供了方便的封装)
    set_seed(cfg.training.seed)

    # ------------------------------------------------------------------
    # 2. 准备数据
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        log.info("Initializing Dataset...")
        
    dataset = H5RobotDataset(**cfg.data)
    collator = VLDCollator(tokenizer_name=cfg.model.text_model_name)
    
    dataloader = DataLoader(
        dataset, collate_fn=collator, **cfg.dataloader
    )
    
    if accelerator.is_main_process:
        log.info(f"Dataset loaded. Size: {len(dataset)}, Batches: {len(dataloader)}")

    # ------------------------------------------------------------------
    # 3. 初始化模型
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        log.info("Initializing Model...")
        
    # 确保 cfg.model 包含 local_loss_weight 和 target_loss_weight
    model = VLDDiffusionSystem(**cfg.model)
    
    # ------------------------------------------------------------------
    # 4. 优化器与调度器
    # ------------------------------------------------------------------
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=cfg.training.learning_rate, 
        weight_decay=cfg.training.weight_decay
    )
    
    steps_per_epoch = len(dataloader) 
    num_training_steps = cfg.training.epochs * steps_per_epoch

    num_warmup_steps = int(num_training_steps * cfg.training.warmup_ratio)

    # 3. 替换 Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    # ------------------------------------------------------------------
    # 5. 使用 Accelerate Prepare
    # ------------------------------------------------------------------
    # 重要：这步会自动处理设备放置和 DDP 包装
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # 恢复检查点 logic
    start_epoch = 0
    if cfg.checkpoint.resume and os.path.exists(cfg.checkpoint.path):
        if accelerator.is_main_process:
            log.info(f"Resuming from checkpoint: {cfg.checkpoint.path}")
            
        # 使用 map_location 确保加载到当前设备的 CPU/GPU
        checkpoint = torch.load(cfg.checkpoint.path, map_location='cpu')
        
        # 加载参数
        accelerator.unwrap_model(model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

    # ------------------------------------------------------------------
    # 6. 训练循环
    # ------------------------------------------------------------------
    if accelerator.is_main_process:
        log.info("Start Training...")
        
    cp_dir = f"/media/hhhar/hhd/har/vlddd/cps/{datetime.now().strftime('%Y_%m_%d_%H:%M')}"
    if accelerator.is_main_process and not osp.exists(cp_dir):
        os.makedirs(cp_dir)

    for epoch in range(start_epoch, cfg.training.epochs):
        model.train()
        epoch_loss = 0.0
        
        # 只在主进程显示进度条，disable=not ...
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{cfg.training.epochs}", 
                     disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(pbar):
            # *** 关键修改：解包数据，新增 is_terminal_chunk ***
            # VLDCollator 返回顺序: imgs, pcs, txts, acts, is_terminal_chunk
            imgs, pcs, txts, acts, is_terminal_chunk = batch
            
            # --------------------------------------------------------------
            # 处理数据移动到设备
            # imgs/pcs/acts 应该已被 accelerator.prepare(dataloader) 自动移动
            # 但 txts (dict) 和 is_terminal_chunk (新的) 需要显式处理
            # --------------------------------------------------------------
            
            # 处理特殊的 txts 结构
            if isinstance(txts, dict):
                # 确保字典里的 tensor 在正确的设备上
                txts = {k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v for k, v in txts.items()}
            elif isinstance(txts, torch.Tensor):
                txts = txts.to(accelerator.device)
            
            # *** 关键修改：is_terminal_chunk 移动到设备 ***
            is_terminal_chunk = is_terminal_chunk.to(accelerator.device) 
            
            optimizer.zero_grad()
            
            # 前向传播
            # *** 关键修改：传入 is_terminal_chunk ***
            loss = model(imgs, pcs, txts, acts, is_terminal_chunk=is_terminal_chunk)
            
            # 反向传播 - 使用 Accelerate
            accelerator.backward(loss)
            
            # 梯度裁剪 - 使用 Accelerate
            if accelerator.sync_gradients:
                # 裁剪原始模型的梯度
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            # 记录日志
            # 使用 accelerator.reduce 聚合多卡 loss 可能会更准确，但简单起见，这里继续累加本卡 loss
            epoch_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            
            if accelerator.is_main_process:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}", 
                    "lr": f"{current_lr:.6f}"
                })
                
                if step % cfg.training.log_interval == 0:
                    # 这里可以添加 TensorBoard logging
                    pass
        
        
        # 计算平均 Loss
        # 注意：这里计算的是单卡平均 Loss，如果需要全局平均，需要使用 accelerator.reduce
        avg_loss = epoch_loss / len(dataloader)
        
        if accelerator.is_main_process:
            log.info(f"Epoch {epoch+1} done. Avg Loss: {avg_loss:.4f}")
        
        # 保存 Checkpoint - 使用 save_checkpoint 封装
        if (epoch + 1) % cfg.training.save_interval == 0:
            save_checkpoint(accelerator, model, optimizer, epoch, cfg, 
                             filename=osp.join(cp_dir,f"checkpoint_epoch_{epoch+1}_{avg_loss:.4f}.pth"))
            
    # 保存最终模型
    save_checkpoint(accelerator, model, optimizer, cfg.training.epochs, cfg, 
                     filename=osp.join(cp_dir, "checkpoint_final.pth"))
    
    if accelerator.is_main_process:
        log.info("Training Finished.")

if __name__ == "__main__":
    main()