from torch.utils.data import DataLoader
from h5_robot_dataset import H5RobotDataset, VLDCollator


# 1. 准备数据
dataset = H5RobotDataset(
    h5_dir="/home/hhhar/liuliu/vld/data/regression/data_res/", # 这里可以写逻辑读取多个文件并 ConcatDataset
    action_stats_path="action_stats.json",
    num_bins=256
)

# 2. 准备整理器 (Collator)
collator = VLDCollator(tokenizer_name='bert-base-uncased')

# 3. 创建 DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=4, 
    shuffle=True, 
    collate_fn=collator, # 关键：处理文本分词
    num_workers=4
)

# 4. 训练循环测试
for batch in dataloader:
    imgs, pcs, txts, acts = batch
    print("Image:", imgs.shape)      # (B, 3, 480, 640)
    print("Points:", pcs.shape)      # (B, 1024, 3)
    print("Actions:", acts.shape)    # (B, 7)
    print("Text Token:", txts) 
    
    # 喂给你的模型
    # loss = model(imgs, pcs, txts, acts)
    break