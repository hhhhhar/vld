import torch
import torch.nn as nn
from transformers import BertTokenizer
from model.dis_dif_mini import DiscreteDiffusionPolicy
from model.backbone import MultiModalBackbone
import os
import sys
sys.path.append("..")

class VLDDiffusionSystem(nn.Module):
    def __init__(self, 
                 action_dim=7,              # 机器人关节数
                 chunk_size=10,             # 预测时间步长
                 num_action_bins=256,       # 动作离散类别数
                 text_model_name='bert-base-uncased',
                 rgb_dim = 2048,
                 pc_feature_dim = 1024,
                 text_dim = 768,
                 diffusion_hidden_dim = 512,
                 diffusion_layers = 6):
        super().__init__()

        # 实例化 Backbone
        self.backbone = MultiModalBackbone(
            text_model_name=text_model_name,
            load_pretrained_resnet=True 
        )
        
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # 计算融合维度
        self.fusion_dim = rgb_dim + pc_feature_dim + text_dim
        
        # 特征投影层
        self.diffusion_hidden_dim = diffusion_hidden_dim
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.diffusion_hidden_dim),
            nn.LayerNorm(self.diffusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 实例化 Diffusion Policy
        self.policy = DiscreteDiffusionPolicy(
            n_joints=action_dim,
            chunk_size=chunk_size,
            num_action_bins=num_action_bins,
            state_dim=self.diffusion_hidden_dim, 
            hidden_dim=self.diffusion_hidden_dim,
            num_layers=diffusion_layers,
            nhead=8
        )

    @property
    def device(self):
        """辅助属性：获取模型当前所在的设备"""
        # 查找 policy 中任意一个参数的 device
        return next(self.policy.parameters()).device

    def get_fused_state(self, image, point_cloud, text_input):
        """
        提取并融合特征
        """
        # 确保 backbone 输出的特征在正确的 device 上 (由 Backbone 内部处理)
        rgb_feat, pc_feat, text_feat = self.backbone(image, point_cloud, text_input)

        # 拼接
        concat_feat = torch.cat([rgb_feat, pc_feat, text_feat], dim=1)

        # 投影
        state_feat = self.fusion_proj(concat_feat) 
        
        return state_feat

    def forward(self, image, point_cloud, text_input, action_tokens):
        """
        训练入口
        """
        # 确保输入的 GT actions 也在正确的 device 上
        if action_tokens.device != self.device:
            action_tokens = action_tokens.to(self.device)

        # 获取状态
        state_feat = self.get_fused_state(image, point_cloud, text_input)
        
        # 计算 Loss
        loss = self.policy(state_feat, action_tokens)
        return loss

    def predict(self, image, point_cloud, text_input, num_steps=10):
        """
        推理入口
        """
        # 获取状态
        state_feat = self.get_fused_state(image, point_cloud, text_input)
        
        # 采样 (sample 内部会自动使用 state_feat.device 创建初始 mask)
        pred_actions = self.policy.sample(state_feat, steps=num_steps)
        
        return pred_actions

# --- 3. 包含 Device 的测试代码 ---
if __name__ == "__main__":
    # 自动检测 device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # 配置
    B = 2
    chunk_size = 12
    joints = 7
    
    # 1. 实例化模型并移动到 device
    model = VLDDiffusionSystem(action_dim=joints, chunk_size=chunk_size)
    model.to(device)  # <--- 关键步骤：将整个模型搬到 GPU
    
    # 2. 构造 Dummy 数据并移动到 device
    img = torch.randn(B, 3, 224, 224).to(device)
    pc = torch.randn(B, 1024, 3).to(device)
    # 文本通常是 List[str] 或 Tokenized Tensor，这里简化处理
    txt = ["instruction"] * B 
    dummy_text_input = tokenizer(
        txt, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    ).to(device)
    
    # GT Action 也必须在 device 上
    gt_actions = torch.randint(0, 256, (B, chunk_size, joints)).to(device)

    print("Model and Data moved to device.")

    # 3. 训练测试
    loss = model(img, pc, dummy_text_input, gt_actions)
    print(f"Training Loss: {loss.item():.4f}")

    # 4. 推理测试
    preds = model.predict(img, pc, dummy_text_input, num_steps=5)
    print(f"Prediction Shape: {preds.shape}") 
    print(f"Prediction Device: {preds.device}") # 验证输出是否也在 GPU 上
    