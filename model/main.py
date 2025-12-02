import torch
import torch.nn as nn
from transformers import BertTokenizer
import sys
sys.path.append("/home/hhhar/liuliu/vld")
from model.dis_dif_mini import DiscreteDiffusionPolicy
from model.backbone import MultiModalBackbone
import os

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
                 diffusion_layers = 6,
                 # --- 混合惩罚参数 ---
                 local_mul_weight=1.0,   # 局部乘法权重
                 global_mul_weight=5.0,  # 全局乘法权重
                 local_add_val=1.0,      # 局部加法值
                 global_add_val=3.0      # 全局加法值
                 ):
        super().__init__()

        # 实例化 Backbone
        self.backbone = MultiModalBackbone(
            text_model_name=text_model_name,
            load_pretrained_resnet=True 
        )
        
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.fusion_dim = rgb_dim + pc_feature_dim + text_dim
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
            nhead=8,
            # 传递所有 4 个参数
            local_mul_weight=local_mul_weight,
            global_mul_weight=global_mul_weight,
            local_add_val=local_add_val,
            global_add_val=global_add_val
        )

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def get_fused_state(self, image, point_cloud, text_input):
        rgb_feat, pc_feat, text_feat = self.backbone(image, point_cloud, text_input)
        concat_feat = torch.cat([rgb_feat, pc_feat, text_feat], dim=1)
        state_feat = self.fusion_proj(concat_feat) 
        return state_feat

    def forward(self, image, point_cloud, text_input, action_tokens, is_terminal_chunk=None):
        if action_tokens.device != self.device:
            action_tokens = action_tokens.to(self.device)
        
        if is_terminal_chunk is not None and is_terminal_chunk.device != self.device:
            is_terminal_chunk = is_terminal_chunk.to(self.device)

        state_feat = self.get_fused_state(image, point_cloud, text_input)
        loss = self.policy(state_feat, action_tokens, is_terminal_chunk)
        return loss

    def predict(self, image, point_cloud, text_input, num_steps=10):
        state_feat = self.get_fused_state(image, point_cloud, text_input)
        pred_actions = self.policy.sample(state_feat, steps=num_steps)
        return pred_actions

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")
    
    model = VLDDiffusionSystem(
        action_dim=7, 
        chunk_size=10,
        local_mul_weight=2.0, 
        global_mul_weight=5.0,
        local_add_val=1.0,
        global_add_val=3.0
    )
    model.to(device)
    print("Model initialized successfully with Mixed Penalty.")