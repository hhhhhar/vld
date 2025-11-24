import torch
import torch.nn as nn
from dis_dif_mini import DiscreteDiffusionPolicy
from backbone import MultiModalBackbone

class VLDDiffusionSystem(nn.Module):
    def __init__(self, 
                 action_dim=7, 
                 num_action_bins=256,
                 text_model_name='bert-base-uncased',
                 rgb_dim = 2048,
                 pc_feature_dim = 1024,
                 text_dim = 768):
        super().__init__()

        # 1. 实例化你的 Backbone
        self.backbone = MultiModalBackbone(
            text_model_name=text_model_name,
            load_pretrained_resnet=True # 演示用
        )
        self.rgb_dim = rgb_dim
        self.pc_feature_dim = pc_feature_dim
        self.text_dim = text_dim

        # 2. 计算融合后的特征维度
        # RGB(2048) + PC(1024) + Text(768) = 3840
        self.fusion_dim = self.rgb_dim + self.pc_feature_dim + self.text_dim
        
        # 3. 特征投影层 (将巨大的拼接特征降维到 Diffusion 需要的维度)
        self.diffusion_hidden_dim = 512
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.fusion_dim, self.diffusion_hidden_dim),
            nn.LayerNorm(self.diffusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

        # 4. 实例化 Diffusion Policy
        self.policy = DiscreteDiffusionPolicy(
            action_dim=action_dim,
            num_action_bins=num_action_bins,
            state_dim=self.diffusion_hidden_dim, # 注意这里接的是投影后的维度
            hidden_dim=self.diffusion_hidden_dim
        )

    def get_fused_state(self, image, point_cloud, text_input):
        """辅助函数：提取特征并融合"""
        # A. 提取三个模态
        rgb_feat, pc_feat, text_feat = self.backbone(image, point_cloud, text_input)

        # B. 拼接 (Concatenate)
        # (B, 2048), (B, 1024), (B, 768) -> (B, 3840)
        concat_feat = torch.cat([rgb_feat, pc_feat, text_feat], dim=1)

        # C. 投影融合 (Projection)
        state_feat = self.fusion_proj(concat_feat) # (B, 512)
        
        return state_feat

    def forward(self, image, point_cloud, text_input, action_tokens):
        """训练入口"""
        # 1. 获取融合后的状态
        state_feat = self.get_fused_state(image, point_cloud, text_input)
        
        # 2. 计算 Diffusion Loss
        loss = self.policy(state_feat, action_tokens)
        return loss

    def predict(self, image, point_cloud, text_input, num_steps=12):
        """推理入口"""
        # 1. 获取融合后的状态
        state_feat = self.get_fused_state(image, point_cloud, text_input)
        
        # 2. 采样动作
        action_tokens = self.policy.sample(state_feat, num_steps=num_steps)
        return action_tokens


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # 1. 初始化模型
    model = VLDDiffusionSystem(action_dim=7, num_action_bins=100).to(device)
    print("模型构建完成！")

    # 2. 构造伪数据
    batch_size = 2
    
    # 图像: (B, 3, 224, 224)
    dummy_img = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 点云: (B, 2048, 3) 假设有2048个点
    dummy_pc = torch.randn(batch_size, 1024, 3).to(device)
    
    # 文本: 这里的字典结构要符合 BERT 的要求
    dummy_text = {
        'input_ids': torch.randint(0, 100, (batch_size, 10)).to(device), # 假设 seq_len=10
        'attention_mask': torch.ones(batch_size, 10).to(device)
    }
    
    # 动作 GT: (B, 7)
    dummy_action = torch.randint(0, 100, (batch_size, 7)).to(device)

    # 3. 测试训练 Forward
    loss = model(dummy_img, dummy_pc, dummy_text, dummy_action)
    print(f"训练 Loss: {loss.item():.4f}")

    # 4. 测试推理 Predict
    print("开始推理...")
    pred_actions = model.predict(dummy_img, dummy_pc, dummy_text)
    print(f"预测动作 Shape: {pred_actions.shape}")
    print(f"预测动作示例: {pred_actions[0].tolist()}")
