import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class DiscreteDiffusionPolicy(nn.Module):
    def __init__(
        self,
        n_joints: int,              # 关节数
        chunk_size: int,            # 预测步长
        num_action_bins: int,       # 动作离散 Bin 数
        state_dim: int,             # 输入特征维度 (例如 512)
        hidden_dim: int = 512,      # Transformer 内部维度
        num_layers: int = 6,
        nhead: int = 8,
        dropout: float = 0.1,
        # --- 乘法权重参数 (Multiplicative) ---
        local_mul_weight: float = 1.0,   # 局部乘法权重 (默认 1.0 表示不加权)
        global_mul_weight: float = 5.0,  # 全局乘法权重 (终端 Chunk)
        # --- 加法惩罚参数 (Additive) ---
        local_add_val: float = 0.0,      # 局部加法惩罚值 (默认 0.0 表示不惩罚)
        global_add_val: float = 3.0      # 全局加法惩罚值 (终端 Chunk)
    ):
        super().__init__()
        self.n_joints = n_joints
        self.chunk_size = chunk_size
        self.total_action_tokens = n_joints * chunk_size
        
        self.mask_token_id = num_action_bins 
        self.vocab_size = num_action_bins + 1 
        
        # 存储参数
        self.local_mul_weight = local_mul_weight
        self.global_mul_weight = global_mul_weight
        self.local_add_val = local_add_val
        self.global_add_val = global_add_val

        # 1. 特征投影层
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)

        # 2. 动作 Embedding
        self.action_embed = nn.Embedding(self.vocab_size, hidden_dim)
        
        # 3. 位置编码
        self.state_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_pos_embed = nn.Parameter(torch.randn(1, self.total_action_tokens, hidden_dim) * 0.02)

        # 4. Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, dim_feedforward=hidden_dim*4, 
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 5. 输出头
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.predict_head = nn.Linear(hidden_dim, num_action_bins)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _process_state(self, state_features):
        if state_features.dim() == 2:
            state_features = state_features.unsqueeze(1)
        state_embeds = self.state_proj(state_features)
        seq_len = state_embeds.shape[1]
        pos_embed = self.state_pos_embed[:, :seq_len, :] if seq_len <= self.state_pos_embed.shape[1] else self.state_pos_embed
        state_embeds = state_embeds + pos_embed
        state_embeds = self.state_norm(state_embeds)
        return state_embeds

    def get_mask_ratio_schedule(self, t_float):
        return math.cos(t_float * math.pi * 0.5)

    def forward(self, state_features: torch.Tensor, action_tokens: torch.Tensor, is_terminal_chunk=None):
        B = action_tokens.shape[0]
        device = action_tokens.device
        L = self.total_action_tokens
        
        # --- A. Transformer Forward ---
        state_embeds = self._process_state(state_features) 
        
        # 准备 Mask
        flat_actions = action_tokens.view(B, L) 
        u = torch.rand(B, device=device)
        mask_ratios = torch.cos(u * math.pi * 0.5)
        num_masked = (mask_ratios * L).long().clamp(min=1, max=L)
        rand_scores = torch.rand(B, L, device=device)
        mask_indices = rand_scores.argsort(dim=1).argsort(dim=1) < num_masked.unsqueeze(1)
        
        input_ids = flat_actions.clone()
        input_ids[mask_indices] = self.mask_token_id
        
        action_embeds = self.action_embed(input_ids) + self.action_pos_embed[:, :L, :]
        x = torch.cat([state_embeds, action_embeds], dim=1)
        x = self.transformer(x)
        
        action_out = x[:, state_embeds.shape[1]:, :] 
        action_out = self.final_norm(action_out)
        logits = self.predict_head(action_out)
        
        # --- B. 混合 Loss 计算 ---
        
        # 1. 基础 Cross Entropy Loss (Batch, L)
        ce_loss_flat = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), 
            flat_actions.reshape(-1), 
            reduction='none'
        )
        ce_loss = ce_loss_flat.view(B, L)

        # 2. 准备两个矩阵：乘法权重矩阵 & 加法惩罚矩阵
        mul_weights = torch.ones_like(ce_loss)  # 默认为 1.0 (不加权)
        add_values = torch.zeros_like(ce_loss)  # 默认为 0.0 (不加罚)

        # 3. 设置局部惩罚 (Local): 所有样本的最后一步
        # 对应的列是最后 n_joints 个
        if self.local_mul_weight != 1.0:
            mul_weights[:, -self.n_joints:] = self.local_mul_weight
        if self.local_add_val != 0.0:
            add_values[:, -self.n_joints:] = self.local_add_val

        # 4. 设置全局惩罚 (Global): 终端样本的最后一步 (覆盖局部设置)
        if is_terminal_chunk is not None:
            terminal_indices = is_terminal_chunk.nonzero(as_tuple=True)[0]
            
            if terminal_indices.numel() > 0:
                row_mask = torch.zeros(B, dtype=torch.bool, device=device)
                row_mask[terminal_indices] = True
                col_mask = torch.zeros(1, L, dtype=torch.bool, device=device)
                col_mask[:, -self.n_joints:] = True 
                
                terminal_action_mask = row_mask.unsqueeze(1) & col_mask
                
                # 覆盖权重
                if self.global_mul_weight != 1.0:
                    mul_weights[terminal_action_mask] = self.global_mul_weight
                if self.global_add_val != 0.0:
                    add_values[terminal_action_mask] = self.global_add_val

        # 5. 计算最终矩阵 Loss
        # Part 1: 加权 CE Loss
        weighted_ce_loss = ce_loss * mul_weights
        
        # Part 2: 加法惩罚 Loss (没到位概率 * 惩罚值)
        # 没到位概率 = 1.0 - p_correct = 1.0 - exp(-ce_loss)
        not_arrived_prob = 1.0 - torch.exp(-ce_loss)
        additive_penalty_loss = not_arrived_prob * add_values
        
        # Combine
        total_loss_matrix = weighted_ce_loss + additive_penalty_loss

        # 6. 应用 Diffusion Mask (只计算被 Mask 掉的部分)
        mask_flat = mask_indices.float() 
        final_loss = (total_loss_matrix * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        
        return final_loss

    @torch.no_grad()
    def sample(self, state_features: torch.Tensor, steps: int = 10):
        B = state_features.shape[0]
        device = state_features.device
        L = self.total_action_tokens
        state_embeds = self._process_state(state_features)
        current_ids = torch.full((B, L), self.mask_token_id, device=device, dtype=torch.long)
        
        for step in range(steps):
            t = (step + 1) / steps
            mask_ratio = self.get_mask_ratio_schedule(t)
            n_mask = int(L * mask_ratio)
            
            action_embeds = self.action_embed(current_ids) + self.action_pos_embed[:, :L, :]
            x = torch.cat([state_embeds, action_embeds], dim=1)
            x = self.transformer(x)
            action_out = x[:, state_embeds.shape[1]:, :]
            action_out = self.final_norm(action_out)
            logits = self.predict_head(action_out)
            
            probs = F.softmax(logits, dim=-1)
            sampled_conf, sampled_ids = probs.max(dim=-1)
            unknown_mask = (current_ids == self.mask_token_id)
            new_ids = torch.where(unknown_mask, sampled_ids, current_ids)
            
            if n_mask == 0:
                current_ids = new_ids
                break
            
            confidence_scores = sampled_conf.clone()
            confidence_scores[~unknown_mask] = 1e9 
            confidence_scores += torch.rand_like(confidence_scores) * 1e-4
            _, mask_indices = torch.topk(confidence_scores, k=n_mask, dim=1, largest=False)
            current_ids = new_ids.clone()
            row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_mask)
            current_ids[row_idx, mask_indices] = self.mask_token_id
           
        return current_ids.view(B, self.chunk_size, self.n_joints)
    
def test_combined_penalty():
    print("\n========== 测试 Mixed Penalty Logic ==========")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 示例配置：
    # 局部：乘法 1.0 (不加权), 加法 1.0
    # 全局：乘法 5.0 (强加权), 加法 3.0
    model = DiscreteDiffusionPolicy(
        n_joints=7, chunk_size=10, num_action_bins=256, state_dim=768,
        local_mul_weight=1.0, local_add_val=1.0,
        global_mul_weight=5.0, global_add_val=3.0
    ).to(device)

    dummy_state = torch.randn(2, 1, 768).to(device)
    dummy_actions = torch.randint(0, 256, (2, 10, 7)).to(device)
    is_terminal = torch.tensor([True, False]).to(device) 

    loss = model(dummy_state, dummy_actions, is_terminal)
    print(f"Combined Loss: {loss.item():.4f}")

if __name__ == "__main__":
    test_combined_penalty()