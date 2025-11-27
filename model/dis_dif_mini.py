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
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_joints = n_joints
        self.chunk_size = chunk_size
        self.total_action_tokens = n_joints * chunk_size
        
        self.mask_token_id = num_action_bins 
        self.vocab_size = num_action_bins + 1 

        # 1. 特征投影层
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.state_norm = nn.LayerNorm(hidden_dim)

        # 2. 动作 Embedding
        self.action_embed = nn.Embedding(self.vocab_size, hidden_dim)
        
        # 3. 位置编码
        # state_pos_embed 长度设为 1，因为你的特征是 1 维的（变成 1 个 token）
        self.state_pos_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.action_pos_embed = nn.Parameter(torch.randn(1, self.total_action_tokens, hidden_dim) * 0.02)

        # 4. Transformer (Decoder-only but bidirectional)
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
        """
        统一处理状态特征，确保它变成 (Batch, Seq_Len=1, Hidden_Dim)
        """
        # 如果输入是 (Batch, Dim)，则扩充为 (Batch, 1, Dim)
        if state_features.dim() == 2:
            state_features = state_features.unsqueeze(1)
        
        # 投影: (B, 1, Dim) -> (B, 1, Hidden)
        state_embeds = self.state_proj(state_features)
        
        # 加上位置编码 (虽然只有一个位置，但这个编码起到了"Modality Embedding"的作用，区分这是图像不是动作)
        # 注意处理广播机制，如果输入是序列也能兼容
        seq_len = state_embeds.shape[1]
        # 如果预设的位置编码不够长(针对变长序列)，截取或重复，这里针对你的情况通常是 seq_len=1
        pos_embed = self.state_pos_embed[:, :seq_len, :] if seq_len <= self.state_pos_embed.shape[1] else self.state_pos_embed
        
        state_embeds = state_embeds + pos_embed
        state_embeds = self.state_norm(state_embeds)
        return state_embeds

    def get_mask_ratio_schedule(self, t_float):
        return math.cos(t_float * math.pi * 0.5)

    def forward(self, state_features: torch.Tensor, action_tokens: torch.Tensor):
        """
        state_features: (Batch, state_dim) <-- 只需要传入 2D Tensor
        action_tokens: (Batch, chunk_size, n_joints)
        """
        B = action_tokens.shape[0]
        device = action_tokens.device
        
        # --- A. 处理 State ---
        state_embeds = self._process_state(state_features) # (B, 1, H)

        # --- B. 准备 Action Mask ---
        flat_actions = action_tokens.view(B, -1)
        L = flat_actions.shape[1]
        
        u = torch.rand(B, device=device)
        mask_ratios = torch.cos(u * math.pi * 0.5)
        num_masked = (mask_ratios * L).long().clamp(min=1, max=L)
        
        rand_scores = torch.rand(B, L, device=device)
        mask_indices = rand_scores.argsort(dim=1).argsort(dim=1) < num_masked.unsqueeze(1)
        
        input_ids = flat_actions.clone()
        input_ids[mask_indices] = self.mask_token_id
        
        action_embeds = self.action_embed(input_ids) + self.action_pos_embed[:, :L, :]
        
        # --- C. 拼接 ---
        # 现在的拼接形状是: (Batch, 1 + L, Hidden_Dim)
        x = torch.cat([state_embeds, action_embeds], dim=1)
        
        # --- D. 预测 ---
        x = self.transformer(x)
        
        # 截取 Action 部分 (从第 1 个位置开始截取，因为第 0 个是 state)
        action_out = x[:, state_embeds.shape[1]:, :] 
        action_out = self.final_norm(action_out)
        logits = self.predict_head(action_out)
        
        # --- E. Loss ---
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), flat_actions.reshape(-1), reduction='none')
        mask_flat = mask_indices.view(-1).float()
        masked_loss = (loss * mask_flat).sum() / (mask_flat.sum() + 1e-6)
        
        return masked_loss

    @torch.no_grad()
    def sample(self, state_features: torch.Tensor, steps: int = 10):
        B = state_features.shape[0]
        device = state_features.device
        L = self.total_action_tokens
        
        # --- A. 处理 State (只算一次) ---
        state_embeds = self._process_state(state_features) # (B, 1, H)
        
        # 初始化：全都是 [MASK]
        current_ids = torch.full((B, L), self.mask_token_id, device=device, dtype=torch.long)
        
        for step in range(steps):
            # 计算当前进度 (0.0 -> 1.0)
            # step=0 -> t=0.1 ... step=9 -> t=1.0
            t = (step + 1) / steps
            
            # 根据进度计算剩余需要 Mask 的比例 (MaskGIT Cosine Schedule)
            # t=0 时 ratio=1.0 (全Mask), t=1 时 ratio=0.0 (全保留)
            mask_ratio = self.get_mask_ratio_schedule(t)
            
            # 计算本轮结束后，需要保留多少个 [MASK]
            n_mask = int(L * mask_ratio)
            
            # --- Forward ---
            # 1. 正常的 Embedding 和 Transformer 前向传播
            action_embeds = self.action_embed(current_ids) + self.action_pos_embed[:, :L, :]
            x = torch.cat([state_embeds, action_embeds], dim=1)
            x = self.transformer(x)
            
            # 2. 拿到预测结果
            action_out = x[:, state_embeds.shape[1]:, :] # 剥离 state
            action_out = self.final_norm(action_out)
            logits = self.predict_head(action_out) # (B, L, vocab)
            
            # --- Sampling Strategy (核心修改) ---
            
            # 3. 获取当前所有位置的预测 Token 和 置信度
            probs = F.softmax(logits, dim=-1)
            sampled_conf, sampled_ids = probs.max(dim=-1) # (B, L)
            
            # 4. 【关键步骤1：填充】
            # 找出当前仍然是 [MASK] 的位置
            unknown_mask = (current_ids == self.mask_token_id)
            
            # 仅在 unknown 的位置填入新的预测值 (sampled_ids)
            # 已知的位置保持原样 (current_ids)，不要覆写！
            new_ids = torch.where(unknown_mask, sampled_ids, current_ids)
            
            # 5. 如果已经到了最后一步 (n_mask == 0)，直接返回填满的结果
            if n_mask == 0:
                current_ids = new_ids
                break
            
            # 6. 【关键步骤2：重新 Mask】
            # 我们需要选出置信度最低的 n_mask 个位置变回 [MASK]
            # 但是！必须保护那些在上一轮就已经确定的 Token，防止它们被误删
            
            # 初始置信度使用当前预测的概率
            confidence_scores = sampled_conf.clone()
            
            # 对于那些本来就"已知"(非 unknown) 的 Token，将其置信度设为无穷大
            # 这样在排序找"最小值"时，它们永远排在最后，不会被选中
            confidence_scores[~unknown_mask] = 1e9 
            
            # 加一点微小的噪音防止 argsort 在置信度相同时的确定性问题（可选）
            confidence_scores += torch.rand_like(confidence_scores) * 1e-4
            
            # 选出置信度最低的 n_mask 个位置的索引
            # topk(largest=False) 返回最小的 k 个值
            _, mask_indices = torch.topk(confidence_scores, k=n_mask, dim=1, largest=False)
            
            # 7. 应用 Mask
            # 这一步只会在那些"刚刚填入但置信度不高"的位置打上 Mask
            current_ids = new_ids.clone()
            row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_mask)
            current_ids[row_idx, mask_indices] = self.mask_token_id
            
        return current_ids.view(B, self.chunk_size, self.n_joints)
    
def test_discrete_diffusion_policy():
    print("\n========== 开始测试 Discrete Diffusion Policy ==========")
    
    # --- 1. 配置参数 ---
    B = 2                   # Batch Size
    Context_Len = 16        # 视觉特征序列长度 (例如压缩过的 Token)
    State_Dim = 768         # 视觉特征维度
    
    N_Joints = 7            # 机器人 7 轴
    Chunk_Size = 10         # 预测未来 10 步
    Vocab_Size = 256        # 离散 Bin 数
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # --- 2. 实例化模型 ---
    model = DiscreteDiffusionPolicy(
        n_joints=N_Joints,
        chunk_size=Chunk_Size,
        num_action_bins=Vocab_Size,
        state_dim=State_Dim,
        hidden_dim=256,
        num_layers=4,
        nhead=4
    ).to(device)

    print(f"Model created. Total Action Tokens = {model.total_action_tokens}")

    # --- 3. 构造虚拟数据 (Training Mode) ---
    # 模拟前置网络输出的特征
    dummy_state = torch.randn(B, Context_Len, State_Dim).to(device)
    
    # 模拟真实的 Ground Truth 动作 (已经离散化为 0-255 的整数)
    # Shape: (Batch, Time, Joints)
    dummy_actions_gt = torch.randint(0, Vocab_Size, (B, Chunk_Size, N_Joints)).to(device)

    # --- 4. 训练前向传播 ---
    loss = model(dummy_state, dummy_actions_gt)
    print(f"Training Loss: {loss.item():.4f}")
    
    loss.backward()
    print("Backward pass successful.")

    # --- 5. 推理采样 (Inference Mode) ---
    print("\nStarting Inference Sampling...")
    model.eval()
    
    with torch.no_grad():
        # 只需要传入 State
        pred_actions = model.sample(dummy_state, steps=10)
        
    print(f"Predicted Action Shape: {pred_actions.shape}")
    print(f"Expected Shape: ({B}, {Chunk_Size}, {N_Joints})")
    
    # 检查是否还有 Mask Token (理想情况下应该没有)
    has_mask = (pred_actions == model.mask_token_id).any()
    print(f"Any mask tokens left? {has_mask.item()}")


if __name__ == "__main__":
    # 假设你已经把上一段的类定义复制到了这里，或者已经import了
    # 这里直接运行测试
    test_discrete_diffusion_policy()