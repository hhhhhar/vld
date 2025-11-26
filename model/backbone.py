import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import BertModel, BertTokenizer
from model.pointnet_plus_cuda import PointNetPlus
import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# --- PointNet++ 占位符 ---
# PointNet++ 的实现相对复杂，通常依赖于专门的库（如 PyTorch Geometric）
# 或者一个详细的自定义实现。
# 这里我们创建一个占位符模块，它定义了预期的输入和输出，
# 您需要用您的实际 PointNet++ 实现替换它。


# --- 多模态前置网络 ---

class MultiModalBackbone(nn.Module):
    def __init__(self, text_model_name='bert-base-uncased', load_pretrained_resnet=True, pc_num = 1024):
        """
        初始化三个模态的特征提取器。
        
        Args:
            pc_feature_dim (int): PointNet++ 输出的特征维度。
            text_model_name (str): Hugging Face 上的 BERT 模型名称。
        """
        super(MultiModalBackbone, self).__init__()

        # 1. ResNet-50 用于 RGB 特征提取
        # 加载预训练的 ResNet-50
        # resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        res_weight_path = "/home/hhhar/liuliu/vld/model/weight/resnet50-11ad3fa6.pth"
        resnet50 = models.resnet50(pretrained=False)
        if load_pretrained_resnet:
            checkpoint = torch.load(res_weight_path, map_location=torch.device('cpu'))
            resnet50.load_state_dict(checkpoint)
        # 移除最后的 avgpool 和 fc 分类层
        # 我们保留到第8层（即最后一个卷积块的输出）
        self.rgb_extractor = nn.Sequential(*list(resnet50.children())[:-2])
        # 添加一个自适应平均池化层，将 (B, 2048, H/32, W/32) 池化为 (B, 2048, 1, 1)
        self.rgb_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rgb_flatten = nn.Flatten()
        # ResNet-50 的输出特征维度为 2048

        # 2. PointNet++ 用于点云特征提取 
        self.pc_extractor = PointNetPlus()

        # 3. BERT 用于文本特征提取
        self.bert_model = BertModel.from_pretrained(text_model_name)
        # BERT (base) 模型的输出特征维度为 768
        self.pc_num = pc_num
        
        print(f"多模态前置网络初始化完成：")
        print(f"  - RGB (ResNet-50) -> 2048 维特征")
        print(f"  - 点云 (PointNet++ ) -> 1024 维特征")
        print(f"  - 文本 ({text_model_name}) -> {self.bert_model.config.hidden_size} 维特征")


    def forward(self, image, point_cloud, text_input):
        """
        前向传播函数。
        
        Args:
            image (torch.Tensor): 批量的图像张量, e.g., (B, 3, 224, 224)
            point_cloud (torch.Tensor): 批量的点云张量, e.g., (B, NumPoints, 3)
            text_input (dict): 来自 BERT tokenizer 的字典，包含:
                                'input_ids': (B, SeqLen)
                                'attention_mask': (B, SeqLen)
                                'token_type_ids': (B, SeqLen) [可选]
        """
        _pc_num = point_cloud.shape[1]
        assert _pc_num == self.pc_num, f"点云输入的点数应为1024, 但收到 {_pc_num} 点"
        # --- 1. 处理 RGB 图像 ---
        x_rgb = self.rgb_extractor(image)
        x_rgb = self.rgb_pool(x_rgb)
        rgb_features = self.rgb_flatten(x_rgb) # 形状: (B, 2048)

        # --- 2. 处理点云 ---
        pc_features = self.pc_extractor(point_cloud, None) # 形状: (B, pc_feature_dim)

        # --- 3. 处理文本 ---
        bert_output = self.bert_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        )
        # 我们使用 [CLS] 标记（序列的第一个标记）的输出来代表整个序列的特征
        text_features = bert_output.last_hidden_state[:, 0, :] # 形状: (B, 768)

        # 返回三个部分的输出
        return rgb_features, pc_features, text_features

# --- 示例用法 ---
if __name__ == "__main__":
    
    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"正在使用设备: {device}")

    # 1. 初始化 Tokenizer 和模型
    # 我们需要 tokenizer 来准备文本输入
    # try:
    tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = MultiModalBackbone(text_model_name='sentence-transformers/all-MiniLM-L6-v2').to(device)
    model.eval() # 设置为评估模式
    # except Exception as e:
    #     print(f"加载预训练模型失败 (可能是网络问题): {e}")
    #     print("将跳过示例用法。")
    #     exit()


    # 2. 准备伪输入数据 (Batch Size = 2)
    batch_size = 2
    
    # 伪图像数据 (B, 3, 224, 224)
    dummy_image = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 伪点云数据 (B, 1024 点, 3 坐标)
    dummy_pc = torch.randn(batch_size, 1024, 3).to(device)
    
    # 伪文本数据
    dummy_text = ["这是一个基于resnet的rgb特征", "另一个是pointnet++的点云特征"]
    
    # 使用 tokenizer 处理文本
    # padding=True: 填充到批次中的最大长度
    # truncation=True: 截断超过模型最大长度的序列
    # return_tensors='pt': 返回 PyTorch 张量
    dummy_text_input = tokenizer(
        dummy_text, 
        padding=True, 
        truncation=True, 
        return_tensors='pt'
    ).to(device)
    
    print("\n--- 准备输入数据 ---")
    print(f"图像输入形状: {dummy_image.shape}")
    print(f"点云输入形状: {dummy_pc.shape}")
    print(f"文本输入 (input_ids) 形状: {dummy_text_input['input_ids'].shape}")

    # 3. 执行前向传播
    with torch.no_grad(): # 在评估时不需要计算梯度
        try:
            rgb_feat, pc_feat, text_feat = model(dummy_image, dummy_pc, dummy_text_input)
            
            print("\n--- 成功！三部分输出特征 ---")
            print(f"RGB 特征形状:    {rgb_feat.shape}")
            print(f"点云 特征形状:  {pc_feat.shape}")
            print(f"文本 特征形状:  {text_feat.shape}")

        except Exception as e:
            print(f"\n模型前向传播失败: {e}")
            print("请检查您的环境、依赖库（特别是 transformers）和输入数据。")