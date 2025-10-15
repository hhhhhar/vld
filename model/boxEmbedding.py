import torch
import torch.nn as nn
import torch.nn.functional as F

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.ln(x)
        x = self.drop(x)
        return x

class BoxEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, scene_size, dim_type=2, use_class_embed=False, num_classes=None, pos_mlp_dim=64):
        """
        in_dim: M (number of box fields)
        out_dim: embedding dim C (must equal token dim you fuse with)
        scene_size: physical size of the scene in each dimension (for normalization)
        """
        super().__init__()
        self.mlp = Mlp(in_dim, out_dim, out_dim)
        self.use_class_embed = use_class_embed
        self.dim_type = dim_type
        if use_class_embed:
            assert num_classes is not None
            self.class_embed = nn.Embedding(num_classes, out_dim)
        # optional small MLP to project (cx,cy[,cz]) to positional bias
        self.pos_mlp = Mlp(dim_type, pos_mlp_dim, out_dim)
        assert dim_type in [2, 3]
        assert scene_size.shape[0] == dim_type
        angle_scale = [1.0] if dim_type == 2 else [1.0, 1.0, 1.0]
        self.scale = torch.cat([scene_size.repeat(2), torch.tensor(angle_scale)] ) # (2*dim_type,)

    def normalize(self, box):
        return box / self.scale

    def forward(self, box, box_mask=None):
        """
        boxes: (B, M)   (e.g. M=5 center-based or 9 for 3D)
        returns: box_emb: (B, out_dim)
        """
        # normalize box to [0,1] based on scene size
        box = self.normalize(box)
        
        # base embedding from geometry
        b_emb = self.mlp(box)  # (B,out_dim)

        # positional bias from (cx,cy,[cz]) (first two dims, assumed normalized)
        coords = box[..., :self.dim_type]  # (B,dim_type)
        pos = self.pos_mlp(coords)  # (B,out_dim)
        b_emb = b_emb + pos

        if box_mask is not None:
            # zero out padded boxes
            b_emb = b_emb * box_mask.unsqueeze(-1).to(b_emb.dtype)

        return b_emb
    

if __name__ == "__main__":
    # simple test
    be = BoxEmbedding(in_dim=5, out_dim=256, scene_size=torch.tensor([10.0, 10.0]), dim_type=2)
    boxes = torch.tensor([[5.0, 5.0, 2.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])
    emb = be(boxes)
    print(f'2d:{emb.shape}')  # should be (2, 256)
    
    be3d = BoxEmbedding(in_dim=9, out_dim=256, scene_size=torch.tensor([10.0, 10.0, 5.0]), dim_type=3)
    boxes3d = torch.tensor([[5.0, 5.0, 2.5, 2.0, 2.0, 1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]])
    emb3d = be3d(boxes3d)
    print(f'3d:{emb3d.shape}')  # should be (2, 256)