import torch
import torch.nn as nn
from einops import rearrange
from timm.models.registry import register_model

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 add_zero_attn=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.add_zero_attn = add_zero_attn
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            q, k, v = map(lambda t: rearrange(
                t, 'b n (h d) -> (b h) n d', h=self.heads), qkv)
            bsz_x_num_heads, _, head_dim = k.shape
            zero_attn_shape = (bsz_x_num_heads, 1, head_dim)
            k = torch.cat(
                [k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
            v = torch.cat(
                [v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
            # Dimension change: (batch_size, num_heads, num_patches+1, head_dim)
            q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
            k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
            v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)

        else:
            q, k, v = map(lambda t: rearrange(
                t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.,
                 add_zero_attn=True):
        # dim: dimension of image feature vector
        # heads: number of multi-head attention heads
        # dim_head: dimension of each head

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.add_zero_attn = add_zero_attn
        self.img_q = nn.Linear(dim, inner_dim, bias=False)
        self.img_k = nn.Linear(dim, inner_dim, bias=False)
        self.img_v = nn.Linear(dim, inner_dim, bias=False)

        self.clinical_q = nn.Linear(dim, inner_dim, bias=False)
        self.clinical_k = nn.Linear(dim, inner_dim, bias=False)
        self.clinical_v = nn.Linear(dim, inner_dim, bias=False)

        self.img_attn_dropout = nn.Dropout(dropout)
        self.clinical_attn_dropout = nn.Dropout(dropout)

        self.attn_dropout_img_to_clinic = nn.Dropout(dropout)
        self.attn_dropout_clinical_to_img = nn.Dropout(dropout)

        self.img_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.clinical_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, img, clinic):
        img_q = self.img_q(img)  # (batch_size, img_patch_num, inner_dim)
        img_k = self.img_k(img)
        img_v = self.img_v(img)

        clinical_q = self.clinical_q(clinic)
        clinical_k = self.clinical_k(clinic)
        clinical_v = self.clinical_v(clinic)
        img_q = rearrange(img_q, 'b n (h d) -> b h n d', h=self.heads)
        clinical_q = rearrange(
            clinical_q, 'b n (h d) -> b h n d', h=self.heads)
        # add zero attention along batch dimension (now first)
        if self.add_zero_attn:
            # Dimension transformation
            img_k = rearrange(img_k, 'b n (h d) -> (b h) n d', h=self.heads)
            img_v = rearrange(img_v, 'b n (h d) -> (b h) n d', h=self.heads)

            clinical_k = rearrange(
                clinical_k, 'b n (h d) -> (b h) n d', h=self.heads)
            clinical_v = rearrange(
                clinical_v, 'b n (h d) -> (b h) n d', h=self.heads)
            bsz_x_num_heads, _, head_dim = img_k.shape
            zero_attn_shape = (bsz_x_num_heads, 1, head_dim)
            img_k = torch.cat(
                (img_k, torch.zeros(zero_attn_shape, device=img_k.device)), dim=1)
            img_v = torch.cat(
                (img_v, torch.zeros(zero_attn_shape, device=img_v.device)), dim=1)

            clinical_k = torch.cat(
                (clinical_k, torch.zeros(zero_attn_shape, device=clinical_k.device)), dim=1)
            clinical_v = torch.cat(
                (clinical_v, torch.zeros(zero_attn_shape, device=clinical_v.device)), dim=1)

            # Dimension transformation
            img_k = rearrange(img_k, '(b h) n d -> b h n d', h=self.heads)
            img_v = rearrange(img_v, '(b h) n d -> b h n d', h=self.heads)

            clinical_k = rearrange(
                clinical_k, '(b h) n d -> b h n d', h=self.heads)
            clinical_v = rearrange(
                clinical_v, '(b h) n d -> b h n d', h=self.heads)

        else:
            # Dimension transformation
            img_k = rearrange(img_k, 'b n (h d) -> b h n d', h=self.heads)
            img_v = rearrange(img_v, 'b n (h d) -> b h n d', h=self.heads)

            clinical_k = rearrange(
                clinical_k, 'b n (h d) -> b h n d', h=self.heads)
            clinical_v = rearrange(
                clinical_v, 'b n (h d) -> b h n d', h=self.heads)

        img_dots = torch.matmul(
            img_q, clinical_k.transpose(-1, -2)) * self.scale
        clinical_dots = torch.matmul(
            clinical_q, img_k.transpose(-1, -2)) * self.scale
        img_self_dots = torch.matmul(
            img_q, img_k.transpose(-1, -2)) * self.scale
        clinical_self_dots = torch.matmul(
            clinical_q, clinical_k.transpose(-1, -2)) * self.scale

        img_clinical_attn = self.softmax(img_dots)
        clinical_img_attn = self.softmax(clinical_dots)
        img_self_attn = self.softmax(img_self_dots)
        clinical_self_attn = self.softmax(clinical_self_dots)

        img_clinical_attn = self.img_attn_dropout(img_clinical_attn)
        clinical_img_attn = self.clinical_attn_dropout(clinical_img_attn)
        img_self_attn = self.attn_dropout_img_to_clinic(img_self_attn)
        clinical_self_attn = self.attn_dropout_clinical_to_img(
            clinical_self_attn)

        img_clinical_out = torch.matmul(img_clinical_attn, clinical_v)
        clinical_img_out = torch.matmul(clinical_img_attn, img_v)
        img_self_out = torch.matmul(img_self_attn, img_v)
        clinical_self_out = torch.matmul(clinical_self_attn, clinical_v)
        img_out_attn = (img_clinical_out+img_self_out)/2.0
        clinical_out_attn = (clinical_img_out+clinical_self_out)/2.0

        # Dimension transformation
        img_out_attn = rearrange(img_out_attn, 'b h n d -> b n (h d)')
        clinical_out_attn = rearrange(
            clinical_out_attn, 'b h n d -> b n (h d)')

        img_out_attn = self.img_out(img_out_attn)
        clinical_out_attn = self.clinical_out(clinical_out_attn)

        return img_out_attn, clinical_out_attn

class Transformer_Cross(nn.Module):
    # Cannot iterate due to cross-attention, so use two transformers instead
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()

        self.img_pre_norm = nn.LayerNorm(dim)
        self.clinical_pre_norm = nn.LayerNorm(dim)

        self.img_ffd = FeedForward(dim, mlp_dim, dropout=dropout)
        self.clinical_ffd = FeedForward(dim, mlp_dim, dropout=dropout)

        self.img_ffd_norm = nn.LayerNorm(dim)
        self.clinical_ffd_norm = nn.LayerNorm(dim)

        self.attn = CrossAttention(
            dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, img, clinic):
        # Using residual network structure
        img_identity = img
        clinical_identity = clinic
        img = self.img_pre_norm(img)
        clinic = self.clinical_pre_norm(clinic)

        img, clinic = self.attn(img, clinic)

        img = img + img_identity
        clinic = clinic + clinical_identity

        img_identity = img
        clinical_identity = clinic

        img = self.img_ffd_norm(img)
        clinic = self.clinical_ffd_norm(clinic)

        img = self.img_ffd(img)
        clinic = self.clinical_ffd(clinic)

        img = img + img_identity
        clinic = clinic + clinical_identity

        return img, clinic

class UncertaintyNetwork(nn.Module):
    def __init__(self, in_channels=768, width=512, depth=3, init_prednet_zero=False) -> None:
        super().__init__()
        layers = [nn.Linear(in_channels, width),
            nn.LeakyReLU()]
        for i in range(depth - 1):
            layers.extend([
                nn.Linear(width, width),
                nn.LeakyReLU()
            ])
        layers.extend([
            nn.Linear(width, 1),
            nn.Softplus()
        ])
        self.unc_module = nn.Sequential(*layers)
        self.EPS = 1e-6

        self.dropout = nn.Dropout(0.1)

        if init_prednet_zero:
            self.unc_module.apply(self.init_weights_zero)

    def forward(self, input):
        return self.dropout(self.EPS + self.unc_module(input))

    def init_weights_zero(model, layer):
        if isinstance(layer, nn.Linear):
            nn.init.zeros_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

@register_model
class MultiScaleClassifierFM(nn.Module):
    def __init__(self, num_classes,
                 fm_model,
                 hidden_size,
                 num_heads=8,
                 dropout_rate=0.2):
        super(MultiScaleClassifierFM, self).__init__()

        self.feature_extraction = fm_model

        for param in self.feature_extraction.parameters():
            param.requires_grad = False

        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.transformer1 = Transformer_Cross(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size//num_heads,
            mlp_dim=hidden_size*2,
            dropout=dropout_rate)

        self.transformer2 = Transformer_Cross(
            dim=hidden_size,
            heads=num_heads,
            dim_head=hidden_size//num_heads,
            mlp_dim=hidden_size*2,
            dropout=dropout_rate)

        self.transformer3 = Transformer(
            dim=hidden_size,
            depth=2,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            mlp_dim=hidden_size * 2,
            dropout=dropout_rate)
        
        self.transformer4 = Transformer(
            dim=hidden_size,
            depth=2,
            heads=num_heads,
            dim_head=hidden_size // num_heads,
            mlp_dim=hidden_size * 2,
            dropout=dropout_rate)

        self.unc_module = UncertaintyNetwork(in_channels=hidden_size)
        
        self.head_norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.normal_(self.head.bias, std=1e-6)

    def forward(self, img_list):
        # Extract image features at different scales
        # img_list: [img_x2, img_x20] 
        img_x20, img_x2 = img_list
        img_embeddings_x2 = self.feature_extraction.forward_features(img_x2)
        img_embeddings_x20 = self.feature_extraction.forward_features(img_x20)
        
        img_embeddings_x2, img_embeddings_x20 = self.transformer1(
            img_embeddings_x2, img_embeddings_x20)

        img_embeddings_x2, img_embeddings_x20 = self.transformer2(
            img_embeddings_x2, img_embeddings_x20)

        # Concatenate features
        # (batch_size, img_patch_num+num_clinic, hidden_size)
        feature_embeddings = torch.cat(
            (img_embeddings_x2, img_embeddings_x20), dim=1)

        # Extract features from concatenated representation
        feature_embeddings = self.transformer3(feature_embeddings)
        feature_embeddings = self.transformer4(feature_embeddings)

        feature_embeddings = self.head_norm(feature_embeddings)
        logits = self.head(feature_embeddings[:, 0])
        unc = self.unc_module(feature_embeddings[:, 0]).flatten()

        return logits, unc, feature_embeddings

