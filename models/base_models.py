import torch.nn as nn
import torch.nn.functional as F
import torch
affine = True

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
#from MedMamba.MedMamba import VSSM
import os 
from .MedViTV2.MedViT import MedViT_base
#from MedViTV2.MedViT import MedViT_large
#import open_clip
import json
from PIL import Image

# -----------------------------------------------------------------------------
# ---------------------- brain age prediction ---------------------------------
# -----------------------------------------------------------------------------

class Imaging_only_brainage(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.conv1_a = nn.Conv3d(in_channels=1, out_channels=16, kernel_size=3, stride=1)
        self.conv1_b = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm1 = nn.BatchNorm3d(16)

        self.conv2_a = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1)
        self.conv2_b = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm2 = nn.BatchNorm3d(32)

        self.conv3_a = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3_b = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        # MaxPool3d(kernel_size=2, stride=1)
        self.batchnorm3 = nn.BatchNorm3d(64)

        self.dropout1 = nn.Dropout3d(0.2)
        self.linear1 = nn.Linear(in_features=39424, out_features=16)
        # relu
        self.linear2 = nn.Linear(in_features=16, out_features=32)
        # relu
        self.linear3 = nn.Linear(in_features=32, out_features=64)
        # relu
        self.final_layer = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        # input shape = [batch size, channels, image shape]
        img, tabular = x
        x = self.conv1_a(img)
        x = F.relu(x)
        x = self.conv1_b(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = self.batchnorm1(x)

        x = self.conv2_a(x)
        x = F.relu(x)
        x = self.conv2_b(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = self.batchnorm2(x)

        x = self.conv3_a(x)
        x = F.relu(x)
        x = self.conv3_b(x)
        x = F.relu(x)
        x = F.max_pool3d(x, kernel_size=2, stride=2)
        x = self.batchnorm3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.final_layer(x)

        return x[:, 0]


# -----------------------------------------------------------------------------
# ------------------------ AD classification ----------------------------------
# -----------------------------------------------------------------------------
def conv3d_bn3d_relu(in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1, conv_bias=True):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias)
        bn3d = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv3d, bn3d, relu)


def conv3d_instn3d_relu(in_channels, out_channels, bn_momentum=0.05, kernel_size=3, stride=1, padding=1, conv_bias=True):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=conv_bias)
        instn3d = nn.InstanceNorm3d(out_channels, affine=affine)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(conv3d, instn3d, relu)


class PreactivResBlock_bn(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum=0.05, dropout=0.0, stride=1, conv_bias=True):
        super().__init__()
        self.bn1 = nn.BatchNorm3d(in_channels, momentum=bn_momentum)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=conv_bias)
        self.bn2 = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=conv_bias)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout3d(p=dropout)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.BatchNorm3d(in_channels, momentum=bn_momentum),
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=conv_bias),
            )
        else:
            self.downsample = None

    def forward(self, x):
        if not (self.downsample is None):
            identity = self.downsample(x)
        else:
            identity = x

        out = x
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += identity
        return out


class PreactivResNet(nn.Module):
    def __init__(self, in_channels=1, n_outputs=3, bn_momentum=0.1, init_features=4, **kwargs):
        super().__init__()

        # cnn_dropout=0.1
        self.conv_bn_relu = conv3d_bn3d_relu(in_channels, init_features, bn_momentum=bn_momentum)
        self.max_pool3d_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.block1 = PreactivResBlock_bn(init_features, 2 * init_features, bn_momentum=bn_momentum, dropout=0.1)
        self.block2 = PreactivResBlock_bn(2 *init_features, 4 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block3 = PreactivResBlock_bn(4 * init_features, 8 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.2)
        self.block4 = PreactivResBlock_bn(8 * init_features, 16 * init_features, bn_momentum=bn_momentum, stride=2, dropout=0.3)
        self.adaptive_avg_pool3d = nn.AdaptiveAvgPool3d(1)
        self.linear_drop1 = nn.Dropout(0.6)
        self.fc1 = nn.Linear(16 * init_features, 4*init_features)
        self.linear_drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(4*init_features, n_outputs)

        self.relu = nn.ReLU()

    def forward(self, x):
        image, tabular = x
        out = self.conv_bn_relu(image)
        out = self.max_pool3d_1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.adaptive_avg_pool3d(out)
        out = out.view(out.size(0), -1)
        out = self.linear_drop1(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.linear_drop2(out)
        out = self.fc2(out)

        return out


class MLP_8_bn_prl(nn.Module):
    def __init__(self, mlp_layers_shapes, **kwargs):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=mlp_layers_shapes[0], out_features=8),
            nn.BatchNorm1d(8),
            nn.PReLU(),
            nn.Linear(in_features=8, out_features=mlp_layers_shapes[-1]),
        )

        print(self)

    def forward(self, x):
        image, tabular = x
        return self.mlp(tabular)

# ----------------- Tabular Component ----------------- #
class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)

class FeedForwardGEGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.proj = nn.Linear(dim, hidden_dim * 2)
        self.act = GEGLU()
        self.out = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = self.proj(x)
        x = self.act(x)
        x = self.out(x)
        return self.dropout(x)

class FTTransformerBlock(nn.Module):
    def __init__(self, dim, heads, attn_dropout, ffn_hidden, ffn_dropout, residual_dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_dropout, batch_first=True)
        self.residual_dropout1 = nn.Dropout(residual_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForwardGEGLU(dim, ffn_hidden, ffn_dropout)
        self.residual_dropout2 = nn.Dropout(residual_dropout)
    def forward(self, x):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h)
        x = x + self.residual_dropout1(attn_out)
        h = self.norm2(x)
        ffn_out = self.ffn(h)
        x = x + self.residual_dropout2(ffn_out)
        return x

class CategoricalFeatureTokenizer(nn.Module):
    def __init__(self, num_categories, token_dim, bias=True):
        super().__init__()
        self.embeddings = nn.Embedding(sum(num_categories), token_dim)
        self.bias = nn.Parameter(torch.zeros(len(num_categories), token_dim)) if bias else None
        category_offsets = torch.tensor([0] + num_categories[:-1]).cumsum(0)
        self.register_buffer("category_offsets", category_offsets, persistent=False)

    def forward(self, x):
        x = self.embeddings(x + self.category_offsets[None])
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, in_features, token_dim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(in_features, token_dim))
        self.bias = nn.Parameter(torch.zeros(in_features, token_dim)) if bias else None

    def forward(self, x):
        x = self.weight[None] * x[..., None]
        if self.bias is not None:
            x = x + self.bias[None]
        return x

class FTTransformer(nn.Module):
    def __init__(
        self,
        num_numerical,
        num_categories,
        token_dim=192,
        hidden_size=192,
        num_blocks=3,
        attention_n_heads=8,
        attention_dropout=0.2,
        residual_dropout=0.0,
        ffn_dropout=0.1,
        ffn_hidden_size=192,
        pooling_mode="cls",
        num_classes=2, 
    ):
        super().__init__()
        # Tokenizers
        self.categorical_feature_tokenizer = CategoricalFeatureTokenizer(num_categories, token_dim) if num_categories else None
        self.numerical_feature_tokenizer = NumericalFeatureTokenizer(num_numerical, token_dim) if num_numerical else None

        # Adapters
        self.categorical_adapter = nn.Linear(token_dim, hidden_size) if num_categories else None
        self.numerical_adapter = nn.Linear(token_dim, hidden_size) if num_numerical else None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Transformer blocks
        self.transformer = nn.ModuleList([
            FTTransformerBlock(
                hidden_size,
                attention_n_heads,
                attention_dropout,
                ffn_hidden_size,
                ffn_dropout,
                residual_dropout
            )
            for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        self.pooling_mode = pooling_mode

    def forward(self, batch, return_feature=False):
        # batch: expects dict with keys "categorical" and/or "numerical"
        if isinstance(batch, tuple):
            numerical, categorical = batch
        else:
            numerical = batch['numerical']
            categorical = batch['categorical']

        B = numerical.shape[0]
        
        '''
        # to print the tabular input sizes
        if 'numerical' in batch:
            print(f"[FTTransformer] Numerical input shape: {batch['numerical'].shape}")
        if 'categorical' in batch:
            print(f"[FTTransformer] Categorical input shape: {batch['categorical'].shape}")
        '''

        multimodal_tokens = []

        if self.categorical_feature_tokenizer and categorical is not None:
            x_cat = self.categorical_feature_tokenizer(categorical)  # use unpacked variable
            x_cat = self.categorical_adapter(x_cat)
            multimodal_tokens.append(x_cat)

        if self.numerical_feature_tokenizer and numerical is not None:
            x_num = self.numerical_feature_tokenizer(numerical)  # use unpacked variable
            x_num = self.numerical_adapter(x_num)
            multimodal_tokens.append(x_num)

        tokens = torch.cat(multimodal_tokens, dim=1)  # (B, total_num_tokens, hidden_size)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, hidden_size)
        tokens = torch.cat([tokens, cls_token], dim=1)  # (B, total_num_tokens+1, hidden_size)

        for block in self.transformer:
            tokens = block(tokens)

        features = tokens[:, -1, :]
        logits = self.head(features)

        if return_feature:
            return features, logits
        return logits

# ----------------- Image Component (MedViTV2) ----------------- #

class MedViTModel(nn.Module):
    def __init__(self, variant='base', pretrained_path=None, freeze=False, device='cpu'):
        super().__init__()

        # select model variant
        if variant == 'base':
            from .MedViTV2.MedViT import MedViT_base
            self.base_model = MedViT_base(pretrained=False)
            output_dim = 768
        elif variant == 'large':
            from .MedViTV2.MedViT import MedViT_large
            self.base_model = MedViT_large(pretrained=False)
            output_dim = 1024
        else:
            raise ValueError(f"Unknown MedViT variant: {variant}. Choose 'base' or 'large'.")

        # feature extractor (remove classifier)
        self.feature_extractor = nn.Sequential(
            self.base_model.stem,
            self.base_model.features,
            self.base_model.norm,
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.base_model.proj_head = nn.Identity()
        
        # load pretrained weights
        if pretrained_path is not None:
            if os.path.exists(pretrained_path):
                ckpt = torch.load(pretrained_path, map_location='cpu', weights_only = True)

                
                if 'model' in ckpt:
                    state_dict = ckpt['model']
                elif 'state_dict' in ckpt:
                    state_dict = ckpt['state_dict']
                else:
                    state_dict = ckpt

                new_state_dict = {}
                for k, v in state_dict.items():
                    k_new = k
                    
                    if k.startswith("model."):
                        k_new = k[len("model."):]
                    if k.startswith("module."):
                        k_new = k[len("module."):]
                    if k.startswith("encoder."):
                        k_new = k[len("encoder."):]
                    # Drop classifier / projection heads
                    if "head" in k_new or "proj_head" in k_new:
                        continue
                    new_state_dict[k_new] = v

                # Load cleaned weights 
                missing, unexpected = self.base_model.load_state_dict(new_state_dict, strict=False)
                total_params = sum(p.numel() for p in self.base_model.parameters())
                loaded_params = sum(v.numel() for k, v in new_state_dict.items() if k in self.base_model.state_dict())

                print(f"Loaded MedViT-{variant} pretrained weights from {pretrained_path}")
                print(f"Loaded params: {loaded_params/1e6:.2f}M / {total_params/1e6:.2f}M "
                    f"({100 * loaded_params/total_params:.2f}% of model)")
                print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
                print("Example missing keys:", missing[:20])

            else:
                print(f" WARNING: Pretrained path not found: {pretrained_path}. Using random init.")
        else:
          
            print(" Skipping pretrained MedViT weights (load_pretrained=False)")
        


        # Optionally freeze backbone
        if freeze:
            for p in self.base_model.parameters():
                p.requires_grad = False

        self.output_dim = output_dim
        self.proj = nn.Linear(output_dim, output_dim)
        self.feature_dim = output_dim
        self.device = device
        self.to(device)

    def forward(self, x, image_valid_num=None, return_feature=False):
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        feats = self.proj(feats)
        feats = feats.view(B, N, -1)

        if image_valid_num is not None:
            mask = torch.arange(N, device=self.device).unsqueeze(0) < image_valid_num.unsqueeze(1)
            feats = (feats * mask.unsqueeze(-1)).sum(dim=1) / image_valid_num.unsqueeze(1).clamp(min=1)
        else:
            feats = feats.mean(dim=1)

        return (feats, None) if return_feature else feats


if __name__ == "__main__":
    pass
