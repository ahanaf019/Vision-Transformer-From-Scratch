import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from models.modules.patch_embedding import PatchEmbedding
from models.modules.shifted_patch_tokenization import ShiftedPatchTokenization
from models.modules.positional_embedding import PositionalEmbedding
from models.modules.transformer_encoder import TransformerEncoder

class SL_ViT(nn.Module):
    def __init__(
            self, 
            in_channels, 
            d_model,
            num_layers,
            num_heads,
            mlp_dim,
            dropout_rate, 
            stochastic_path_rate,
            image_size,
            patch_size,
            shift_ratio,
            num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.image_size = image_size
        self.patch_size = patch_size
        self.shift_ratio = shift_ratio
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.mlp_dim = mlp_dim
        self.num_classes = num_classes
        
        self.patch_embedding = ShiftedPatchTokenization(
            in_channels=in_channels,
            d_model=d_model,
            patch_size=patch_size,
            shift_ratio=shift_ratio
        )

        self.cls_token = nn.Parameter(
            torch.zeros(size=[1, d_model])
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        sequence_len = (image_size // patch_size) ** 2
        self.positional_embedding = PositionalEmbedding(
            d_model=d_model,
            sequence_len=sequence_len + 1
        )

        steps = torch.linspace(0, stochastic_path_rate, num_layers)
        layers = [TransformerEncoder(
            d_model=d_model, 
            num_heads=num_heads, 
            mlp_dim=mlp_dim, 
            dropout_rate=dropout_rate,
            stochastic_path_rate=steps[i],
            attention_type='locality_attention'
            ) for i in range(num_layers)]
        self.encoder = nn.Sequential(*layers)

        self.classifier = nn.Linear(d_model, num_classes)

    
    def forward(self, x: torch.Tensor):
        x = self.patch_embedding(x)
        x = torch.cat([self.cls_token.expand(x.size(0), -1, -1), x], dim=1)
        x = self.positional_embedding(x)
        x = self.encoder(x)
        return self.classifier(x[:, 0, :])


# model = ViT(
#     in_channels=3,
#     d_model=768,
#     image_size=384,
#     patch_size=16,
#     num_heads=12,
#     num_layers=12,
#     dropout_rate=0.1,
#     mlp_dim=3072,
#     num_classes=1000
# ).to('cuda')
# summary(model, input_size=[5, 3, 256, 256])