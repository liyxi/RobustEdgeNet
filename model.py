import logging
from functools import partial

import cv2
import numpy as np
import torch
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg
from timm.models.registry import register_model
from timm.models.vision_transformer import PatchEmbed, Block, checkpoint_filter_fn
from timm.models.vision_transformer import VisionTransformer
from torch import nn

_logger = logging.getLogger(__name__)


class RobustVisionTransformer(VisionTransformer):

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 global_pool='token',
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 init_values=None,
                 class_token=True,
                 no_embed_class=False,
                 pre_norm=False,
                 fc_norm=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 weight_init='',
                 embed_layer=PatchEmbed,
                 norm_layer=None,
                 act_layer=None,
                 block_fn=Block,
                 start_index=0,
                 interval=1,
                 zero_conv_channel=128
                 ):
        super(RobustVisionTransformer, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,  # 0.,
            attn_drop_rate=attn_drop_rate,  # 0.,
            drop_path_rate=drop_path_rate,  # 0.,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
        )

        self.start_index = start_index
        self.interval = interval
        self.zero_conv_channel = zero_conv_channel
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self._build_side_branch(
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            init_values=init_values,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn

        )
        self.patch_embed_c = embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
            bias=not pre_norm,  # disable bias if pre-norm is used (e.g. CLIP)
        )

        self.cls_token_c = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        num_patches = self.patch_embed.num_patches
        embed_len_c = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed_c = nn.Parameter(torch.randn(1, embed_len_c, embed_dim) * .02)
        self.pos_drop_c = nn.Dropout(p=drop_rate)
        self.norm_pre_c = norm_layer(embed_dim) if pre_norm else nn.Identity()

        self.train_transforms_norm = None
        self.eval_transforms_norm = None
        self.train_transforms_norm_c = None
        self.eval_transforms_norm_c = None

    def _build_side_branch(
            self,
            depth: int,
            num_heads: int,
            mlp_ratio: float,
            qkv_bias: bool,
            init_values: float,
            drop_rate: float,
            attn_drop_rate: float,
            drop_path_rate: float,
            norm_layer,
            act_layer,
            block_fn
    ):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Initialize the branch as an empty list
        zero_conv_in = []
        zero_conv_out = []
        blocks = []

        # Populate the branch with alternating zero convolutions and blocks
        # for i in range(depth):
        for i in range(self.start_index, depth, self.interval):
            count = 0
            # zero convolution for input
            zero_conv = nn.Linear(self.embed_dim, self.zero_conv_channel)
            nn.init.zeros_(zero_conv.weight)
            nn.init.zeros_(zero_conv.bias)
            zero_conv_in.append(zero_conv)
            # zero convolution for output
            zero_conv = nn.Linear(self.zero_conv_channel, self.embed_dim)
            nn.init.zeros_(zero_conv.weight)
            nn.init.zeros_(zero_conv.bias)
            zero_conv_out.append(zero_conv)

            # Add a block layer
            blocks.append(block_fn(
                dim=self.zero_conv_channel,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[count],
                norm_layer=norm_layer,
                act_layer=act_layer
            ))
            count += 1

        self.side_zero_conv_in = nn.Sequential(*zero_conv_in)
        self.side_zero_conv_out = nn.Sequential(*zero_conv_out)
        self.side_blocks = nn.Sequential(*blocks)

    def _pos_embed_c(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pos_embed_c
            if self.cls_token is not None:
                x = torch.cat((self.cls_token_c.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token_c.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pos_embed_c
        return self.pos_drop(x)

    def forward_features(self, x, c):

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        # x = self.patch_drop(x)
        x = self.norm_pre(x)
        c = self.patch_embed_c(c)
        c = self._pos_embed_c(c)
        # c = self.patch_drop(c)
        c = self.norm_pre_c(c)
        next_index = self.start_index
        j = 0
        for i in range(len(self.blocks)):
            if i == next_index:
                c = self.side_zero_conv_in[j](c)
                c = self.side_blocks[j](c)
                c = self.side_zero_conv_out[j](c)
                x = self.blocks[i](x) + c
                next_index += self.interval
                j += 1
            else:
                x = self.blocks[i](x)

        x = self.norm(x)
        return x

    def forward(self, x, c):  # noqa
        self.freeze_blocks()

        if (self.training is True) and self.train_transforms_norm is not None:
            x = self.train_transforms_norm(x)
            c = self.train_transforms_norm_c(c)
        elif self.eval_transforms_norm is not None:
            x = self.eval_transforms_norm(x)
            c = self.eval_transforms_norm_c(c)

        x = self.forward_features(x, c)
        x = self.forward_head(x)
        return x

    def freeze_blocks(self, freeze_head=False):
        for param in self.patch_embed.parameters():
            param.requires_grad = False

        self.pos_embed.requires_grad = False

        for param in self.norm_pre.parameters():
            param.requires_grad = False

        for param in self.blocks.parameters():
            param.requires_grad = False

        if freeze_head:
            for param in self.head.parameters():
                param.requires_grad = False


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        RobustVisionTransformer, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs
    )
    return model


@register_model
def robust_vit_base_patch16_224(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224', pretrained=pretrained, **model_kwargs)
    return model
