import torch
import torch.nn as nn
from omegaconf import DictConfig

from models.base import Model

from models.modules import PositionalEncoding, TimestepEmbedder
from models.modules import SceneMapEncoderDecoder, SceneMapEncoder
from models.functions import load_and_freeze_clip_model, encode_text_clip, \
    load_and_freeze_bert_model, encode_text_bert, get_lang_feat_dim_type
from utils.misc import compute_repr_dimesion

from models.trick.mamba_block import *
# from models.trick.mamba_cross import *
# from models.trick.mamba_block_AdaLN import *
# from models.trick.rwkv_block import BidirectionalRWKVBlock
from models.trick.dit_block import DiTBlock, DiTBlockWithCrossAttention, ConditionEmbedder, FinalLayer


@Model.register()
class CMDM(nn.Module):

    def __init__(self, cfg: DictConfig, *args, **kwargs):
        super().__init__()
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'

        self.motion_type = cfg.data_repr
        self.motion_dim = cfg.input_feats
        self.latent_dim = cfg.latent_dim
        self.mask_motion = cfg.mask_motion

        self.arch = cfg.arch

        ## time embedding
        self.time_emb_dim = cfg.time_emb_dim
        self.timestep_embedder = TimestepEmbedder(self.latent_dim, self.time_emb_dim, max_len=1000)

        ## contact
        self.contact_type = cfg.contact_model.contact_type
        self.contact_dim = compute_repr_dimesion(self.contact_type)
        self.planes = cfg.contact_model.planes

        if self.arch == 'trans_enc':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_dec':
            SceneMapModule = SceneMapEncoderDecoder
        elif self.arch == 'trans_mamba':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_mamba_cross':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'trans_rwkv':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'dit':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'dit_cross':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        elif self.arch == 'bimamba':
            SceneMapModule = SceneMapEncoder
            self.contact_adapter = nn.Linear(self.planes[-1], self.latent_dim, bias=True)
        else:
            raise NotImplementedError

        self.contact_encoder = SceneMapModule(
            point_feat_dim=self.contact_dim,
            planes=self.planes,
            blocks=cfg.contact_model.blocks,
            num_points=cfg.contact_model.num_points,
        )

        ## text
        self.text_model_name = cfg.text_model.version
        self.text_max_length = cfg.text_model.max_length
        self.text_feat_dim, self.text_feat_type = get_lang_feat_dim_type(self.text_model_name)

        if self.text_feat_type == 'clip':
            self.text_model = load_and_freeze_clip_model(self.text_model_name)
        elif self.text_feat_type == 'bert':
            self.tokenizer, self.text_model = load_and_freeze_bert_model(self.text_model_name)
        else:
            raise NotImplementedError
        self.language_adapter = nn.Linear(self.text_feat_dim, self.latent_dim, bias=True)

        ## model architecture
        self.motion_adapter = nn.Linear(self.motion_dim, self.latent_dim, bias=True)
        self.positional_encoder = PositionalEncoding(self.latent_dim, dropout=0.1, max_len=5000)

        self.num_layers = cfg.num_layers

        # ================== Architecture Building Blocks ==================

        if self.arch == 'trans_enc':
            self.self_attn_layer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=self.latent_dim,
                    nhead=cfg.num_heads,
                    dim_feedforward=cfg.dim_feedforward,
                    dropout=cfg.dropout,
                    activation='gelu',
                    batch_first=True,
                ),
                enable_nested_tensor=False,
                num_layers=sum(cfg.num_layers),
            )

        elif self.arch == 'trans_mamba':
            total_layers = sum(cfg.num_layers)
            # mamba_layers = getattr(cfg, 'mamba_layers', 1)
            mamba_layers = 1

            if mamba_layers > total_layers:
                raise ValueError("cfg.mamba_layers 不能大于总层数")

            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            self.encoder_layers = nn.ModuleList()

            for idx in range(total_layers):
                if idx < total_layers - mamba_layers:
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.latent_dim,
                        nhead=cfg.num_heads,
                        dim_feedforward=cfg.dim_feedforward,
                        dropout=cfg.dropout,
                        activation='gelu',
                        batch_first=True,
                    )
                else:
                    layer = BidirectionalMambaBlock(
                        d_model=self.latent_dim,
                        d_state=getattr(cfg, 'mamba_d_state', 16),
                        d_conv=getattr(cfg, 'mamba_d_conv', 4),
                        expand=getattr(cfg, 'mamba_expand', 2),
                        mlp_ratio=mlp_ratio,
                        drop=cfg.dropout,
                        drop_path=getattr(cfg, 'mamba_drop_path', 0.0),
                    )
                self.encoder_layers.append(layer)

        elif self.arch == 'trans_mamba_cross':
            total_layers = sum(cfg.num_layers)
            mamba_layers = getattr(cfg, 'mamba_layers', 0)

            if mamba_layers > total_layers:
                raise ValueError("cfg.mamba_layers 不能大于总层数")

            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            self.encoder_layers = nn.ModuleList()

            for idx in range(total_layers):
                if idx < total_layers - mamba_layers:
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.latent_dim,
                        nhead=cfg.num_heads,
                        dim_feedforward=cfg.dim_feedforward,
                        dropout=cfg.dropout,
                        activation='gelu',
                        batch_first=True,
                    )
                else:
                    layer = BidirectionalMambaBlock(
                        d_model=self.latent_dim,
                        d_state=getattr(cfg, 'mamba_d_state', 16),
                        d_conv=getattr(cfg, 'mamba_d_conv', 4),
                        expand=getattr(cfg, 'mamba_expand', 2),
                        mlp_ratio=mlp_ratio,
                        drop=cfg.dropout,
                        drop_path=getattr(cfg, 'mamba_drop_path', 0.0),
                    )
                self.encoder_layers.append(layer)

        # -----------------------------------------------------------
        # [BiMamba Branch] Bi-directional Mamba architecture
        # -----------------------------------------------------------
        elif self.arch == 'bimamba':
            total_layers = sum(cfg.num_layers)
            mamba_layers = getattr(cfg, 'mamba_layers', 2)

            if mamba_layers > total_layers:
                raise ValueError("cfg.mamba_layers 不能大于总层数")

            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            self.encoder_layers = nn.ModuleList()

            for idx in range(total_layers):
                if idx < total_layers - mamba_layers:
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.latent_dim,
                        nhead=cfg.num_heads,
                        dim_feedforward=cfg.dim_feedforward,
                        dropout=cfg.dropout,
                        activation='gelu',
                        batch_first=True,
                    )
                else:
                    layer = BidirectionalMambaBlock(
                        d_model=self.latent_dim,
                        d_state=getattr(cfg, 'mamba_d_state', 16),
                        d_conv=getattr(cfg, 'mamba_d_conv', 4),
                        expand=getattr(cfg, 'mamba_expand', 2),
                        mlp_ratio=mlp_ratio,
                        drop=cfg.dropout,
                        drop_path=getattr(cfg, 'mamba_drop_path', 0.05),
                    )
                self.encoder_layers.append(layer)

            print(f"\n{'=' * 20} CMDM Architecture Info {'=' * 20}")
            print(f"Arch: {self.arch}")
            print(f"Total Layers: {len(self.encoder_layers)}")
            for i, layer in enumerate(self.encoder_layers):
                if isinstance(layer, nn.TransformerEncoderLayer):
                    print(f"  Layer {i+1}: TransformerEncoderLayer")
                else:
                    print(f"  Layer {i+1}: BidirectionalMambaBlock")
            print(f"{'=' * 64}\n")

        # -----------------------------------------------------------
        # [RWKV Branch] 正确初始化
        # -----------------------------------------------------------
        elif self.arch == 'trans_rwkv':
            total_layers = sum(cfg.num_layers)
            rwkv_layers = getattr(cfg, 'rwkv_layers', 1)

            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            self.encoder_layers = nn.ModuleList()

            for idx in range(total_layers):
                if idx < total_layers - rwkv_layers:
                    layer = nn.TransformerEncoderLayer(
                        d_model=self.latent_dim,
                        nhead=cfg.num_heads,
                        dim_feedforward=cfg.dim_feedforward,
                        dropout=cfg.dropout,
                        activation='gelu',
                        batch_first=True,
                    )
                else:
                    layer = BidirectionalRWKVBlock(
                        d_model=self.latent_dim,
                        drop_path=getattr(cfg, 'rwkv_drop_path', 0.1),
                        layer_id=idx
                    )
                self.encoder_layers.append(layer)

            print(f"\n{'=' * 20} CMDM Architecture Info {'=' * 20}")
            print(f"Arch: {self.arch}")
            print(f"Total Layers: {len(self.encoder_layers)}")
            print(f"RWKV Layers:  {rwkv_layers} (Bidirectional)")
            print(f"{'=' * 64}\n")

        elif self.arch == 'trans_dec':
            self.self_attn_layers = nn.ModuleList()
            self.kv_mappling_layers = nn.ModuleList()
            self.cross_attn_layers = nn.ModuleList()
            for i, n in enumerate(self.num_layers):
                self.self_attn_layers.append(
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        ),
                        num_layers=n,
                    )
                )

                if i != len(self.num_layers) - 1:
                    self.kv_mappling_layers.append(
                        nn.Sequential(
                            nn.Linear(self.planes[-1 - i], self.latent_dim, bias=True),
                            nn.LayerNorm(self.latent_dim),
                        )
                    )
                    self.cross_attn_layers.append(
                        nn.TransformerDecoderLayer(
                            d_model=self.latent_dim,
                            nhead=cfg.num_heads,
                            dim_feedforward=cfg.dim_feedforward,
                            dropout=cfg.dropout,
                            activation='gelu',
                            batch_first=True,
                        )
                    )



        # -----------------------------------------------------------
        # [DiT· Branch] DiT-style architecture with AdaLN
        # -----------------------------------------------------------
        elif self.arch == 'dit':
            total_layers = sum(cfg.num_layers)
            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            drop_path_rate = getattr(cfg, 'dit_drop_path', 0.1)

            # Condition embedder: fuses time + text + contact into single vector
            self.cond_embedder = ConditionEmbedder(
                input_dim=self.latent_dim,
                output_dim=self.latent_dim,
                use_cross_attn_pooling=getattr(cfg, 'dit_use_cross_attn_pooling', True)
            )

            # DiT blocks with AdaLN
            self.dit_blocks = nn.ModuleList([
                DiTBlock(
                    hidden_size=self.latent_dim,
                    num_heads=cfg.num_heads,
                    cond_dim=self.latent_dim,
                    mlp_ratio=mlp_ratio,
                    drop=cfg.dropout,
                    drop_path=drop_path_rate * (i / (total_layers - 1)) if total_layers > 1 else 0.0
                )
                for i in range(total_layers)
            ])

            # Final layer with AdaLN
            self.final_layer = FinalLayer(
                hidden_size=self.latent_dim,
                output_size=self.motion_dim,
                cond_dim=self.latent_dim
            )

            print(f"\n{'=' * 20} CMDM DiT Architecture Info {'=' * 20}")
            print(f"Arch: {self.arch}")
            print(f"Total DiT Blocks: {total_layers}")
            print(f"Latent Dim: {self.latent_dim}")
            print(f"MLP Ratio: {mlp_ratio}")
            print(f"Drop Path Rate: {drop_path_rate}")
            print(f"{'=' * 64}\n")

        # -----------------------------------------------------------
        # [DiT + Cross-Attention Branch]
        # -----------------------------------------------------------
        elif self.arch == 'dit_cross':
            total_layers = sum(cfg.num_layers)
            mlp_ratio = cfg.dim_feedforward / float(self.latent_dim)
            drop_path_rate = getattr(cfg, 'dit_drop_path', 0.1)

            # Condition embedder for AdaLN (time + text only, contact via cross-attn)
            self.cond_embedder = ConditionEmbedder(
                input_dim=self.latent_dim,
                output_dim=self.latent_dim,
                use_cross_attn_pooling=False  # We'll use full cross-attention instead
            )

            # DiT blocks with both AdaLN and cross-attention to contact
            self.dit_blocks = nn.ModuleList([
                DiTBlockWithCrossAttention(
                    hidden_size=self.latent_dim,
                    num_heads=cfg.num_heads,
                    cond_dim=self.latent_dim,
                    context_dim=self.latent_dim,  # contact features
                    mlp_ratio=mlp_ratio,
                    drop=cfg.dropout,
                    drop_path=drop_path_rate * (i / (total_layers - 1)) if total_layers > 1 else 0.0
                )
                for i in range(total_layers)
            ])

            # Final layer
            self.final_layer = FinalLayer(
                hidden_size=self.latent_dim,
                output_size=self.motion_dim,
                cond_dim=self.latent_dim
            )

            print(f"\n{'=' * 20} CMDM DiT+Cross Architecture Info {'=' * 20}")
            print(f"Arch: {self.arch}")
            print(f"Total DiT Blocks: {total_layers}")
            print(f"With Cross-Attention to Contact Features")
            print(f"{'=' * 64}\n")

        else:
            raise NotImplementedError

        self.motion_layer = nn.Linear(self.latent_dim, self.motion_dim, bias=True)

    def forward(self, x, timesteps, **kwargs):
        """ Forward pass of the model. """

        ## time embedding
        time_emb = self.timestep_embedder(timesteps)  # [bs, 1, latent_dim]
        time_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)

        ## text embedding
        if self.text_feat_type == 'clip':
            text_emb = encode_text_clip(self.text_model, kwargs['c_text'], max_length=self.text_max_length,
                                        device=self.device)
            text_emb = text_emb.unsqueeze(1).detach().float()
            text_mask = torch.zeros((x.shape[0], 1), dtype=torch.bool, device=self.device)
        elif self.text_feat_type == 'bert':
            text_emb, text_mask = encode_text_bert(self.tokenizer, self.text_model, kwargs['c_text'],
                                                   max_length=self.text_max_length, device=self.device)
            text_mask = ~(text_mask.to(torch.bool))  # 0 for valid, 1 for invalid
        else:
            raise NotImplementedError

        if 'c_text_mask' in kwargs:
            text_mask = torch.logical_or(text_mask, kwargs['c_text_mask'].repeat(1, text_mask.shape[1]))
        if 'c_text_erase' in kwargs:
            text_emb = text_emb * (1. - kwargs['c_text_erase'].unsqueeze(-1).float())
        text_emb = self.language_adapter(text_emb)  # [bs, 1, latent_dim]

        ## encode contact
        cont_emb = self.contact_encoder(kwargs['c_pc_xyz'], kwargs['c_pc_contact'])
        if hasattr(self, 'contact_adapter'):  # trans_enc
            cont_mask = torch.zeros((x.shape[0], cont_emb.shape[1]), dtype=torch.bool, device=self.device)
            if 'c_pc_mask' in kwargs:
                cont_mask = torch.logical_or(cont_mask, kwargs['c_pc_mask'].repeat(1, cont_mask.shape[1]))
            if 'c_pc_erase' in kwargs:
                cont_emb = cont_emb * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
            cont_emb = self.contact_adapter(cont_emb)

        ## motion embedding
        x = self.motion_adapter(x)  # [bs, seq_len, latent_dim]

        # ================== Main Forward Logic ==================
        if self.arch in ['trans_enc', 'trans_mamba', 'trans_mamba_cross', 'trans_rwkv', 'bimamba']:
            x = torch.cat([time_emb, text_emb, cont_emb, x], dim=1)
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, cont_mask, kwargs['x_mask']], dim=1)

            # --- Branch: Transformer Only ---
            if self.arch == 'trans_enc':
                x = self.self_attn_layer(x, src_key_padding_mask=x_mask)

            # --- Branch: Mamba + Cross ---
            elif self.arch == 'trans_mamba_cross':
                for layer in self.encoder_layers:
                    if isinstance(layer, BidirectionalMambaBlock):
                        x = layer(
                            x,
                            padding_mask=x_mask,
                            context=text_emb,
                            context_mask=text_mask
                        )
                    elif isinstance(layer, nn.TransformerEncoderLayer):
                        x = layer(x, src_key_padding_mask=x_mask)
                    else:
                        x = layer(x)

            # --- Branch: Mamba Original ---
            elif self.arch == 'trans_mamba':
                for layer in self.encoder_layers:
                    if isinstance(layer, BidirectionalMambaBlock):
                        x = layer(x, padding_mask=x_mask)
                    elif isinstance(layer, nn.TransformerEncoderLayer):
                        x = layer(x, src_key_padding_mask=x_mask)
                    else:
                        x = layer(x)
            # --- Branch: BiMamba (3 Trans + 2 BiMamba) ---
            elif self.arch == 'bimamba':
                for idx, layer in enumerate(self.encoder_layers):
                    if isinstance(layer, BidirectionalMambaBlock):
                        # BiMamba层：接收条件注入
                        x = layer(x, padding_mask=x_mask)
                    elif isinstance(layer, nn.TransformerEncoderLayer):
                        # Transformer层：标准处理
                        x = layer(x, src_key_padding_mask=x_mask)
                    else:
                        x = layer(x)

            # --- Branch: RWKV (With Context for Top-3) ---
            elif self.arch == 'trans_rwkv':
                for layer in self.encoder_layers:
                    if isinstance(layer, BidirectionalRWKVBlock):
                        # [关键] 传入 context 以激活 Cross-Attention
                        x = layer(
                            x,
                            padding_mask=x_mask,
                            context=text_emb,  # [BS, 1, D]
                            context_mask=text_mask  # [BS, 1]
                        )
                    elif isinstance(layer, nn.TransformerEncoderLayer):
                        x = layer(x, src_key_padding_mask=x_mask)
                    else:
                        x = layer(x)

            non_motion_token = time_mask.shape[1] + text_mask.shape[1] + cont_mask.shape[1]
            x = x[:, non_motion_token:, :]

        elif self.arch == 'trans_dec':
            x = torch.cat([time_emb, text_emb, x], dim=1)
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            x_mask = None
            if self.mask_motion:
                x_mask = torch.cat([time_mask, text_mask, kwargs['x_mask']], dim=1)
            for i in range(len(self.num_layers)):
                x = self.self_attn_layers[i](x, src_key_padding_mask=x_mask)
                if i != len(self.num_layers) - 1:
                    mem = cont_emb[i]
                    mem_mask = torch.zeros((x.shape[0], mem.shape[1]), dtype=torch.bool, device=self.device)
                    if 'c_pc_mask' in kwargs:
                        mem_mask = torch.logical_or(mem_mask, kwargs['c_pc_mask'].repeat(1, mem_mask.shape[1]))
                    if 'c_pc_erase' in kwargs:
                        mem = mem * (1. - kwargs['c_pc_erase'].unsqueeze(-1).float())
                    mem = self.kv_mappling_layers[i](mem)
                    x = self.cross_attn_layers[i](x, mem, tgt_key_padding_mask=x_mask, memory_key_padding_mask=mem_mask)

            non_motion_token = time_mask.shape[1] + text_mask.shape[1]
            x = x[:, non_motion_token:, :]

        # -----------------------------------------------------------
        # [DiT Branch] Forward with AdaLN conditioning
        # -----------------------------------------------------------
        elif self.arch == 'dit':
            # Prepare motion with positional encoding
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            # Prepare motion mask
            x_mask = kwargs.get('x_mask', None) if self.mask_motion else None

            # Fuse conditions into single vector: time + text + contact -> cond
            # time_emb: [B, 1, D], text_emb: [B, 1, D], cont_emb: [B, N, D]
            cond = self.cond_embedder(
                time_emb.squeeze(1),  # [B, D]
                text_emb,              # [B, 1, D]
                cont_emb,              # [B, N, D]
                contact_mask=cont_mask if self.mask_motion else None
            )  # [B, D]

            # Apply DiT blocks with AdaLN
            for block in self.dit_blocks:
                x = block(x, cond, padding_mask=x_mask)

            # Final layer (also uses AdaLN)
            x = self.final_layer(x, cond)
            return x  # Skip motion_layer, final_layer already projects to motion_dim

        # -----------------------------------------------------------
        # [DiT + Cross-Attention Branch]
        # -----------------------------------------------------------
        elif self.arch == 'dit_cross':
            # Prepare motion with positional encoding
            x = self.positional_encoder(x.permute(1, 0, 2)).permute(1, 0, 2)

            # Prepare masks
            x_mask = kwargs.get('x_mask', None) if self.mask_motion else None

            # Fuse time + text into condition vector (contact goes to cross-attention)
            cond = self.cond_embedder(
                time_emb.squeeze(1),  # [B, D]
                text_emb,              # [B, 1, D]
                cont_emb,              # [B, N, D] - will be mean-pooled for AdaLN
                contact_mask=cont_mask if self.mask_motion else None
            )  # [B, D]

            # Apply DiT blocks with AdaLN + cross-attention to contact
            for block in self.dit_blocks:
                x = block(
                    x, cond,
                    context=cont_emb,  # Full contact features for cross-attention
                    padding_mask=x_mask,
                    context_mask=cont_mask if self.mask_motion else None
                )

            # Final layer
            x = self.final_layer(x, cond)
            return x  # Skip motion_layer



        else:
            raise NotImplementedError

        x = self.motion_layer(x)
        return x