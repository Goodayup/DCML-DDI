import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList
from torch_geometric.nn import GATConv, SAGPooling, global_add_pool
from torch.nn import LayerNorm

from layers import IntraGraphAttention, InterGraphAttention, CoAttentionLayer, RESCAL


class MVN_DDI_Block(nn.Module):
    def __init__(self, n_heads, in_features, head_out_feats):
        super().__init__()
        self.out_dim = head_out_feats * n_heads

        self.feature_conv = GATConv(in_features, head_out_feats, n_heads)
        self.post_gat_norm = LayerNorm(self.out_dim)
        self.dropout = nn.Dropout(p=0.3)  

        self.intraAtt = IntraGraphAttention(self.out_dim, out_dim=self.out_dim, heads=1)
        self.interAtt = InterGraphAttention(self.out_dim, out_dim=self.out_dim, heads=1)

        self.readout = SAGPooling(self.out_dim, min_score=-1)
        self.reduce_fc = nn.Linear(self.out_dim * 2, self.out_dim)

    def forward(self, h_data, t_data, b_graph):
        h_data.x = self.feature_conv(h_data.x, h_data.edge_index)
        h_data.x = self.dropout(self.post_gat_norm(h_data.x))  

        t_data.x = self.feature_conv(t_data.x, t_data.edge_index)
        t_data.x = self.dropout(self.post_gat_norm(t_data.x))  

        h_intra = self.intraAtt(h_data)
        t_intra = self.intraAtt(t_data)
        h_inter, t_inter = self.interAtt(h_data, t_data, b_graph)

        h_fused = self.reduce_fc(torch.cat([h_intra, h_inter], dim=1))
        t_fused = self.reduce_fc(torch.cat([t_intra, t_inter], dim=1))

        h_data.x = h_fused
        t_data.x = t_fused

        h_x, _, _, h_batch, _, _ = self.readout(h_data.x, h_data.edge_index, batch=h_data.batch, edge_attr=None)
        t_x, _, _, t_batch, _, _ = self.readout(t_data.x, t_data.edge_index, batch=t_data.batch, edge_attr=None)

        h_graph_emb = global_add_pool(h_x, h_batch)
        t_graph_emb = global_add_pool(t_x, t_batch)

        return h_data, t_data, h_graph_emb, t_graph_emb, h_intra, t_intra, h_inter, t_inter



class ContrastiveModel(nn.Module):
    def __init__(self, in_dim, projection_dim=128, temperature=0.5, dropout=0.3):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

        '''self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, projection_dim)
        )'''
        self.projector = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
)


    def forward(self, h_intra, t_intra, h_inter, t_inter):
        h_intra_proj = F.normalize(self.projector(self.dropout(h_intra)), dim=-1)
        t_intra_proj = F.normalize(self.projector(self.dropout(t_intra)), dim=-1)
        h_inter_proj = F.normalize(self.projector(self.dropout(h_inter)), dim=-1)
        t_inter_proj = F.normalize(self.projector(self.dropout(t_inter)), dim=-1)

        loss_h = self.nt_xent_loss(h_intra_proj, h_inter_proj)
        loss_t = self.nt_xent_loss(t_intra_proj, t_inter_proj)
        contrastive_loss = (loss_h + loss_t) / 2
        return contrastive_loss, h_intra_proj, t_intra_proj, h_inter_proj, t_inter_proj

    def nt_xent_loss(self, z1, z2):
        batch_size = z1.size(0)
        temperature = self.temperature
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, dim=1)
        sim_matrix = torch.matmul(z, z.T) / temperature

        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        #mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        #sim_matrix = sim_matrix.masked_fill(mask, -1e9)
        sim_matrix = sim_matrix - torch.eye(2 * batch_size, device=z.device) * 1e9


        loss = F.cross_entropy(sim_matrix, labels)
        return loss
class GatedFusion(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GatedFusion, self).__init__()
        self.gate = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )
        self.output_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x1, x2):
        gate_value = self.gate(torch.cat([x1, x2], dim=-1))
        fused = gate_value * x1 + (1 - gate_value) * x2
        return self.output_proj(fused)

class MVN_DDI(nn.Module):
    def __init__(self, in_features, hidd_dim, kge_dim, rel_total, heads_out_feat_params, blocks_params, projection_dim=128):
        super().__init__()
        self.initial_norm = LayerNorm(in_features)
        self.blocks = ModuleList()
        self.net_norms = ModuleList()
        self.block_dropout = nn.Dropout(p=0.3)


        for head_out_feats, n_heads in zip(heads_out_feat_params, blocks_params):
            block = MVN_DDI_Block(n_heads, in_features, head_out_feats)
            self.blocks.append(block)
            self.net_norms.append(LayerNorm(block.out_dim))
            in_features = block.out_dim

        self.fusion_mlp = nn.Sequential(
            nn.Linear(in_features + projection_dim, hidd_dim),
            nn.LayerNorm(hidd_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidd_dim, hidd_dim),
            nn.LayerNorm(hidd_dim),
            nn.ReLU()
        )

        self.kge_dim = kge_dim
        self.co_attention = CoAttentionLayer(n_features=self.kge_dim)
        self.KGE = RESCAL(rel_total, kge_dim)

        self.use_contrastive = True
        self.contrastive_model = ContrastiveModel(in_dim=in_features, projection_dim=projection_dim)

        self.contrastive_fusion_mlp = nn.Sequential(
            nn.Linear(projection_dim * 2, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU()
        )

    def forward(self, triples):
        h_data, t_data, rels, b_graph = triples

        h_data.x = self.initial_norm(h_data.x)
        t_data.x = self.initial_norm(t_data.x)

        pooled_h_list, pooled_t_list = [], []

        for i, block in enumerate(self.blocks):
            h_data, t_data, h_pool, t_pool, h_intra, t_intra, h_inter, t_inter = block(h_data, t_data, b_graph)

            h_data.x = F.elu(self.net_norms[i](h_data.x))
            t_data.x = F.elu(self.net_norms[i](t_data.x))

            h_data.x = self.block_dropout(h_data.x)
            t_data.x = self.block_dropout(t_data.x)

            pooled_h_list.append(h_pool)
            pooled_t_list.append(t_pool)

        pooled_h = torch.mean(torch.stack(pooled_h_list, dim=0), dim=0)
        pooled_t = torch.mean(torch.stack(pooled_t_list, dim=0), dim=0)

        contrastive_loss = torch.tensor(0.0, device=h_data.x.device)
        if self.use_contrastive:
            contrastive_loss, h_intra_proj, t_intra_proj, h_inter_proj, t_inter_proj = self.contrastive_model(
                h_intra, t_intra, h_inter, t_inter)

            h_intra_proj = global_add_pool(h_intra_proj, h_data.batch)
            h_inter_proj = global_add_pool(h_inter_proj, h_data.batch)
            t_intra_proj = global_add_pool(t_intra_proj, t_data.batch)
            t_inter_proj = global_add_pool(t_inter_proj, t_data.batch)

            h_contrastive = self.contrastive_fusion_mlp(torch.cat([h_intra_proj, h_inter_proj], dim=-1))
            t_contrastive = self.contrastive_fusion_mlp(torch.cat([t_intra_proj, t_inter_proj], dim=-1))
        else:
            batch_size = pooled_h.size(0)
            h_contrastive = t_contrastive = torch.zeros(batch_size, 128).to(pooled_h.device)

        fused_h = self.fusion_mlp(torch.cat([pooled_h, h_contrastive], dim=-1))
        fused_t = self.fusion_mlp(torch.cat([pooled_t, t_contrastive], dim=-1))
        

        fused_h = F.layer_norm(fused_h, fused_h.shape[1:])
        fused_t = F.layer_norm(fused_t, fused_t.shape[1:])

        repr_h = fused_h
        repr_t = fused_t

        attentions = self.co_attention(repr_h, repr_t)
        scores = self.KGE(repr_h, repr_t, rels, attentions)

        return scores, contrastive_loss
