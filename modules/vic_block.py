import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DenseDilatedKnnGraph, BasicConv, batched_index_select
from timm.models.layers import DropPath

class ViCBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        kernel_size=9,
        dilation=1,
        act='relu',
        norm=None,
        bias=True,
        stochastic=False,
        epsilon=0.0,
        r=1,
        n=196,
        drop_path=0.0,
        relative_pos=False,
        num_hyperedges=64,
        laplacian_alpha=0.1,
        hypergraph_k=9,
        use_structure_induction=True,
        e2v_ratio=2.0,

        hyperedge_dropout=False,
        e_drop_rate=0.2,
    ):
        super(ViCBlock, self).__init__()
        self.n = n
        self.r = r
        self.num_hyperedges = num_hyperedges
        self.hypergraph_k = hypergraph_k
        self.e2v_ratio = e2v_ratio

        self.hyperedge_dropout = hyperedge_dropout
        self.e_drop_rate = float(e_drop_rate)

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.dilated_knn_graph = DenseDilatedKnnGraph(kernel_size, dilation, stochastic, epsilon)

        self.laplacian_alpha = laplacian_alpha
        self.use_structure_induction = use_structure_induction

        self.hyperedge_dim = in_channels
        self.scale = 1.0 / (self.hyperedge_dim ** 0.5)

        self.hyperedge_embeddings = nn.Parameter(torch.zeros(num_hyperedges, self.hyperedge_dim))
        nn.init.normal_(self.hyperedge_embeddings, mean=0.0, std=0.02)
        self.hyperedge_embeddings.data = F.normalize(self.hyperedge_embeddings.data, p=2, dim=1)

        self.fc_hg = BasicConv([in_channels * 2, in_channels], act, norm, bias)
        self.residual_alpha = nn.Parameter(torch.tensor(0.0))

        self.fc2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, stride=1, padding=0),
            nn.BatchNorm2d(in_channels),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.relative_pos = None
        if relative_pos:
            from .utils import get_2d_relative_pos_embed
            import numpy as np
            relative_pos_tensor = torch.from_numpy(
                np.float32(get_2d_relative_pos_embed(in_channels, int(n ** 0.5)))
            ).unsqueeze(0).unsqueeze(1)
            relative_pos_tensor = F.interpolate(
                relative_pos_tensor, size=(n, n // (r * r)), mode='bicubic', align_corners=False
            )
            self.relative_pos = nn.Parameter(-relative_pos_tensor.squeeze(1), requires_grad=False)

    def _get_relative_pos(self, relative_pos, H, W):
        if relative_pos is None or H * W == self.n:
            return relative_pos
        N = H * W
        N_reduced = N // (self.r * self.r)
        return F.interpolate(relative_pos.unsqueeze(0), size=(N, N_reduced), mode="bicubic").squeeze(0)

    @staticmethod
    def laplacian(x, x_neighbors, alpha):
        x_agg = torch.mean(x_neighbors, dim=-1, keepdim=True)
        return (1.0 - alpha) * x + alpha * x_agg

    @staticmethod
    def _make_keep_mask(shape, drop_rate, device):
        if drop_rate <= 0.0:
            return torch.ones(shape, device=device, dtype=torch.bool)

        mask = (torch.rand(shape, device=device) > drop_rate)

        all_dropped = ~mask.any(dim=-1)
        if all_dropped.any():
            K = shape[-1]

            rand_keep = torch.randint(0, K, all_dropped.shape, device=device)
            mask2 = mask.reshape(-1, K)
            all2 = all_dropped.reshape(-1)
            rand2 = rand_keep.reshape(-1)
            rows = torch.nonzero(all2, as_tuple=False).squeeze(1)
            if rows.numel() > 0:
                mask2[rows, rand2[rows]] = True
        return mask

    def e2v_mapping(self, X_g):
        B, N, C = X_g.shape
        E = F.normalize(self.hyperedge_embeddings, p=2, dim=1)
        A = X_g @ E.t() * self.scale

        k = min(self.hypergraph_k, N)
        A_t = A.transpose(1, 2)
        topk_values, topk_indices = torch.topk(A_t, k=k, dim=2, sorted=False)
        return topk_indices, topk_values

    def v2e_mapping(self, X, hyperedge_features):
        B, N, C = X.shape
        M = hyperedge_features.shape[1]

        hyperedge_features_norm = F.normalize(hyperedge_features, p=2, dim=2)
        A = X @ hyperedge_features_norm.transpose(1, 2) * self.scale

        k_v2e = max(1, int(self.hypergraph_k / self.e2v_ratio))
        k_v2e = min(k_v2e, M)
        _, v2e_indices = torch.topk(A, k=k_v2e, dim=2, sorted=False)
        return v2e_indices

    def v2e_aggregate(self, X, topk_indices, topk_values):
        B, N, C = X.shape
        M, k = topk_indices.shape[1], topk_indices.shape[2]

        if self.hyperedge_dropout and self.training and (0.0 < self.e_drop_rate < 1.0):
            mask = self._make_keep_mask((B, M, k), self.e_drop_rate, topk_values.device)
            masked_logits = topk_values.masked_fill(~mask, float('-inf'))

            topk_weights = F.softmax(masked_logits, dim=2)
        else:
            topk_weights = F.softmax(topk_values, dim=2)

        Mk = topk_indices.numel() // B
        idx = topk_indices.reshape(B, Mk).unsqueeze(-1).expand(B, Mk, C)
        selected_v = X.gather(dim=1, index=idx).reshape(B, M, k, C)

        hyperedge_features = torch.sum(selected_v * topk_weights.unsqueeze(-1), dim=2)
        return hyperedge_features

    @staticmethod
    def v2v_diff(x_reshaped, x_neighbors, k_g1):
        x_i = x_reshaped.expand(-1, -1, -1, k_g1)
        max_rel, _ = torch.max(x_neighbors - x_i, dim=-1, keepdim=True)
        return max_rel

    def v2e_diff(self, x_vertices_2d, hyperedge_features, v2e_indices):
        B, N, C = x_vertices_2d.shape
        k_v2e = v2e_indices.shape[2]

        Nk = v2e_indices.numel() // B
        idx_expand = v2e_indices.reshape(B, Nk).unsqueeze(-1).expand(B, Nk, C)
        connected_hyper = hyperedge_features.gather(dim=1, index=idx_expand).reshape(B, N, k_v2e, C)

        rel = connected_hyper - x_vertices_2d.unsqueeze(2)

        if self.hyperedge_dropout and self.training and (0.0 < self.e_drop_rate < 1.0):
            mask = self._make_keep_mask((B, N, k_v2e), self.e_drop_rate, rel.device)

            rel = rel.masked_fill(~mask.unsqueeze(-1), float('-inf'))

        max_rel, _ = torch.max(rel, dim=2)
        max_rel = max_rel.transpose(1, 2).unsqueeze(-1)
        return max_rel

    def forward(self, x):
        _tmp = x
        B, _, H, W = x.shape

        x = self.fc1(x)
        B, C, H, W = x.shape
        relative_pos = self._get_relative_pos(self.relative_pos, H, W)

        N = H * W
        x_reshaped = x.reshape(B, C, N, 1).contiguous()

        y = None
        if self.r > 1:
            y = F.avg_pool2d(x, self.r, self.r).reshape(B, C, -1, 1).contiguous()

        edge_index_G1 = self.dilated_knn_graph(x_reshaped, y, relative_pos)

        if y is not None:
            x_neighbors_G1 = batched_index_select(y, edge_index_G1[0])
        else:
            x_neighbors_G1 = batched_index_select(x_reshaped, edge_index_G1[0])

        x_reshaped_2d = x_reshaped.reshape(B, C, N).transpose(1, 2).contiguous()

        if self.use_structure_induction:
            x_g = self.laplacian(x_reshaped, x_neighbors_G1, self.laplacian_alpha).reshape(B, C, N).transpose(1, 2).contiguous()
        else:
            x_g = x_reshaped_2d

        topk_indices, topk_values = self.e2v_mapping(x_g)

        hyperedge_features = self.v2e_aggregate(x_reshaped_2d, topk_indices, topk_values)

        v2e_indices = self.v2e_mapping(x_g, hyperedge_features)

        k_g1 = edge_index_G1.shape[-1]
        max_rel_v2v = self.v2v_diff(x_reshaped, x_neighbors_G1, k_g1)

        max_rel_v2e = self.v2e_diff(x_reshaped_2d, hyperedge_features, v2e_indices)

        max_rel = torch.max(max_rel_v2v, max_rel_v2e)

        x_concat = torch.cat([x_reshaped, max_rel], dim=1)
        x_conv = self.fc_hg(x_concat)

        alpha = torch.sigmoid(self.residual_alpha)
        x_conv = (1.0 - alpha) * x_reshaped + alpha * x_conv
        x_conv = x_conv.reshape(B, C, H, W).contiguous()

        x_out = self.fc2(x_conv)
        x_out = _tmp + self.drop_path(x_out)
        return x_out
