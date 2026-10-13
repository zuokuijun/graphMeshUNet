""""
Author: zuo kuijun
data: 2026/5/4
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, BatchNorm
from torch_geometric.nn import graclus, max_pool
from torch_geometric.utils import normalized_cut
from torch_geometric.data import Data
from torch_scatter import scatter_mean, scatter_max

# ══════════════════════════════════════════════════════════════════
# 基础块
# ══════════════════════════════════════════════════════════════════
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GINEBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.conv = GINEConv(mlp, train_eps=True)
        self.bn   = nn.BatchNorm1d(hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        out = self.conv(x, edge_index, edge_attr)
        return F.relu(self.bn(out) + x)


class LearnableLaplacianSmoothing(nn.Module):
    def __init__(self, num_iters=3):
        super().__init__()
        self.num_iters = num_iters
        self.lambdas = nn.Parameter(torch.full((num_iters,), 0.5))

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)
        for i in range(self.num_iters):
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])
            lam = torch.sigmoid(self.lambdas[i])
            x   = x + lam * (agg / deg - x)
        return x


class LearnableLaplacianSmoothing1(nn.Module):
    def __init__(self, num_iters=3, curvature_pow=2.0):
        super().__init__()
        self.num_iters     = num_iters
        self.curvature_pow = curvature_pow                  # ← 新增，控制凹坑保护强度
        self.lambdas = nn.Parameter(torch.full((num_iters,), 0.5))

    def forward(self, x, edge_index, num_nodes):
        row, col = edge_index
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, col, torch.ones(col.size(0), device=x.device))
        deg = deg.clamp(min=1).unsqueeze(1)

        for i in range(self.num_iters):
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, col.unsqueeze(1).expand(-1, x.size(1)), x[row])
            neighbor_mean = agg / deg                       # [N, F]

            # ── 新增：估计局部曲率 ──────────────────────
            delta     = (x - neighbor_mean).norm(dim=-1, keepdim=True)   # [N, 1]
            d_min     = delta.min()
            d_max     = delta.max().clamp(min=d_min + 1e-8)
            curvature = (delta - d_min) / (d_max - d_min)  # [N, 1]  归一化到 [0,1]

            # 曲率高（凹坑边缘）→ suppress 趋近 0，少平滑
            # 曲率低（平坦区域）→ suppress 趋近 1，正常平滑
            suppress  = (1.0 - curvature).clamp(min=0) ** self.curvature_pow

            # ── 原来一行，乘上 suppress 即可 ────────────
            lam = torch.sigmoid(self.lambdas[i])
            x   = x + lam * suppress * (neighbor_mean - x) # ← 唯一改动

        return x

class AdaptiveLaplacianSmoothing(nn.Module):
    def __init__(self, num_iters=3, k_init=10.0):
        """
        Args:
            num_iters: 迭代次数。
            k_init: 控制对几何特征的保护强度。k 越大，对大曲率区域（凹坑边缘）的保护越强。
        """
        super().__init__()
        self.num_iters = num_iters
        # 可学习的平滑基础强度
        self.lambdas = nn.Parameter(torch.full((num_iters,), 0.5))
        # 可学习的敏感度系数 k (使用 softplus 保证其始终为正)
        self.k = nn.Parameter(torch.tensor([k_init]))

    def forward(self, x, edge_index):
        """
        x: 顶点坐标 (N, 3)
        edge_index: 边索引 (2, E)
        """
        row, col = edge_index
        num_nodes = x.size(0)
        device = x.device

        # 1. 预先计算度矩阵 (Degree)，即每个点有多少个邻居
        # 这里假设 edge_index 在训练过程中拓扑不变
        with torch.no_grad():
            deg = torch.zeros(num_nodes, device=device)
            # col 指向中心点，统计每个中心点出现了多少次
            deg.scatter_add_(0, col, torch.ones(col.size(0), device=device))
            deg = deg.clamp(min=1).unsqueeze(1)  # 防止除零 (N, 1)

        # 2. 迭代平滑
        for i in range(self.num_iters):
            # 聚合邻居坐标: 计算 \sum x_neighbor
            # 我们将 row (邻居) 的坐标加到 col (中心) 的位置上
            agg = torch.zeros_like(x)
            # 这里的 index 需要扩展到 (E, 3) 维度
            idx_expanded = col.unsqueeze(1).expand(-1, 3)
            agg.scatter_add_(0, idx_expanded, x[row])

            # 计算局部拉普拉斯偏移量: (邻居平均值 - 当前值)
            delta = (agg / deg) - x  # (N, 3)

            # --- 核心自适应逻辑 ---
            # 计算偏移量的模长，代表局部曲率突变的程度
            variation = torch.norm(delta, p=2, dim=-1, keepdim=True)  # (N, 1)

            # 使用指数函数进行衰减保护：
            # 当 variation 很小时（平滑区域），exp(-k*v) 接近 1，执行正常平滑
            # 当 variation 很大时（凹坑边缘），exp(-k*v) 趋近 0，平滑被关闭
            k_val = torch.nn.functional.softplus(self.k)
            adaptive_mask = torch.exp(-variation * k_val)

            # 结合可学习的基础平滑系数
            lam = torch.sigmoid(self.lambdas[i])

            # 更新坐标
            x = x + lam * adaptive_mask * delta

        return x
# ══════════════════════════════════════════════════════════════════
# 池化工具（修复 cluster 索引越界）
# ══════════════════════════════════════════════════════════════════

def pool_graph(x, edge_index, edge_attr, batch):
    """
    graclus 池化 + 粗化图重建

    关键修复：
        graclus 返回的 cluster 可能不连续，如 [0,2,2,5,5,...]
        必须用 torch.unique 重映射为连续索引 [0,1,1,2,2,...]
        否则 x_coarse[cluster] 必然越界
    """
    num_nodes = x.size(0)
    device    = x.device

    # ① graclus 聚类
    weight  = normalized_cut(
        edge_index,
        torch.ones(edge_index.size(1), device=device),
        num_nodes=num_nodes,
    )
    cluster_raw = graclus(edge_index, weight, num_nodes=num_nodes)

    # ② 将 cluster 重映射为连续整数索引
    _, cluster = torch.unique(cluster_raw, return_inverse=True)
    # 现在 cluster 的最大值 == 粗化节点数 - 1，不再越界

    num_coarse = int(cluster.max().item()) + 1

    # ③ 粗化节点特征：取簇内均值（也可用 scatter_max）
    x_coarse = scatter_mean(x, cluster, dim=0, dim_size=num_coarse)  # (N_c, D)

    # ④ 粗化 batch：取簇内众数（用均值取整即可，同一 batch 同一簇）
    batch_coarse = scatter_mean(
        batch.float(), cluster, dim=0, dim_size=num_coarse
    ).long()                                                          # (N_c,)

    # ⑤ 粗化 edge_index：把细图的边映射到粗节点，去自环，去重
    row_c = cluster[edge_index[0]]
    col_c = cluster[edge_index[1]]
    mask  = row_c != col_c                    # 去除自环（同簇内部边）

    if mask.sum() > 0:
        row_c_m = row_c[mask]
        col_c_m = col_c[mask]
        ea_m    = edge_attr[mask]             # (E', D)

        # 用线性索引唯一化粗化边，并对边特征取均值
        edge_id       = row_c_m * num_coarse + col_c_m
        unique_ids, inv = torch.unique(edge_id, return_inverse=True)

        num_unique    = unique_ids.size(0)
        ea_coarse     = torch.zeros(num_unique, ea_m.size(1), device=device)
        ea_coarse.scatter_add_(
            0, inv.unsqueeze(1).expand(-1, ea_m.size(1)), ea_m
        )
        count = torch.zeros(num_unique, 1, device=device)
        count.scatter_add_(0, inv.unsqueeze(1),
                           torch.ones(inv.size(0), 1, device=device))
        ea_coarse = ea_coarse / count.clamp(min=1)

        new_row = unique_ids // num_coarse
        new_col = unique_ids %  num_coarse
        ei_coarse = torch.stack([new_row, new_col], dim=0)            # (2, E_c)

    else:
        # 极端情况：所有节点合并进一个超节点
        ei_coarse = torch.zeros(2, 0, dtype=torch.long, device=device)
        ea_coarse = torch.zeros(0, edge_attr.size(1), device=device)

    return x_coarse, ei_coarse, ea_coarse, batch_coarse, cluster


# ══════════════════════════════════════════════════════════════════
# 完整图 U-Net
# ══════════════════════════════════════════════════════════════════

class MeshUNet(nn.Module):
    """
    输入 (N, node_in_dim) / (E, edge_in_dim)
        │
    node_embed / edge_embed  →  hidden_dim
        │
    Encoder-0: 2×GINEBlock → skip[0] → pool_graph → L0
    Encoder-1: 2×GINEBlock → skip[1] → pool_graph → L1
    Encoder-2: 2×GINEBlock → skip[2] → pool_graph → L2
                                                      │
                         Bottleneck: 4×GINEBlock ─────┘
                                                      │
    Decoder-2: cluster[2].upsample + cat(skip[2]) + 2×GINEBlock
    Decoder-1: cluster[1].upsample + cat(skip[1]) + 2×GINEBlock
    Decoder-0: cluster[0].upsample + cat(skip[0]) + 2×GINEBlock
        │
    LearnableLaplacianSmoothing
        │
    classifier → (N, field_out_dim)
    """

    def __init__(self,
                 node_in_dim:   int,
                 edge_in_dim:   int,
                 hidden_dim:    int = 128,
                 field_out_dim: int = 3,
                 depth:         int = 3):
        super().__init__()
        self.depth = depth

        # ── 嵌入 ─────────────────── ───────────────────────────────
        self.node_embedding = nn.Sequential(
            nn.Linear(node_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.aerodynamics = nn.Linear(1, hidden_dim)
        self.aerodynamics_encoder = nn.Linear(hidden_dim*2, hidden_dim)

        # edge_in_dim → hidden_dim；池化后边特征始终保持 hidden_dim
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim)

        # ── 编码器 ─────────────────────────────────────────────────
        self.enc_convs = nn.ModuleList([
            nn.ModuleList([GINEBlock(hidden_dim), GINEBlock(hidden_dim)])
            for _ in range(depth)
        ])

        # ── 瓶颈 ───────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList(
            [GINEBlock(hidden_dim) for _ in range(4)]
        )

        # ── 解码器 ─────────────────────────────────────────────────
        self.dec_projs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim) for _ in range(depth)
        ])
        self.dec_convs = nn.ModuleList([
            nn.ModuleList([GINEBlock(hidden_dim), GINEBlock(hidden_dim)])
            for _ in range(depth)
        ])

        # ── 输出 ───────────────────────────────────────────────────
        self.smooth = LearnableLaplacianSmoothing(num_iters=3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, field_out_dim),
        )

    def encoder(self, x, edge_index, edge_attr, batch):
        # ② 编码
        skips, ei_list, ea_list, batch_list, cluster_list = [], [], [], [], []

        cur_x, cur_ei, cur_ea, cur_batch = x, edge_index, edge_attr, batch

        for i in range(self.depth):
            cur_x = self.enc_convs[i][0](cur_x, cur_ei, cur_ea)
            cur_x = self.enc_convs[i][1](cur_x, cur_ei, cur_ea)

            # 保存细粒度信息
            skips.append(cur_x)
            ei_list.append(cur_ei)
            ea_list.append(cur_ea)
            batch_list.append(cur_batch)

            # 下采样
            cur_x, cur_ei, cur_ea, cur_batch, cluster = pool_graph(
                cur_x, cur_ei, cur_ea, cur_batch
            )
            cluster_list.append(cluster)  # cluster: 细索引 → 粗索引
        return skips, ei_list, ea_list, batch_list, cluster_list, cur_x, cur_ei, cur_ea

    def decoder(self, cur_x, cluster_list, skips, ei_list, ea_list):
        # ④ 解码
        for i in range(self.depth - 1, -1, -1):
            cluster = cluster_list[i]  # (N_fine,) → 粗节点索引


            x_up = cur_x[cluster]  # (N_fine, hidden)
            x_cat = torch.cat([x_up, skips[i]], dim=-1)
            cur_x = F.relu(self.dec_projs[i](x_cat))

            cur_x = self.dec_convs[i][0](cur_x, ei_list[i], ea_list[i])
            cur_x = self.dec_convs[i][1](cur_x, ei_list[i], ea_list[i])
        return cur_x

    def forward(self, x, edge_index, edge_attr, cd, batch=None):
        N = x.size(0)
        if batch is None:
            batch = torch.zeros(N, dtype=torch.long, device=x.device)

        # ① 嵌入
        x         = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)   # 统一到 hidden_dim

        skips, ei_list, ea_list, batch_list, cluster_list, cur_x, cur_ei, cur_ea = self.encoder(x, edge_index, edge_attr, batch)

        # ③ 瓶颈
        for btn in self.bottleneck:
            cur_x = btn(cur_x, cur_ei, cur_ea)


        ###################### 阻力系数编码 ###########################
        cd_encoding = self.aerodynamics(cd)
        cd_vector = cd_encoding.unsqueeze(0).expand(cur_x.size(0), cd_encoding.size(-1))
        #############################################################
        out = torch.cat([cur_x, cd_vector], dim=-1)

        out = self.aerodynamics_encoder(out)      #(212, 128)

        print(out.shape)
        out = self.decoder(out, cluster_list, skips, ei_list, ea_list)

        # out = self.smooth(out, edge_index, N)
        smooth = LearnableLaplacianSmoothing1().to(device)
        smooth_out = smooth(out, edge_index, out.shape[0])
        # self.classifier(out)
        return self.classifier(smooth_out)

    # # 推理： 通过调节阻力系数生成新的点云
    # @torch.no_grad()
    # def generate(self, x, edge_index, edge_attr, batch):
    #     self.encoder(x, edge_index, edge_attr, batch)


# ══════════════════════════════════════════════════════════════════
# 实例化
# ══════════════════════════════════════════════════════════════════
#
# if __name__ == "__main__":
#     N = 240_000
#     E = 1_440_000
#
#     x          = torch.randn(N, 3)
#     edge_index = torch.randint(0, N, (2, E))
#     edge_attr  = torch.randn(E, 4)    # 边特征维度 = 4
#     cd = torch.randn(1)
#     print(cd)
#     model = MeshUNet(
#         node_in_dim   = 3,
#         edge_in_dim   = 4,   # ← 与 edge_attr.shape[1] 一致
#         hidden_dim    = 128,
#         field_out_dim = 3,
#         depth         = 3,
#     )
#
#     out = model(x, edge_index, edge_attr, cd)
#     print(f"输出: {out.shape}")   # (240000, 3)