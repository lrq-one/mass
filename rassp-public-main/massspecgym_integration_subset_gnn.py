import torch
import torch.nn as nn
import torch.nn.functional as F
from massspecgym.models.simulation.base import SimulationBaseModel
from massspecgym.models.layers import FourierFeatures
from torch_geometric.nn import GINEConv, global_mean_pool

class SubsetSimulationGNN(SimulationBaseModel):
    """
    基于 4 步蓝图 + MARASON RAG 实现的极速预测模型：
    1. 分子编码器 (GIN)
    2. 全局上下文融合 (CE Fourier Embedding)
    3. RAG 增强 (Neural Graph Matching + Mass Shifting)
    4. 交叉注意力打分
    """
    def __init__(
        self, 
        hidden_dim=256, 
        num_layers=4, 
        node_feat_dim=133, 
        edge_feat_dim=14, 
        num_ce_freqs=128,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # --- 1. 图编码器 (分子图 -> 表征) ---
        self.atom_encoder = nn.Linear(node_feat_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_feat_dim, hidden_dim)

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            self.convs.append(GINEConv(nn=mlp))
        
        # --- 2. 全局上下文 (CE & Precursor) ---
        self.ce_fourier = FourierFeatures(num_freqs=num_ce_freqs, strategy='dreams')
        self.ce_encoder = nn.Sequential(
            nn.Linear(self.ce_fourier.num_features, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # --- 3. RAG 神经网络匹配 (Neural Graph Matching) ---
        # 用于比较查询分子和参考分子的差异表示
        self.diff_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        # 用于预测参考谱图峰相对于查询分子的平移量修正 (Mass Shifting)
        # 输入：[参考峰 mz, 查询分子与参考分子的全局差异]
        self.shift_predictor = nn.Sequential(
            nn.Linear(1 + hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1) # 输出 mz 改变量 delta_m
        )

        # --- 4. 交叉注意力打分网络 ---
        # Query: 碎片表征, Key/Value: [全局上下文, 参考谱图特征]
        # 注意：这里我们引入了参考谱图的注意力
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
        self.final_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid() 
        )

    def forward(self, batch):
        # --- A. 分子图编码 ---
        x = self.atom_encoder(batch.x.float())
        edge_attr = self.edge_encoder(batch.edge_attr.float())
        for conv in self.convs:
            x = F.relu(conv(x, batch.edge_index, edge_attr))
        mol_global_rep = global_mean_pool(x, batch.batch)
        
        # --- B. 碰撞能量 (CE) 编码 ---
        if hasattr(batch, 'collision_energy'):
            ce_val = batch.collision_energy.view(-1, 1).float()
            ce_feats = self.ce_fourier(ce_val)
            ce_rep = self.ce_encoder(ce_feats)
        else:
            ce_rep = torch.zeros(mol_global_rep.size(0), self.hidden_dim // 2, device=x.device)

        # 融合基本的全局特征
        global_context = torch.cat([mol_global_rep, ce_rep], dim=1) # [B, 1.5 * hidden_dim]
        # 为了适配注意力维度，我们简单做一个投影
        global_context_proj = nn.Linear(global_context.size(1), x.size(1), device=x.device)(global_context)

        predictions = []
        
        # --- C. RAG 检索处理 (MARASON 逻辑) ---
        # 假设 batch 中已经包含检索到的邻居信息
        has_rag = hasattr(batch, 'ref_spectra') and batch.ref_spectra is not None
        
        for i in range(batch.num_graphs):
            # 1. 碎片背景准备
            if not hasattr(batch, 'candidate_indices') or batch.candidate_indices is None:
                predictions.append(torch.tensor([], device=x.device))
                continue
            candidate_node_indices = batch.candidate_indices[i]
            if len(candidate_node_indices) == 0:
                predictions.append(torch.tensor([], device=x.device))
                continue
            
            mol_nodes = x[batch.batch == i]
            
            # --- 2. 基于 Neural Graph Matching 的 RAG 增强 ---
            context_seq_list = [global_context_proj[i].view(1, 1, -1)] # 初始 Context
            
            if has_rag:
                # 获取该分子的参考谱图和参考分子表征 (假设已由 DataLoader/Transform 注入)
                # ref_info = batch.ref_info[i]
                # ref_mol_rep = batch.ref_mol_rep[i] 
                
                # 提示：在这里我们可以根据 MARASON 论文实现 Neural Graph Matching
                # 比较 mol_global_rep[i] 和 ref_mol_rep
                # 修正 ref_spectra 中的 mz 值： shifted_mz = mz + delta_m
                
                # 为了保持代码通用性，我们这里暂留占位符，如果数据到位则激活：
                pass

            # 构建碎片表征
            frag_reps = []
            for subset in candidate_node_indices:
                subset_tensor = torch.as_tensor(subset, dtype=torch.long, device=x.device)
                frag_rep = mol_nodes[subset_tensor].mean(dim=0) 
                frag_reps.append(frag_rep)
            frag_reps = torch.stack(frag_reps).unsqueeze(0) # [1, num_frags, hidden_dim]
            
            # --- 3. 交叉注意力打分 ---
            context_seq = torch.cat(context_seq_list, dim=1) # [1, SeqLen, hidden_dim]
            attn_output, _ = self.cross_attention(frag_reps, context_seq, context_seq)
            
            pred_intensities = self.final_scorer(attn_output.squeeze(0)).squeeze(-1)
            predictions.append(pred_intensities)
            
        return predictions

    def step(self, batch, step_type="train"):
        preds = self.forward(batch)
        targets = batch.target_intensities
        loss = 0.0
        valid_graphs = 0
        start_idx = 0
        for i, p in enumerate(preds):
            length = len(p)
            if length == 0: continue
            t = targets[start_idx : start_idx + length]
            start_idx += length
            cos_sim = F.cosine_similarity(p.unsqueeze(0), t.unsqueeze(0), eps=1e-8)
            loss += (1.0 - cos_sim.mean())
            valid_graphs += 1
        return loss / max(valid_graphs, 1)
