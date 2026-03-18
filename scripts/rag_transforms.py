import torch
import pickle
import os
from torch_geometric.transforms import BaseTransform
from rdkit import Chem
from massspecgym.simulation_utils.feat_utils import MolGraphFeaturizer

class AddRetrievalContext(BaseTransform):
    """
    数据增强变换：为每个分子注入其检索到的参考分子及谱图信息。
    遵循 MARASON 风格的 RAG 机制。
    """
    def __init__(
        self, 
        index_path='reference_spectra_index.pkl', 
        top_k=1,
        node_feat_dim=133,
        edge_feat_dim=14
    ):
        self.top_k = top_k
        self.index_path = index_path
        self.ref_db = None
        
        # 定义参考分子的图特征提取器 (需与查询分子保持一致)
        self.featurizer = MolGraphFeaturizer()

    def _load_db(self):
        if self.ref_db is None and os.path.exists(self.index_path):
            with open(self.index_path, 'rb') as f:
                self.ref_db = pickle.load(f)

    def __call__(self, data):
        self._load_db()
        if self.ref_db is None:
            return data
            
        # 假设 MassSpecGym 的 Dataset 已经在 data.candidates_smiles 中存了检索建议
        # 或者我们使用 data.inchikey 在索引中查找相似物 (此处采用简化逻辑)
        
        # 情况 A: Dataset 提供了检索到的 ID
        retrieved_keys = getattr(data, 'candidates_inchikey', [])
        if len(retrieved_keys) == 0:
            # 备选: 直接从索引中拿一个相似的 (演示逻辑)
            return data
            
        ref_keys = retrieved_keys[:self.top_k]
        
        ref_spectra_mzs = []
        ref_spectra_ints = []
        ref_mol_graphs = []
        
        for k in ref_keys:
            if k in self.ref_db:
                info = self.ref_db[k]
                ref_spectra_mzs.append(torch.from_numpy(info['mz']))
                ref_spectra_ints.append(torch.from_numpy(info['intensity']))
                
                # 构建参考分子的 PyG 图表示 (用于 Neural Graph Matching)
                mol = Chem.MolFromSmiles(info['smiles'])
                if mol:
                    ref_g = self.featurizer.get_pyg_graph(mol)
                    ref_mol_graphs.append(ref_g)
        
        # 挂载到 Data 对象上
        data.ref_spectra_mzs = ref_spectra_mzs
        data.ref_spectra_ints = ref_spectra_ints
        data.ref_mol_graphs = ref_mol_graphs # List of Data objects
        
        return data
