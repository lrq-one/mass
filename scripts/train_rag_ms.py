import torch
from torch.utils.data import DataLoader
from torch_geometric.transforms import Compose
import os
import sys

# 1. 动态处理路径：包含连字符的目录 (如 rassp-public-main) 无法直接 import
# 我们将其绝对路径加入 sys.path
current_script_path = os.path.abspath(__file__)
scripts_dir = os.path.dirname(current_script_path)
root_dir = os.path.dirname(scripts_dir)
rassp_dir = os.path.join(root_dir, "rassp-public-main")

if rassp_dir not in sys.path:
    sys.path.append(rassp_dir)
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# 2. 绝对导入 (不使用点号)
try:
    from massspecgym_integration_subset_gnn import SubsetSimulationGNN
    from massspecgym_integration_transforms import AddCandidateFragmentsTransform
    from rag_transforms import AddRetrievalContext
except ImportError as e:
    print(f"Import Error: {e}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# 3. MassSpecGym 环境检查
try:
    from massspecgym.data.datasets import SimulationDataset
except ImportError:
    print("Warning: massspecgym not found. Please ensure it is installed in your conda/pip environment.")
    class SimulationDataset:
        def __init__(self, *args, **kwargs): pass

def train_rag_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 组合变换
    transform = Compose([
        AddCandidateFragmentsTransform(num_breaks=3, tolerance_ppm=10),
        AddRetrievalContext(index_path='reference_spectra_index.pkl', top_k=1)
    ])
    
    # 初始化模型 (4-步蓝图)
    model = SubsetSimulationGNN(
        hidden_dim=256,
        num_layers=4,
        node_feat_dim=133, 
        edge_feat_dim=14
    ).to(device)
    
    print("Model initialized successfully.")

if __name__ == "__main__":
    print("=== MassSpecGym RAG + RASSP Training Pipeline ===")
    train_rag_model()
