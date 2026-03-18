import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data
from rassp.msutil.vertsubsetgen import BreakAndRearrangeAllH, BreakAndRearrangeFast

def match_spectra(candidate_mzs, spectrum_mzs, spectrum_intensities, tolerance_ppm=10):
    """把实验真实质谱与这些候选碎片进行比对（容差比如 10ppm）"""
    target_intensities = np.zeros(len(candidate_mzs))
    # 兼容 list、numpy 或 tensor 的输入
    spectrum_mzs = np.array(spectrum_mzs) if isinstance(spectrum_mzs, (list, tuple)) else spectrum_mzs.numpy() if hasattr(spectrum_mzs, 'numpy') else spectrum_mzs
    spectrum_ints = np.array(spectrum_intensities) if isinstance(spectrum_intensities, (list, tuple)) else spectrum_intensities.numpy() if hasattr(spectrum_intensities, 'numpy') else spectrum_intensities

    for i, cmz in enumerate(candidate_mzs):
        diffs = np.abs(spectrum_mzs - cmz) / cmz * 1e6
        match_idx = np.where(diffs <= tolerance_ppm)[0]
        if len(match_idx) > 0:
            target_intensities[i] = np.max(spectrum_ints[match_idx])
    return target_intensities

class AddCandidateFragmentsTransform:
    def __init__(self, num_breaks=3, min_size=1, tolerance_ppm=10, use_fast=True):
        self.num_breaks = num_breaks
        self.min_size = min_size
        self.tolerance_ppm = tolerance_ppm
        
        # 兼容性设计: Try to use Fast Cython version, fallback to python version if not available
        try:
            if use_fast:
                self.sampler = BreakAndRearrangeFast(num_breaks=num_breaks, min_size=min_size)
            else:
                self.sampler = BreakAndRearrangeAllH(num_breaks=num_breaks, min_size=min_size)
        except NameError:
            print("Warning: BreakdownAndRearrangeFast not available. Falling back to Python implementation.")
            self.sampler = BreakAndRearrangeAllH(num_breaks=num_breaks, min_size=min_size)

    def __call__(self, data: Data) -> Data:
        smiles = data.smiles
        mol = Chem.MolFromSmiles(smiles)
        mol = Chem.AddHs(mol)
        
        # 1. 用 RASSP 跑出所有可能的连通子图（碎片）
        # 返回形状为 (num_subsets, num_atoms) 的 0/1 矩阵
        try:
            subsets_mask = self.sampler(mol)
        except Exception as e:
            # 兜底：如果有些特别大的分子在 C++ 版本中崩溃，也回退到纯 Python 处理或兜底掩码
            print(f"Sampling failed for SMILES {smiles}: {e}")
            subsets_mask = None

        if subsets_mask is None or len(subsets_mask) == 0:
            subsets_mask = np.ones((1, mol.GetNumAtoms()), dtype=np.uint8)
            
        candidate_formulas = []
        candidate_mzs = []
        candidate_indices = []
        
        atom_masses = [a.GetMass() for a in mol.GetAtoms()]
        atom_symbols = [a.GetSymbol() for a in mol.GetAtoms()]

        for mask in subsets_mask:
            # 获取该碎片包含哪些原子的索引列表
            indices = np.where(mask == 1)[0].tolist()
            candidate_indices.append(indices)
            
            mass = sum(atom_masses[i] for i in indices)
            candidate_mzs.append(mass)
            
            # (可选) 如果你后续还需要用到化学式本身，这里也可以记录
            formula = "".join(sorted([atom_symbols[i] for i in indices]))
            candidate_formulas.append(formula)
            
        candidate_mzs = np.array(candidate_mzs)

        # 2. 把实验真实质谱与这些候选碎片进行比对
        # 注意对应你准备好的 massspecgym 数据里面的键名 (通常为 data.mz, data.intensity)
        target_intensities = match_spectra(candidate_mzs, data.mz, data.intensity, self.tolerance_ppm) 
        
        # 3. 把这些信息存入图数据对象中传给下游
        data.candidate_formulas = candidate_formulas
        # 将列表序列化，因为 PyG 的 batch default_collate 对于长度不一的嵌套列表会报错
        # 常用 trick: 由于每个分子的候选碎片数量不同、每个子图包含的原子数也不同
        # 我们作为 object 或 custom attr 存入
        data.candidate_indices = candidate_indices 
        data.candidate_mzs = candidate_mzs.tolist()
        data.target_intensities = torch.tensor(target_intensities, dtype=torch.float)
        
        return data
