import numpy as np
import torch
import torch.utils.data
from rdkit import Chem

from . import atom_features
from . import molecule_features
from . import util
from .. import msutil
"""
特征化流程整合
类	作用
MolFeaturizer	主特征化器，将分子转化为模型所需的全套输入字典（原子特征向量、邻接矩阵、化学式特征、顶点子集等）
PredFeaturizer	标签特征化器，将稀疏谱 (m/z, intensity) 转化为分 bin 的直方图向量作为训练标签
create_subset_generator()	工厂函数，创建顶点子集生成器（BandR=断键+氢重排、B=纯断键）
"""

def create_mol_featurizer(spect_bin_config, featurizer_config):

    return MolFeaturizer(bin_config = spect_bin_config,
                         **featurizer_config)

def create_pred_featurizer(spect_bin_config, pred_featurizer_config):

    return PredFeaturizer(bin_config = spect_bin_config,
                         **pred_featurizer_config)

class MolFeaturizer(torch.utils.data.Dataset):
    def __init__(
            self, 
            MAX_N,
            bin_config, 
            feat_vert_args = {}, 
            adj_args = {},
            mol_args = {},
            explicit_formulae_config = {},
            formula_frag_count_config = {},
            max_conf_sample = 1, 
            spect_assign = True,
            extra_features = None,
            sparse_spect = False, 
            sparse_peak_num = 128,
            round_mass_to_int = True,
            removeHs = False,
            element_oh = [],
            subset_gen_config = {},
            vert_subset_samples_n = 0, 
            MAX_EDGE_N=64,
            spect_input_sparse = False,         
        ):
        self.MAX_N = MAX_N

        self.feat_vert_args = feat_vert_args
        self.adj_args = adj_args
        self.mol_args = mol_args

        self.spect_bin_config = bin_config
        
        self.formula_frag_count_config = formula_frag_count_config

        self.spect_assign = spect_assign
        self.extra_features = extra_features
        self.max_conf_sample = max_conf_sample

        self.sparse_spect = sparse_spect
        self.sparse_peak_num = sparse_peak_num

        self.round_mass_to_int= round_mass_to_int

        if explicit_formulae_config != {}:
            self.pff = molecule_features.PossibleFormulaFeaturizer(**explicit_formulae_config)
        else:
            self.pff = None

        self.removeHs = removeHs

        self.element_oh = element_oh

        self.vert_subset_samples_n = vert_subset_samples_n
        self.MAX_EDGE_N = MAX_EDGE_N

        self.spect_input_sparse = spect_input_sparse

        self.mp2b = msutil.binutils.create_peaks_to_bins(self.spect_bin_config)

        self.subset_gen_config = subset_gen_config
        
    @property
    def n_feats(self):
        # Dynamically determine the number of features by featurizing a dummy molecule
        test_mol = Chem.MolFromSmiles('C')
        # We need to make sure we don't require conformers if they aren't provided
        feat_args = self.feat_vert_args.copy()
        feat_args['feat_pos'] = False # Force false for dimension check to avoid conformer requirement
        f_vec = atom_features.feat_tensor_atom(test_mol, **feat_args)
        
        # If the actual featurizer uses pos, we add 3
        actual_n = f_vec.shape[1]
        if self.feat_vert_args.get('feat_pos', True):
            actual_n += 3
        return actual_n
        
    def __call__(self, mol, collision_energy=None, adduct_type=None):

        if self.removeHs:
            mol = Chem.RemoveHs(mol)
        
        n_bonds = mol.GetNumBonds()
        
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=0, 
                                                **self.feat_vert_args)
        DATA_N = f_vect.shape[0]
        
        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32)
        vect_feat[:DATA_N] = f_vect
        
        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, :adj_nopad.shape[1], :adj_nopad.shape[2]] = adj_nopad

        adj_oh_nopad = molecule_features.feat_mol_adj(mol, split_weights=[1.0, 1.5, 2.0, 3.0], 
                                                      edge_weighted=False, norm_adj=False, add_identity=False)

        adj_oh = torch.zeros((adj_oh_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj_oh[:, :adj_oh_nopad.shape[1], :adj_oh_nopad.shape[2]] = adj_oh_nopad
                
        atomicnos = util.get_nos(mol)

        # input mask
        input_mask = torch.zeros(self.MAX_N) 
        input_mask[:DATA_N] = 1.0
        
        v = {
            'vect_feat' : vect_feat, 
            'adj' : adj,
            'adj_oh' : adj_oh,
            'input_mask' : input_mask, 
        }

        if collision_energy is not None:
            v['collision_energy'] = np.array([collision_energy], dtype=np.float32)
        
        if adduct_type is not None:
            v['adduct_type'] = np.array([adduct_type], dtype=np.int64)

        if self.sparse_spect:
            # Note: spect_out is not defined here in the original code, 
            # this seems like a legacy block that might need fixing if used.
            pass
            
        if self.pff is not None:
            formulae_feats, formulae_peaks = self.pff(mol)

            assert np.max(formulae_feats) <= 255

            # Bin the mass peaks into the app
            formulae_peaks_mass_idx, formulae_peaks_intensity =\
                self.mp2b(formulae_peaks) 
            
            v['formulae_features'] = formulae_feats.astype(np.uint8)
            v['formulae_peaks'] = formulae_peaks
            v['formulae_peaks_mass_idx'] = formulae_peaks_mass_idx.astype(np.int64)
            v['formulae_peaks_intensity'] = formulae_peaks_intensity
            
        # atomicno one-hot matrix
        if len(self.element_oh ) > 0 :
            element_oh_mat = np.zeros((self.MAX_N, len(self.element_oh)), dtype=np.float32)
            for ei, e in enumerate(self.element_oh):
                element_oh_mat[:DATA_N, ei] = (atomicnos == e)

            v['vert_element_oh'] = element_oh_mat


        if self.vert_subset_samples_n > 0:
            SUBSET_SAMPLES_N = self.vert_subset_samples_n
            formulae_features = v['formulae_features']

            vert_element_oh = v['vert_element_oh']

            vert_subset_generator = create_subset_generator(**self.subset_gen_config)
            
            vertsubsets = vert_subset_generator(mol)
            
            atom_subsets = np.zeros((SUBSET_SAMPLES_N, self.MAX_N))
            
            # Ensure full molecule is always the first subset (Precursor Guard)
            full_mol_subset = np.ones(DATA_N)
            
            # Combine full molecule with generated subsets, avoiding duplicates if possible
            # but simple approach is to just put it at index 0.
            atom_subsets[0, :DATA_N] = full_mol_subset
            
            number_of_subsets = min(len(vertsubsets), SUBSET_SAMPLES_N - 1)
            if number_of_subsets > 0:
                atom_subsets[1:1+number_of_subsets, :DATA_N] = \
                    vertsubsets[np.random.permutation(len(vertsubsets))[:number_of_subsets]]

            atom_subsets_formula_idx = util.get_subset_peaks_idx_from_formulae_fast(formulae_features,
                                                                                    vert_element_oh,
                                                                                    atom_subsets)

            assert atom_subsets_formula_idx.shape == (SUBSET_SAMPLES_N,)

            v['atom_subsets'] = atom_subsets
            v['atom_subsets_peaks_mass_idx'] = formulae_peaks_mass_idx[atom_subsets_formula_idx].astype(np.int64)
            v['atom_subsets_peaks_intensity'] = formulae_peaks_intensity[atom_subsets_formula_idx]
            v['atom_subsets_formula_idx'] = atom_subsets_formula_idx.astype(np.int64)

        return v


class PredFeaturizer:
    """
    create output data to predict
    """
    def __init__(self, bin_config, **kwargs):
        self.spect_bin_config = bin_config

    def __call__(self, mol, sparse_spect):
        """
        sparse_spect is N x 2 of (mz, intensity) 
        note that there can be duplicate m/zs

        """
        record_spect = np.stack(sparse_spect)
        spect_idx, spect_p, spect_out = self.spect_bin_config.histogram(record_spect[:, 0],
                                                                        record_spect[:, 1])
        spect_out = spect_out.astype(np.float32)
        return {'spect': spect_out}


def create_subset_generator(name, **config):
    if name == 'BandR':
        def run(mol):
            if mol.GetNumAtoms() <= 64:
                return msutil.vertsubsetgen.BreakAndRearrangeFast(**config)(mol)
            else:
                # fallback to the slow version
                return msutil.vertsubsetgen.BreakAndRearrangeAllH(**config)(mol)
        return run
    elif name == 'B':
        def run(mol):
            return msutil.vertsubsetgen.EnumerateBreaks(**config)(mol)
        return run
    else:
        raise NotImplementedError(f"unknown subset generator {name}")
