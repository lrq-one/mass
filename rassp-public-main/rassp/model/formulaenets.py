"""
Lightweight utility module for formula-based network components.
This file contains only the essential parts required by SubsetNet/MSSubsetNet.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def create_mass_matrix_oh(masses, SPECT_BIN_N, mass_intensities=None):
    """
    Create a one-hot mass matrix from a list of formula masses.
    """
    BATCH_N, POSSIBLE_FORMULAE_N = masses.shape
    out = torch.zeros((BATCH_N, POSSIBLE_FORMULAE_N, SPECT_BIN_N)).to(masses.device)
    a = out.reshape(-1, SPECT_BIN_N)
    if mass_intensities is None:
        mass_intensities = torch.ones_like(masses).float().to(masses.device)
    b = masses.reshape(-1, 1)
    c = mass_intensities.reshape(-1, 1)
    a.scatter_(1, b.long(), c)
    return a.reshape(out.shape)

class StructuredOneHot(nn.Module):
    """
    Handles structured one-hot encoding for molecular formulas.
    """
    def __init__(self, oh_sizes, cumulative=False):
        super(StructuredOneHot, self).__init__()
        self.oh_sizes = oh_sizes
        self.cumulative = cumulative 
        if self.cumulative:
            n = np.sum(oh_sizes)
            accum_mat = np.zeros((n, n), dtype=np.int64)
            offset = 0
            for i, s in enumerate(oh_sizes):
                e = np.tril(np.ones((s, s)))
                accum_mat[offset:offset+s, offset:offset+s, ] = e
                offset += s
            a = torch.Tensor(accum_mat).long()
            self.register_buffer('accum_mat', a)
            
    def forward(self, data):
        data = data.long()
        oh_list = [F.one_hot(data[:, i], gs) for i, gs in enumerate(self.oh_sizes)]
        oh = torch.cat(oh_list, -1)
        if self.cumulative:
            return oh.float() @ self.accum_mat.float()
        else:
            return oh.float()

def create_mass_matrix_sparse(masses, SPECT_BIN_N, mass_intensities=None):
    """
    Sparse version of the mass matrix creation.
    """
    BATCH_N, POSSIBLE_FORMULAE_N = masses.shape
    if mass_intensities is None:
        mass_intensities = torch.ones_like(masses).float()
    
    # We create a flattened sparse representation
    # This is a simplified version compatible with mat_matrix_sparse_mm
    indices = torch.stack([
        torch.arange(BATCH_N).view(-1, 1).expand(-1, POSSIBLE_FORMULAE_N).reshape(-1),
        torch.arange(POSSIBLE_FORMULAE_N).view(1, -1).expand(BATCH_N, -1).reshape(-1),
        masses.reshape(-1)
    ]).to(masses.device)
    
    values = mass_intensities.reshape(-1)
    # Using coalesce to handle potential duplicate indices if any
    return torch.sparse_coo_tensor(indices, values, (BATCH_N, POSSIBLE_FORMULAE_N, SPECT_BIN_N)).coalesce()

def mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs):
    """
    Matrix multiplication with sparse mass matrix.
    """
    # formulae_probs: B x F
    # sparse_mass_matrix: B x F x S (COO sparse)
    # Result: B x S
    
    # Since torch doesn't support batch sparse matmul easily for 3D sparse tensors,
    # we use the indices to perform a weighted sum.
    indices = sparse_mass_matrix.indices() # 3 x NNZ
    values = sparse_mass_matrix.values()   # NNZ
    
    batch_idx = indices[0]
    formula_idx = indices[1]
    bin_idx = indices[2]
    
    # weighting probabilities: B x F
    # we need the probability for each (batch_idx, formula_idx)
    weights = formulae_probs[batch_idx, formula_idx]
    
    weighted_values = values * weights
    
    # Now scatter to output B x S
    num_bins = sparse_mass_matrix.shape[2]
    batch_size = sparse_mass_matrix.shape[0]
    
    res = torch.zeros((batch_size, num_bins), device=formulae_probs.device)
    # Use index_add to accumulate
    res.index_put_((batch_idx, bin_idx), weighted_values, accumulate=True)
    
    return res
