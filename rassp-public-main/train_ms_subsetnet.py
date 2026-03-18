import torch
from rassp import featurize, dataset, util, netutil
from rassp.model import subsetnets, losses
import os
import yaml

def train_mssubsetnet():
    # 1. Configuration
    config = {
        'MAX_N': 64,
        'internal_d': 512,
        'ce_emb_dim': 32,
        'adduct_emb_dim': 16,
        'num_adducts': 10,
        'lr': 1e-4,
        'batch_size': 32,
        'epochs': 100,
        'subset_gen': {'name': 'BandR', 'num_breaks': 3},
        'vert_subset_samples_n': 1024,
    }
    
    spect_bin_config = {
        'first_bin_center': 1.0,
        'bin_width': 1.0,
        'bin_number': 1024
    }
    spect_bin = featurize.msutil.binutils.create_spectrum_bins(**spect_bin_config)
    
    # Use default vertical feature arguments but ensure we have enough atom types for MassSpecGym
    feat_vert_args = netutil.dict_combine(netutil.default_feat_vert_args, {
        'feat_atomicno_onehot': [1, 6, 7, 8, 9, 15, 16, 17, 34, 35, 53] # Added Se, Br, I for completeness
    })

    featurizer_config = {
        'MAX_N': config['MAX_N'],
        'feat_vert_args': feat_vert_args,
        'adj_args': netutil.default_adj_args,
        'vert_subset_samples_n': config['vert_subset_samples_n'],
        'subset_gen_config': config['subset_gen'],
        'element_oh': feat_vert_args['feat_atomicno_onehot'],
        'explicit_formulae_config': {
            'formula_possible_atomicno': feat_vert_args['feat_atomicno_onehot'],
            'clip_mass': 1023, # Match new bin_number
            'use_highres': True
        }
    }
    
    # Matching formula_oh_sizes to the number of elements in element_oh
    formula_oh_sizes = [50] * len(feat_vert_args['feat_atomicno_onehot'])

    # 2. Dataset
    train_ds = dataset.ParquetDataset(
        "data/massspecgym/train.parquet",
        spect_bin,
        featurizer_config,
        {} # pred_config
    )
    
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=4)
    
    # 3. Model
    n_feats = train_ds.featurizer.n_feats
    
    model = subsetnets.MSSubsetNet(
        g_feat_in=n_feats,
        spect_bin=spect_bin,
        ce_emb_dim=config['ce_emb_dim'],
        adduct_emb_dim=config['adduct_emb_dim'],
        num_adducts=config['num_adducts'],
        internal_d=config['internal_d'],
        formula_oh_sizes=formula_oh_sizes
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # 4. Optimization
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = losses.SDPLoss()
    
    # 5. Training Loop
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
        
    print("Starting training...")
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for i, batch in enumerate(train_dl):
            batch = {k: util.move(v, torch.cuda.is_available()) for k, v in batch.items()}
            
            optimizer.zero_grad()
            res = model(**batch)
            loss = criterion(res, batch['spect'])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            if i % 10 == 0:
                print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch} finished. Average Loss: {total_loss/len(train_dl):.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), f"checkpoints/mssubsetnet_epoch_{epoch}.pt")

if __name__ == "__main__":
    train_mssubsetnet()
