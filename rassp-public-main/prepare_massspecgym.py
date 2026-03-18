import pandas as pd
import numpy as np
from rdkit import Chem
import os

def prepare_data(tsv_path, output_parquet_path):
    print(f"Reading {tsv_path}...")
    df = pd.read_table(tsv_path)
    
    # MassSpecGym columns: smiles, precursor_mz, adduct, collision_energy, mzs, intensities
    # We need to transform 'mzs' and 'intensities' strings to a unified peaks numpy array
    
    def parse_peaks(row):
        # MassSpecGym format: mzs and intensities can be space or comma separated strings
        def to_list(s):
            if not isinstance(s, str): return []
            # Replace commas with spaces to handle both formats
            return [float(x) for x in s.replace(',', ' ').split()]
            
        mzs = to_list(row.get('mzs', ''))
        intensities = to_list(row.get('intensities', ''))
        
        if len(mzs) != len(intensities) or len(mzs) == 0:
            return []
        return np.stack([mzs, intensities], axis=1).astype(np.float32).tolist()

    print("Parsing peaks and molecules...")
    # Combine mzs and intensities into 'spect'
    df['spect'] = df.apply(parse_peaks, axis=1)
    df['rdmol'] = df['smiles'].apply(lambda x: Chem.MolFromSmiles(x).ToBinary() if isinstance(x, str) else None)
    
    # Drop rows with invalid molecules or empty spectra
    df = df.dropna(subset=['rdmol'])
    
    # Collision energy is often '30.0' or '10.0;20.0;30.0'
    def parse_ce(ce):
        try:
            if isinstance(ce, str) and ';' in ce:
                return np.mean([float(x) for x in ce.split(';')])
            return float(ce)
        except:
            return 35.0 # default fallback
    
    if 'collision_energy' in df.columns:
        df['collision_energy'] = df['collision_energy'].apply(parse_ce)
    else:
        df['collision_energy'] = 35.0 # default if missing
    
    # Adduct handling
    if 'adduct' not in df.columns:
        df['adduct'] = '[M+H]+' # default

    # Save to parquet
    print(f"Saving to {output_parquet_path}...")
    df[['rdmol', 'spect', 'collision_energy', 'adduct']].to_parquet(output_parquet_path)
    print("Done.")

if __name__ == "__main__":
    output_dir = "data/massspecgym"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for individual split files first
    splits_found = False
    for split_name in ["train", "val", "test"]:
        tsv_path = os.path.join("data", f"MassSpecGym_{split_name}.tsv")
        if os.path.exists(tsv_path):
            prepare_data(tsv_path, os.path.join(output_dir, f"{split_name}.parquet"))
            splits_found = True
            
    # If no split files, handle the main file
    main_tsv = os.path.join("data", "MassSpecGym.tsv")
    if os.path.exists(main_tsv):
        print(f"Processing main file {main_tsv}...")
        # Prepare the full version
        prepare_data(main_tsv, os.path.join(output_dir, "train_full.parquet"))
        
        # If we didn't find specific split files, we split the main one
        if not splits_found:
            df = pd.read_table(main_tsv)
            
            # Check for split source: first 'split' column, then 'fold' column
            split_col = None
            if 'split' in df.columns:
                split_col = 'split'
            elif 'fold' in df.columns:
                # Check if 'fold' contains the expected string labels
                if any(df['fold'].isin(['train', 'val', 'test'])):
                    split_col = 'fold'
            
            if split_col:
                print(f"Detected split labels in column '{split_col}'. Partitioning data...")
                for s in ['train', 'val', 'test']:
                    sub_df = df[df[split_col] == s]
                    if not sub_df.empty:
                        # Temporary save and process
                        tmp_path = f"data/tmp_{s}.tsv"
                        sub_df.to_csv(tmp_path, sep='\t', index=False)
                        prepare_data(tmp_path, os.path.join(output_dir, f"{s}.parquet"))
                        os.remove(tmp_path)
            else:
                # Random split if no column found
                print("No split labels found. Performing random split (80/10/10)...")
                from sklearn.model_selection import train_test_split
                train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
                val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)
                
                for name, sdf in [("train", train_df), ("val", val_df), ("test", test_df)]:
                    tmp_path = f"data/tmp_{name}.tsv"
                    sdf.to_csv(tmp_path, sep='\t', index=False)
                    prepare_data(tmp_path, os.path.join(output_dir, f"{name}.parquet"))
                    os.remove(tmp_path)
    elif not splits_found:
        print("No MassSpecGym TSV files found in data/ directory.")
