import os

# python rassp/forward_evaluate_pipeline.py
# evaluate a trained model against a dataset and produce spectral predictions in `forward.preds`
FORWARD_EVAL_EXPERIMENTS = {
  # MS-SubsetNet evaluation on MassSpecGym
  'massspecgym': {
    'dataset' : 'data/massspecgym/test.parquet',
    'cv_method' : {
      'how': 'morgan_fingerprint_mod', 
      'mod' : 10,
      'test': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    'normalize_pred': True,
    'streaming_save': True,
    'checkpoint': 'checkpoints/mssubsetnet_best',
    'batch_size': 32,
    'epoch': 0,
    'mol_id_type': str,
  },
}

# python analysis_pipeline.py
# given a set of spectral predictions + the true predictions, compute metrics over the entire set and produce output in `results.metrics`
DATA_DIR = "."
WORKING_DIR = "results.metrics"
td = lambda x: os.path.join(WORKING_DIR, x)
ANALYSIS_EXPERIMENTS = {
  'massspecgym': {
    'true_spect' : 'data/massspecgym/test.parquet',
    'pred_spect' : f'./forward.preds/massspecgym.spect.sqlite',
    'phases': ['test'],
  },
}

# python library_match_pipeline.py
# given a set of spectral predictions, compute library matching metrics
LIBRARY_MATCH_EXPERIMENTS = {
  'massspecgym': {
    'main_library': 'data/massspecgym/train_full.parquet',
    'query_library': 'data/massspecgym/test.parquet',
    'exp_name': 'massspecgym',
  },
}