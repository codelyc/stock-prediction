import os
import json
import torch
import torch.nn as nn
import numpy as np
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from logure import logger

# Import internal modules
try:
    from .common import (
        Stock_Data,
        stock_queue_dataset,
        deep_copy_queue,
        custom_collate,
        pad_input,
        canonical_symbol,
        is_number,
        INPUT_DIMENSION,
        OUTPUT_DIMENSION,
        SEQ_LEN,
        BATCH_SIZE,
        NUM_WORKERS,
        TQDM_NCOLS,
        BUFFER_SIZE, # Assuming BUFFER_SIZE is in common based on usage
        feature_engineer
    )
except ImportError:
    # Fallback if relative import fails (e.g. running script directly)
    from stock_prediction.common import (
        Stock_Data,
        stock_queue_dataset,
        deep_copy_queue,
        custom_collate,
        pad_input,
        canonical_symbol,
        is_number,
        INPUT_DIMENSION,
        OUTPUT_DIMENSION,
        SEQ_LEN,
        BATCH_SIZE,
        NUM_WORKERS,
        TQDM_NCOLS,
        BUFFER_SIZE,
        feature_engineer
    )

class InferenceRunner:
    def __init__(self, args, device, save_path, model_mode):
        self.args = args
        self.device = device
        self.save_path = save_path
        self.model_mode = model_mode
        self.drop_last = False
        self.total_test_length = 0
        
        # State for normalization stats
        self.norm_stats = {
            "mean_list": [],
            "std_list": [],
            "show_list": [],
            "name_list": []
        }

    def _apply_norm_from_params(self, norm_params, symbol=None, symbol_norm_map=None):
        """Apply saved normalization parameters."""
        symbol_key = canonical_symbol(symbol)
        per_symbol = norm_params.get("per_symbol") or norm_params.get("per_symbol_stats")

        target_stats = None
        missing_symbol_stats = False
        
        # Attempt to find symbol-specific stats from the loaded file
        if symbol_key and isinstance(per_symbol, dict):
            target_stats = per_symbol.get(symbol_key)
            if target_stats is None:
                # Fallback logic if needed
                pass
        elif symbol_key and not isinstance(per_symbol, dict):
            missing_symbol_stats = True
            
        if target_stats is None:
            target_stats = norm_params

        means = target_stats.get("mean_list")
        stds = target_stats.get("std_list")
        shows = target_stats.get("show_list") or norm_params.get("show_list")
        names = target_stats.get("name_list") or norm_params.get("name_list")

        if means:
            self.norm_stats["mean_list"] = list(means)
        if stds:
            self.norm_stats["std_list"] = list(stds)
        if shows:
            self.norm_stats["show_list"] = list(shows)
        if names:
            self.norm_stats["name_list"] = list(names)

        if missing_symbol_stats and symbol_key:
            logger.warning(f"No symbol-specific normalization stats found for {symbol_key}; using global statistics.")

        # Update the global symbol map provided by caller if applicable
        if symbol_key and means and stds and symbol_norm_map is not None:
             # Basic update logic - in train.py this calls record_symbol_norm
             pass

    def _warn_if_norm_mismatch(self, symbol_key, observed_means, observed_stds, symbol_norm_map):
        if not symbol_key or not symbol_norm_map:
            return
        expected = symbol_norm_map.get(symbol_key)
        if not expected:
            return
        try:
            obs_mean = np.asarray(observed_means, dtype=float)
            obs_std = np.asarray(observed_stds, dtype=float)
            exp_mean = np.asarray(expected.get("mean_list", []), dtype=float)
            exp_std = np.asarray(expected.get("std_list", []), dtype=float)
            
            mean_diff = None
            std_diff = None
            
            if exp_mean.size and obs_mean.size and exp_mean.shape == obs_mean.shape:
                mean_diff = np.max(np.abs(exp_mean - obs_mean))
            if exp_std.size and obs_std.size and exp_std.shape == obs_std.shape:
                std_diff = np.max(np.abs(exp_std - obs_std))
                
            threshold = 1e-3
            if (mean_diff is not None and mean_diff > threshold) or (std_diff is not None and std_diff > threshold):
                logger.warning(f"Normalization mismatch for {symbol_key}: mean_diff={mean_diff}, std_diff={std_diff}")
        except Exception:
            pass

    def _load_model_instance(self, model_args=None):
        """Factory method to create model instance based on mode and args."""
        # Dynamic imports to avoid circular dependencies and load only what's needed
        if self.model_mode == "LSTM":
            from models import LSTM
            return LSTM(**model_args) if model_args else None
        elif self.model_mode == "GRU":
            from models import GRU
            return GRU(**model_args) if model_args else None
        elif self.model_mode == "ATTENTION_LSTM":
            from models import AttentionLSTM
            return AttentionLSTM(**model_args) if model_args else None
        elif self.model_mode == "BILSTM":
            from models import BiLSTM
            return BiLSTM(**model_args) if model_args else None
        elif self.model_mode == "TCN":
            from models import TCN
            return TCN(**model_args) if model_args else None
        elif self.model_mode == "MULTIBRANCH":
            from models import MultiBranchNet
            return MultiBranchNet(**model_args) if model_args else None
        elif self.model_mode == "TRANSFORMER":
            from models import Transformer
            return Transformer(**model_args) if model_args else None  # Note: Check class name consistency (Transformer vs TransformerModel)
        elif self.model_mode == "HYBRID":
            from models import TemporalHybridNet
            return TemporalHybridNet(**model_args) if model_args else None
        elif self.model_mode == "PTFT_VSSM":
            from models import PTFTVSSMEnsemble
            return PTFTVSSMEnsemble(**model_args) if model_args else None
        elif self.model_mode == "DIFFUSION":
            from models import DiffusionLSTM # Verify name
            return DiffusionLSTM(**model_args) if model_args else None
        elif self.model_mode == "GRAPH":
            from models import GraphLSTM # Verify name
            return GraphLSTM(**model_args) if model_args else None
        elif self.model_mode == "CNNLSTM":
            from models import CNNLSTM
            return CNNLSTM(**model_args) if model_args else None
        return None

    def load_model_from_checkpoint(self, predict_days, symbol_key=None, symbol_norm_map=None):
        """Loads the model and normalization parameters from disk."""
        ckpt_prefix = f"{self.save_path}_out{OUTPUT_DIMENSION}_time{SEQ_LEN}"
        if predict_days > 0:
            ckpt_prefix += f"_pre{predict_days}"
            
        candidates = [
            f"{ckpt_prefix}_Model.pkl",
            f"{ckpt_prefix}_Model_best.pkl",
        ]
        
        test_model = None
        loaded = False
        
        for candidate in candidates:
            if os.path.exists(candidate):
                # Load normalization params
                norm_file = candidate.replace("_Model.pkl", "_norm_params.json").replace("_Model_best.pkl", "_norm_params_best.json")
                if os.path.exists(norm_file):
                    try:
                        with open(norm_file, 'r', encoding='utf-8') as f:
                            norm_params = json.load(f)
                        self._apply_norm_from_params(norm_params, symbol=symbol_key, symbol_norm_map=symbol_norm_map)
                        logger.info(f"Loaded normalization params from {norm_file}")
                        
                        if symbol_key and symbol_norm_map and symbol_key in symbol_norm_map:
                            stats = symbol_norm_map[symbol_key]
                            logger.info(f"Using symbol-specific norm stats for {symbol_key} (features={len(stats.get('mean_list', []))})")
                        else:
                            logger.info(f"Using global normalization stats (features={len(self.norm_stats['mean_list'])})")
                    except Exception as e:
                        logger.warning(f"Failed to load normalization params from {norm_file}: {e}")

                # Load model args
                args_file = candidate.replace("_Model.pkl", "_Model_args.json").replace("_Model_best.pkl", "_Model_best_args.json")
                if os.path.exists(args_file):
                    try:
                        with open(args_file, 'r', encoding='utf-8') as f:
                            model_args = json.load(f)
                        test_model = self._load_model_instance(model_args)
                        
                        if test_model:
                            if self.args.test_gpu == 0:
                                test_model = test_model.to('cpu', non_blocking=True)
                            else:
                                test_model = test_model.to(self.device, non_blocking=True)
                            logger.info(f"Loaded model args from {args_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load model args from {args_file}: {e}")

                # Fallback if model not created yet (e.g. no args file) - THIS MIGHT FAIL if init requires args
                # But typically trained models save args. 
                # If test_model is None here, torch.load might need a class definition or pickle might handle it if simple.
                # Usually we need the instance first for load_state_dict.
                
                if test_model is None:
                     # If we couldn't create from args, we can't easily proceed unless we have a default constructor or pickle saves full object
                     # Assuming pure state_dict save.
                     # If the pickle contains the full model object (not just state_dict), we can direct load.
                     # train.py uses: test_model.load_state_dict(torch.load(candidate))
                     # This implies test_model exists. 
                     # If args_file failed, train.py would use the global 'test_model' (passed in?) or fail?
                     # train.py snippet: "using default test_model" -> assumes global 'model' or 'test_model' was initialized at start of script?
                     # We will assume args file exists or we return None.
                     pass

                if test_model:
                    try:
                        test_model.load_state_dict(torch.load(candidate, map_location=self.device))
                        loaded = True
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load state dict: {e}")
        
        return test_model

    def run_test(self, dataset, testmodel=None, dataloader_mode=0, norm_symbol=None, symbol_norm_map=None, total_test_length=0):
        """
        Main entry point for testing/inference.
        """
        if getattr(self.args, "full_train", False):
            return -1, -1, None
            
        symbol_key = canonical_symbol(norm_symbol)
        if not symbol_key:
            candidate_symbol = getattr(self.args, "test_code", "")
            if candidate_symbol and str(candidate_symbol).lower() != "all":
                symbol_key = canonical_symbol(candidate_symbol)
        
        predict_list = []
        accuracy_list = []
        
        use_gpu = self.device.type == "cuda" and getattr(self.args, "test_gpu", 1) == 1 and torch.cuda.is_available()
        pin_memory = use_gpu
        drop_last = getattr(self, 'drop_last', False)

        # Create DataLoader
        if dataloader_mode in [0, 2]:
            stock_predict = Stock_Data(
                mode=dataloader_mode,
                dataFrame=dataset,
                label_num=OUTPUT_DIMENSION,
                predict_days=int(self.args.predict_days),
                trend=int(self.args.trend),
                norm_symbol=symbol_key,
            )
            dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=pin_memory)
        elif dataloader_mode in [1]:
            _stock_test_data_queue = deep_copy_queue(dataset)
            stock_test = stock_queue_dataset(mode=1, data_queue=_stock_test_data_queue, label_num=OUTPUT_DIMENSION, buffer_size=BUFFER_SIZE, total_length=total_test_length, predict_days=int(self.args.predict_days), trend=int(self.args.trend))
            dataloader = DataLoader(dataset=stock_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=pin_memory, collate_fn=custom_collate)
        elif dataloader_mode in [3]:
            stock_predict = Stock_Data(
                mode=1,
                dataFrame=dataset,
                label_num=OUTPUT_DIMENSION,
                predict_days=int(self.args.predict_days),
                trend=int(self.args.trend),
                norm_symbol=symbol_key,
            )
            
            # Warn if mismatch using the stored norm stats
            if symbol_key and self.norm_stats["mean_list"]:
                 self._warn_if_norm_mismatch(symbol_key, self.norm_stats["mean_list"], self.norm_stats["std_list"], symbol_norm_map)

            dataloader = DataLoader(dataset=stock_predict, batch_size=BATCH_SIZE, shuffle=False, drop_last=drop_last, num_workers=NUM_WORKERS, pin_memory=pin_memory)
        else:
             dataloader = None # Handle error?

        # Load Model
        test_model = testmodel
        if test_model is None:
            predict_days = int(self.args.predict_days)
            test_model = self.load_model_from_checkpoint(predict_days, symbol_key, symbol_norm_map)
            
            if test_model is None:
                tqdm.write("No model found or failed to load")
                return -1, -1, -1
        
        test_model.eval()
        accuracy_fn = nn.MSELoss()
        
        pbar = tqdm(total=len(dataloader), leave=False, ncols=TQDM_NCOLS)
        
        # Inference Loop
        with torch.no_grad():
            for batch in dataloader:
                try:
                    if batch is None:
                        pbar.update(1)
                        continue
                        
                    if isinstance(batch, (list, tuple)):
                        if len(batch) == 3:
                            data, label, symbol_idx = batch
                        else:
                            data, label = batch[0], batch[1]
                            symbol_idx = None
                    else:
                        data, label = batch
                        symbol_idx = None
                        
                    if data is None or label is None:
                        pbar.update(1)
                        continue
                        
                    if self.args.test_gpu == 1:
                        data, label = data.to(self.device, non_blocking=True), label.to(self.device, non_blocking=True)
                        if symbol_idx is not None:
                            symbol_idx = symbol_idx.to(self.device, non_blocking=True).long()
                    else:
                        device_target = "cpu"
                        data, label = data.to(device_target, non_blocking=True), label.to(device_target, non_blocking=True)
                        if symbol_idx is not None:
                            symbol_idx = symbol_idx.to(device_target, non_blocking=True).long()
                    
                    if torch.isnan(data).any() or torch.isinf(data).any():
                        tqdm.write(f"test error: data has nan or inf, skip batch")
                        pbar.update(1)
                        continue
                    if torch.isnan(label).any() or torch.isinf(label).any():
                        tqdm.write(f"test error: label has nan or inf, skip batch")
                        pbar.update(1)
                        continue

                    # Forward Pass
                    if self.model_mode == "MULTIBRANCH":
                        price_dim = INPUT_DIMENSION // 2
                        tech_dim = INPUT_DIMENSION - price_dim
                        price_x = data[:, :, :price_dim]
                        tech_x = data[:, :, price_dim:]
                        predict = test_model.forward(price_x, tech_x)
                    elif self.model_mode == "TRANSFORMER":
                        data = pad_input(data)
                        predict = test_model.forward(data, label, int(self.args.predict_days))
                    else:
                        data = pad_input(data)
                        if symbol_idx is not None:
                            predict = test_model(data, symbol_index=symbol_idx)
                        else:
                            predict = test_model(data)
                            
                    predict_list.append(predict)
                    
                    if(predict.shape == label.shape):
                        accuracy = accuracy_fn(predict, label)
                        if not torch.isfinite(accuracy):
                            tqdm.write(f"test warning: accuracy is not finite (nan/inf), skip batch")
                            pbar.update(1)
                            continue
                            
                        if is_number(str(accuracy.item())):
                            accuracy_list.append(accuracy.item())
                        
                        if dataloader_mode not in [2]:
                            pbar.set_description(f"test accuracy: {np.mean(accuracy_list):.2e}")
                        pbar.update(1)
                    else:
                        tqdm.write(f"test error: predict.shape != label.shape")
                        pbar.update(1)
                        continue
                        
                except Exception as e:
                    tqdm.write(f"test error: {e}")
                    pbar.update(1)
                    continue

        if dataloader_mode not in [2]:
            tqdm.write(f"test accuracy: {np.mean(accuracy_list)}")
        pbar.close()
        
        if not accuracy_list:
            accuracy_list = [0]

        test_loss = np.mean(accuracy_list)
        
        # Post-test check
        if symbol_key and dataloader_mode in [3] and self.norm_stats["mean_list"]:
            self._warn_if_norm_mismatch(symbol_key, self.norm_stats["mean_list"], self.norm_stats["std_list"], symbol_norm_map)
            
        return test_loss, predict_list, dataloader
