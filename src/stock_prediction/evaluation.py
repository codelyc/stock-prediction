import os
import sys
import copy
import json
import queue
import glob
import dill
import platform
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import torch
from logure import logger

# Internal modules
from stock_prediction.common import (
    load_data, 
    ensure_queue_compatibility, 
    normalize_date_column,
    data_queue,
    PLOT_FEATURE_COLUMNS,
    SEQ_LEN,
    TQDM_NCOLS,
    INPUT_DIMENSION,
    OUTPUT_DIMENSION,
    TRAIN_WEIGHT,
    feature_engineer,
    NoneDataFrame,
    plot_feature_comparison,
    use_list
)
from stock_prediction.diagnostics import (
    evaluate_feature_metrics,
    load_bias_corrections,
    save_bias_corrections,
    apply_bias_corrections_to_dataframe,
    metrics_report,
    distribution_report,
    STD_RATIO_WARNING,
    BIAS_WARNING
)
from stock_prediction.inference import InferenceRunner

class Evaluator:
    def __init__(self, args, config, device, save_path, runner: InferenceRunner):
        self.args = args
        self.config = config
        self.device = device
        self.save_path = save_path
        self.runner = runner
        self.model_mode = args.model.upper()
        self.train_pkl_path = config.train_pkl_path
        self.png_path = config.png_path
        self.PKL = False if args.pkl <= 0 else True

    def resolve_plot_window(self, total_length: int) -> int:
        """Return plot window length; plot_days==0 means full history."""
        plot_days_value = int(getattr(self.args, "plot_days", 30))
        if plot_days_value == 0:
            return max(total_length, 0)
        if total_length <= 0:
            return max(plot_days_value, 1)
        return max(1, min(plot_days_value, total_length))

    def plot_loss_curve(self, loss_list, cnname="Stock"):
        try:
            if not loss_list:
                logger.warning("loss_curve: loss_list is empty, skip plotting")
                return
            save_dir = Path(self.png_path) / "train_loss"
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Matplotlib config for non-interactive backend
            plt.figure(figsize=(12, 6))
            steps = np.arange(1, len(loss_list) + 1)
            plt.plot(steps, np.array(loss_list), label="Training Loss", linewidth=1.2)
            plt.ylabel("MSELoss")
            plt.xlabel("Iteration")
            plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
            plt.legend()
            
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            img_path = save_dir / f"{cnname}_{self.model_mode}_{timestamp}_train_loss.png"
            plt.savefig(img_path, dpi=600)
            plt.close()
            logger.info(f"Training loss figure saved: {img_path}")
        except Exception as e:
            logger.error(f"Error: loss_curve {e}")

    def run_prediction(self, test_codes):
        logger.info(f"test_code={test_codes}")
        data = NoneDataFrame
        
        # Load Data
        if self.PKL is False:
            load_data(test_codes, data_queue=data_queue)
            try:
                data = data_queue.get(timeout=30)
            except queue.Empty:
                logger.error("Error: data_queue is empty")
                return
        else:
            with open(self.train_pkl_path, 'rb') as f:
                _pkl_queue = ensure_queue_compatibility(dill.load(f))
            # Find the matching code in the pkl queue
            while not _pkl_queue.empty():
                try:
                    item = _pkl_queue.get(timeout=1)
                    if str(item['ts_code'].iloc[0]).zfill(6) == str(test_codes[0]):
                        data = item
                        break
                except queue.Empty:
                    break
        
        data = normalize_date_column(data)

        if data.empty or (isinstance(data, pd.DataFrame) and data.empty) or (hasattr(data, 'iloc') and data["ts_code"].iloc[0] == "None"):
            logger.error("Error: data is empty or ts_code is None")
            return

        # Double check match
        if str(data['ts_code'].iloc[0]).zfill(6) != str(test_codes[0]):
            logger.error("Error: ts_code is not match")
            return

        predict_size = int(data.shape[0])
        if predict_size < SEQ_LEN:
            logger.error("Error: train_size is too small or too large")
            return

        predict_data = normalize_date_column(copy.deepcopy(data))
        spliced_data = normalize_date_column(copy.deepcopy(data))
        history_window = self.resolve_plot_window(spliced_data.shape[0])
        predicted_rows: list[dict] = []
        
        symbol_code = str(test_codes[0]).split('.')[0].zfill(6)
        test_path = str(Path(self.config.stock_daily_path) / f"{symbol_code}.csv")
        
        # Scenario 1: Iterative Prediction (predict_days > 0)
        if int(self.args.predict_days) <= 0:
            predict_days = abs(int(self.args.predict_days)) or 1
            pbar = tqdm(total=predict_days, leave=False, ncols=TQDM_NCOLS)
            
            while predict_days > 0:
                lastdate = pd.to_datetime(predict_data["Date"].iloc[0])
                feature_frame = predict_data.drop(columns=['ts_code', 'Date']).copy()
                feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
                
                # Call InferenceRunner
                test_loss, predict_list, _ = self.runner.run_test(
                    feature_frame, dataloader_mode=2, norm_symbol=symbol_code
                )
                
                if test_loss == -1 and predict_list == -1:
                    tqdm.write("Error: Inference failed")
                    break
                
                # Process predictions
                rows = []
                for items in predict_list:
                    items = items.to("cpu", non_blocking=True)
                    for idxs in items:
                        rows.append(idxs.tolist())
                
                # Append predicted data
                date_obj = lastdate + timedelta(days=1)
                tmp_data = [test_codes[0], date_obj]
                
                if rows:
                    if len(rows) > 0 and len(rows[0]) > 0:
                         # Assume last prediction is the one we want? (Logic from original predict)
                         # Original: tmp_data.extend(rows[0]) if rows else ... wait
                         # In original code: `for idxs in items: rows.append...` then `if rows: tmp_data.extend(rows[-1])`? Splicing logic seems implicit.
                         # Let's trust rows[-1] corresponds to the latest prediction if sequential.
                         # Actually original code: `tmp_data.extend(rows[0])`. 
                         # Since dataloader_mode=2 usually returns 1 item?
                         tmp_data.extend(rows[0])
                
                # Fill missing columns if any? (Original code loops len(tmp_data)-2 to df_mean)
                _splice_df = spliced_data.drop(columns=['ts_code', 'Date'])
                df_mean = _splice_df.mean().tolist()
                
                if len(tmp_data) < len(spliced_data.columns):
                     for index in range(len(tmp_data) - 2, len(df_mean)):
                        tmp_data.append(df_mean[index])
                        
                tmp_df = pd.DataFrame([tmp_data], columns=spliced_data.columns)
                predicted_rows.append(tmp_df.iloc[0].to_dict())
                
                # Prep for next iteration
                predict_data = pd.concat([tmp_df, spliced_data], axis=0, ignore_index=True)
                spliced_data = normalize_date_column(copy.deepcopy(predict_data))
                predict_data['Date'] = pd.to_datetime(predict_data['Date'])
                
                # Convert types and rename for compatibility
                cols_to_float = ['Open', 'High', 'Low', 'Close', 'change', 'pct_change', 'Volume', 'amount', 'amplitude', 'exchange_rate']
                existing_cols = [c for c in cols_to_float if c in predict_data.columns]
                predict_data[existing_cols] = predict_data[existing_cols].astype('float64')
                
                # If we need to write back to CSV for reload? Original code does `predict_data.to_csv(test_path)` then reload.
                # This seems inefficient but preserves exact behaviour of 'new data arrival'.
                predict_days -= 1
                pbar.update(1)
                
            pbar.close()
            
            # Post-prediction processing
            full_df = spliced_data.copy()
            full_df['Date'] = pd.to_datetime(full_df['Date'])
            predicted_df = pd.DataFrame(predicted_rows)
            if not predicted_df.empty:
                predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
            
            history_df = full_df.iloc[len(predicted_rows):len(predicted_rows) + history_window].copy()
            history_df['Date'] = pd.to_datetime(history_df['Date'])
            history_df = history_df.sort_values('Date').tail(history_window)

            bias_corrections = load_bias_corrections(symbol_code, self.model_mode)
            if bias_corrections and not predicted_df.empty:
                apply_bias_corrections_to_dataframe(predicted_df, bias_corrections)

            self._save_prediction_results(symbol_code, history_df, predicted_df)
            return

        else:
            # Scenario 2: Batch Prediction (predict_days > 0, usually implies test set evaluation or interval prediction)
            # Logic from "else" block of predict()
            normalized_predict = normalize_date_column(predict_data)
            feature_frame = normalized_predict.drop(columns=['ts_code', 'Date']).copy()
            feature_frame = feature_frame.fillna(feature_frame.median(numeric_only=True))
            
            test_loss, predict_list, _ = self.runner.run_test(
                feature_frame, dataloader_mode=2, norm_symbol=symbol_code
            )
            
            predictions = []
            for items in predict_list:
                items = items.to("cpu", non_blocking=True)
                for idxs in items:
                     predictions.append(idxs.tolist())
            
            # Resolve column names from global state... managed via runner or we assume standard
            # We need standard feature names.
            # In original code, `name_list` and `show_list` are global.
            # We can get them from runner.norm_stats
            
            stats = self.runner.norm_stats
            name_list = stats.get('name_list', [])
            show_list = stats.get('show_list', [])
            
            if not name_list or not show_list:
                 # Fallback if uninitialized
                 logger.warning("Name list or show list empty, using numeric indices")
                 pred_columns = [str(i) for i in range(INPUT_DIMENSION)] # Approx?
            else:
                 selected_features = [name_list[idx] for idx, flag in enumerate(show_list) if flag == 1]
                 rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
                 pred_columns = [rename_map.get(name, name.title()) for name in selected_features]

            # Adjust if dimension mismatch
            if predictions and len(pred_columns) != len(predictions[0]):
                 logger.warning(f"Prediction dim {len(predictions[0])} != Columns {len(pred_columns)}")
                 # Truncate or use indices
                 pred_columns = [f"Feat_{i}" for i in range(len(predictions[0]))]

            pred_df = pd.DataFrame(predictions, columns=pred_columns)
            actual_df = normalize_date_column(predict_data)
            actual_df['Date'] = pd.to_datetime(actual_df['Date'])
            actual_df = actual_df.sort_values('Date')
            
            window = self.resolve_plot_window(len(actual_df))
            if window > 0:
                actual_df = actual_df.tail(window)
            if not pred_df.empty:
                pred_df = pred_df.tail(len(actual_df))
                
            # Align Dates
            if len(pred_df) <= len(actual_df):
                 pred_df['Date'] = actual_df['Date'].iloc[:len(pred_df)].values
            
            bias_corrections = load_bias_corrections(symbol_code, self.model_mode)
            if bias_corrections and not pred_df.empty:
                apply_bias_corrections_to_dataframe(pred_df, bias_corrections)

            self._save_prediction_results(symbol_code, actual_df, pred_df, prefix="predict")
            return

    def _save_prediction_results(self, symbol_code, history_df, predicted_df, prefix="predict"):
        """Helper to plot and save metrics."""
        # Plotting
        for col in PLOT_FEATURE_COLUMNS:
            if col not in history_df.columns and col not in predicted_df.columns:
                continue
            
            history_series = pd.Series(dtype=float)
            if col in history_df.columns:
                history_series = history_df.set_index('Date')[col].astype(float).dropna()
            
            prediction_series = pd.Series(dtype=float)
            if not predicted_df.empty and col in predicted_df.columns:
                prediction_series = predicted_df.set_index('Date')[col].astype(float).dropna()
                
            plot_feature_comparison(
                symbol_code,
                self.model_mode,
                col,
                history_series,
                prediction_series,
                Path(self.png_path) / prefix,
                prefix=prefix,
            )
            
        # Metrics
        metrics_out = {}
        distribution_out = {}
        for col in PLOT_FEATURE_COLUMNS:
             if col in history_df.columns and col in predicted_df.columns:
                 history_series = history_df.set_index('Date')[col].astype(float).dropna()
                 prediction_series = predicted_df.set_index('Date')[col].astype(float).dropna()
                 evaluate_feature_metrics(col, history_series, prediction_series, metrics_out, distribution_out)
        
        if distribution_out:
            save_bias_corrections(symbol_code, self.model_mode, distribution_out, smoothing=0.5)
            
        if metrics_out:
            output_dir = Path("output")
            output_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d%H%M%S")
            metrics_path = output_dir / f"metrics_{symbol_code}_{self.model_mode}_{ts}.json"
            payload = {
                "regression": metrics_out,
                "distribution": distribution_out,
                "meta": {
                    "std_ratio_warning": STD_RATIO_WARNING,
                    "bias_warning": BIAS_WARNING,
                    "generated_at": ts,
                    "symbol": symbol_code,
                    "mode": self.model_mode,
                },
            }
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"Metrics saved: {metrics_path}")

    def run_contrast_routine(self, test_codes):
        if getattr(self.args, "full_train", False):
            logger.info("full_train enabled, skip contrast_lines.")
            return

        data = NoneDataFrame
        if self.PKL is False:
            load_data(test_codes, data_queue=data_queue)
            try:
                data = data_queue.get(timeout=30)
            except queue.Empty:
                logger.error("Error: data_queue is empty")
                return
        else:
            # Need to scan pkl for the code
            with open(self.train_pkl_path, 'rb') as f:
                _pkl_queue = ensure_queue_compatibility(dill.load(f))
            while not _pkl_queue.empty():
                try:
                    item = _pkl_queue.get(timeout=1)
                    if str(item['ts_code'].iloc[0]).zfill(6) in test_codes:
                        data = item
                        break
                except queue.Empty:
                    break
            
            if data is NoneDataFrame:
                logger.error("Error: data is None (Found no matching code in PKL)")
                return

        data = normalize_date_column(data)
        
        # Symbol Mapping Logic
        ts_code_value = None
        if 'ts_code' in data.columns and not data.empty:
            ts_code_value = str(data['ts_code'].iloc[0])

        if feature_engineer.settings.use_symbol_embedding and ts_code_value:
             if not hasattr(feature_engineer, 'symbol_to_id') or not feature_engineer.symbol_to_id:
                  symbol_id = hash(ts_code_value) % 4096 
             else:
                  symbol_id = feature_engineer.symbol_to_id.get(ts_code_value, 0)
             data['_symbol_index'] = symbol_id
             logger.info(f"Added _symbol_index={symbol_id} for ts_code={ts_code_value}")

        feature_data = data.drop(columns=['ts_code', 'Date'], errors='ignore').copy()
        feature_data = feature_data.fillna(feature_data.median(numeric_only=True))
        logger.info(f"test_code={test_codes}")
        
        if feature_data.empty:
             logger.error("Error: data is empty")
             return -1
             
        train_size = int(TRAIN_WEIGHT * (feature_data.shape[0]))
        if train_size < SEQ_LEN or train_size + SEQ_LEN > feature_data.shape[0]:
            logger.error("Error: train_size is too small or too large")
            return -1

        # Test on Full Data or Split? Original code uses copy of feature_data as Test_data
        Test_data = copy.deepcopy(feature_data) 
        
        # Call InferenceRunner
        test_loss, predict_list, dataloader = self.runner.run_test(
            Test_data, dataloader_mode=3, norm_symbol=ts_code_value
        )
        
        if test_loss == -1 and predict_list == -1:
            logger.error("Error: No model exists")
            return -1

        # Reconstruct Real vs Pred
        real_list = []
        prediction_list = []
        
        # Fetch stats for denormalization
        stats = self.runner.norm_stats
        raw_mean = stats.get('mean_list', [])
        raw_std = stats.get('std_list', [])
        
        # Filter mean/std for output columns only (defined by use_list)
        use_mean = []
        use_std = []
        if raw_mean and raw_std and len(raw_mean) >= len(use_list):
            for idx, flag in enumerate(use_list):
                if flag == 1 and idx < len(raw_mean):
                    use_mean.append(raw_mean[idx])
                    use_std.append(raw_std[idx])
            use_mean = np.array(use_mean)
            use_std = np.array(use_std)
        else:
            # Fallback if dimensions don't match expectation
            use_mean = np.array(raw_mean) if raw_mean else np.array([])
            use_std = np.array(raw_std) if raw_std else np.array([])
        
        # Iterate dataloader to get GROUND TRUTH
        for i, batch in enumerate(dataloader):
             if isinstance(batch, (list, tuple)):
                  if len(batch) == 3:
                      _, label, _ = batch
                  else:
                      _, label = batch[0], batch[1]
             else:
                  _, label = batch
            
             # Denormalize labels
             for idx in range(label.shape[0]):
                  if len(use_mean) > 0 and len(use_std) > 0:
                      item = label[idx].cpu().detach().numpy() * use_std + use_mean
                      real_list.append(item)
                  else:
                      real_list.append(label[idx].cpu().detach().numpy())

        # Process Predictions
        for items in predict_list:
             items = items.to("cpu", non_blocking=True)
             for idxs in items:
                  if len(use_mean) > 0 and len(use_std) > 0:
                      res = idxs.detach().numpy() * use_std + use_mean
                      prediction_list.append(res)
                  else:
                      prediction_list.append(idxs.detach().numpy())

        # Prep DataFrame for Plotting
        stats = self.runner.norm_stats
        name_list = stats.get('name_list', [])
        show_list = stats.get('show_list', [])
        
        if not name_list or not show_list:
             selected_features = [f"feat_{i}" for i in range(INPUT_DIMENSION)] 
             pred_columns = selected_features
        else:
             selected_features = [name_list[idx] for idx, flag in enumerate(show_list) if flag == 1]
             rename_map = {"open": "Open", "high": "High", "low": "Low", "close": "Close"}
             pred_columns = [rename_map.get(name, name.title()) for name in selected_features]

        real_array = np.array(real_list)
        pred_array = np.array(prediction_list)
        min_len = min(len(real_array), len(pred_array))
        
        if min_len == 0:
             logger.error("Error: No valid results")
             return
             
        plot_window = self.resolve_plot_window(min_len)
        real_array = real_array[:min_len][-plot_window:]
        pred_array = pred_array[:min_len][-plot_window:]
        
        real_df = pd.DataFrame(real_array, columns=pred_columns)
        pred_df = pd.DataFrame(pred_array, columns=pred_columns)
        
        # Add Dates
        if data is not None and "Date" in data.columns:
            date_series = pd.to_datetime(data['Date']).iloc[:min_len][-plot_window:]
            real_df['Date'] = date_series.values
            pred_df['Date'] = date_series.values
            real_df.sort_values('Date', inplace=True)
            pred_df.sort_values('Date', inplace=True)
        
        # Bias Correction
        symbol_code = str(test_codes[0]).split('.')[0].zfill(6)
        bias_corrections = load_bias_corrections(symbol_code, self.model_mode)
        if bias_corrections and not pred_df.empty:
            apply_bias_corrections_to_dataframe(pred_df, bias_corrections)

        self._save_prediction_results(symbol_code, real_df, pred_df, prefix="test")
        
        # Return for metrics calculation
        y_true = None
        y_pred = None
        if 'Close' in real_df.columns and 'Close' in pred_df.columns:
            y_true = real_df.set_index('Date')['Close'].astype(float).dropna().values
            y_pred = pred_df.set_index('Date')['Close'].astype(float).dropna().values
            
        return (y_true, y_pred) if y_true is not None else None
