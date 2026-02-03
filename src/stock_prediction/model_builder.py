import os
from datetime import datetime
from pathlib import Path

from logure import logger
import torch
import torch.nn as nn
import torch.optim as optim

# Import models
from models import (
    LSTM,
    AttentionLSTM,
    BiLSTM,
    TCN,
    MultiBranchNet,
    TransformerModel,
    CNNLSTM,
    TemporalHybridNet,
    PTFTVSSMEnsemble,
    PTFTVSSMLoss,
    DiffusionForecaster,
    GraphTemporalModel,
    HybridLoss,
)

# Import shared config and common utils
from stock_prediction.common import (
    INPUT_DIMENSION, OUTPUT_DIMENSION, SEQ_LEN, D_MODEL, NHEAD, 
    LEARNING_RATE, WEIGHT_DECAY, WARMUP_STEPS, CustomSchedule,
    feature_engineer
)
from stock_prediction.hybrid_config import get_adaptive_hybrid_config

class ModelBuilder:
    def __init__(self, args, config, device, symbol=None):
        self.args = args
        self.config = config
        self.device = device
        self.symbol = symbol
        self.model_mode = args.model.upper()
        
        # Paths
        self.lstm_path = config.lstm_path
        self.transformer_path = config.transformer_path
        self.cnnlstm_path = config.cnnlstm_path
        
        # Dimensions
        self.input_dim = INPUT_DIMENSION
        self.output_dim = OUTPUT_DIMENSION
        
        # Symbol Embedding Settings
        self.feature_settings = getattr(config, "features", None)
        self.symbol_embed_enabled = bool(getattr(self.feature_settings, "use_symbol_embedding", False))
        self.symbol_embed_dim = int(getattr(self.feature_settings, "symbol_embedding_dim", 16))
        self.symbol_embed_max = int(os.getenv("SYMBOL_EMBED_MAX", "4096"))
        
        _symbol_vocab = len(feature_engineer.get_symbol_indices()) if self.symbol_embed_enabled else 0
        if self.symbol_embed_enabled:
            self.symbol_vocab_size = _symbol_vocab if _symbol_vocab > 0 else self.symbol_embed_max
            self.symbol_vocab_size = min(max(self.symbol_vocab_size, 1), self.symbol_embed_max)
        else:
            self.symbol_vocab_size = max(_symbol_vocab, 1)

    def build(self):
        """Builder method to construct model, optimizer and criterion."""
        
        predict_days = int(self.args.predict_days)
        
        if self.model_mode == "LSTM":
            model = LSTM(input_dim=self.input_dim)
            model._init_args = dict(input_dim=self.input_dim)
            test_model = LSTM(input_dim=self.input_dim)
            test_model._init_args = dict(input_dim=self.input_dim)
            save_path = self.lstm_path
            criterion = nn.MSELoss()
            
        elif self.model_mode == "ATTENTION_LSTM":
            model = AttentionLSTM(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            model._init_args = dict(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            test_model = AttentionLSTM(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            test_model._init_args = dict(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            save_path = "output/attention_lstm"
            criterion = nn.MSELoss()
            
        elif self.model_mode == "BILSTM":
            model = BiLSTM(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            model._init_args = dict(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            test_model = BiLSTM(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            test_model._init_args = dict(input_dim=self.input_dim, hidden_dim=128, num_layers=2, output_dim=self.output_dim)
            save_path = "output/bilstm"
            criterion = nn.MSELoss()
            
        elif self.model_mode == "TCN":
            model = TCN(input_dim=self.input_dim, output_dim=self.output_dim, num_channels=[64, 64, 64], kernel_size=3)
            model._init_args = dict(input_dim=self.input_dim, output_dim=self.output_dim, num_channels=[64, 64, 64], kernel_size=3)
            test_model = TCN(input_dim=self.input_dim, output_dim=self.output_dim, num_channels=[64, 64, 64], kernel_size=3)
            test_model._init_args = dict(input_dim=self.input_dim, output_dim=self.output_dim, num_channels=[64, 64, 64], kernel_size=3)
            save_path = "output/tcn"
            criterion = nn.MSELoss()
            
        elif self.model_mode == "MULTIBRANCH":
            price_dim = self.input_dim // 2
            tech_dim = self.input_dim - price_dim
            model = MultiBranchNet(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=self.output_dim)
            model._init_args = dict(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=self.output_dim)
            test_model = MultiBranchNet(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=self.output_dim)
            test_model._init_args = dict(price_dim=price_dim, tech_dim=tech_dim, hidden_dim=64, output_dim=self.output_dim)
            save_path = "output/multibranch"
            criterion = nn.MSELoss()
            
        elif self.model_mode == "TRANSFORMER":
            model = TransformerModel(input_dim=self.input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=6, 
                                   dim_feedforward=2048, output_dim=self.output_dim, max_len=SEQ_LEN, mode=0)
            model._init_args = dict(input_dim=self.input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=6,
                                   dim_feedforward=2048, output_dim=self.output_dim, max_len=SEQ_LEN, mode=0)
            test_model = TransformerModel(input_dim=self.input_dim, d_model=D_MODEL, nhead=NHEAD, 
                                        num_layers=6, dim_feedforward=2048, output_dim=self.output_dim, max_len=SEQ_LEN, mode=1)
            test_model._init_args = dict(input_dim=self.input_dim, d_model=D_MODEL, nhead=NHEAD, num_layers=6,
                                        dim_feedforward=2048, output_dim=self.output_dim, max_len=SEQ_LEN, mode=1)
            save_path = self.transformer_path
            criterion = nn.MSELoss()
            
        elif self.model_mode == "HYBRID":
            hybrid_steps = abs(predict_days) if predict_days > 0 else 1
            
            # Use shared config utility
            size_hint = getattr(self.args, "hybrid_size", "auto")
            estimated_data_size = int(os.getenv("TRAIN_DATA_SIZE", "1000")) 
            hybrid_config = get_adaptive_hybrid_config(size_hint, estimated_data_size)
            
            model = TemporalHybridNet(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=hybrid_config["hidden_dim"],
                predict_steps=hybrid_steps,
                branch_config=hybrid_config["branch_config"],
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=hybrid_config["hidden_dim"],
                predict_steps=hybrid_steps,
                branch_config=hybrid_config["branch_config"],
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model = TemporalHybridNet(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=hybrid_config["hidden_dim"],
                predict_steps=hybrid_steps,
                branch_config=hybrid_config["branch_config"],
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                hidden_dim=hybrid_config["hidden_dim"],
                predict_steps=hybrid_steps,
                branch_config=hybrid_config["branch_config"],
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            save_path = str(self.config.get_model_path("HYBRID", self.symbol))
            criterion = HybridLoss(
                model,
                mse_weight=float(getattr(self.args, "hybrid_mse_weight", 1.0)),
                quantile_weight=float(getattr(self.args, "hybrid_quantile_weight", 0.1)),
                direction_weight=float(getattr(self.args, "hybrid_direction_weight", 0.05)),
                regime_weight=float(getattr(self.args, "hybrid_regime_weight", 0.05)),
                volatility_weight=float(getattr(self.args, "hybrid_volatility_weight", 0.12)),
                extreme_weight=float(getattr(self.args, "hybrid_extreme_weight", 0.02)),
                mean_weight=float(getattr(self.args, "hybrid_mean_weight", 0.05)),
                return_weight=float(getattr(self.args, "hybrid_return_weight", 0.08)),
            )

        elif self.model_mode == "PTFT_VSSM":
            ensemble_steps = abs(predict_days) if predict_days > 0 else 1
            model = PTFTVSSMEnsemble(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=ensemble_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=ensemble_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model = PTFTVSSMEnsemble(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=ensemble_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=ensemble_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            save_path = str(self.config.get_model_path("PTFT_VSSM", self.symbol))
            criterion = PTFTVSSMLoss(model, mse_weight=1.0, kl_weight=1e-3)
            
        elif self.model_mode == "DIFFUSION":
            diffusion_steps = abs(predict_days) if predict_days > 0 else 1
            model = DiffusionForecaster(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=diffusion_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=diffusion_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model = DiffusionForecaster(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=diffusion_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=diffusion_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            save_path = str(self.config.get_model_path("DIFFUSION", self.symbol))
            criterion = nn.MSELoss()
            
        elif self.model_mode == "GRAPH":
            graph_steps = abs(predict_days) if predict_days > 0 else 1
            model = GraphTemporalModel(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=graph_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=graph_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model = GraphTemporalModel(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=graph_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            test_model._init_args = dict(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                predict_steps=graph_steps,
                use_symbol_embedding=self.symbol_embed_enabled,
                symbol_embedding_dim=self.symbol_embed_dim,
                max_symbols=self.symbol_vocab_size,
            )
            save_path = str(self.config.get_model_path("GRAPH", self.symbol))
            criterion = nn.MSELoss()
            
        elif self.model_mode == "CNNLSTM":
            assert abs(abs(predict_days)) > 0, "Error: predict_days must be greater than 0"
            model = CNNLSTM(input_dim=self.input_dim, num_classes=self.output_dim, predict_days=abs(predict_days))
            model._init_args = dict(input_dim=self.input_dim, num_classes=self.output_dim, predict_days=abs(predict_days))
            test_model = CNNLSTM(input_dim=self.input_dim, num_classes=self.output_dim, predict_days=abs(predict_days))
            test_model._init_args = dict(input_dim=self.input_dim, num_classes=self.output_dim, predict_days=abs(predict_days))
            save_path = self.cnnlstm_path
            criterion = nn.MSELoss()
            
        else:
            logger.error(f"No such model: {self.model_mode}")
            exit(0)

        # Move to device
        model = model.to(self.device, non_blocking=True)
        if self.args.test_gpu == 0:
            test_model = test_model.to('cpu', non_blocking=True)
        else:
            test_model = test_model.to(self.device, non_blocking=True)
            
        if torch.cuda.device_count() > 1:
            logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
            model = nn.DataParallel(model)
            if self.args.test_gpu == 1:
                test_model = nn.DataParallel(test_model)
        elif torch.cuda.is_available():
            logger.info("Let's use 1 GPU!")
        else:
            logger.info("Let's use CPU!")

        logger.info(f"Model:\n{model}")
        
        # Optimizer & Scheduler
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = CustomSchedule(d_model=D_MODEL, warmup_steps=WARMUP_STEPS, optimizer=optimizer)

        # Load Checkpoints if available
        self._load_checkpoints(model, optimizer, save_path, predict_days)

        return model, test_model, optimizer, criterion, scheduler, save_path

    def _load_checkpoints(self, model, optimizer, save_path, predict_days):
        """Helper to load existing weights."""
        if predict_days > 0:
            ckpt_path = f"{save_path}_out{self.output_dim}_time{SEQ_LEN}_pre{predict_days}"
        else:
            ckpt_path = f"{save_path}_out{self.output_dim}_time{SEQ_LEN}"
            
        model_file = f"{ckpt_path}_Model.pkl"
        optim_file = f"{ckpt_path}_Optimizer.pkl"
        
        if os.path.exists(model_file) and os.path.exists(optim_file):
            logger.info("Load model and optimizer from file")
            try:
                # Handle DataParallel wrapping
                state_dict = torch.load(model_file, map_location=self.device)
                if isinstance(model, nn.DataParallel):
                    model.module.load_state_dict(state_dict)
                else:
                    model.load_state_dict(state_dict)
                    
                optimizer.load_state_dict(torch.load(optim_file, map_location=self.device))
            except Exception as e:
                logger.error(f"Error loading checkpoints: {e}, training from scratch.")
        else:
            logger.info("No model and optimizer file, train from scratch")
