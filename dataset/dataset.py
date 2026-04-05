import torch
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
# Suppress HuggingFace warning about sequences longer than model_max_length;
# we handle truncation manually for training (random-window augmentation).
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)


class AgnosticDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512, is_train: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train

        self.df = dataframe.reset_index(drop=True)

        required_cols = {'code', 'label', 'agnostic_features'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Dataframe missing required columns: {required_cols - set(self.df.columns)}")

        self.features_matrix = np.array(self.df['agnostic_features'].tolist(), dtype=np.float32)
        self.num_samples = len(self.df)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        logger.info(f"AgnosticDataset | {'TRAIN' if is_train else 'VAL/TEST'} | Samples: {self.num_samples}")

    def __len__(self):
        return self.num_samples

    def _normalize_features(self, feature_vector: torch.Tensor) -> torch.Tensor:
        x = feature_vector.clone()
        x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=0.0)
        for i in [0, 1, 7]:
            if i < x.shape[0]:
                x[i] = torch.log1p(x[i])
        return torch.clamp(x, min=0.0, max=100.0)

    def __getitem__(self, idx):
        code = str(self.df.iat[idx, self.df.columns.get_loc('code')])
        label = int(self.df.iat[idx, self.df.columns.get_loc('label')])

        norm_feats = self._normalize_features(
            torch.tensor(self.features_matrix[idx], dtype=torch.float32)
        )

        if self.is_train:
            # Encode full sequence for random-window augmentation
            input_ids = self.tokenizer.encode(code, add_special_tokens=True, truncation=False)
            total_len = len(input_ids)
            if total_len > self.max_length:
                start_token = input_ids[0]
                max_start_idx = total_len - self.max_length + 1
                random_start = np.random.randint(1, max_start_idx)
                final_input_ids = [start_token] + input_ids[random_start: random_start + self.max_length - 1]
            else:
                final_input_ids = input_ids
        else:
            # Let tokenizer truncate directly — avoids long-sequence warning
            final_input_ids = self.tokenizer.encode(
                code, add_special_tokens=True, truncation=True, max_length=self.max_length
            )

        processed_len = len(final_input_ids)
        padding_needed = self.max_length - processed_len

        if padding_needed > 0:
            final_input_ids = final_input_ids + [self.pad_token_id] * padding_needed
            attention_mask = [1] * processed_len + [0] * padding_needed
        else:
            attention_mask = [1] * self.max_length

        return {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "extra_features": norm_feats,
            "labels": torch.tensor(label, dtype=torch.long),
        }


class SimpleTextDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer, max_length: int = 512, is_train: bool = False, is_use_agnostic: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_train = is_train
        self.is_use_agnostic = is_use_agnostic

        self.df = dataframe.reset_index(drop=True)

        required_cols = {'code', 'label'}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"Dataframe missing required columns: {required_cols - set(self.df.columns)}")

        self.num_samples = len(self.df)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        if self.is_use_agnostic and 'agnostic_features' in self.df.columns:
            self.features_matrix = np.array(self.df['agnostic_features'].tolist(), dtype=np.float32)

        logger.info(f"SimpleTextDataset | {'TRAIN' if is_train else 'VAL/TEST'} | Samples: {self.num_samples}")

    def _get_agnostic_features(self, idx: int) -> torch.Tensor:
        def _normalize_features(feature_vector: torch.Tensor) -> torch.Tensor:
            x = feature_vector.clone()
            x = torch.nan_to_num(x, nan=0.0, posinf=100.0, neginf=0.0)
            for i in [0, 1, 7]:
                if i < x.shape[0]:
                    x[i] = torch.log1p(x[i])
            return torch.clamp(x, min=0.0, max=100.0)
        
        if 'agnostic_features' in self.df.columns:
            self.features_matrix = np.array(self.df['agnostic_features'].tolist(), dtype=np.float32)
            return _normalize_features(torch.tensor(self.features_matrix[idx], dtype=torch.float32))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        code = str(self.df.iat[idx, self.df.columns.get_loc('code')])
        label = int(self.df.iat[idx, self.df.columns.get_loc('label')])

        if self.is_train:
            # Encode full sequence for random-window augmentation
            input_ids = self.tokenizer.encode(code, add_special_tokens=True, truncation=False)
            total_len = len(input_ids)
            if total_len > self.max_length:
                start_token = input_ids[0]
                max_start_idx = total_len - self.max_length + 1
                random_start = np.random.randint(1, max_start_idx)
                final_input_ids = [start_token] + input_ids[random_start: random_start + self.max_length - 1]
            else:
                final_input_ids = input_ids
        else:
            # Let tokenizer truncate directly — avoids long-sequence warning
            final_input_ids = self.tokenizer.encode(
                code, add_special_tokens=True, truncation=True, max_length=self.max_length
            )

        processed_len = len(final_input_ids)
        padding_needed = self.max_length - processed_len

        if padding_needed > 0:
            final_input_ids = final_input_ids + [self.pad_token_id] * padding_needed
            attention_mask = [1] * processed_len + [0] * padding_needed
        else:
            attention_mask = [1] * self.max_length
            
        response = {
            "input_ids": torch.tensor(final_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long),
        }
        
        if self.is_use_agnostic:
            extra_features = self._get_agnostic_features(idx)
            response["extra_features"] = extra_features
        return response


def load_data(config, tokenizer):
    model_type = config.get("model", {}).get("model_type", "hybrid")
    max_len = config["data"]["max_length"]
    max_length = config["data"]["max_length"]
    tokenizer.model_max_length = max_length
    
    perplexity_model_name_save = (
        config.get('data', {})
        .get('perplexity_model', 'model')
        .rsplit('/', 1)[-1]
        .replace('.', '_')
        .replace('-', '_')
        .lower()
    )
    
    data_dir = config["data"]["data_dir"]
    logger.info(f"Loading raw parquet files from {data_dir} for DroidDetect...")
    train_df = pd.read_parquet(f"{data_dir}/train_processed_{perplexity_model_name_save}.parquet")
    logger.info(f"Loaded train data successfully at: {data_dir}/train_processed_{perplexity_model_name_save}.parquet")
    val_df = pd.read_parquet(f"{data_dir}/val_processed_{perplexity_model_name_save}.parquet")
    logger.info(f"Loaded validation data successfully at: {data_dir}/val_processed_{perplexity_model_name_save}.parquet")

    train_df = train_df.dropna(subset=['label']).reset_index(drop=True)
    val_df = val_df.dropna(subset=['label']).reset_index(drop=True)
    is_use_agnostic = config.get("data", {}).get("use_agnostic_features", False)
    if model_type == "droiddetect":
        return (
            SimpleTextDataset(train_df, tokenizer, max_length=max_len, is_train=True, is_use_agnostic=is_use_agnostic),
            SimpleTextDataset(val_df, tokenizer, max_length=max_len, is_train=False, is_use_agnostic=is_use_agnostic),
        )
    else:
        return (
            AgnosticDataset(train_df, tokenizer, max_length=max_len, is_train=True),
            AgnosticDataset(val_df, tokenizer, max_length=max_len, is_train=False),
        )
