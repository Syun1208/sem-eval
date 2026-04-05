import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from pytorch_metric_learning import losses, miners


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits, dim=1)
        p = torch.exp(log_p)

        p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
        log_p_t = log_p.gather(1, targets.view(-1, 1)).squeeze(1)

        focal_weight = (1.0 - p_t) ** self.gamma

        if self.label_smoothing > 0:
            smooth_loss = -log_p.mean(dim=1)
            loss = focal_weight * (
                (1 - self.label_smoothing) * (-log_p_t)
                + self.label_smoothing * smooth_loss
            )
        else:
            loss = focal_weight * (-log_p_t)

        return loss.mean()


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, attention_mask):
        hidden_states = hidden_states.to(self.dense.weight.dtype)
        x = torch.tanh(self.dense(hidden_states))
        x = self.dropout(x)
        scores = self.out_proj(x).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(hidden_states * attn_weights, dim=1)


class FeatureGatingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.BatchNorm1d(output_dim * 2),
            nn.Mish(),
            nn.Dropout(dropout_rate),
            nn.Linear(output_dim * 2, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.Mish(),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.net(x)


class HybridClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})
        
        self.use_agnostic_features = data_cfg.get("use_agnostic_features", False)

        self.num_labels = model_cfg.get("num_labels", 2)
        self.num_handcrafted = data_cfg.get("num_handcrafted_features", 11)
        model_name = model_cfg.get("base_model", "microsoft/unixcoder-base")
        projection_dim = model_cfg.get("projection_dim", model_cfg.get("deberta_projection_dim", 32))

        print(f"Initializing HybridClassifier | Backbone: {model_name} | Features: {self.num_handcrafted}")

        hf_config = AutoConfig.from_pretrained(model_name)
        hf_config.hidden_dropout_prob = 0.2
        hf_config.attention_probs_dropout_prob = 0.2

        self.base_model = AutoModel.from_pretrained(model_name, config=hf_config)
        self.text_projection = nn.Linear(hf_config.hidden_size, projection_dim)

        if model_cfg.get("gradient_checkpointing", False):
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.hidden_size = hf_config.hidden_size
        self.pooler = AttentionPooler(self.hidden_size)
        self.feature_encoder = FeatureGatingNetwork(input_dim=self.num_handcrafted, output_dim=128)

        if self.use_agnostic_features:
            fusion_dim = self.hidden_size + 128
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.Mish(),
                nn.Dropout(0.3),
                nn.Linear(fusion_dim // 2, self.num_labels)
            )
        else:
            self.classifier = nn.Linear(projection_dim, self.num_labels)


        self._init_weights()

    def _init_weights(self):
        for m in self.feature_encoder.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, input_ids, attention_mask, extra_features, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_agnostic_features:
            style_embedding = self.feature_encoder(extra_features)
            text_embedding = self.pooler(outputs.last_hidden_state, attention_mask)

            combined_features = torch.cat([text_embedding, style_embedding], dim=1)
        else:
            combined_features = F.relu(self.text_projection(self.pooler(outputs.last_hidden_state, attention_mask)))
            combined_features = F.dropout(combined_features, p=0.3, training=self.training)
        logits = self.classifier(combined_features)

        loss = None
        if labels is not None:
            focal_gamma = self.config.get("training", {}).get("focal_gamma", 2.0)
            loss = FocalLoss(gamma=focal_gamma, label_smoothing=0.1)(logits, labels.view(-1)).unsqueeze(0)

        return logits, loss, combined_features


class TLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_cfg = config.get("model", {})
        data_cfg = config.get("data", {})


        model_name = model_cfg.get("base_model", model_cfg.get("droiddetect_base_model", "project-droid/DroidDetect-Base"))
        projection_dim = model_cfg.get("projection_dim", model_cfg.get("droiddetect_projection_dim", 128))
        self.num_classes = model_cfg.get("droiddetect_num_labels", 4)
        self.lambda_contrast = model_cfg.get("triplet_lambda", model_cfg.get("droiddetect_triplet_lambda", 0.1))

        print(f"Initializing TLModel | Backbone: {model_name} | Classes: {self.num_classes}")

        DROIDDETECT_BASE = "project-droid/DroidDetect-Base"
        MODERNBERT_BASE  = "answerdotai/ModernBERT-base"

        hf_config = AutoConfig.from_pretrained(MODERNBERT_BASE)
        self.text_encoder = AutoModel.from_pretrained(MODERNBERT_BASE, config=hf_config)

        if model_name == DROIDDETECT_BASE:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(repo_id=DROIDDETECT_BASE, filename="pytorch_model.bin")
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
            encoder_prefix = "text_encoder."
            encoder_sd = {
                k[len(encoder_prefix):]: v
                for k, v in state_dict.items()
                if k.startswith(encoder_prefix)
            }
            missing, _ = self.text_encoder.load_state_dict(encoder_sd, strict=False)
            if missing:
                print(f"[TLModel] Missing keys when loading DroidDetect encoder: {missing[:5]}")
            print(f"[TLModel] Loaded DroidDetect-Base encoder weights from {ckpt_path}")

        if model_cfg.get("gradient_checkpointing", False):
            self.text_encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            
        self.num_labels = model_cfg.get("num_labels", 2)
        self.text_projection = nn.Linear(hf_config.hidden_size, projection_dim)
        self._triplet_miner = miners.BatchHardMiner()
        self._triplet_loss_fn = losses.TripletMarginLoss(smooth_loss=True)
        self.class_weights = None

        self.num_handcrafted = data_cfg.get("num_handcrafted_features", 11)
        if data_cfg.get("use_agnostic_features", False):
            self.hidden_size = hf_config.hidden_size
            self.pooler = AttentionPooler(self.hidden_size)
            self.feature_encoder = FeatureGatingNetwork(input_dim=self.num_handcrafted, output_dim=128)
            fusion_dim = self.hidden_size + 128
            self.classifier = nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim // 2),
                nn.LayerNorm(fusion_dim // 2),
                nn.Mish(),
                nn.Dropout(0.3),
                nn.Linear(fusion_dim // 2, self.num_labels)
            )
        else:
            self.classifier = nn.Linear(projection_dim, self.num_classes)
        print(f"Initializing HybridClassifier | Backbone: {model_name} | Features: {self.num_handcrafted}")

    def set_class_weights(self, class_weights: torch.Tensor):
        self.class_weights = class_weights
        
    def forward(self, input_ids=None, attention_mask=None, extra_features=None, labels=None):
        hidden_states = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state

        if extra_features is not None:
            style_embedding = self.feature_encoder(extra_features)
            text_embedding = self.pooler(hidden_states, attention_mask)
            projected_text = torch.cat([text_embedding, style_embedding], dim=-1)
        else:
            sentence_embeddings = hidden_states.mean(dim=1)
            projected_text = F.relu(self.text_projection(sentence_embeddings))

        logits = self.classifier(projected_text)
        output = {"logits": logits, "fused_embedding": projected_text}

        if labels is not None:
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)

            labels = labels.long()

            loss_fct_ce = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
            )
            cross_entropy_loss = loss_fct_ce(logits, labels)

            hard_pairs = self._triplet_miner(projected_text, labels)
            contrastive_loss = self._triplet_loss_fn(projected_text, labels, hard_pairs)
            loss = cross_entropy_loss + self.lambda_contrast * contrastive_loss

            output["loss"] = loss.unsqueeze(0)
            output["cross_entropy_loss"] = cross_entropy_loss.unsqueeze(0)
            output["contrastive_loss"] = contrastive_loss.unsqueeze(0)

        return output


def build_model(config):
    model_type = config.get("model", {}).get("model_type", "hybrid")
    if model_type == "droiddetect":
        return TLModel(config)
    return HybridClassifier(config)


def get_model_name(config):
    model_type = config.get("model", {}).get("model_type", "hybrid")
    if model_type == "droiddetect":
        return config["model"].get("base_model", config["model"].get("droiddetect_base_model", "project-droid/DroidDetect-Base"))
    return config["model"].get("base_model", "microsoft/deberta-v3-base")


def get_label_names():
    return ["Human", "AI"]
