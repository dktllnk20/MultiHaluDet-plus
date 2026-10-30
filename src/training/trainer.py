import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from src.models.multihaludet import MultiHaluDet
from src.models.losses import FocalLoss, AsymmetricLoss, ContrastiveLoss, PairwiseAUCLoss
from src.training.augmentations import mixup_data, cutmix_data

EVAL_BATCH_SIZE = 1024

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.step = 0
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        # Warm-up: a low decay in the first steps lets the average leave the initial weights.
        decay = min(self.decay, (1 + self.step) / (10 + self.step))
        self.step += 1
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(decay).add_(param.data, alpha=1.0 - decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        self.backup = {}

def _to_tensors(device, *arrays):
    return [torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device) for a in arrays]

@torch.inference_mode()
def _forward_batches(model, seq, glob, **kwargs):
    model.eval()
    return torch.cat([model(s, g, **kwargs)
                      for s, g in zip(seq.split(EVAL_BATCH_SIZE), glob.split(EVAL_BATCH_SIZE))])

def train_deep_model_fold(X_seq_tr, X_glob_tr, y_tr, X_seq_val, X_glob_val, y_val, config, log_prefix=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = config.use_amp and device.type == "cuda"

    n_pos = float(np.sum(y_tr))
    n_neg = len(y_tr) - n_pos

    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], device=device)

    # The features are small: keep them on the device and index batches directly.
    seq_tr, glob_tr, tgt_tr, seq_val, glob_val = _to_tensors(
        device, X_seq_tr, X_glob_tr, y_tr, X_seq_val, X_glob_val)
    tgt_tr = tgt_tr.unsqueeze(1)
    y_val = np.asarray(y_val)

    model = MultiHaluDet(
        seq_dim=X_seq_tr.shape[2],
        global_dim=X_glob_tr.shape[1],
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        num_llm_layers=X_seq_tr.shape[1],
        scales=config.scales,
        dropout=config.dropout,
        layer_delta=config.use_layer_delta,
        scale_layer_weights=config.scale_layer_weights
    ).to(device)

    bce_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    focal_crit = FocalLoss(alpha_pos=config.focal_alpha_pos, alpha_neg=config.focal_alpha_neg, gamma=config.focal_gamma)
    asym_crit = AsymmetricLoss(gamma_neg=3, gamma_pos=1, clip=0.03)
    cont_crit = ContrastiveLoss(config.contrastive_temp)
    rank_crit = PairwiseAUCLoss(config.rank_loss_tau)

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=config.min_lr)

    if config.use_ema:
        ema = EMA(model, decay=config.ema_decay)

    best_auc = -1.0
    best_model = copy.deepcopy(model.state_dict())
    patience_counter = 0

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        model.train()

        if epoch < config.warmup_epochs:
            warmup_lr = config.learning_rate * (epoch + 1) / config.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        for idx in torch.randperm(len(tgt_tr), device=device).split(config.batch_size):
            s, g, t = seq_tr[idx], glob_tr[idx], tgt_tr[idx]
            optimizer.zero_grad()

            aug_choice = random.random()
            mixed = True
            if aug_choice < 0.33 and config.use_mixup:
                s, g, t_a, t_b, lam = mixup_data(s, g, t, config.mixup_alpha)
            elif aug_choice < 0.66 and config.use_cutmix:
                s, g, t_a, t_b, lam = cutmix_data(s, g, t, config.cutmix_alpha)
            else:
                mixed = False

            # Only the forward pass runs in mixed precision; the losses stay in float32.
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                output = model(s, g, return_embedding=not mixed)

            if mixed:
                out = output.float()
                loss = lam * bce_crit(out, t_a) + (1 - lam) * bce_crit(out, t_b)
            else:
                out, emb = output[0].float(), output[1].float()
                t_smooth = t * (1 - config.label_smoothing) + 0.5 * config.label_smoothing
                loss = 0.45 * bce_crit(out, t_smooth) + 0.35 * focal_crit(out, t) + 0.20 * asym_crit(out, t)
                loss += config.contrastive_weight * cont_crit(emb, t)
                if config.rank_loss_weight > 0:
                    loss += config.rank_loss_weight * rank_crit(out, t)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            if config.use_ema:
                ema.update()

        if config.use_ema:
            ema.apply_shadow()

        probs = torch.sigmoid(_forward_batches(model, seq_val, glob_val)).cpu().numpy().ravel()

        if config.use_ema:
            ema.restore()

        # A one-class fold or NaN outputs have no usable AUC; score them as chance.
        probs = np.nan_to_num(probs, nan=0.5)
        auc_score = roc_auc_score(y_val, probs) if len(np.unique(y_val)) > 1 else 0.5
        if epoch >= config.warmup_epochs:
            scheduler.step(auc_score)

        if auc_score > best_auc:
            best_auc = auc_score
            if config.use_ema:
                ema.apply_shadow()
                best_model = copy.deepcopy(model.state_dict())
                ema.restore()
            else:
                best_model = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        print(f"{log_prefix}epoch {epoch + 1:02d}/{config.epochs} "
              f"{time.perf_counter() - epoch_start:.1f}s auc {auc_score:.4f} best {best_auc:.4f} "
              f"lr {optimizer.param_groups[0]['lr']:.1e}", flush=True)

        if patience_counter >= config.patience:
            print(f"{log_prefix}early stop at epoch {epoch + 1}", flush=True)
            break

    model.load_state_dict(best_model)
    return model, best_auc

def extract_features_batch(model, X_seq, X_glob):
    device = next(model.parameters()).device
    seq, glob = _to_tensors(device, X_seq, X_glob)
    return _forward_batches(model, seq, glob, return_features=True).cpu().numpy()

def extract_logits_batch(model, X_seq, X_glob):
    """One logit per row, shape [N]. Comparable across fold networks, unlike the 576-d features."""
    device = next(model.parameters()).device
    seq, glob = _to_tensors(device, X_seq, X_glob)
    return _forward_batches(model, seq, glob).cpu().numpy().ravel()
