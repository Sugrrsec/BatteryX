"""
BatteryX Core Engine (batteryx_core.py)
=======================================
Single source of truth for BatteryX:
- Cell schema & causal feature engineering
- Water-filling balanced dataset weighting
- Graph bank with outcome provenance & vectorized causal retrieval
- Multimodal neural architecture (CurveCNN, SeqEnc, CellGAT, BatteryX)
- Causal monotone projection
- Split-conformal calibration & evaluation metrics
- Training engine with ModelEMA & patience early stopping
- Paired cell-level bootstrap confidence intervals
"""

import os
import sys
import math
import copy
import random
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F

CHEMS = ['LCO', 'LFP', 'NMC', 'NCA', 'Mixed']
FORMS = ['cylindrical', 'pouch', 'prismatic', 'unknown']
COND_NUM = ['temp_C', 'c_rate_charge', 'c_rate_discharge', 'v_max', 'v_min', 'nominal_Ah']

# Causal history features: feature 0 is non-capacity; features 1-7 are capacity-derived
F_NAMES = [
    'norm_cycle',        # 0: k / 1000.0 (non-capacity)
    'cap_proxy',         # 1: q_k / nominal_Ah (capacity-derived)
    'fade_q0',           # 2: (q_0 - q_k) / q_0 (capacity-derived)
    'fade_rate_10',      # 3: (q_{k-10} - q_k) / c_step_10 (capacity-derived)
    'fade_rate_25',      # 4: (q_{k-25} - q_k) / c_step_25 (capacity-derived)
    'fade_rate_50',      # 5: (q_{k-50} - q_k) / c_step_50 (capacity-derived)
    'cum_throughput',    # 6: sum(q_0..k) / nominal_Ah (capacity-derived)
    'early_descriptor'   # 7: mean(q_0..min(k,20)) / nominal_Ah (capacity-derived)
]




# Explicit input-provenance manifest for the capacity-free evaluation.
# The SOH/RUL labels remain targets; they are never included in model inputs.
CAPACITY_FREE_SPEC = {
    'retained_seq_features': ['norm_cycle'],
    'excluded_seq_features': F_NAMES[1:],
    'retained_cond_features': [name for name in COND_NUM if name != 'nominal_Ah'] + CHEMS + FORMS,
    'excluded_cond_features': ['nominal_Ah'],
    'retained_curve_channels': ['q_tilde_shape', 'dq_dv_shape'],
    'excluded_curve_channels': ['nominal_capacity_scaling'],
    'retained_bank_features': ['temp_C', 'c_rate_charge', 'c_rate_discharge', 'v_max', 'v_min',
                               'chemistry_one_hot', 'form_one_hot'],
    'excluded_bank_features': ['nominal_Ah', 'log_life', 'mean_fade', 'soh_25', 'soh_50', 'soh_75', 'early_cap'],
    'note': 'All measured-capacity-derived inputs and all bank outcomes are excluded. SOH remains a supervised target only. Curves retain span-normalized electrochemical shape [0, 1].'
}

# ==============================================================================
# 1. DATA SCHEMA & CAUSAL FEATURE EXTRACTION
# ==============================================================================

@dataclass
class Cell:
    cell_id: str
    dataset: str
    chem: str
    form: str
    cond_num: np.ndarray        # [6] Numerical operating conditions
    q_dis: np.ndarray           # [K] Discharge capacities (Ah)
    cycles: np.ndarray          # [K] Cycle indices
    soh: np.ndarray             # [K] SOH labels (q_dis / q0)
    rul: np.ndarray             # [K] Ground truth RUL (exact or lower bound)
    eol_cycle: Optional[int]    # Ground truth EOL cycle index (if exact RUL known)
    censored: bool              # True if EOL not observed
    curves: np.ndarray          # [K, L, 2] Delta-Q and dQ/dV curves
    features: np.ndarray        # [K, 8] Causal history features
    v_grid: np.ndarray          # [L] Discretized voltage points
    q0: float                   # Initial discharge capacity (Ah)

    def __hash__(self):
        return hash(self.cell_id)

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.cell_id == other.cell_id
        return False


def normalize_form_factor(form_raw: Any) -> str:
    """Normalizes arbitrary form factor strings. Never defaults silently (Fix D14)."""
    f = str(form_raw).lower()
    if any(k in f for k in ['cyl', '18650', '21700']):
        return 'cylindrical'
    elif 'pouch' in f:
        return 'pouch'
    elif any(k in f for k in ['prism', 'can']):
        return 'prismatic'
    return 'unknown'


def cycle_summary(v, q, temp, q0, vmax, vmin, L=64, window_frac=(0.10, 0.90)):
    """Converts raw discharge (V, Q, T) curve into normalized shape Q_tilde(V) in [0, 1] and dQ_tilde/dV."""
    v = np.asarray(v, dtype=float)
    q = np.asarray(q, dtype=float)
    valid = np.isfinite(v) & np.isfinite(q)
    v, q = v[valid], q[valid]
    if len(v) < 10:
        return None

    lo = vmin + window_frac[0] * (vmax - vmin)
    hi = vmin + window_frac[1] * (vmax - vmin)
    if hi <= lo:
        lo, hi = vmin, vmax
    v_grid = np.linspace(lo, hi, L)

    sort_idx = np.argsort(v)
    v_sort = v[sort_idx]
    q_sort = q[sort_idx]
    q_interp = np.interp(v_grid, v_sort, q_sort, left=0.0, right=float(q_sort.max()))

    # Per-cycle span normalization is independent of initial capacity q0.
    # A near-flat/invalid span carries no reliable shape information; encode
    # that case as zeros rather than falling back to q0 (which would leak it).
    q_min = float(q_sort.min())
    q_span = float(q_sort.max() - q_min)
    if np.isfinite(q_span) and q_span > 1e-4:
        q_tilde = np.clip((q_interp - q_min) / q_span, 0.0, 1.0)
    else:
        q_tilde = np.zeros_like(q_interp, dtype=np.float64)
    dq_dv = np.gradient(q_tilde, v_grid)
    dq_dv = uniform_filter1d(dq_dv, size=max(1, L // 16))

    curves = np.stack([q_tilde, dq_dv], axis=-1)  # [L, 2]
    return curves, v_grid


def make_cell(cell_id, dataset, chem, form, cond_num, cycles, q_dis, curves_arr, v_grid,
              min_cycles=20, eol_soh=0.80, eol_persistence=3):
    """Create a validated cell. EOL requires a persistent threshold crossing.

    A cell is right-censored when no run of `eol_persistence` consecutive
    measured cycles is at/below eol_soh. This prevents one isolated noisy
    capacity point from assigning EOL. The persistence rule is configurable.
    """
    cycles = np.asarray(cycles, dtype=np.int32).reshape(-1)
    q_dis = np.asarray(q_dis, dtype=np.float32).reshape(-1)
    cond_num = np.asarray(cond_num, dtype=np.float32).reshape(-1)
    curves = np.asarray(curves_arr, dtype=np.float32)
    v_grid = np.asarray(v_grid, dtype=np.float32).reshape(-1)

    if len(cycles) < int(min_cycles):
        return None
    if len(cycles) != len(q_dis) or curves.ndim != 3 or curves.shape[0] != len(cycles):
        raise ValueError(f"{cell_id}: cycles, capacities and curves are not aligned")
    if cond_num.shape != (6,):
        raise ValueError(f"{cell_id}: cond_num must contain exactly 6 values")
    if curves.shape[2] != 2 or curves.shape[1] != len(v_grid):
        raise ValueError(f"{cell_id}: curves must have shape [K, L, 2] aligned to v_grid")
    if len(v_grid) < 10 or not np.isfinite(v_grid).all():
        raise ValueError(f"{cell_id}: invalid voltage grid")
    if not np.isfinite(cycles).all() or np.any(cycles <= 0) or np.any(np.diff(cycles) <= 0):
        raise ValueError(f"{cell_id}: cycle indices must be positive and strictly increasing")
    if not np.isfinite(q_dis).all() or np.any(q_dis <= 0):
        raise ValueError(f"{cell_id}: discharge capacities must be finite and positive")
    if not np.isfinite(curves).all():
        raise ValueError(f"{cell_id}: non-finite curve values; clean the raw curve rather than hiding invalid labels")

    q0 = float(q_dis[0])
    nom_ah = float(cond_num[5])
    if not np.isfinite(nom_ah) or nom_ah <= 0 or q0 <= 0.05 * nom_ah:
        return None

    soh = q_dis / q0
    below = soh <= float(eol_soh)
    persistence = max(1, int(eol_persistence))
    eol_idx = None
    if len(below) >= persistence:
        for i in range(0, len(below) - persistence + 1):
            if bool(np.all(below[i:i + persistence])):
                eol_idx = i
                break

    if eol_idx is not None:
        eol_cycle = int(cycles[eol_idx])
        rul = np.maximum(0.0, (eol_cycle - cycles).astype(np.float32))
        censored = False
    else:
        eol_cycle = None
        censored = True
        rul = np.maximum(0.0, (float(cycles[-1]) - cycles).astype(np.float32))

    K = len(cycles)
    feats = np.zeros((K, 8), dtype=np.float32)
    for k in range(K):
        c_k = float(cycles[k])
        q_k = float(q_dis[k])
        feats[k, 0] = c_k / 1000.0
        feats[k, 1] = q_k / max(nom_ah, 1e-4)
        feats[k, 2] = (q0 - q_k) / max(q0, 1e-4)
        k10 = max(0, k - 10)
        k25 = max(0, k - 25)
        k50 = max(0, k - 50)
        feats[k, 3] = (float(q_dis[k10]) - q_k) / max(1.0, float(cycles[k] - cycles[k10]))
        feats[k, 4] = (float(q_dis[k25]) - q_k) / max(1.0, float(cycles[k] - cycles[k25]))
        feats[k, 5] = (float(q_dis[k50]) - q_k) / max(1.0, float(cycles[k] - cycles[k50]))
        feats[k, 6] = float(np.sum(q_dis[:k + 1])) / max(nom_ah, 1e-4)
        feats[k, 7] = float(np.mean(q_dis[:min(k + 1, 20)])) / max(nom_ah, 1e-4)

    return Cell(
        cell_id=str(cell_id), dataset=str(dataset), chem=str(chem), form=str(form),
        cond_num=cond_num, q_dis=q_dis, cycles=cycles, soh=soh.astype(np.float32),
        rul=rul.astype(np.float32), eol_cycle=eol_cycle, censored=bool(censored),
        curves=curves, features=feats, v_grid=v_grid, q0=q0
    )


# ==============================================================================
# 2. BALANCED WATER-FILLING DATASET WEIGHTING (Requirement C2)
# ==============================================================================

def solve_water_filling_weights(fit_cells: List[Cell], cap: float = 0.25) -> Tuple[Dict[str, float], Dict[str, float], bool]:
    """
    Solves dataset share_i = min(0.25, lambda * raw_share_i) such that sum(shares) == 1.0.
    Feasible only if D >= 4. If D < 4, uses equal shares (1/D) and reports infeasibility.
    Returns: (cell_weights, sample_weights, is_feasible).
    """
    ds_counts: Dict[str, int] = {}
    for c in fit_cells:
        ds_counts[c.dataset] = ds_counts.get(c.dataset, 0) + 1

    D = len(ds_counts)
    total_fit_cells = len(fit_cells)

    if D < 4:
        # Mathematically infeasible to cap every dataset at 25% when D < 4
        is_feasible = False
        ds_shares = {ds: 1.0 / D for ds in ds_counts}
    else:
        is_feasible = True
        raw_shares = {ds: count / total_fit_cells for ds, count in ds_counts.items()}
        # Iterative water-filling bisection for lambda
        low_l = 0.0
        high_l = 100.0
        for _ in range(100):
            mid_l = (low_l + high_l) / 2.0
            shares = {ds: min(cap, mid_l * r_s) for ds, r_s in raw_shares.items()}
            tot = sum(shares.values())
            if tot < 1.0:
                low_l = mid_l
            else:
                high_l = mid_l

        ds_shares = {ds: min(cap, low_l * raw_shares[ds]) for ds in ds_counts}
        # Final exact normalization
        s_sum = sum(ds_shares.values())
        ds_shares = {ds: s / s_sum for ds, s in ds_shares.items()}

    # Compute per-cell weight = ds_share / n_fit_cells_of_dataset
    cell_weights = {}
    for c in fit_cells:
        cell_weights[c.cell_id] = ds_shares[c.dataset] / ds_counts[c.dataset]

    # Verify cell weights sum to 1.0
    c_sum = sum(cell_weights.values())
    cell_weights = {cid: w / c_sum for cid, w in cell_weights.items()}

    return ds_shares, cell_weights, is_feasible


# ==============================================================================
# 3. NORMALIZATION STATS & OUTCOME-BEARING GRAPH BANK (Requirements F, G)
# ==============================================================================

def fit_stats(fit_cells: List[Cell], eol_soh: float = 0.80) -> Dict[str, Any]:
    """Fit-only robust imputation/scaling statistics. Targets are never imputed."""
    if not fit_cells:
        raise ValueError("Cannot fit normalization statistics on an empty fit split.")

    cond = np.vstack([np.asarray(c.cond_num, dtype=np.float64).reshape(1, -1) for c in fit_cells])
    feat = np.vstack([np.asarray(c.features, dtype=np.float64).reshape(-1, np.asarray(c.features).shape[-1]) for c in fit_cells])

    def _fit_matrix(x, label):
        if x.ndim != 2 or not np.isfinite(x).any(axis=0).any():
            raise ValueError(f"{label}: no finite input values available.")
        med = np.zeros(x.shape[1], dtype=np.float64)
        filled = x.copy()
        for j in range(x.shape[1]):
            good = np.isfinite(x[:, j])
            med[j] = float(np.median(x[good, j])) if good.any() else 0.0
            filled[~good, j] = med[j]
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-5), std, 1.0)
        return med, mean, std

    cond_impute, cond_mean, cond_std = _fit_matrix(cond, 'Condition features')
    feat_impute, feat_mean, feat_std = _fit_matrix(feat, 'Sequence features')

    soh_parts = [np.asarray(c.soh, dtype=np.float64).reshape(-1) for c in fit_cells]
    all_soh = np.concatenate(soh_parts)
    if not np.isfinite(all_soh).all():
        raise ValueError("Non-finite SOH targets found in fit cells; target labels must be fixed or excluded, not imputed.")

    log_rul = []
    for c in fit_cells:
        if not c.censored:
            r = np.asarray(c.rul, dtype=np.float64)
            valid = np.isfinite(r) & (r > 0)
            if valid.any():
                log_rul.extend(np.log1p(r[valid]).tolist())
    if not log_rul:
        raise ValueError("Fit split contains no exact positive RUL targets for normalization.")
    log_rul = np.asarray(log_rul, dtype=np.float64)

    return {
        'cond_mean': cond_mean.tolist(), 'cond_std': cond_std.tolist(),
        'feat_mean': feat_mean.tolist(), 'feat_std': feat_std.tolist(),
        'cond_impute': cond_impute.tolist(), 'feat_impute': feat_impute.tolist(),
        'soh_mean': float(np.mean(all_soh)), 'soh_std': max(float(np.std(all_soh)), 1e-5),
        'log_rul_mean': float(np.mean(log_rul)), 'log_rul_std': max(float(np.std(log_rul)), 1e-5),
        'eol_soh': float(eol_soh)
    }


def build_bank(fit_cells: List[Cell], stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Constructs descriptor and outcome bank strictly from fit cells (Requirement G).
    Bank features: 15 descriptor dims + 6 outcome dims = 21 dims total.
    """
    c_means = np.array(stats['cond_mean'])
    c_stds = np.array(stats['cond_std'])

    descs = []
    outcomes = []
    cell_ids = []
    datasets = []
    chems = []

    for c in fit_cells:
        c_raw = np.asarray(c.cond_num, dtype=np.float32).copy()
        c_med = np.asarray(stats.get('cond_impute', c_means), dtype=np.float32)
        c_raw[~np.isfinite(c_raw)] = c_med[~np.isfinite(c_raw)]
        c_norm = (c_raw - c_means) / c_stds
        chem_oh = np.zeros(len(CHEMS), dtype=np.float32)
        if c.chem in CHEMS:
            chem_oh[CHEMS.index(c.chem)] = 1.0
        form_oh = np.zeros(len(FORMS), dtype=np.float32)
        if c.form in FORMS:
            form_oh[FORMS.index(c.form)] = 1.0

        # Descriptor: normalized conditions (6) + chem (5) + form (4) = 15 dims
        desc = np.concatenate([c_norm, chem_oh, form_oh]).astype(np.float32)

        # Outcomes: log total life, mean fade rate, SOH at 25/50/75%, early capacity = 6 dims
        total_life = float(c.eol_cycle) if c.eol_cycle is not None else float(c.cycles[-1])
        log_life = math.log1p(total_life)
        mean_fade = float(c.q0 - c.q_dis[-1]) / max(1.0, float(len(c.cycles))) * 100.0

        k25 = min(len(c.soh) - 1, max(0, int(len(c.soh) * 0.25)))
        k50 = min(len(c.soh) - 1, max(0, int(len(c.soh) * 0.50)))
        k75 = min(len(c.soh) - 1, max(0, int(len(c.soh) * 0.75)))
        soh_25 = float(c.soh[k25])
        soh_50 = float(c.soh[k50])
        soh_75 = float(c.soh[k75])
        early_cap = float(c.q0) / max(float(c.cond_num[5]), 1e-4)

        outcome_vec = np.array([log_life, mean_fade, soh_25, soh_50, soh_75, early_cap], dtype=np.float32)

        descs.append(desc)
        outcomes.append(outcome_vec)
        cell_ids.append(c.cell_id)
        datasets.append(c.dataset)
        chems.append(c.chem)

    desc_tensor = torch.tensor(np.array(descs), dtype=torch.float32)
    outcome_tensor = torch.tensor(np.array(outcomes), dtype=torch.float32)
    # Combined bank feature tensor [N_fit, 21]
    combined_features = torch.cat([desc_tensor, outcome_tensor], dim=-1)

    return {
        'descriptors': desc_tensor,          # [N_fit, 15]
        'outcomes': outcome_tensor,          # [N_fit, 6]
        'bank_features': combined_features,  # [N_fit, 21]
        'cell_ids': cell_ids,
        'datasets': datasets,
        'chems': chems
    }


# ==============================================================================
# 4. CAUSAL VECTORIZED BATTERY POOL (Requirements F, G, D11)
# ==============================================================================

class BatteryPool(torch.utils.data.Dataset):
    """
    Fully pre-tensorized multimodal dataset.
    Precomputes cell-level top-k graph neighbors once per cell (754 cells)
    instead of running torch.topk on the CPU for every cycle sample.
    """
    def __init__(self, cells: List[Cell], stats: Dict[str, Any], bank: Dict[str, Any],
                 knn: int = 5, is_train: bool = False, p_domain_mask: float = 0.5,
                 cell_weights: Optional[Dict[str, float]] = None, stride: int = 1,
                 capacity_free: bool = False):
        self.cells = cells
        self.stats = stats
        self.bank = bank
        self.knn = knn
        self.is_train = is_train
        self.p_domain_mask = p_domain_mask
        self.capacity_free = capacity_free
        self.feature_manifest = copy.deepcopy(CAPACITY_FREE_SPEC) if capacity_free else {
            'retained_seq_features': list(F_NAMES), 'excluded_seq_features': [],
            'retained_cond_features': list(COND_NUM) + CHEMS + FORMS, 'excluded_cond_features': [],
            'retained_curve_channels': ['q_tilde_shape', 'dq_dv_shape'], 'excluded_curve_channels': [],
            'retained_bank_features': ['descriptors_15d', 'outcomes_6d'], 'excluded_bank_features': []
        }
        self.samples = []
        self._build_index(stride, cell_weights)

    def _pad_neighbors(self, feats: torch.Tensor) -> torch.Tensor:
        if feats.shape[0] < self.knn:
            feat_dim = feats.shape[1] if feats.shape[0] > 0 else self.bank['bank_features'].size(-1)
            pad = torch.zeros(self.knn - feats.shape[0], feat_dim, dtype=feats.dtype)
            feats = torch.cat([feats, pad], dim=0)
        if self.capacity_free:
            feats = feats.clone()
            feats[:, 5] = 0.0
            feats[:, 15:] = 0.0
        return feats

    def _build_index(self, stride: int, cell_weights: Optional[Dict[str, float]]):
        c_mean = np.array(self.stats['cond_mean'], dtype=np.float32)
        c_std = np.array(self.stats['cond_std'], dtype=np.float32)
        f_mean = np.array(self.stats['feat_mean'], dtype=np.float32)
        f_std = np.array(self.stats['feat_std'], dtype=np.float32)

        bank_descs = self.bank['descriptors'].clone()
        bank_feats_all = self.bank['bank_features'].clone()
        if self.capacity_free:
            bank_descs[:, 5] = 0.0
            bank_feats_all[:, 5] = 0.0
            bank_feats_all[:, 15:] = 0.0

        bank_ds = self.bank['datasets']
        bank_cids = self.bank['cell_ids']

        for c_idx, cell in enumerate(self.cells):
            K = len(cell.cycles)
            cond_raw = np.asarray(cell.cond_num, dtype=np.float32).copy()
            cond_med = np.asarray(self.stats.get('cond_impute', c_mean), dtype=np.float32)
            cond_raw[~np.isfinite(cond_raw)] = cond_med[~np.isfinite(cond_raw)]
            c_norm = (cond_raw - c_mean) / c_std
            if self.capacity_free:
                c_norm[5] = 0.0

            chem_oh = np.zeros(len(CHEMS), dtype=np.float32)
            if cell.chem in CHEMS:
                chem_oh[CHEMS.index(cell.chem)] = 1.0
            form_oh = np.zeros(len(FORMS), dtype=np.float32)
            if cell.form in FORMS:
                form_oh[FORMS.index(cell.form)] = 1.0
            cond_vec = np.concatenate([c_norm, chem_oh, form_oh]).astype(np.float32)

            feat_raw = np.asarray(cell.features, dtype=np.float32).copy()
            feat_med = np.asarray(self.stats.get('feat_impute', f_mean), dtype=np.float32)
            bad_feat = ~np.isfinite(feat_raw)
            feat_raw[bad_feat] = np.broadcast_to(feat_med, feat_raw.shape)[bad_feat]
            norm_feats = (feat_raw - f_mean) / f_std
            if self.capacity_free:
                norm_feats[:, 1:] = 0.0

            # Precompute neighbor feature tensors ONCE per cell (instead of per cycle sample!)
            self_mask = torch.tensor([b_cid != cell.cell_id for b_cid in bank_cids], dtype=torch.bool)
            same_ds_mask = torch.tensor([b_ds == cell.dataset for b_ds in bank_ds], dtype=torch.bool)
            q_desc = torch.tensor(cond_vec, dtype=torch.float32).unsqueeze(0)
            dists_base = torch.norm(bank_descs - q_desc, dim=-1)
            dists_base[~self_mask] = float('inf')

            k_all = min(self.knn, int(torch.isfinite(dists_base).sum().item()))
            neigh_all = bank_feats_all[torch.topk(dists_base, k_all, largest=False).indices] if k_all > 0 else torch.zeros(0, bank_feats_all.size(-1))
            neigh_all = self._pad_neighbors(neigh_all)

            dists_masked = dists_base.clone()
            dists_masked[same_ds_mask] = float('inf')
            k_mask = min(self.knn, int(torch.isfinite(dists_masked).sum().item()))
            neigh_masked = bank_feats_all[torch.topk(dists_masked, k_mask, largest=False).indices] if k_mask > 0 else torch.zeros(0, bank_feats_all.size(-1))
            neigh_masked = self._pad_neighbors(neigh_masked)

            cell_w = cell_weights.get(cell.cell_id, 1.0 / len(self.cells)) if cell_weights else 1.0 / len(self.cells)
            cyc_step = stride if not self.is_train else max(1, K // 80)
            sampled_cycles = list(range(0, K, cyc_step))
            sample_w = cell_w / max(1, len(sampled_cycles))

            for k in sampled_cycles:
                w_start = max(0, k - 64 + 1)
                seq_chunk = norm_feats[w_start:k+1]
                if len(seq_chunk) < 64:
                    pad = np.zeros((64 - len(seq_chunk), 8), dtype=np.float32)
                    seq_chunk = np.vstack([pad, seq_chunk])

                curve = np.nan_to_num(
                    np.asarray(cell.curves[k], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
                ).transpose(1, 0)

                soh_val = float(cell.soh[k])
                rul_val = float(cell.rul[k])
                if not (np.isfinite(soh_val) and np.isfinite(rul_val)):
                    continue
                total_life = float(cell.eol_cycle) if cell.eol_cycle is not None else float(cell.cycles[-1])
                pre_eol = bool(cell.censored or (cell.eol_cycle is not None and int(cell.cycles[k]) < int(cell.eol_cycle)))
                rul_frac = max(0.0, min(1.0, rul_val / max(total_life, 1.0)))
                soh_z = (soh_val - self.stats['soh_mean']) / self.stats['soh_std']
                rul_z = (math.log1p(max(0.0, rul_val)) - self.stats['log_rul_mean']) / self.stats['log_rul_std']

                self.samples.append({
                    'cond': torch.from_numpy(cond_vec),
                    'seq': torch.from_numpy(seq_chunk),
                    'curve': torch.from_numpy(curve),
                    'neigh_all': neigh_all,
                    'neigh_masked': neigh_masked,
                    'soh_z': torch.tensor(soh_z, dtype=torch.float32),
                    'rul_z': torch.tensor(rul_z, dtype=torch.float32),
                    'rul_frac': torch.tensor(rul_frac, dtype=torch.float32),
                    'soh_raw': torch.tensor(soh_val, dtype=torch.float32),
                    'rul_raw': torch.tensor(rul_val, dtype=torch.float32),
                    'censored': torch.tensor(1.0 if cell.censored else 0.0, dtype=torch.float32),
                    'exact_mask': torch.tensor(0.0 if cell.censored else 1.0, dtype=torch.float32),
                    'pre_eol_mask': torch.tensor(1.0 if pre_eol else 0.0, dtype=torch.float32),
                    'total_life': torch.tensor(total_life, dtype=torch.float32),
                    'sample_w': torch.tensor(sample_w, dtype=torch.float32),
                    'cell_int_id': torch.tensor(c_idx, dtype=torch.long),
                    'cycle': torch.tensor(int(cell.cycles[k]), dtype=torch.long),
                    'cell_id': str(cell.cell_id),
                    'dataset': str(cell.dataset),
                    'chem': str(cell.chem),
                    'cycle_k': int(cell.cycles[k]),
                    'soh': soh_val,
                    'rul': rul_val,
                    'cond_vec': cond_vec,
                    'dists': dists_base,
                    'self_mask': self_mask,
                    'same_ds_mask': same_ds_mask
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        use_masked = self.is_train and (random.random() < self.p_domain_mask)
        neighbors = s['neigh_masked'] if use_masked else s['neigh_all']
        return {
            'cond': s['cond'],
            'seq': s['seq'],
            'curve': s['curve'],
            'neighbors': neighbors,
            'soh_z': s['soh_z'],
            'rul_z': s['rul_z'],
            'rul_frac': s['rul_frac'],
            'soh_raw': s['soh_raw'],
            'rul_raw': s['rul_raw'],
            'censored': s['censored'],
            'exact_mask': s['exact_mask'],
            'pre_eol_mask': s['pre_eol_mask'],
            'total_life': s['total_life'],
            'sample_w': s['sample_w'],
            'cell_int_id': s['cell_int_id'],
            'cycle': s['cycle'],
            'cell_id': s['cell_id'],
            'dataset': s['dataset'],
            'chem': s['chem']
        }


def assert_capacity_free_batch(batch: Dict[str, Any]) -> None:
    """Assert that a collated capacity-free batch contains no prohibited model inputs."""
    assert torch.all(batch['seq'][:, :, 1:] == 0), 'Capacity-derived sequence inputs are nonzero.'
    if not torch.all(batch['curve'] == 0):
        c0 = batch['curve'][:, 0, :]
        assert c0.min() >= -0.05 and c0.max() <= 1.05, 'Curve channel 0 is not span-normalized shape in [0, 1]!'
    assert torch.all(batch['cond'][:, 5] == 0), 'Nominal capacity condition input is nonzero.'
    assert torch.all(batch['neighbors'][:, :, 5] == 0), 'Nominal capacity graph descriptor is nonzero.'
    assert torch.all(batch['neighbors'][:, :, 15:] == 0), 'Outcome-derived graph inputs are nonzero.'

# ==============================================================================
# 5. NEURAL ARCHITECTURES & MONOTONE PROJECTION (Requirement A, N)
# ==============================================================================

class CurveCNN(nn.Module):
    """1D-CNN processing continuous Delta-Q(V) and differential capacity curves."""
    def __init__(self, in_channels=2, d_model=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(4),
            nn.Flatten(),
            nn.Linear(64 * 4, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        return self.net(x)


class SeqEnc(nn.Module):
    """Temporal encoder processing historical cycle sequences with positional encoding."""
    def __init__(self, in_dim=8, d_model=64, seq_type='transformer', nhead=4, num_layers=2):
        super().__init__()
        self.seq_type = seq_type
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, d_model) * 0.02)
        if seq_type == 'transformer':
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                dropout=0.1, activation='gelu', batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        else:
            self.encoder = nn.LSTM(
                d_model, d_model // 2, num_layers=num_layers,
                batch_first=True, bidirectional=True
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.proj(x)
        if self.seq_type == 'transformer':
            seq_len = x.size(1)
            h = h + self.pos_emb[:, :seq_len, :]
            out = self.encoder(h)
        else:
            out, _ = self.encoder(h)
        return self.norm(out[:, -1, :])


class CellGAT(nn.Module):
    """Multi-head Graph Attention receiving neighbor descriptors + outcomes (21 dims)."""
    def __init__(self, desc_dim=21, d_model=64, heads=4):
        super().__init__()
        self.heads = heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(desc_dim, d_model)
        self.v_proj = nn.Linear(desc_dim, d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, batch_first=True)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, target_rep, neighbor_feats):
        # target_rep: [B, d_model], neighbor_feats: [B, K, 21]
        B = target_rep.size(0)
        q = self.q_proj(target_rep).unsqueeze(1)  # [B, 1, d_model]
        k = self.k_proj(neighbor_feats)           # [B, K, d_model]
        v = self.v_proj(neighbor_feats)           # [B, K, d_model]

        valid = neighbor_feats.abs().sum(dim=-1) > 0  # [B, K]
        no_neighbors = ~valid.any(dim=1)             # [B]

        padding_mask = ~valid
        if no_neighbors.any():
            padding_mask = padding_mask.clone()
            padding_mask[no_neighbors] = False

        attn_out, _ = self.mha(query=q, key=k, value=v, key_padding_mask=padding_mask)
        attn_out = attn_out.squeeze(1)  # [B, d_model]
        attn_out = self.out_proj(attn_out)
        attn_out = attn_out.masked_fill(no_neighbors.unsqueeze(-1), 0.0)
        return self.norm(target_rep + attn_out)


class BatteryX(nn.Module):
    """Unified multimodal degradation prognostics network with coupled SOH-RUL heads."""
    def __init__(self, cond_dim=15, desc_dim=21, d_model=64, seq_type='transformer',
                 knn=5, heads=4, use_cnn=True, use_seq=True, use_tab=True,
                 use_graph=True, use_monotone_head=False, stats: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.use_cnn = use_cnn
        self.use_seq = use_seq
        self.use_tab = use_tab
        self.use_graph = use_graph
        self.use_monotone_head = use_monotone_head

        # Normalized EOL margin anchor
        if stats is not None and 'soh_mean' in stats and 'soh_std' in stats:
            eol_soh = float(stats.get('eol_soh', 0.80))
            eol_soh_z = (eol_soh - float(stats['soh_mean'])) / max(float(stats['soh_std']), 1e-4)
        else:
            eol_soh_z = -1.25  # Standard default corresponding to 0.80 SOH threshold
        self.register_buffer('eol_soh_z', torch.tensor(eol_soh_z, dtype=torch.float32))

        if use_cnn:
            self.cnn = CurveCNN(in_channels=2, d_model=d_model)
        if use_seq:
            self.seq = SeqEnc(in_dim=8, d_model=d_model, seq_type=seq_type, nhead=heads)
        if use_tab:
            self.tab = nn.Sequential(
                nn.Linear(cond_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(d_model)
            )

        n_branches = sum([use_cnn, use_seq, use_tab])
        fuse_in = max(1, n_branches * d_model)
        self.fusion = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

        if use_graph:
            self.gat = CellGAT(desc_dim=desc_dim, d_model=d_model, heads=heads)

        self.soh_mu = nn.Linear(d_model, 1)
        # Condition RUL heads explicitly on rep and predicted distance to EOL threshold
        self.rul_mu = nn.Linear(d_model + 1, 1)
        self.rul_frac_logits = nn.Linear(d_model + 1, 1)

        if use_monotone_head:
            self.mono_delta = nn.Sequential(
                nn.Linear(d_model, 1),
                nn.Softplus()
            )

    def forward(self, batch):
        feats = []
        if self.use_cnn:
            feats.append(self.cnn(batch['curve']))
        if self.use_seq:
            feats.append(self.seq(batch['seq']))
        if self.use_tab:
            feats.append(self.tab(batch['cond']))

        h = torch.cat(feats, dim=-1) if len(feats) > 0 else torch.zeros(batch['cond'].size(0), 64, device=batch['cond'].device)
        rep = self.fusion(h)

        if self.use_graph and 'neighbors' in batch:
            rep = self.gat(rep, batch['neighbors'])

        soh_pred = self.soh_mu(rep).squeeze(-1)
        margin_to_eol = (soh_pred - self.eol_soh_z).unsqueeze(-1)
        rul_in = torch.cat([rep, margin_to_eol], dim=-1)
        rul_pred = self.rul_mu(rul_in).squeeze(-1)
        frac_logits = self.rul_frac_logits(rul_in).squeeze(-1)

        out = {
            'soh_z': soh_pred,
            'rul_z': rul_pred,
            'rul_frac_logits': frac_logits,
            'rep': rep
        }
        if self.use_monotone_head:
            out['delta'] = self.mono_delta(rep).squeeze(-1)
        return out


def causal_monotone_project(soh_raw: np.ndarray, window: int = 5) -> np.ndarray:
    """Causal monotone projection enforcing irreversible fade over past cycles strictly."""
    soh_raw = np.asarray(soh_raw, dtype=float)
    K = len(soh_raw)
    if K <= 1:
        return soh_raw.copy()

    alpha = 2.0 / (window + 1.0)
    ema = np.zeros(K, dtype=float)
    ema[0] = soh_raw[0]
    for i in range(1, K):
        ema[i] = alpha * soh_raw[i] + (1.0 - alpha) * ema[i - 1]

    proj = np.zeros(K, dtype=float)
    running_min = ema[0]
    for i in range(K):
        running_min = min(running_min, ema[i])
        proj[i] = running_min
    return proj


# ==============================================================================
# 6. LOSSES & SPLIT-CONFORMAL CALIBRATION (Requirements C2, I)
# ==============================================================================

def combined_loss(outputs, batch, stats, lambda_phys=1.0, huber_delta=1.0, lambda_cens=0.5,
                  use_weights=True, return_metrics=False):
    """Vectorized composite loss. Set return_metrics=True only when scalar logs are needed."""
    device = outputs['soh_z'].device
    pred_soh = outputs['soh_z'].float()
    pred_rul = outputs['rul_z'].float()
    target_soh = batch['soh_z'].float()
    target_rul = batch['rul_z'].float()

    weights = batch['sample_w'].float() if use_weights and ('sample_w' in batch) else torch.ones_like(target_soh)
    weights = weights / weights.mean().clamp_min(1e-6)

    raw_l_soh = F.huber_loss(pred_soh, target_soh, delta=huber_delta, reduction='none')
    l_soh = (raw_l_soh * weights).mean()

    pre_eol = batch.get('pre_eol_mask', torch.ones_like(batch['exact_mask'])).float()
    exact_mask = batch['exact_mask'].float() * pre_eol
    w_exact = weights * exact_mask
    exact_denom = w_exact.sum().clamp_min(1e-6)
    raw_l_rul = F.huber_loss(pred_rul, target_rul, delta=huber_delta, reduction='none')
    l_rul = (raw_l_rul * w_exact).sum() / exact_denom
    raw_l_frac = F.binary_cross_entropy_with_logits(
        outputs['rul_frac_logits'].float(), batch['rul_frac'].float(), reduction='none'
    )
    l_frac = (raw_l_frac * w_exact).sum() / exact_denom

    cens_mask = batch['censored'].float()
    w_cens = weights * cens_mask
    cens_denom = w_cens.sum().clamp_min(1e-6)
    viol = F.relu(target_rul - pred_rul)
    l_cens = (viol * w_cens).sum() / cens_denom

    # Stable two-pass lexicographic sort: cycle first, then cell ID. This avoids
    # composite-key collisions and keeps all work on the tensor's current device.
    if 'cell_int_id' in batch and 'cycle' in batch and len(batch['cycle']) > 1:
        cell_ids = batch['cell_int_id'].to(device=device, dtype=torch.long)
        cycles = batch['cycle'].to(device=device, dtype=torch.long)
        by_cycle = torch.argsort(cycles, stable=True)
        order = by_cycle[torch.argsort(cell_ids[by_cycle], stable=True)]
        same_cell = cell_ids[order][1:] == cell_ids[order][:-1]
        diffs = pred_soh[order][1:] - pred_soh[order][:-1]
        same_float = same_cell.to(dtype=pred_soh.dtype)
        l_phys = (F.relu(diffs) * same_float).sum() / same_float.sum().clamp_min(1.0)
    else:
        l_phys = pred_soh.new_zeros(())

    # Optimize a small cycle-space term alongside log-RUL to penalize large
    # absolute lifetime misses. Normalize using the fit partition only.
    reference_rul = max(1.0, math.expm1(min(float(stats['log_rul_mean']), 12.0)))
    cycle_log = pred_rul * float(stats['log_rul_std']) + float(stats['log_rul_mean'])
    # Cap the training-only exponential to bound gradients on pathological batches.
    relative_error = (torch.expm1(cycle_log.clamp(0.0, 16.0)) - batch['rul_raw'].float()) / reference_rul
    l_cycle = (F.smooth_l1_loss(relative_error, torch.zeros_like(relative_error), reduction='none')
               * w_exact).sum() / exact_denom

    l_delta = pred_soh.new_zeros(())
    if 'delta' in outputs and 'cell_int_id' in batch and 'cycle' in batch and len(batch['cycle']) > 1:
        left, right = order[:-1], order[1:]
        gap = (cycles[right] - cycles[left]).float()
        valid = (same_cell & (gap > 0)).float()
        target_rate = F.relu(target_soh[left] - target_soh[right]) / gap.clamp_min(1.0)
        delta = outputs['delta'].float()
        pred_rate = 0.5 * (delta[left] + delta[right])
        rate_error = F.smooth_l1_loss(torch.log1p(pred_rate), torch.log1p(target_rate), reduction='none')
        l_delta = (rate_error * valid).sum() / valid.sum().clamp_min(1.0)

    total_loss = l_soh + l_rul + 0.2 * l_frac + lambda_cens * l_cens + lambda_phys * l_phys + 0.15 * l_cycle + 0.05 * l_delta
    if return_metrics:
        metrics = {
            'loss': float(total_loss.detach().item()),
            'l_soh': float(l_soh.detach().item()),
            'l_rul': float(l_rul.detach().item()),
            'l_cens': float(l_cens.detach().item()),
            'l_phys': float(l_phys.detach().item()),
            'l_cycle': float(l_cycle.detach().item()),
            'l_fade_rate': float(l_delta.detach().item())
        }
    else:
        metrics = {}
    return total_loss, metrics

def compute_rul_interval(rul_z: float, q_conformal: Optional[float], stats: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Computes lower bound, upper bound, and interval width in actual RUL cycles.
    Inverse transform from normalized log-RUL space: z = (log(1 + RUL) - mu) / sigma.
    Handles NaN, infinity, and finite quantiles separately.
    Lower bound is clipped to zero.
    """
    if q_conformal is None or math.isnan(q_conformal):
        return float('nan'), float('nan'), float('nan')
    if math.isinf(q_conformal):
        return 0.0, float('inf'), float('inf')

    mu = float(stats['log_rul_mean'])
    sigma = float(stats['log_rul_std'])
    z_lo = float(rul_z) - float(q_conformal)
    z_hi = float(rul_z) + float(q_conformal)

    log_lo = z_lo * sigma + mu
    log_hi = z_hi * sigma + mu

    r_lo = float(max(0.0, math.expm1(min(log_lo, 80.0))))
    r_hi = float('inf') if log_hi > 80.0 else float(max(0.0, math.expm1(log_hi)))
    width = float(r_hi - r_lo) if math.isfinite(r_hi) else float('inf')
    return r_lo, r_hi, width


def required_conformal_cells(alpha: float, min_floor: int = 9) -> Tuple[int, int]:
    """Return (minimum finite-quantile cells, mathematical finite-quantile threshold)."""
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    n_min_finite = int(math.ceil((1.0 - float(alpha)) / float(alpha)))
    return max(int(min_floor), n_min_finite), n_min_finite


def calibrate_conformal_rul_detailed(model, cal_pool, stats, alpha: float = 0.10,
                                     allow_insufficient: bool = False) -> Dict[str, Any]:
    """Cell-level simultaneous split-conformal calibration with explicit status metadata."""
    return conformal_from_cell_scores(collect_conformal_cell_scores(model, cal_pool), alpha, allow_insufficient)


def collect_conformal_cell_scores(model, cal_pool):
    """Collect calibration residuals once for all requested confidence levels."""
    model.eval()
    loader = torch.utils.data.DataLoader(cal_pool, batch_size=512, shuffle=False)
    cell_errors: Dict[str, List[float]] = {}
    with torch.no_grad():
        for batch in loader:
            device = next(model.parameters()).device
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            output = model(batch_dev)
            pred_z = output['rul_z'].detach().cpu().numpy()
            true_z = batch['rul_z'].numpy()
            exact = batch['exact_mask'].numpy()
            pre_eol = batch.get('pre_eol_mask', torch.ones_like(batch['exact_mask'])).numpy()
            cell_ids = batch['cell_id']
            for cid, pred, true, is_exact, before_eol in zip(cell_ids, pred_z, true_z, exact, pre_eol):
                if is_exact > 0.5 and before_eol > 0.5:
                    cell_errors.setdefault(str(cid), []).append(abs(float(pred - true)))

    return cell_errors


def conformal_from_cell_scores(cell_errors, alpha=0.10, allow_insufficient=False):
    n_cal = len(cell_errors)
    n_min_required, n_min_finite = required_conformal_cells(alpha, min_floor=9)
    invalid_score_cells = [cid for cid, scores in cell_errors.items() if not scores or not np.isfinite(np.asarray(scores, dtype=float)).all()]
    if invalid_score_cells:
        info = {'alpha': float(alpha), 'n_cal': int(n_cal), 'n_min_required': int(n_min_required),
                'n_min_finite': int(n_min_finite), 'rank': int(math.ceil((n_cal + 1) * (1.0 - float(alpha)))),
                'calibration_status': 'invalid_nonfinite_scores', 'quantile_status': 'invalid_nonfinite_scores',
                'q_conformal': float('nan'), 'reason': f'Non-finite calibration scores for cells: {invalid_score_cells[:5]}'}
        if not allow_insufficient:
            raise FloatingPointError(info['reason'])
        return info
    rank = int(math.ceil((n_cal + 1) * (1.0 - float(alpha))))
    base = {
        'alpha': float(alpha), 'n_cal': int(n_cal), 'n_min_required': int(n_min_required),
        'n_min_finite': int(n_min_finite), 'rank': int(rank)
    }
    if n_cal < 9:
        info = {**base, 'calibration_status': 'insufficient_calibration_cells',
                'quantile_status': 'invalid_insufficient_data', 'q_conformal': float('nan'),
                'reason': f'Only {n_cal} eligible exact-RUL calibration cells; {n_min_required} required for a finite quantile at alpha={alpha}.'}
        if not allow_insufficient:
            raise RuntimeError(info['reason'])
        return info

    cell_scores = sorted(max(scores) for scores in cell_errors.values() if scores)
    if rank > n_cal:
        return {**base, 'calibration_status': 'unbounded_interval',
                'quantile_status': 'unbounded_interval', 'q_conformal': float('inf'),
                'reason': f'Conformal rank {rank} exceeds {n_cal} eligible calibration cells.'}

    q_value = float(cell_scores[rank - 1])
    return {**base, 'calibration_status': 'finite_interval', 'quantile_status': 'finite',
            'q_conformal': q_value, 'reason': 'OK'}


def calibrate_conformal_rul(model, cal_pool, stats, alpha=0.10, is_target_domain=False) -> float:
    """Compatibility wrapper. Raises for <9 eligible cells; returns infinity only for rank>n."""
    info = calibrate_conformal_rul_detailed(model, cal_pool, stats, alpha=alpha, allow_insufficient=False)
    return float(info['q_conformal'])

# ==============================================================================
# 7. TRAINING ENGINE WITH PATIENCE EARLY STOPPING (Requirements E, Q)
# ==============================================================================

class ModelEMA:
    """Exponential Moving Average of model parameters."""
    def __init__(self, model, decay=0.995):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            average, current = [], []
            for name, value in model.state_dict().items():
                shadow = self.shadow[name]
                if torch.is_floating_point(value):
                    average.append(shadow)
                    current.append(value)
                else:
                    shadow.copy_(value)
            if average:
                torch._foreach_lerp_(average, current, 1.0 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)


def set_eol_threshold_from_stats(model, stats):
    """Set the normalized EOL anchor from the current fit-fold statistics."""
    if not hasattr(model, 'eol_soh_z'):
        return None
    soh_mean = float(stats['soh_mean'])
    soh_std = max(float(stats['soh_std']), 1e-4)
    eol_soh = float(stats.get('eol_soh', 0.80))
    eol_z = (eol_soh - soh_mean) / soh_std
    if not math.isfinite(eol_z):
        raise ValueError('Non-finite normalized EOL threshold; check fit-fold SOH statistics.')
    with torch.no_grad():
        model.eol_soh_z.copy_(model.eol_soh_z.new_tensor(eol_z))
    return eol_z


def train_model(train_pool, val_pool, stats, model=None, epochs=25, lr=1e-3, wd=1e-4,
                patience=7, log_interval=5, lambda_phys=1.0, huber_delta=1.0,
                device='cpu', batch_size=256, num_workers=None, use_amp=None) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train BatteryX with fold-correct EOL anchoring, EMA, AMP and efficient input transfer."""
    import time
    from contextlib import nullcontext

    if len(train_pool) == 0:
        raise ValueError('Training pool is empty; revise the cell-level split or data filters.')
    if len(val_pool) == 0:
        raise ValueError('Validation pool is empty; a validation partition is required for early stopping.')

    device_obj = torch.device(device)
    if model is None:
        model = BatteryX(stats=stats).to(device_obj)
    else:
        model = model.to(device_obj)
    # Essential: many benchmark/ablation constructors pass a model without stats.
    # Re-anchor its RUL heads to this run's fit-only normalization before EMA copies it.
    set_eol_threshold_from_stats(model, stats)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, threshold=1e-4, min_lr=lr * 0.1
    )
    ema = ModelEMA(model, decay=0.995)
    ema_model = copy.deepcopy(model)

    use_amp = (device_obj.type == 'cuda') if use_amp is None else bool(use_amp and device_obj.type == 'cuda')
    try:
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if num_workers is None:
        num_workers = int(os.environ.get('BATTERYX_NUM_WORKERS', '2' if device_obj.type == 'cuda' and os.name != 'nt' else '0'))
    num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError('num_workers must be >= 0.')
    pin_memory = (device_obj.type == 'cuda')
    loader_options = {'num_workers': num_workers, 'pin_memory': pin_memory}
    if num_workers > 0:
        loader_options.update({'persistent_workers': True, 'prefetch_factor': 2})

    train_loader = torch.utils.data.DataLoader(
        train_pool, batch_size=min(int(batch_size), len(train_pool)), shuffle=True, **loader_options
    )
    val_loader = torch.utils.data.DataLoader(
        val_pool, batch_size=min(max(512, int(batch_size)), len(val_pool)), shuffle=False, **loader_options
    )

    best_metric = float('inf')
    best_state = None
    patience_counter = 0
    history = {key: [] for key in (
        'train_loss', 'val_metric', 'val_soh_mae', 'val_rul_mae_cycles',
        'val_rul_median_life_relative_error', 'val_exact_cycles', 'learning_rate',
        'train_seconds', 'validation_seconds', 'train_samples_per_second', 'is_best'
    )}

    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        ep_loss = torch.zeros((), device=device_obj, dtype=torch.float32)
        epoch_finite = torch.ones((), device=device_obj, dtype=torch.bool)
        n_b = 0

        for b in train_loader:
            b_dev = {
                k: v.to(device_obj, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
                for k, v in b.items()
            }
            optimizer.zero_grad(set_to_none=True)
            amp_context = (torch.autocast(device_type='cuda', dtype=torch.float16)
                           if use_amp else nullcontext())
            with amp_context:
                out = model(b_dev)
                loss, _ = combined_loss(
                    out, b_dev, stats, lambda_phys=lambda_phys,
                    huber_delta=huber_delta, return_metrics=False
                )
            epoch_finite = epoch_finite & torch.isfinite(loss.detach())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            ep_loss = ep_loss + loss.detach().float()
            n_b += 1

        # One device synchronization per epoch, not several for every minibatch.
        epoch_values = torch.stack([ep_loss, epoch_finite.float()]).detach().cpu().tolist()
        if epoch_values[1] < 0.5:
            raise FloatingPointError('Non-finite training loss observed in this epoch; training stopped.')
        train_loss = float(epoch_values[0]) / max(1, n_b)
        history['train_loss'].append(train_loss)
        t_epoch = time.time() - t0

        # Validate on EMA weights using one persistent evaluation model.
        ema.apply(ema_model)
        ema_model.eval()

        val_start = time.perf_counter()
        val_soh_errs, val_norm_rul_errs, val_rul_errs = [], [], []
        with torch.inference_mode():
            for b in val_loader:
                b_dev = {
                    k: v.to(device_obj, non_blocking=pin_memory) if isinstance(v, torch.Tensor) else v
                    for k, v in b.items()
                }
                out = ema_model(b_dev)
                soh_p = out['soh_z'].float().cpu().numpy() * stats['soh_std'] + stats['soh_mean']
                val_soh_errs.extend(np.abs(soh_p - b['soh_raw'].numpy()))

                rul_p = np.expm1(np.maximum(0.0, out['rul_z'].float().cpu().numpy() * stats['log_rul_std'] + stats['log_rul_mean']))
                pre = b.get('pre_eol_mask', torch.ones_like(b['exact_mask'])).numpy()
                eligible = (b['exact_mask'].numpy() > 0.5) & (pre > 0.5)
                error = np.abs(rul_p[eligible] - b['rul_raw'].numpy()[eligible])
                val_rul_errs.extend(error)
                val_norm_rul_errs.extend(error / np.maximum(1.0, b['total_life'].numpy()[eligible]))

        if not val_soh_errs:
            raise ValueError('Validation pool produced no finite SOH errors.')
        val_soh_mae = float(np.mean(val_soh_errs))
        val_norm_rul = float(np.median(val_norm_rul_errs)) if val_norm_rul_errs else float('nan')
        comp_metric = val_soh_mae + val_norm_rul if np.isfinite(val_norm_rul) else val_soh_mae
        if not np.isfinite(comp_metric) or not np.isfinite(val_soh_mae):
            raise FloatingPointError('Non-finite validation predictions; checkpoint selection stopped.')
        val_rul_mae = float(np.mean(val_rul_errs)) if val_rul_errs else float('nan')
        history['val_metric'].append(comp_metric)
        history['val_soh_mae'].append(val_soh_mae)
        history['val_rul_mae_cycles'].append(val_rul_mae)
        history['val_rul_median_life_relative_error'].append(val_norm_rul)
        history['val_exact_cycles'].append(len(val_rul_errs))
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        history['train_seconds'].append(t_epoch)
        history['validation_seconds'].append(time.perf_counter() - val_start)
        history['train_samples_per_second'].append(len(train_pool) / max(t_epoch, 1e-9))
        history['is_best'].append(comp_metric < best_metric)
        scheduler.step(comp_metric)

        if comp_metric < best_metric:
            best_metric = comp_metric
            best_state = copy.deepcopy(ema_model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if ep % log_interval == 0 or ep == epochs:
            print(f"   [Epoch {ep:2d}/{epochs:2d}] Loss {train_loss:.4f} | "
                  f"Val score {comp_metric:.4f} | SOH MAE {val_soh_mae * 100:.2f} pp | "
                  f"RUL MAE {val_rul_mae:.1f} cycles | LR {history['learning_rate'][-1]:.2e} | "
                  f"Train/val {t_epoch:.1f}/{history['validation_seconds'][-1]:.1f}s")
        if patience_counter >= patience:
            print(f"   Early stopping triggered at epoch {ep} (patience={patience}).")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history

def evaluate_pool(model, pool, stats, q_conformal: Optional[float] = None, save_csv_path: Optional[str] = None,
                  protocol: str = 'in_dist', seed: int = 42) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Evaluates pool, records prediction store with complete metadata, and returns metrics."""
    model.eval()
    loader = torch.utils.data.DataLoader(pool, batch_size=512, shuffle=False)
    cell_data: Dict[str, List[Dict[str, Any]]] = {}

    with torch.no_grad():
        for b in loader:
            b_dev = {k: v.to(next(model.parameters()).device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
            out = model(b_dev)
            soh_pred = out['soh_z'].cpu().numpy() * stats['soh_std'] + stats['soh_mean']
            rul_z = out['rul_z'].float().cpu().numpy()
            rul_log = rul_z * stats['log_rul_std'] + stats['log_rul_mean']
            if not (np.isfinite(soh_pred).all() and np.isfinite(rul_log).all()):
                raise FloatingPointError('Non-finite predictions encountered during evaluation.')
            rul_pred = np.expm1(np.clip(rul_log, 0.0, 80.0))

            cycle = b['cycle'].numpy()
            soh_true = b['soh_raw'].numpy()
            rul_true = b['rul_raw'].numpy()
            exact = b['exact_mask'].numpy()
            pre_eol = b.get('pre_eol_mask', torch.ones_like(b['exact_mask'])).numpy()
            life = b['total_life'].numpy()
            for i, cell_id in enumerate(b['cell_id']):
                cid = str(cell_id)
                cell_data.setdefault(cid, []).append({
                    'cycle': int(cycle[i]),
                    'soh_pred_raw': float(soh_pred[i]),
                    'soh_true': float(soh_true[i]),
                    'rul_pred': float(rul_pred[i]),
                    'rul_true': float(rul_true[i]),
                    'rul_z_pred': float(rul_z[i]),
                    'exact_mask': float(exact[i]),
                    'pre_eol_mask': float(pre_eol[i]),
                    'total_life': float(life[i]),
                    'dataset': str(b['dataset'][i]),
                    'chem': str(b['chem'][i]),
                    'protocol': protocol,
                    'seed': int(seed)
                })

    pred_records = []
    all_soh_raw_errs, all_soh_proj_errs = [], []
    all_rul_errs, all_norm_rul_errs, all_mdape_errs = [], [], []
    covered_cycles = 0
    total_exact_cycles = 0
    covered_cells = 0
    total_exact_cells = 0
    widths, rel_widths = [], []

    for cid, samples in cell_data.items():
        samples = sorted(samples, key=lambda x: x['cycle'])
        soh_raw = np.array([s['soh_pred_raw'] for s in samples])
        soh_true = np.array([s['soh_true'] for s in samples])
        soh_proj = causal_monotone_project(soh_raw)

        all_soh_raw_errs.extend(np.abs(soh_raw - soh_true))
        all_soh_proj_errs.extend(np.abs(soh_proj - soh_true))

        exact_s = [s for s in samples if s['exact_mask'] > 0.5 and s.get('pre_eol_mask', 1.0) > 0.5]
        cell_cov = True if exact_s else False

        for idx, s in enumerate(samples):
            r_p = s['rul_pred']
            r_t = s['rul_true']
            is_exact = s['exact_mask'] > 0.5 and s.get('pre_eol_mask', 1.0) > 0.5

            if is_exact:
                total_exact_cycles += 1
                err = abs(r_p - r_t)
                all_rul_errs.append(err)
                all_norm_rul_errs.append(err / max(1.0, s['total_life']))
                all_mdape_errs.append(err / max(1.0, r_t))

            r_lo, r_hi, w = compute_rul_interval(s['rul_z_pred'], q_conformal, stats)

            if is_exact:
                if math.isfinite(w):
                    widths.append(w)
                    rel_widths.append(w / max(1.0, r_t))
                    if (r_t >= r_lo) and (r_t <= r_hi):
                        covered_cycles += 1
                    else:
                        cell_cov = False
                elif math.isinf(w):
                    covered_cycles += 1
                else:
                    cell_cov = False

            pred_records.append({
                'cell_id': cid,
                'dataset': s['dataset'],
                'chem': s['chem'],
                'cycle': s['cycle'],
                'protocol': s['protocol'],
                'seed': s['seed'],
                'soh_true': s['soh_true'],
                'soh_pred_raw': float(soh_raw[idx]),
                'soh_pred_proj': float(soh_proj[idx]),
                'rul_true': r_t,
                'rul_pred': r_p,
                'rul_z_pred': s['rul_z_pred'],
                'interval_lo': r_lo,
                'interval_hi': r_hi,
                'interval_width': w,
                'interval_status': ('invalid_insufficient_data' if q_conformal is None or math.isnan(float(q_conformal)) else ('unbounded_interval' if math.isinf(float(q_conformal)) else 'finite')),
                'exact_mask': s['exact_mask'],
                'pre_eol_mask': s.get('pre_eol_mask', 1.0),
                'total_life': s['total_life']
            })

        if exact_s:
            total_exact_cells += 1
            if cell_cov:
                covered_cells += 1

    pred_df = pd.DataFrame(pred_records)
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path) or '.', exist_ok=True)
        pred_df.to_csv(save_csv_path, index=False)

    if q_conformal is None or math.isnan(float(q_conformal)):
        cycle_cov_pct = float('nan')
        cell_cov_pct = float('nan')
        mean_w = float('nan')
        mean_rel_w = float('nan')
        conformal_status = 'invalid_insufficient_data'
    elif math.isinf(float(q_conformal)):
        # An unbounded interval trivially covers all points but is not useful
        # evidence of calibration quality. Do not report 100% coverage or width.
        cycle_cov_pct = float('nan')
        cell_cov_pct = float('nan')
        mean_w = float('nan')
        mean_rel_w = float('nan')
        conformal_status = 'unbounded_interval'
    else:
        cycle_cov_pct = (covered_cycles / max(1, total_exact_cycles)) * 100.0 if total_exact_cycles else float('nan')
        cell_cov_pct = (covered_cells / max(1, total_exact_cells)) * 100.0 if total_exact_cells else float('nan')
        mean_w = float(np.mean(widths)) if widths else float('nan')
        mean_rel_w = float(np.mean(rel_widths)) if rel_widths else float('nan')
        conformal_status = 'finite'

    metrics = {
        'SOH_MAE_raw': float(np.mean(all_soh_raw_errs)) if all_soh_raw_errs else float('nan'),
        'SOH_MAE_proj': float(np.mean(all_soh_proj_errs)) if all_soh_proj_errs else float('nan'),
        'RUL_MAE': float(np.mean(all_rul_errs)) if all_rul_errs else float('nan'),
        'RUL_PerCell_MAE_Life%': float(np.mean(all_norm_rul_errs)) * 100.0 if all_norm_rul_errs else float('nan'),
        'RUL_MdAPE%': float(np.median(all_mdape_errs)) * 100.0 if all_mdape_errs else float('nan'),
        'Conformal_Cycle_Cov%': cycle_cov_pct,
        'Conformal_Cell_Cov%': cell_cov_pct,
        'Conformal_Width_Cycles': mean_w,
        'Conformal_Rel_Width%': mean_rel_w * 100.0 if math.isfinite(mean_rel_w) else mean_rel_w,
        'Conformal_Status': conformal_status
    }
    return metrics, pred_df


@torch.no_grad()
def predict_pool(model, pool, q_conformal: Optional[float] = None, batch_size: int = 512) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Infers physical SOH, RUL, and conformal interval bounds on a BatteryPool."""
    model.eval()
    loader = torch.utils.data.DataLoader(pool, batch_size=batch_size, shuffle=False)
    device = next(model.parameters()).device
    st = pool.stats

    soh_raw_list = []
    rul_list = []
    rul_lo_list = []
    rul_hi_list = []

    for b in loader:
        b_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in b.items()}
        out = model(b_dev)
        soh_z = out['soh_z'].cpu().numpy()
        rul_z = out['rul_z'].cpu().numpy()

        soh_raw = np.clip(soh_z * st['soh_std'] + st['soh_mean'], 0.0, 1.05)
        rul_log = rul_z * st['log_rul_std'] + st['log_rul_mean']
        if not (np.isfinite(soh_z).all() and np.isfinite(rul_z).all() and np.isfinite(rul_log).all()):
            raise FloatingPointError('Non-finite predictions encountered during inference.')
        rul_val = np.expm1(np.clip(rul_log, 0.0, 80.0))

        soh_raw_list.extend(soh_raw)
        rul_list.extend(rul_val)

        for r_z in rul_z:
            r_lo, r_hi, _ = compute_rul_interval(float(r_z), q_conformal, st)
            rul_lo_list.append(r_lo)
            rul_hi_list.append(r_hi)

    return (np.array(soh_raw_list, dtype=np.float32),
            np.array(rul_list, dtype=np.float32),
            np.array(rul_lo_list, dtype=np.float32),
            np.array(rul_hi_list, dtype=np.float32))



def paired_cell_bootstrap_ci(errs_a: Dict[str, float], errs_b: Dict[str, float],
                             n_boot: int = 1000, seed: int = 42) -> Tuple[float, float, float, str]:
    """
    Paired cell-level bootstrap resampling over common eligible test cells (Requirement M).
    Returns: (diff_mean, ci_lo, ci_hi, significance_label).
    """
    common_cids = sorted(list(set(errs_a.keys()).intersection(set(errs_b.keys()))))
    if len(common_cids) == 0:
        return 0.0, 0.0, 0.0, "INSUFFICIENT DATA"

    diffs = np.array([errs_a[cid] - errs_b[cid] for cid in common_cids])
    diff_mean = float(np.mean(diffs))

    rng = np.random.RandomState(seed)
    boot_means = []
    n = len(common_cids)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_means.append(float(np.mean(diffs[idx])))

    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    if (ci_lo <= 0.0) and (ci_hi >= 0.0):
        label = "NO SIGNIFICANT DIFFERENCE"
    elif ci_hi < 0.0:
        label = "BETTER"
    else:
        label = "WORSE"

    return diff_mean, ci_lo, ci_hi, label
