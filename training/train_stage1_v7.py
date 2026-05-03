import os

import sys

import json

import math

import random

import argparse

from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(current_dir)

if project_root not in sys.path:

    sys.path.append(project_root)

from models.TARCE_HACS import TRACE

def set_seed(seed: int = 42):

    random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

def parse_int_list(s: str):

    if s is None or len(s.strip()) == 0:

        return []

    return [int(x.strip()) for x in s.split(",") if len(x.strip()) > 0]

def masked_mean(x, mask, dim=1, eps=1e-12):

    mask = mask.float()

    denom = mask.sum(dim=dim, keepdim=True).clamp_min(eps)

    return (x * mask.unsqueeze(-1)).sum(dim=dim) / denom

def trajectory_mean_pool(step_vecs, step_mask):

    step_vecs = F.normalize(step_vecs, p=2, dim=-1)

    traj = masked_mean(step_vecs, step_mask, dim=1)

    return F.normalize(traj, p=2, dim=-1)

def trajectory_usage_pool(step_weights, step_mask, eps=1e-12):

    mask = step_mask.float().unsqueeze(-1)

    denom = mask.sum(dim=1).clamp_min(eps)

    usage = (step_weights * mask).sum(dim=1) / denom

    return usage / usage.sum(dim=1, keepdim=True).clamp_min(eps)

class TRACEStage1TrajectoryDataset(Dataset):

    def __init__(

        self,

        jsonl_path,

        tokenizer,

        max_step_length=128,

        max_traj_length=22,

        exclude_train_hazards=None,

        eval_unseen_hazards_only=False,

        split="train",

    ):

        self.tokenizer = tokenizer

        self.max_step_length = max_step_length

        self.max_traj_length = max_traj_length

        self.samples = []

        exclude_train_hazards = set(exclude_train_hazards or [])

        with open(jsonl_path, "r", encoding="utf-8") as f:

            raw = [json.loads(line) for line in f]

        dropped = 0

        kept_benign = 0

        kept_harmful = 0

        kept_unseen_harmful = 0

        lang_counter   = defaultdict(int)

        source_counter = defaultdict(int)

        for ridx, record in enumerate(raw):

            steps = record.get("steps", [])[:max_traj_length]

            if not steps:

                continue

            labels  = record.get("labels", {})

            meta    = record.get("meta", {})

            binary_label = int(labels.get("binary", -1))

            hazard_type  = int(labels.get("hazard_type", -1))

            source       = record.get("source", "unknown")

            lang         = meta.get("lang", "en")

            if split == "train" and binary_label == 1 and hazard_type in exclude_train_hazards:

                dropped += 1

                continue

            if split != "train" and eval_unseen_hazards_only:

                is_sdb = (source == "safedialbench")

                if not is_sdb and binary_label == 1 and hazard_type not in exclude_train_hazards:

                    dropped += 1

                    continue

            self.samples.append({

                "steps":            steps,

                "binary_label":     binary_label,

                "hazard_type":      hazard_type,

                "source":           source,

                "lang":             lang,

                "record_idx":       ridx,

                "is_unseen_hazard": int(

                    binary_label == 1 and hazard_type in exclude_train_hazards

                ),

            })

            if binary_label == 0:

                kept_benign += 1

            elif binary_label == 1:

                kept_harmful += 1

                if hazard_type in exclude_train_hazards:

                    kept_unseen_harmful += 1

            lang_counter[lang]     += 1

            source_counter[source] += 1

        print(f"[Dataset] {jsonl_path} ({split}): {len(self.samples)} trajectories")

        print(f"  kept_benign={kept_benign} | kept_harmful={kept_harmful} | dropped={dropped}")

        if len(exclude_train_hazards) > 0:

            print(f"  unseen_hazard_ids={sorted(list(exclude_train_hazards))}")

            print(f"  kept_unseen_harmful={kept_unseen_harmful}")

        print(f"  by_lang   = {dict(lang_counter)}")

        print(f"  by_source = {dict(source_counter)}")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        item = self.samples[idx]

        ids_list, mask_list = [], []

        for s in item["steps"]:

            enc = self.tokenizer(

                s,

                max_length=self.max_step_length,

                padding="max_length",

                truncation=True,

                return_tensors="pt",

            )

            ids_list.append(enc["input_ids"].squeeze(0))

            mask_list.append(enc["attention_mask"].squeeze(0))

        return {

            "input_ids_list":       ids_list,

            "attention_mask_list":  mask_list,

            "binary_label":         torch.tensor(item["binary_label"],     dtype=torch.long),

            "hazard_type":          torch.tensor(item["hazard_type"],       dtype=torch.long),

            "record_idx":           torch.tensor(item["record_idx"],        dtype=torch.long),

            "is_unseen_hazard":     torch.tensor(item["is_unseen_hazard"],  dtype=torch.long),

            "source":               item["source"],

            "lang":                 item["lang"],

        }

def stage1_collate_fn(batch):

    max_steps = max(len(b["input_ids_list"]) for b in batch)

    step_len  = batch[0]["input_ids_list"][0].shape[0]

    input_ids, attention_mask, step_mask = [], [], []

    for b in batch:

        n   = len(b["input_ids_list"])

        ids = b["input_ids_list"]

        ams = b["attention_mask_list"]

        if n < max_steps:

            pad = max_steps - n

            ids = ids + [torch.zeros(step_len, dtype=torch.long) for _ in range(pad)]

            ams = ams + [torch.zeros(step_len, dtype=torch.long) for _ in range(pad)]

        input_ids.append(torch.stack(ids))

        attention_mask.append(torch.stack(ams))

        sm = torch.zeros(max_steps, dtype=torch.long)

        sm[:n] = 1

        step_mask.append(sm)

    return {

        "input_ids":        torch.stack(input_ids),

        "attention_mask":   torch.stack(attention_mask),

        "step_mask":        torch.stack(step_mask),

        "binary_label":     torch.stack([b["binary_label"]    for b in batch]),

        "hazard_type":      torch.stack([b["hazard_type"]      for b in batch]),

        "record_idx":       torch.stack([b["record_idx"]       for b in batch]),

        "is_unseen_hazard": torch.stack([b["is_unseen_hazard"] for b in batch]),

        "sources":          [b["source"] for b in batch],

        "langs":            [b["lang"]   for b in batch],

    }

def loss_recon_weighted(

    flat_euc_raw:  torch.Tensor,

    flat_base_raw: torch.Tensor,

    flat_binary:   torch.Tensor,

    cos_w:         float = 0.7,

    mse_w:         float = 0.3,

    neg_weight:    float = 7.7,

) -> torch.Tensor:

    e = F.normalize(flat_euc_raw,  p=2, dim=1)

    b = F.normalize(flat_base_raw, p=2, dim=1)

    cos_term = 1.0 - (e * b).sum(dim=1)

    mse_term = ((flat_euc_raw - flat_base_raw) ** 2).mean(dim=1)

    per_step = cos_w * cos_term + mse_w * mse_term

    w = torch.where(

        flat_binary == 0,

        torch.full_like(per_step, neg_weight),

        torch.ones_like(per_step),

    )

    return (per_step * w).mean()

def loss_orth(atom_bank: torch.Tensor) -> torch.Tensor:

    a    = F.normalize(atom_bank, p=2, dim=1)

    gram = a @ a.t()

    eye  = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)

    return ((gram - eye) ** 2).mean()

def _sample_triplet_pairs(y: torch.Tensor, device: torch.device):

    anchors, pos_list, neg_list = [], [], []

    for i in range(y.size(0)):

        pos_mask    = (y == y[i])

        pos_mask[i] = False

        neg_mask    = (y != y[i])

        if pos_mask.sum() < 1 or neg_mask.sum() < 1:

            continue

        pi = pos_mask.nonzero(as_tuple=True)[0][

            torch.randint(int(pos_mask.sum()), (1,), device=device)

        ]

        ni = neg_mask.nonzero(as_tuple=True)[0][

            torch.randint(int(neg_mask.sum()), (1,), device=device)

        ]

        anchors.append(i)

        pos_list.append(pi[0].item())

        neg_list.append(ni[0].item())

    if not anchors:

        return None, None, None

    return (

        torch.tensor(anchors,   device=device),

        torch.tensor(pos_list,  device=device),

        torch.tensor(neg_list,  device=device),

    )

def loss_traj_triplet(

    traj_euc:      torch.Tensor,

    traj_usage:    torch.Tensor,

    binary_labels: torch.Tensor,

    emb_margin:    float = 0.3,

    usage_margin:  float = 0.1,

    usage_weight:  float = 0.3,

) -> tuple:

    ze = F.normalize(traj_euc,   p=2, dim=-1)

    zu = F.normalize(traj_usage, p=2, dim=-1)

    y  = binary_labels

    if y.size(0) < 3:

        z = traj_euc.sum() * 0.0

        return z, z, z, z, z

    ai, pi, ni = _sample_triplet_pairs(y, ze.device)

    if ai is None:

        z = traj_euc.sum() * 0.0

        return z, z, z, z, z

    sp_emb = (ze[ai] * ze[pi]).sum(dim=1)

    sn_emb = (ze[ai] * ze[ni]).sum(dim=1)

    l_emb  = F.relu(emb_margin - sp_emb + sn_emb).mean()

    sp_use = (zu[ai] * zu[pi]).sum(dim=1)

    sn_use = (zu[ai] * zu[ni]).sum(dim=1)

    l_use  = F.relu(usage_margin - sp_use + sn_use).mean()

    loss = l_emb + usage_weight * l_use

    return (

        loss,

        sp_emb.detach().mean(),

        sn_emb.detach().mean(),

        sp_use.detach().mean(),

        sn_use.detach().mean(),

    )

@torch.no_grad()

def weight_stats(weights: torch.Tensor, eps: float = 1e-12) -> dict:

    w    = weights.clamp_min(eps)

    ent  = -(w * w.log()).sum(dim=1)

    keff = ent.exp()

    return {

        "eff_k_mean": float(keff.mean()),

        "eff_k_p10":  float(torch.quantile(keff, 0.1)),

        "w_max_mean": float(w.max(dim=1).values.mean()),

    }

@torch.no_grad()

def usage_center_gap(traj_usage: torch.Tensor, labels: torch.Tensor) -> dict:

    valid = (labels == 0) | (labels == 1)

    if valid.sum() < 2:

        return {"center_cos": 1.0, "center_l2": 0.0}

    u  = traj_usage[valid]

    y  = labels[valid]

    m0, m1 = y == 0, y == 1

    if m0.sum() < 1 or m1.sum() < 1:

        return {"center_cos": 1.0, "center_l2": 0.0}

    mu0 = u[m0].mean(0, keepdim=True)

    mu1 = u[m1].mean(0, keepdim=True)

    cos = float((F.normalize(mu0, p=2, dim=1) * F.normalize(mu1, p=2, dim=1)).sum())

    l2  = float(torch.norm(mu0 - mu1, p=2))

    return {"center_cos": cos, "center_l2": l2}

@torch.no_grad()

def group_knn1_binary_accuracy(

    z:              torch.Tensor,

    binary:         torch.Tensor,

    group_ids,

    min_group_size: int = 1,

) -> dict:

    if z.size(0) <= 1:

        return {}

    sim     = z @ z.t()

    sim.fill_diagonal_(-1e9)

    nn_idx  = sim.argmax(dim=1)

    nn_bin  = binary[nn_idx]

    group2idx = defaultdict(list)

    for i, g in enumerate(group_ids):

        group2idx[g].append(i)

    out = {}

    for g, idxs in group2idx.items():

        if len(idxs) < min_group_size:

            continue

        idxs_t = torch.tensor(idxs, dtype=torch.long)

        acc    = (nn_bin[idxs_t] == binary[idxs_t]).float().mean().item()

        out[str(g)] = float(acc)

    return out

@torch.no_grad()

def group_geometry_stats(

    z:              torch.Tensor,

    binary:         torch.Tensor,

    hazard:         torch.Tensor,

    sources,

    langs,

    min_group_size: int = 1,

) -> dict:

    out = {}

    z   = F.normalize(z, p=2, dim=-1)

    mb  = (binary == 0)

    mh  = (binary == 1)

    if mh.sum() > 0:

        harm_center  = F.normalize(z[mh].mean(0, keepdim=True), p=2, dim=1)

        cos_to_harm  = (z * harm_center).sum(dim=1)

    else:

        cos_to_harm  = torch.zeros(z.size(0))

    if mb.sum() > 0:

        benign_center  = F.normalize(z[mb].mean(0, keepdim=True), p=2, dim=1)

        cos_to_benign  = (z * benign_center).sum(dim=1)

    else:

        cos_to_benign  = torch.zeros(z.size(0))

    hazard2idx = defaultdict(list)

    for i, hv in enumerate(hazard.tolist()):

        hazard2idx[int(hv)].append(i)

    out["per_hazard_count"]              = {}

    out["per_hazard_harm_fraction"]      = {}

    out["per_hazard_mean_cos_to_harm"]   = {}

    out["per_hazard_mean_cos_to_benign"] = {}

    for hv, idxs in hazard2idx.items():

        if len(idxs) < min_group_size:

            continue

        idxs_t = torch.tensor(idxs, dtype=torch.long)

        out["per_hazard_count"][str(hv)]              = len(idxs)

        out["per_hazard_harm_fraction"][str(hv)]      = float(binary[idxs_t].float().mean())

        out["per_hazard_mean_cos_to_harm"][str(hv)]   = float(cos_to_harm[idxs_t].mean())

        out["per_hazard_mean_cos_to_benign"][str(hv)] = float(cos_to_benign[idxs_t].mean())

    source2idx = defaultdict(list)

    for i, sv in enumerate(sources):

        source2idx[str(sv)].append(i)

    out["per_source_count"]              = {}

    out["per_source_harm_fraction"]      = {}

    out["per_source_mean_cos_to_harm"]   = {}

    out["per_source_mean_cos_to_benign"] = {}

    for sv, idxs in source2idx.items():

        if len(idxs) < min_group_size:

            continue

        idxs_t = torch.tensor(idxs, dtype=torch.long)

        out["per_source_count"][sv]              = len(idxs)

        out["per_source_harm_fraction"][sv]      = float(binary[idxs_t].float().mean())

        out["per_source_mean_cos_to_harm"][sv]   = float(cos_to_harm[idxs_t].mean())

        out["per_source_mean_cos_to_benign"][sv] = float(cos_to_benign[idxs_t].mean())

    lang2idx = defaultdict(list)

    for i, lv in enumerate(langs):

        lang2idx[str(lv)].append(i)

    out["per_lang_count"]              = {}

    out["per_lang_harm_fraction"]      = {}

    out["per_lang_mean_cos_to_harm"]   = {}

    out["per_lang_mean_cos_to_benign"] = {}

    for lv, idxs in lang2idx.items():

        if len(idxs) < min_group_size:

            continue

        idxs_t = torch.tensor(idxs, dtype=torch.long)

        out["per_lang_count"][lv]              = len(idxs)

        out["per_lang_harm_fraction"][lv]      = float(binary[idxs_t].float().mean())

        out["per_lang_mean_cos_to_harm"][lv]   = float(cos_to_harm[idxs_t].mean())

        out["per_lang_mean_cos_to_benign"][lv] = float(cos_to_benign[idxs_t].mean())

    return out

@torch.no_grad()

def masked_knn1_acc(z: torch.Tensor, binary: torch.Tensor, mask: torch.Tensor):

    if z.size(0) <= 1 or mask.sum() == 0:

        return None

    sim    = z @ z.t()

    sim.fill_diagonal_(-1e9)

    nn_idx = sim.argmax(dim=1)

    pred   = binary[nn_idx]

    return float((pred[mask] == binary[mask]).float().mean())

@torch.no_grad()

def seen_unseen_geometry_stats(

    z:                torch.Tensor,

    binary:           torch.Tensor,

    is_unseen_hazard: torch.Tensor,

) -> dict:

    out = {}

    z   = F.normalize(z, p=2, dim=-1)

    mb  = (binary == 0)

    mh  = (binary == 1)

    mu  = (is_unseen_hazard == 1) & mh

    ms  = (is_unseen_hazard == 0) & mh

    if mh.sum() > 0:

        harm_center  = F.normalize(z[mh].mean(0, keepdim=True), p=2, dim=1)

        cos_to_harm  = (z * harm_center).sum(dim=1)

        out["seen_harmful_mean_cos_to_harm"]   = float(cos_to_harm[ms].mean()) if ms.sum() > 0 else None

        out["unseen_harmful_mean_cos_to_harm"] = float(cos_to_harm[mu].mean()) if mu.sum() > 0 else None

        out["benign_mean_cos_to_harm"]         = float(cos_to_harm[mb].mean()) if mb.sum() > 0 else None

    if mb.sum() > 0:

        benign_center  = F.normalize(z[mb].mean(0, keepdim=True), p=2, dim=1)

        cos_to_benign  = (z * benign_center).sum(dim=1)

        out["seen_harmful_mean_cos_to_benign"]   = float(cos_to_benign[ms].mean()) if ms.sum() > 0 else None

        out["unseen_harmful_mean_cos_to_benign"] = float(cos_to_benign[mu].mean()) if mu.sum() > 0 else None

        out["benign_mean_cos_to_benign"]         = float(cos_to_benign[mb].mean()) if mb.sum() > 0 else None

    out["seen_harmful_count"]   = int(ms.sum())

    out["unseen_harmful_count"] = int(mu.sum())

    out["benign_count"]         = int(mb.sum())

    out["seen_harmful_knn1_bin_acc"]   = masked_knn1_acc(z, binary, ms)

    out["unseen_harmful_knn1_bin_acc"] = masked_knn1_acc(z, binary, mu)

    out["benign_knn1_bin_acc"]         = masked_knn1_acc(z, binary, mb)

    return out

def encode_steps(model, batch, device):

    input_ids    = batch["input_ids"].to(device)

    attention_mask = batch["attention_mask"].to(device)

    step_mask    = batch["step_mask"].to(device)

    B, T, L     = input_ids.shape

    flat_valid  = step_mask.view(B * T) > 0

    enc = model.encoder._encode(

        input_ids.view(B * T, L)[flat_valid],

        attention_mask.view(B * T, L)[flat_valid],

    )

    D = enc["base_raw"].size(-1)

    def _scatter(src):

        buf = torch.zeros(B * T, src.size(-1), device=device)

        buf[flat_valid] = src

        return buf

    return {

        "step_base_raw":       _scatter(enc["base_raw"]).view(B, T, D),

        "step_base_emb":       _scatter(enc["base_emb"]).view(B, T, D),

        "step_euc_raw":        _scatter(enc["euc_raw"]).view(B, T, D),

        "step_euc_vec":        _scatter(enc["euc_vec"]).view(B, T, D),

        "step_weights":        _scatter(enc["weights"]).view(B, T, enc["weights"].size(-1)),

        "flat_weights_valid":  enc["weights"],

        "step_mask":           step_mask,

    }

def _expand_binary_to_steps(binary: torch.Tensor, step_mask: torch.Tensor) -> torch.Tensor:

    B, T   = step_mask.shape

    binary_bt = binary.unsqueeze(1).expand(B, T)

    flat_valid = step_mask.view(B * T) > 0

    return binary_bt.reshape(B * T)[flat_valid]

@torch.no_grad()

def evaluate(model, loader, device, config):

    model.eval()

    metrics = defaultdict(float)

    n = 0

    all_z       = []

    all_bin     = []

    all_hazard  = []

    all_sources = []

    all_langs   = []

    all_unseen  = []

    for batch in tqdm(loader, desc="Eval", leave=False):

        binary           = batch["binary_label"].to(device)

        hazard           = batch["hazard_type"].to(device)

        is_unseen_hazard = batch["is_unseen_hazard"].to(device)

        enc = encode_steps(model, batch, device)

        flat_valid    = enc["step_mask"].view(-1) > 0

        flat_base_raw = enc["step_base_raw"].view(-1, enc["step_base_raw"].size(-1))[flat_valid]

        flat_euc_raw  = enc["step_euc_raw"].view(-1,  enc["step_euc_raw"].size(-1))[flat_valid]

        flat_binary   = _expand_binary_to_steps(binary, enc["step_mask"])

        traj_euc   = trajectory_mean_pool(enc["step_euc_vec"], enc["step_mask"])

        traj_usage = trajectory_usage_pool(enc["step_weights"], enc["step_mask"])

        traj_base  = trajectory_mean_pool(enc["step_base_emb"], enc["step_mask"])

        l_rec  = loss_recon_weighted(

            flat_euc_raw, flat_base_raw, flat_binary,

            neg_weight=config.get("neg_recon_weight", 7.7),

        )

        l_orth = loss_orth(model.encoder.atom_bank)

        l_tri, sp_emb, sn_emb, sp_use, sn_use = loss_traj_triplet(

            traj_euc, traj_usage, binary,

            emb_margin=config["emb_margin"],

            usage_margin=config["usage_margin"],

            usage_weight=config["usage_weight"],

        )

        metrics["rec"]     += l_rec.item()

        metrics["orth"]    += l_orth.item()

        metrics["triplet"] += l_tri.item()

        metrics["sp_emb"]  += float(sp_emb)

        metrics["sn_emb"]  += float(sn_emb)

        metrics["sp_use"]  += float(sp_use)

        metrics["sn_use"]  += float(sn_use)

        ws  = weight_stats(enc["flat_weights_valid"])

        gap = usage_center_gap(traj_usage, binary)

        for k, v in {**ws, **gap}.items():

            metrics[k] += v

        metrics["align_cos"] += float((traj_euc * traj_base).sum(dim=1).mean())

        all_z.append(traj_euc.cpu())

        all_bin.append(binary.cpu())

        all_hazard.append(hazard.cpu())

        all_unseen.append(is_unseen_hazard.cpu())

        all_sources.extend(batch["sources"])

        all_langs.extend(batch["langs"])

        n += 1

    out = {k: v / max(n, 1) for k, v in metrics.items()}

    if all_z:

        z = F.normalize(torch.cat(all_z), p=2, dim=-1)

        b = torch.cat(all_bin)

        h = torch.cat(all_hazard)

        u = torch.cat(all_unseen)

        sim = z @ z.t()

        sim.fill_diagonal_(-1e9)

        out["knn1_bin_acc"] = float((b[sim.argmax(dim=1)] == b).float().mean())

        mb, mh = b == 0, b == 1

        if mb.any() and mh.any():

            pb = F.normalize(z[mb].mean(0, keepdim=True), p=2, dim=1)

            ph = F.normalize(z[mh].mean(0, keepdim=True), p=2, dim=1)

            out["harm_benign_cos"] = float((pb * ph).sum())

            out["harm_benign_l2"]  = float(torch.norm(pb - ph, p=2))

        if config.get("enable_unseen_analysis", False):

            unseen_stats = seen_unseen_geometry_stats(z, b, u)

            out.update(unseen_stats)

            out["per_hazard_knn1_bin_acc"] = group_knn1_binary_accuracy(

                z, b, h.tolist(),

                min_group_size=config.get("min_group_size", 1),

            )

            out["per_source_knn1_bin_acc"] = group_knn1_binary_accuracy(

                z, b, all_sources,

                min_group_size=config.get("min_group_size", 1),

            )

            out["per_lang_knn1_bin_acc"] = group_knn1_binary_accuracy(

                z, b, all_langs,

                min_group_size=config.get("min_group_size", 1),

            )

            geo = group_geometry_stats(

                z=z, binary=b, hazard=h,

                sources=all_sources,

                langs=all_langs,

                min_group_size=config.get("min_group_size", 1),

            )

            out.update(geo)

    return out

def train_stage1(config, device):

    print("\n" + "=" * 72)

    print("  TRACE Stage1 v7: Recon(weighted) + Binary-Triplet + Orth + Lang")

    print("=" * 72)

    print(f"  w_rec={config['w_rec']}  w_triplet={config['w_triplet']}  w_orth={config['w_orth']}")

    print(f"  neg_recon_weight={config.get('neg_recon_weight', 7.7)}")

    print(f"  emb_margin={config['emb_margin']}  usage_margin={config['usage_margin']}  usage_weight={config['usage_weight']}")

    print(f"  temp={config['temperature']}  num_atoms={config['num_atoms']}  topk={config['topk_atoms']}")

    print(f"  max_traj_length={config['max_traj_length']}")

    print(f"  exclude_train_hazards={config.get('exclude_train_hazards', '')}")

    print(f"  eval_unseen_hazards_only={config.get('eval_unseen_hazards_only', False)}")

    print("=" * 72)

    tokenizer             = AutoTokenizer.from_pretrained(config["model_name"])

    exclude_train_hazards = parse_int_list(config.get("exclude_train_hazards", ""))

    def make_loader(path, shuffle, drop_last, split):

        ds = TRACEStage1TrajectoryDataset(

            path,

            tokenizer,

            max_step_length=config["max_step_length"],

            max_traj_length=config["max_traj_length"],

            exclude_train_hazards=exclude_train_hazards,

            eval_unseen_hazards_only=(

                split != "train" and config.get("eval_unseen_hazards_only", False)

            ),

            split=split,

        )

        return DataLoader(

            ds,

            batch_size=config["batch_size"],

            shuffle=shuffle,

            collate_fn=stage1_collate_fn,

            num_workers=config.get("num_workers", 0),

            pin_memory=(config.get("num_workers", 0) > 0),

            drop_last=drop_last,

        )

    train_loader = make_loader(config["train_jsonl"], shuffle=True,  drop_last=True,  split="train")

    val_loader   = make_loader(config["val_jsonl"],   shuffle=False, drop_last=False, split="val")

    model = TRACE(

        model_name_or_path=config["model_name"],

        num_atoms=config["num_atoms"],

        temperature=config["temperature"],

        num_attack_classes=config.get("num_attack_classes", 12),

        coact_rank=config.get("coact_rank", 16),

        topk_atoms=config.get("topk_atoms", 128),

        accumulator_mode=config.get("accumulator_mode", "attention"),

        window_size=config.get("window_size", 10),

    ).to(device)

    for p in model.encoder.embedding_model.parameters():

        p.requires_grad = False

    for mod in [model.accumulator, model.coact_computer,

                model.prototype_bank, model.zeroshot_scorer]:

        for p in mod.parameters():

            p.requires_grad = False

    model.risk_scale.requires_grad = False

    if hasattr(model, "zs_scale"):

        model.zs_scale.requires_grad = False

    atom_params = [

        model.encoder.atom_bank,

        model.encoder.query_proj.weight,

        model.encoder.query_proj.bias,

    ]

    print(f"🔧 Trainable: {sum(p.numel() for p in atom_params):,} params")

    optimizer = torch.optim.AdamW(

        [{"params": atom_params, "lr": config["atom_lr"]}],

        weight_decay=config["weight_decay"],

    )

    total_steps  = len(train_loader) * config["num_epochs"]

    warmup_steps = int(total_steps * config["warmup_ratio"])

    def lr_lambda(step):

        if step < warmup_steps:

            return step / max(warmup_steps, 1)

        p = (step - warmup_steps) / max(total_steps - warmup_steps, 1)

        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * p)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    os.makedirs(config["save_dir"], exist_ok=True)

    with open(os.path.join(config["save_dir"], "config.json"), "w") as f:

        json.dump(config, f, indent=2)

    best_score = -1e9

    for epoch in range(config["num_epochs"]):

        model.train()

        ep = defaultdict(float)

        nb = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")

        for batch in pbar:

            optimizer.zero_grad(set_to_none=True)

            binary = batch["binary_label"].to(device)

            enc    = encode_steps(model, batch, device)

            flat_valid    = enc["step_mask"].view(-1) > 0

            flat_base_raw = enc["step_base_raw"].view(-1, enc["step_base_raw"].size(-1))[flat_valid]

            flat_euc_raw  = enc["step_euc_raw"].view(-1,  enc["step_euc_raw"].size(-1))[flat_valid]

            flat_binary   = _expand_binary_to_steps(binary, enc["step_mask"])

            traj_euc   = trajectory_mean_pool(enc["step_euc_vec"], enc["step_mask"])

            traj_usage = trajectory_usage_pool(enc["step_weights"], enc["step_mask"])

            l_rec  = loss_recon_weighted(

                flat_euc_raw, flat_base_raw, flat_binary,

                neg_weight=config.get("neg_recon_weight", 7.7),

            )

            l_orth = loss_orth(model.encoder.atom_bank)

            l_tri, sp_emb, sn_emb, sp_use, sn_use = loss_traj_triplet(

                traj_euc, traj_usage, binary,

                emb_margin=config["emb_margin"],

                usage_margin=config["usage_margin"],

                usage_weight=config["usage_weight"],

            )

            loss = (

                config["w_rec"]     * l_rec

                + config["w_triplet"] * l_tri

                + config["w_orth"]    * l_orth

            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(atom_params, config["grad_clip"])

            optimizer.step()

            scheduler.step()

            ws  = weight_stats(enc["flat_weights_valid"])

            gap = usage_center_gap(traj_usage.detach(), binary.detach())

            ep["total"]      += loss.item()

            ep["rec"]        += l_rec.item()

            ep["triplet"]    += l_tri.item()

            ep["orth"]       += l_orth.item()

            ep["sp_emb"]     += float(sp_emb)

            ep["sn_emb"]     += float(sn_emb)

            ep["sp_use"]     += float(sp_use)

            ep["sn_use"]     += float(sn_use)

            ep["center_cos"] += gap["center_cos"]

            ep["center_l2"]  += gap["center_l2"]

            ep["eff_k_mean"] += ws["eff_k_mean"]

            ep["w_max_mean"] += ws["w_max_mean"]

            nb += 1

            pbar.set_postfix({

                "rec":  f"{l_rec.item():.3f}",

                "tri":  f"{l_tri.item():.3f}",

                "se+":  f"{float(sp_emb):.3f}",

                "se-":  f"{float(sn_emb):.3f}",

                "su+":  f"{float(sp_use):.3f}",

                "su-":  f"{float(sn_use):.3f}",

                "kEff": f"{ws['eff_k_mean']:.1f}",

                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",

            })

        n = max(nb, 1)

        print(f"\nEpoch {epoch+1} | " + " | ".join(f"{k}={v/n:.4f}" for k, v in ep.items()))

        val = evaluate(model, val_loader, device, config)

        scalar_items, nested_items = [], []

        for k, v in val.items():

            if isinstance(v, (int, float)) and v is not None:

                scalar_items.append(f"{k}={v:.4f}")

            else:

                nested_items.append(k)

        print("Val | " + " | ".join(scalar_items))

        if nested_items:

            print("Val nested keys | " + ", ".join(nested_items))

        emb_gap = val.get("sp_emb", 0.0) - val.get("sn_emb", 0.0)

        use_gap = val.get("sp_use", 0.0) - val.get("sn_use", 0.0)

        score   = (

            -1.0 * val.get("rec",       1.0)

            -0.8 * val.get("triplet",   1.0)

            +1.0 * emb_gap

            +0.5 * use_gap

            +0.5 * val.get("center_l2", 0.0)

            +0.2 * val.get("align_cos", 0.0)

        )

        if score > best_score:

            best_score = score

            ckpt_path  = os.path.join(config["save_dir"], "stage1_best.pth")

            torch.save({

                "epoch":       epoch + 1,

                "state_dict":  model.state_dict(),

                "optimizer":   optimizer.state_dict(),

                "score":       score,

                "config":      config,

                "val_metrics": val,

            }, ckpt_path)

            print(f"✅ Best saved (score={score:.4f}) → {ckpt_path}")

        torch.save({

            "epoch":       epoch + 1,

            "state_dict":  model.state_dict(),

            "score":       score,

            "val_metrics": val,

        }, os.path.join(config["save_dir"], f"stage1_epoch{epoch+1}.pth"))

    print(f"\nStage1 v7 complete. Best score = {best_score:.4f}")

    return best_score

def main():

    parser = argparse.ArgumentParser(description="TRACE Stage1 v7 Training")

    parser.add_argument("--train_jsonl", required=True,

                        help="训练集 JSONL，如 data/processed_multisource/train.jsonl")

    parser.add_argument("--val_jsonl",   required=True,

                        help="验证集 JSONL，如 data/processed_multisource/val.jsonl")

    parser.add_argument("--model_name",         default="BAAI/bge-base-en-v1.5")

    parser.add_argument("--num_atoms",           type=int,   default=1024)

    parser.add_argument("--temperature",         type=float, default=0.5)

    parser.add_argument("--num_attack_classes",  type=int,   default=12)

    parser.add_argument("--coact_rank",          type=int,   default=16)

    parser.add_argument("--topk_atoms",          type=int,   default=128)

    parser.add_argument("--accumulator_mode",    default="attention")

    parser.add_argument("--window_size",         type=int,   default=10)

    parser.add_argument("--num_epochs",          type=int,   default=20)

    parser.add_argument("--batch_size",          type=int,   default=16)

    parser.add_argument("--atom_lr",             type=float, default=3e-4)

    parser.add_argument("--weight_decay",        type=float, default=1e-4)

    parser.add_argument("--grad_clip",           type=float, default=1.0)

    parser.add_argument("--warmup_ratio",        type=float, default=0.05)

    parser.add_argument("--max_step_length",     type=int,   default=128)

    parser.add_argument("--max_traj_length",     type=int,   default=22,

                        help="v7: 默认 22，为 SDB_ZH max_steps=20 留余量")

    parser.add_argument("--num_workers",         type=int,   default=0)

    parser.add_argument("--seed",                type=int,   default=42)

    parser.add_argument("--w_rec",               type=float, default=1.0)

    parser.add_argument("--w_triplet",           type=float, default=0.5)

    parser.add_argument("--w_orth",              type=float, default=0.05)

    parser.add_argument("--neg_recon_weight",    type=float, default=7.7,

                        help="v7: benign step 在重建 loss 中的上调权重，"

                             "默认 7.7（对应 8:1 正负比）")

    parser.add_argument("--emb_margin",          type=float, default=0.3,

                        help="embedding 空间 triplet margin（主信号）")

    parser.add_argument("--usage_margin",        type=float, default=0.1,

                        help="usage 空间 triplet margin（辅助信号）")

    parser.add_argument("--usage_weight",        type=float, default=0.3,

                        help="usage triplet 相对于 emb triplet 的权重")

    parser.add_argument("--exclude_train_hazards", type=str, default="6,7",

                        help="训练时排除的 hazard_type，逗号分隔。"

                             "默认 '6,7'（privilege_escalation, resource_abuse）")

    parser.add_argument("--eval_unseen_hazards_only", action="store_true",

                        help="验证时只保留 benign + held-out hazards（ATBench only）。"

                             "v7: SDB 样本自动豁免，不受此过滤影响")

    parser.add_argument("--enable_unseen_analysis",   action="store_true",

                        help="输出 seen/unseen/per-hazard/per-source/per-lang 分析")

    parser.add_argument("--min_group_size",      type=int,   default=3)

    parser.add_argument("--save_dir", default="./results/trace_stage1_v7")

    args   = parser.parse_args()

    config = vars(args)

    set_seed(config["seed"])

    os.makedirs(config["save_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    train_stage1(config, device)

if __name__ == "__main__":

    main()
