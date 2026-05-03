import os, sys, json, argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

current_dir  = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from models.TARCE_HACS import TRACE


# ─────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────
class PrefixTrajectoryDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_step_length=128, max_traj_length=22):
        self.tokenizer = tokenizer
        self.max_step_length = max_step_length
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                steps = r.get("steps", [])[:max_traj_length]
                if not steps:
                    continue
                self.samples.append({
                    "steps":  steps,
                    "binary": int(r["labels"].get("binary", -1)),
                    "hazard": int(r["labels"].get("hazard_type", -1)),
                    "source": r.get("source", "unknown"),
                    "lang":   r.get("meta", {}).get("lang", "en"),
                })

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
                return_tensors="pt"
            )
            ids_list.append(enc["input_ids"].squeeze(0))
            mask_list.append(enc["attention_mask"].squeeze(0))
        return {
            "ids_list":  ids_list,
            "mask_list": mask_list,
            "binary": torch.tensor(item["binary"], dtype=torch.long),
            "hazard": torch.tensor(item["hazard"],  dtype=torch.long),
            "source": item["source"],
            "lang":   item["lang"],
        }


def collate_fn(batch):
    max_steps = max(len(b["ids_list"]) for b in batch)
    step_len  = batch[0]["ids_list"][0].shape[0]
    input_ids, attn_mask, step_mask = [], [], []
    binaries, hazards, sources, langs = [], [], [], []

    for b in batch:
        n   = len(b["ids_list"])
        pad = torch.zeros(step_len, dtype=torch.long)
        input_ids.append(torch.stack(b["ids_list"]  + [pad] * (max_steps - n)))
        attn_mask.append(torch.stack(b["mask_list"] + [pad] * (max_steps - n)))
        sm = torch.zeros(max_steps, dtype=torch.long)
        sm[:n] = 1
        step_mask.append(sm)
        binaries.append(b["binary"])
        hazards.append(b["hazard"])
        sources.append(b["source"])
        langs.append(b["lang"])

    return {
        "input_ids":      torch.stack(input_ids),
        "attention_mask": torch.stack(attn_mask),
        "step_mask":      torch.stack(step_mask),
        "binary":         torch.stack(binaries),
        "hazard":         torch.stack(hazards),
        "sources":        sources,
        "langs":          langs,
    }


@dataclass
class SplitPack:
    name:     str
    step_euc: np.ndarray
    traj_euc: np.ndarray
    y:        np.ndarray
    h:        np.ndarray
    traj_len: np.ndarray
    sources:  list
    langs:    list


def merge_packs(name: str, pack_a: SplitPack, pack_b: SplitPack) -> SplitPack:
    """将两个 SplitPack 在样本维度拼接（用于 harmful+benign 配对）"""
    # step_euc 的 T 维度可能不同，需要 pad 到相同长度
    T_a = pack_a.step_euc.shape[1]
    T_b = pack_b.step_euc.shape[1]
    T   = max(T_a, T_b)
    D   = pack_a.step_euc.shape[2]

    def pad_T(arr, T_target):
        if arr.shape[1] == T_target:
            return arr
        pad = np.zeros((arr.shape[0], T_target - arr.shape[1], D), dtype=arr.dtype)
        return np.concatenate([arr, pad], axis=1)

    return SplitPack(
        name     = name,
        step_euc = np.concatenate([pad_T(pack_a.step_euc, T),
                                   pad_T(pack_b.step_euc, T)], axis=0),
        traj_euc = np.concatenate([pack_a.traj_euc, pack_b.traj_euc], axis=0),
        y        = np.concatenate([pack_a.y,        pack_b.y],        axis=0),
        h        = np.concatenate([pack_a.h,        pack_b.h],        axis=0),
        traj_len = np.concatenate([pack_a.traj_len, pack_b.traj_len], axis=0),
        sources  = pack_a.sources + pack_b.sources,
        langs    = pack_a.langs   + pack_b.langs,
    )


# ─────────────────────────────────────────────────────────────
# Extraction
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_trace_embeddings(model, loader, device, desc=""):
    model.eval()
    step_euc_list, traj_euc_list = [], []
    all_y, all_h, all_len, all_sources, all_langs = [], [], [], [], []
    printed = False

    for batch in tqdm(loader, desc=desc, leave=False):
        ids = batch["input_ids"].to(device)
        ams = batch["attention_mask"].to(device)
        sm  = batch["step_mask"].to(device)
        B, T, L = ids.shape

        flat_valid = sm.view(B * T) > 0
        enc = model.encoder._encode(
            ids.view(B * T, L)[flat_valid],
            ams.view(B * T, L)[flat_valid]
        )

        if not printed:
            print(f"\n[DEBUG:{desc}] enc keys: {list(enc.keys())}")
            for k, v in enc.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {tuple(v.shape)}  "
                          f"min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
            printed = True

        step_valid = F.normalize(enc["euc_vec"], p=2, dim=-1)
        D = step_valid.shape[-1]
        buf = torch.zeros(B * T, D, device=device)
        buf[flat_valid] = step_valid
        step_euc = buf.view(B, T, D)

        mask_f   = sm.float().unsqueeze(-1)
        traj_euc = (step_euc * mask_f).sum(1) / mask_f.sum(1).clamp_min(1e-12)
        traj_euc = F.normalize(traj_euc, p=2, dim=-1)

        for i in range(B):
            n = int(sm[i].sum().item())
            step_euc_list.append(step_euc[i, :n].detach().cpu())
            traj_euc_list.append(traj_euc[i].detach().cpu())
            all_y.append(batch["binary"][i].item())
            all_h.append(batch["hazard"][i].item())
            all_len.append(n)
            all_sources.append(batch["sources"][i])
            all_langs.append(batch["langs"][i])

    max_len = max(all_len)
    D = traj_euc_list[0].shape[0]
    N = len(step_euc_list)
    step_euc_pad = torch.zeros(N, max_len, D)
    for i, se in enumerate(step_euc_list):
        step_euc_pad[i, :se.shape[0]] = se

    traj_euc = F.normalize(torch.stack(traj_euc_list, dim=0), p=2, dim=-1)

    return {
        "step_euc": step_euc_pad.numpy(),
        "traj_euc": traj_euc.numpy(),
        "binary":   np.array(all_y),
        "hazard":   np.array(all_h),
        "traj_len": np.array(all_len),
        "sources":  all_sources,
        "langs":    all_langs,
    }


# ─────────────────────────────────────────────────────────────
# Prefix utilities
# ─────────────────────────────────────────────────────────────
def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def cosine_to(z, c):
    c = c / (np.linalg.norm(c) + 1e-12)
    return np.sum(z * c[None, :], axis=1)


def prefix_traj_from_steps(step_euc, traj_len, k, use_full_for_short=False):
    """
    use_full_for_short=True：对于 traj_len < k 的样本，
      用其完整轨迹均值代替（而非丢弃），避免 benign 短轨迹被过滤。
      适用于 paired 模式中 benign 步数远少于 harmful 的情况。
    use_full_for_short=False（默认）：原始行为，过滤掉短样本。
    """
    if not use_full_for_short:
        mask = traj_len >= k
        z = step_euc[mask, :k, :].mean(axis=1)
        z = normalize_rows(z)
        return z, mask

    # use_full_for_short 模式：所有样本均参与
    N, T, D = step_euc.shape
    z_out = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        actual_k = min(k, traj_len[i])
        z_out[i] = step_euc[i, :actual_k, :].mean(axis=0)
    z_out = normalize_rows(z_out)
    mask = np.ones(N, dtype=bool)
    return z_out, mask


def prefix_last_step(step_euc, traj_len, k, use_full_for_short=False):
    if not use_full_for_short:
        mask = traj_len >= k
        z = step_euc[mask, k - 1, :]
        z = normalize_rows(z)
        return z, mask

    N, T, D = step_euc.shape
    z_out = np.zeros((N, D), dtype=np.float32)
    for i in range(N):
        actual_k = min(k, traj_len[i])
        z_out[i] = step_euc[i, actual_k - 1, :]
    z_out = normalize_rows(z_out)
    mask = np.ones(N, dtype=bool)
    return z_out, mask


def prefix_stepwise_margin_max(step_euc, traj_len, k, harm_c, benign_c,
                                use_full_for_short=False):
    harm   = harm_c   / (np.linalg.norm(harm_c)   + 1e-12)
    benign = benign_c / (np.linalg.norm(benign_c) + 1e-12)

    if not use_full_for_short:
        mask  = traj_len >= k
        steps = step_euc[mask, :k, :]
        m = (np.tensordot(steps, harm,   axes=([2], [0]))
           - np.tensordot(steps, benign, axes=([2], [0])))
        return m.max(axis=1), mask

    N = len(traj_len)
    scores = np.zeros(N, dtype=np.float32)
    for i in range(N):
        actual_k = min(k, traj_len[i])
        steps_i  = step_euc[i, :actual_k, :]
        m_i = steps_i @ harm - steps_i @ benign
        scores[i] = m_i.max()
    mask = np.ones(N, dtype=bool)
    return scores, mask


def tpr_at_fpr(y_true, scores, fpr_limit=0.1):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return None
    fpr, tpr, _ = roc_curve(y_true, scores)
    ok = np.where(fpr <= fpr_limit)[0]
    if len(ok) == 0:
        return 0.0
    return float(np.max(tpr[ok]))


def safe_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def safe_ap(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def summarize_prefix_curve(metrics_by_k, target_tpr=0.8):
    ks   = sorted(metrics_by_k.keys())
    aucs = [metrics_by_k[k]["auc"] for k in ks if metrics_by_k[k]["auc"] is not None]
    audc = float(np.mean(aucs)) if len(aucs) else None
    eds  = None
    for k in ks:
        t = metrics_by_k[k]["tpr_at_fpr_0.1"]
        if t is not None and t >= target_tpr:
            eds = int(k)
            break
    return {"AUDC": audc, "EDS": eds, "max_k": int(max(ks)) if ks else None}


# ─────────────────────────────────────────────────────────────
# Per-split evaluation core
# ─────────────────────────────────────────────────────────────
def evaluate_split(
    split_pack:          SplitPack,
    harm_c:              np.ndarray,
    benign_c:            np.ndarray,
    max_k:               int,
    target_tpr:          float,
    methods:             list,
    use_full_for_short:  bool = False,
) -> dict:
    all_results = {}

    for method in methods:
        print(f"\n  [{split_pack.name}] Method: {method}"
              + (" [use_full_for_short]" if use_full_for_short else ""))
        metrics_by_k = {}

        for k in range(1, max_k + 1):
            if method == "trace_center_margin":
                z, mask = prefix_traj_from_steps(
                    split_pack.step_euc, split_pack.traj_len, k, use_full_for_short)
                y     = split_pack.y[mask]
                score = cosine_to(z, harm_c) - cosine_to(z, benign_c)

            elif method == "trace_benign_departure":
                z, mask = prefix_traj_from_steps(
                    split_pack.step_euc, split_pack.traj_len, k, use_full_for_short)
                y     = split_pack.y[mask]
                score = 1.0 - cosine_to(z, benign_c)

            elif method == "last_step_center_margin":
                z, mask = prefix_last_step(
                    split_pack.step_euc, split_pack.traj_len, k, use_full_for_short)
                y     = split_pack.y[mask]
                score = cosine_to(z, harm_c) - cosine_to(z, benign_c)

            elif method == "stepwise_max_center_margin":
                score, mask = prefix_stepwise_margin_max(
                    split_pack.step_euc, split_pack.traj_len, k, harm_c, benign_c,
                    use_full_for_short)
                y = split_pack.y[mask]

            else:
                raise ValueError(f"Unknown method: {method}")

            auc = safe_auc(y, score)
            ap  = safe_ap(y, score)
            tpr = tpr_at_fpr(y, score, fpr_limit=0.1)

            harm_mean   = float(np.mean(score[y == 1])) if np.any(y == 1) else None
            benign_mean = float(np.mean(score[y == 0])) if np.any(y == 0) else None
            delta = (None if (harm_mean is None or benign_mean is None)
                     else float(harm_mean - benign_mean))

            metrics_by_k[k] = {
                "n":               int(len(y)),
                "harm_count":      int(np.sum(y == 1)),
                "benign_count":    int(np.sum(y == 0)),
                "auc":             auc,
                "ap":              ap,
                "tpr_at_fpr_0.1":  tpr,
                "harm_mean_score":   harm_mean,
                "benign_mean_score": benign_mean,
                "delta_mean_score":  delta,
            }

            auc_s = "None" if auc   is None else f"{auc:.4f}"
            ap_s  = "None" if ap    is None else f"{ap:.4f}"
            tpr_s = "None" if tpr   is None else f"{tpr:.4f}"
            dlt_s = "None" if delta is None else f"{delta:.4f}"
            print(f"    k={k:>2d} | n={len(y):>5d} "
                  f"(h={int(np.sum(y==1)):>5d} b={int(np.sum(y==0)):>5d}) | "
                  f"auc={auc_s} | ap={ap_s} | tpr@0.1={tpr_s} | Δ={dlt_s}")

        summary = summarize_prefix_curve(metrics_by_k, target_tpr=target_tpr)
        audc_s  = "None" if summary["AUDC"] is None else f"{summary['AUDC']:.4f}"
        print(f"    -> AUDC={audc_s} | EDS={summary['EDS']}")

        all_results[method] = {"metrics_by_k": metrics_by_k, "summary": summary}

    return all_results


# ─────────────────────────────────────────────────────────────
# Score-only analysis（全 harmful 或全 benign 集）
# ─────────────────────────────────────────────────────────────
def score_distribution_analysis(
    split_pack: SplitPack,
    harm_c:     np.ndarray,
    benign_c:   np.ndarray,
    max_k:      int,
) -> dict:
    """
    对单类集合（全 harmful 或全 benign）报告 score 分布统计，
    替代无法计算的 AUC/AP。
    """
    results = {}
    label = "harmful" if np.all(split_pack.y == 1) else "benign"

    for k in range(1, max_k + 1):
        z, mask = prefix_traj_from_steps(
            split_pack.step_euc, split_pack.traj_len, k,
            use_full_for_short=True)   # 全量，不丢弃短样本
        score_margin   = cosine_to(z, harm_c) - cosine_to(z, benign_c)
        score_departure = 1.0 - cosine_to(z, benign_c)

        results[k] = {
            "n":                     int(mask.sum()),
            "label":                 label,
            "margin_mean":           float(np.mean(score_margin)),
            "margin_std":            float(np.std(score_margin)),
            "margin_p25":            float(np.percentile(score_margin, 25)),
            "margin_p50":            float(np.percentile(score_margin, 50)),
            "margin_p75":            float(np.percentile(score_margin, 75)),
            "departure_mean":        float(np.mean(score_departure)),
            "departure_std":         float(np.std(score_departure)),
            # 对 harmful 集：报告 score > 0 的比例（即"被正确激活"的比例）
            # 对 benign 集：报告 score < 0 的比例（即"被正确压制"的比例）
            "frac_positive_margin":  float(np.mean(score_margin > 0)),
        }

    return results


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    pa = argparse.ArgumentParser(description="TRACE Prefix-AUC Evaluation v2")

    pa.add_argument("--ckpt_path",    required=True)
    pa.add_argument("--center_jsonl", required=True,
                    help="用于计算 harm/benign 类中心（通常为 train.jsonl）")
    pa.add_argument("--eval_jsonl",   nargs="+", required=True,
                    help="待评估的 JSONL 列表（可多个）")
    pa.add_argument("--eval_names",   nargs="+", default=None)

    # 配对模式：将全 harmful 集与全 benign 集合并后计算 AUC
    pa.add_argument("--benign_jsonl", default=None,
                    help="全 benign 集（tb_test.jsonl），"
                         "用于与全 harmful 的 eval_jsonl 配对计算 AUC")
    pa.add_argument("--use_full_for_short", action="store_true",
                    help="对步数不足 k 的样本使用完整轨迹均值，"
                         "避免 benign 短轨迹在大 k 时被全部过滤")

    pa.add_argument("--output_dir",         default="analysis/prefix_eval")
    pa.add_argument("--batch_size",         type=int,   default=32)
    pa.add_argument("--max_traj_length",    type=int,   default=22)
    pa.add_argument("--max_prefix",         type=int,   default=20)
    pa.add_argument("--target_tpr",         type=float, default=0.8)
    args = pa.parse_args()

    if args.eval_names is None:
        args.eval_names = [
            os.path.splitext(os.path.basename(p))[0]
            for p in args.eval_jsonl
        ]
    if len(args.eval_names) != len(args.eval_jsonl):
        pa.error("--eval_names 数量必须与 --eval_jsonl 相同")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    tok  = AutoTokenizer.from_pretrained(cfg["model_name"])

    model = TRACE(
        model_name_or_path=cfg["model_name"],
        num_atoms=cfg["num_atoms"],
        temperature=cfg["temperature"],
        num_attack_classes=cfg.get("num_attack_classes", 12),
        coact_rank=cfg.get("coact_rank", 16),
        topk_atoms=cfg.get("topk_atoms", 128),
        accumulator_mode=cfg.get("accumulator_mode", "attention"),
        window_size=cfg.get("window_size", 10),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"Loaded: {args.ckpt_path}")

    def make_loader(path):
        ds = PrefixTrajectoryDataset(path, tok, max_traj_length=args.max_traj_length)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate_fn, num_workers=0)

    def pack(name, raw):
        return SplitPack(
            name=name, step_euc=raw["step_euc"], traj_euc=raw["traj_euc"],
            y=raw["binary"], h=raw["hazard"], traj_len=raw["traj_len"],
            sources=raw["sources"], langs=raw["langs"])

    # ── Step 1: 类中心 ────────────────────────────────────────
    print(f"\n[Step 1] Computing class centers from: {args.center_jsonl}")
    center_pack = pack("center",
        extract_trace_embeddings(model, make_loader(args.center_jsonl), device, "center"))

    harm_mask   = center_pack.y == 1
    benign_mask = center_pack.y == 0
    if not harm_mask.any():
        raise RuntimeError("center_jsonl 中没有 harmful 样本")
    if not benign_mask.any():
        raise RuntimeError("center_jsonl 中没有 benign 样本")

    harm_c   = center_pack.traj_euc[harm_mask].mean(axis=0)
    harm_c  /= np.linalg.norm(harm_c)   + 1e-12
    benign_c = center_pack.traj_euc[benign_mask].mean(axis=0)
    benign_c /= np.linalg.norm(benign_c) + 1e-12

    print(f"  harm_n={harm_mask.sum()}  benign_n={benign_mask.sum()}")
    print(f"  harm/benign center cosine = {float(np.dot(harm_c, benign_c)):.4f}")
    for src in sorted(set(center_pack.sources)):
        sm = np.array([s == src for s in center_pack.sources])
        sh = sm & harm_mask;  sb = sm & benign_mask
        if sh.any() and sb.any():
            ch = center_pack.traj_euc[sh].mean(0); ch /= np.linalg.norm(ch)+1e-12
            cb = center_pack.traj_euc[sb].mean(0); cb /= np.linalg.norm(cb)+1e-12
            print(f"  [{src}] harm_n={sh.sum()} benign_n={sb.sum()} "
                  f"center_cos={float(np.dot(ch,cb)):.4f}")

    # ── Step 2: 预加载 benign 集（用于配对）──────────────────
    benign_pack = None
    if args.benign_jsonl:
        print(f"\n[Step 2] Loading benign pool: {args.benign_jsonl}")
        benign_pack = pack("benign_pool",
            extract_trace_embeddings(model, make_loader(args.benign_jsonl), device, "benign"))
        print(f"  benign_pool: n={len(benign_pack.y)}, "
              f"all_benign={np.all(benign_pack.y==0)}")

    methods = [
        "trace_center_margin",
        "trace_benign_departure",
        "last_step_center_margin",
        "stepwise_max_center_margin",
    ]

    # ── Step 3: 逐测试集评估 ──────────────────────────────────
    all_splits_results  = {}
    all_score_dist      = {}

    for eval_path, eval_name in zip(args.eval_jsonl, args.eval_names):
        print(f"\n{'='*80}")
        print(f"[Step 3] Evaluating: {eval_name}  ({eval_path})")

        eval_pack = pack(eval_name,
            extract_trace_embeddings(model, make_loader(eval_path), device, eval_name))

        n_harm   = int(np.sum(eval_pack.y == 1))
        n_benign = int(np.sum(eval_pack.y == 0))
        print(f"  n_total={len(eval_pack.y)}  n_harm={n_harm}  n_benign={n_benign}")

        lang_counts = {}
        for lg in eval_pack.langs:
            lang_counts[lg] = lang_counts.get(lg, 0) + 1
        print(f"  lang distribution: {lang_counts}")

        max_k_data = int(min(eval_pack.step_euc.shape[1],
                             np.max(eval_pack.traj_len)))
        max_k = max_k_data if args.max_prefix <= 0 else min(args.max_prefix, max_k_data)

        is_pure_harmful = (n_harm > 0 and n_benign == 0)
        is_pure_benign  = (n_harm == 0 and n_benign > 0)
        has_both        = (n_harm > 0 and n_benign > 0)

        # ── 情况 A：两类都有，直接评估 ──────────────────────
        if has_both:
            print(f"  Mode: direct eval (k=1..{max_k})")
            split_results = evaluate_split(
                eval_pack, harm_c, benign_c, max_k,
                args.target_tpr, methods,
                use_full_for_short=args.use_full_for_short,
            )
            all_splits_results[eval_name] = split_results

        # ── 情况 B：全 harmful，配对 benign 后评估 ───────────
        elif is_pure_harmful:
            print(f"  ⚠️  全为 harmful")

            # B1: score 分布分析（无需 benign）
            print(f"  Mode B1: score distribution (k=1..{max_k})")
            score_dist = score_distribution_analysis(eval_pack, harm_c, benign_c, max_k)
            all_score_dist[eval_name] = score_dist
            # 打印摘要
            print(f"  {'k':>3}  {'n':>6}  {'margin_mean':>12}  "
                  f"{'margin_p50':>10}  {'frac_pos':>9}")
            for k, d in score_dist.items():
                print(f"  {k:>3}  {d['n']:>6}  {d['margin_mean']:>12.4f}  "
                      f"{d['margin_p50']:>10.4f}  {d['frac_positive_margin']:>9.4f}")

            # B2: 若提供了 benign_pool，配对后计算 AUC
            if benign_pack is not None:
                paired_name = f"{eval_name}+benign"
                paired_pack = merge_packs(paired_name, eval_pack, benign_pack)
                max_k_paired = min(max_k, int(np.max(paired_pack.traj_len)))
                print(f"\n  Mode B2: paired AUC with benign_pool "
                      f"(n_harm={n_harm}, n_benign={len(benign_pack.y)}, "
                      f"k=1..{max_k_paired})")
                split_results = evaluate_split(
                    paired_pack, harm_c, benign_c, max_k_paired,
                    args.target_tpr, methods,
                    use_full_for_short=True,   # 配对模式强制使用 full_for_short
                )
                all_splits_results[paired_name] = split_results
            else:
                print("  (提供 --benign_jsonl 可启用配对 AUC 计算)")

        # ── 情况 C：全 benign，报告 score 分布（FPR 基线）───
        elif is_pure_benign:
            print(f"  ⚠️  全为 benign（FPR 基线集）")
            print(f"  Mode C: score distribution (k=1..{max_k})")
            score_dist = score_distribution_analysis(eval_pack, harm_c, benign_c, max_k)
            all_score_dist[eval_name] = score_dist
            print(f"  {'k':>3}  {'n':>6}  {'margin_mean':>12}  "
                  f"{'margin_p50':>10}  {'frac_pos(FPR)':>13}")
            for k, d in score_dist.items():
                print(f"  {k:>3}  {d['n']:>6}  {d['margin_mean']:>12.4f}  "
                      f"{d['margin_p50']:>10.4f}  {d['frac_positive_margin']:>13.4f}")

    # ── Step 4: 保存结果 ──────────────────────────────────────
    for split_name, results in all_splits_results.items():
        out_path = os.path.join(args.output_dir, f"prefix_metrics_{split_name}.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved: {out_path}")

    if all_score_dist:
        dist_path = os.path.join(args.output_dir, "score_distributions.json")
        with open(dist_path, "w") as f:
            json.dump(all_score_dist, f, indent=2)
        print(f"Saved: {dist_path}")

    summary_all = {
        split_name: {m: obj["summary"] for m, obj in results.items()}
        for split_name, results in all_splits_results.items()
    }
    summary_path = os.path.join(args.output_dir, "prefix_summary_all.json")
    with open(summary_path, "w") as f:
        json.dump(summary_all, f, indent=2)
    print(f"Saved: {summary_path}")

    # 打印汇总表
    if summary_all:
        print("\n" + "=" * 90)
        print("SUMMARY TABLE  (AUDC / EDS)")
        print("=" * 90)
        print(f"{'split':<30}" + "  ".join(f"{m[:22]:<22}" for m in methods))
        print("-" * 90)
        for split_name in summary_all:
            row = f"{split_name:<30}"
            for method in methods:
                s = summary_all[split_name][method]
                audc_s = "None" if s["AUDC"] is None else f"{s['AUDC']:.4f}"
                eds_s  = "None" if s["EDS"]  is None else str(s["EDS"])
                row += f"{audc_s}/{eds_s:<5}  "
            print(row)


if __name__ == "__main__":
    main()