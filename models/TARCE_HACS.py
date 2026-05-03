import torch

import torch.nn as nn

import torch.nn.functional as F

from transformers import AutoModel

import math

_LN2 = math.log(2.0)

class H_ACS_Encoder(nn.Module):

    def __init__(self, model_name_or_path: str,

                 num_atoms: int = 1024, temperature: float = 0.7, routing_topk: int = 128):

        super().__init__()

        self.embedding_model = AutoModel.from_pretrained(model_name_or_path)

        self.embed_dim   = int(self.embedding_model.config.hidden_size)

        self.num_atoms   = int(num_atoms)

        self.temperature = float(temperature)

        self.routing_topk = routing_topk

        self.atom_bank  = nn.Parameter(

            torch.randn(self.num_atoms, self.embed_dim) * 0.1)

        self.query_proj = nn.Linear(self.embed_dim, self.embed_dim)

        with torch.no_grad():

            d = self.embed_dim

            self.query_proj.weight.copy_(

                torch.eye(d) + torch.randn(d, d) * 0.01)

            self.query_proj.bias.zero_()

    @property

    def hidden_size(self):

        return self.embed_dim

    @property

    def atom_dim(self):

        return self.embed_dim

    def get_output_dims(self):

        return {

            "embed_dim": self.embed_dim,

            "hidden_size": self.embed_dim,

            "num_atoms": self.num_atoms,

        }

    @staticmethod

    def _mean_pool(last_hidden_state, attention_mask):

        mask = attention_mask.unsqueeze(-1).float()

        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1e-9)

    def _weights_from_base_(self, base_raw):

        q      = self.query_proj(base_raw)

        scores = q @ self.atom_bank.t()

        return F.softmax(scores / self.temperature, dim=-1), scores

    def _weights_from_base(self, base_raw, topk=None):

        q = F.normalize(self.query_proj(base_raw), p=2, dim=1)

        a = F.normalize(self.atom_bank, p=2, dim=1)

        scores = q @ a.t()

        if topk is None or topk >= scores.size(1):

            weights = F.softmax(scores / self.temperature, dim=-1)

            return weights, scores

        k = max(1, int(topk))

        topv, topi = torch.topk(scores, k=k, dim=-1)

        topw = F.softmax(topv / self.temperature, dim=-1)

        weights = torch.zeros_like(scores)

        weights.scatter_(1, topi, topw)

        return weights, scores

    def _reconstruct(self, weights):

        euc_raw = weights @ self.atom_bank

        return euc_raw, F.normalize(euc_raw, p=2, dim=1)

    def _encode(self, input_ids, attention_mask):

        out = self.embedding_model(input_ids=input_ids,

                                   attention_mask=attention_mask)

        base_raw = self._mean_pool(out.last_hidden_state, attention_mask)

        base_emb = F.normalize(base_raw, p=2, dim=1)

        weights, scores = self._weights_from_base(base_raw, topk=self.routing_topk)

        euc_raw, euc_vec = self._reconstruct(weights)

        return {

            "base_raw": base_raw,

            "base_emb": base_emb,

            "weights": weights,

            "scores": scores,

            "euc_raw": euc_raw,

            "euc_vec": euc_vec,

        }

    @staticmethod

    def _l1_renorm(x, eps=1e-12):

        return x / x.sum(dim=1, keepdim=True).clamp_min(eps)

    @staticmethod

    def _top_p_mask(weights, top_p):

        if top_p is None or top_p >= 1.0:

            return weights

        sorted_w, idx = torch.sort(weights, dim=1, descending=True)

        cdf  = torch.cumsum(sorted_w, dim=1)

        keep = cdf <= float(top_p)

        keep[:, 0] = True

        masked = torch.zeros_like(weights)

        masked.scatter_(1, idx, sorted_w * keep.float())

        return masked

    def steer_weights(self, weights, concept_profile,

                      steering_lambda=0.0, adaptive_alpha=True,

                      top_p=1.0, eps=1e-12):

        if steering_lambda <= 0.0 or concept_profile is None:

            z = torch.zeros((weights.size(0), 1),

                            device=weights.device, dtype=weights.dtype)

            return weights, z, z

        prof = concept_profile

        if prof.dim() == 1:

            prof = prof.unsqueeze(0)

        prof  = F.relu(self._l1_renorm(

            prof.to(weights.device, weights.dtype), eps))

        alpha = ((weights * prof).sum(1, keepdim=True) if adaptive_alpha

                 else torch.ones((weights.size(0), 1),

                                 device=weights.device, dtype=weights.dtype))

        steered = F.relu(weights - float(steering_lambda) * alpha * prof)

        steered = self._l1_renorm(self._top_p_mask(steered, top_p), eps)

        return steered, alpha, (weights - steered).clamp_min(0).sum(1, keepdim=True)

    @torch.no_grad()

    def encode(self, input_ids, attention_mask):

        return self._encode(input_ids, attention_mask)

class TemporalAtomAccumulator(nn.Module):

    def __init__(self, num_atoms: int, embed_dim: int,

                 mode: str = "attention", decay_gamma: float = 0.9,

                 window_size: int = 10):

        super().__init__()

        self.num_atoms   = num_atoms

        self.mode        = mode

        self.decay_gamma = decay_gamma

        self.window_size = window_size

        if mode == "attention":

            self.attn_q_proj = nn.Linear(num_atoms, num_atoms // 4)

            self.attn_k_proj = nn.Linear(num_atoms, num_atoms // 4)

            self.attn_scale  = (num_atoms // 4) ** -0.5

    def forward(self, weights_seq, traj_lengths=None):

        acc = (self._decay_accumulate(weights_seq)

               if self.mode == "decay"

               else self._attention_accumulate(weights_seq))

        if traj_lengths is not None:

            B, T, N    = acc.shape

            step_idx   = torch.arange(T, device=acc.device).unsqueeze(0)

            valid_mask = step_idx < traj_lengths.unsqueeze(1)

            acc        = acc * valid_mask.unsqueeze(-1).float()

        return acc

    def _decay_accumulate(self, weights_seq):

        B, T, N = weights_seq.shape

        W       = self.window_size

        padded  = F.pad(weights_seq, (0, 0, W - 1, 0))

        windows = padded.unfold(1, W, 1).permute(0, 1, 3, 2)

        decay   = torch.tensor(

            [self.decay_gamma ** (W - 1 - i) for i in range(W)],

            device=weights_seq.device, dtype=weights_seq.dtype)

        decay   = decay / decay.sum()

        return (windows * decay.view(1, 1, W, 1)).sum(dim=2)

    def _attention_accumulate(self, weights_seq):

        B, T, N = weights_seq.shape

        W       = self.window_size

        Q       = self.attn_q_proj(weights_seq)

        K       = self.attn_k_proj(weights_seq)

        V       = weights_seq

        scores  = torch.bmm(Q, K.transpose(1, 2)) * self.attn_scale

        idx     = torch.arange(T, device=scores.device)

        causal  = idx.unsqueeze(0) <= idx.unsqueeze(1)

        window  = (idx.unsqueeze(0) - idx.unsqueeze(1)) < W

        scores  = scores.masked_fill(

            ~(causal & window).unsqueeze(0), float("-inf"))

        attn    = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)

        return torch.bmm(attn, V)

class CoActivationComputer(nn.Module):

    def __init__(self, num_atoms: int, topk_atoms: int = 64):

        super().__init__()

        self.num_atoms  = num_atoms

        self.topk_atoms = topk_atoms

    def forward(self, accumulated, t=None):

        if accumulated.dim() == 2:

            return self._compute(accumulated)

        if t is not None:

            return self._compute(accumulated[:, t, :])

        return [self._compute(accumulated[:, i, :])

                for i in range(accumulated.size(1))]

    def _compute(self, A):

        B, N = A.shape

        K    = min(self.topk_atoms, N)

        topk_vals, topk_idx = torch.topk(A, K, dim=1)

        return CoActState(topk_vals=topk_vals, topk_idx=topk_idx,

                          full_A=A, B=B, N=N, K=K)

class CoActState:

    def __init__(self, topk_vals, topk_idx, full_A, B, N, K):

        self.topk_vals = topk_vals

        self.topk_idx  = topk_idx

        self.full_A    = full_A

        self.B = B;  self.N = N;  self.K = K

    def frob_inner_lowrank(self, U):

        B, K    = self.topk_vals.shape

        C, N, R = U.shape

        U_t     = U.permute(1, 2, 0)

        U_topk  = U_t[self.topk_idx]

        Au      = torch.einsum("bk,bkrc->brc", self.topk_vals, U_topk)

        return (Au ** 2).sum(dim=1) / (K * R + 1e-9)

    def to_dense(self):

        B, K = self.topk_vals.shape

        N    = self.N

        M    = torch.zeros(B, N, N, device=self.topk_vals.device,

                           dtype=self.topk_vals.dtype)

        row_idx  = self.topk_idx.unsqueeze(2).expand(B, K, K)

        col_idx  = self.topk_idx.unsqueeze(1).expand(B, K, K)

        vals     = self.topk_vals.unsqueeze(2) * self.topk_vals.unsqueeze(1)

        flat_idx = (row_idx * N + col_idx).view(B, K * K)

        M.view(B, N * N).scatter_add_(1, flat_idx, vals.view(B, K * K))

        return M

class AttackPrototypeBank(nn.Module):

    def __init__(self, num_classes: int, num_atoms: int,

                 coact_rank: int = 16, proto_temp: float = 0.1):

        super().__init__()

        self.num_classes = num_classes

        self.num_atoms   = num_atoms

        self.coact_rank  = coact_rank

        self.proto_temp  = proto_temp

        atoms_per_class = max(1, num_atoms // num_classes)

        proto_init      = torch.zeros(num_classes, num_atoms)

        perm            = torch.randperm(num_atoms)

        for c in range(num_classes):

            start = (c * atoms_per_class) % num_atoms

            idx   = perm[start: start + atoms_per_class]

            if len(idx) < atoms_per_class:

                idx = torch.cat([idx, perm[: atoms_per_class - len(idx)]])

            proto_init[c, idx] = 1.0

        self.proto_weights = nn.Parameter(

            proto_init + torch.randn(num_classes, num_atoms) * 0.01)

        self.proto_coact_U = nn.Parameter(

            torch.randn(num_classes, num_atoms, coact_rank) * 0.1)

        self.proto_span    = nn.Parameter(

            torch.randn(num_classes) * 1.0 + 5.0)

    def get_proto_weights(self):

        return F.softplus(self.proto_weights) - _LN2

    def get_proto_weights_raw(self):

        return self.proto_weights

    def get_proto_coact_U(self):

        return self.proto_coact_U

    def match_score(self, coact_state, span_t,

                    w_weight=1.0, w_coact=1.0, w_span=0.3, max_span=20.0):

        A_t     = coact_state.full_A

        proto_w = self.get_proto_weights()

        score_w = A_t @ proto_w.t()

        score_w = score_w - score_w.mean(dim=1, keepdim=True)

        score_coact = coact_state.frob_inner_lowrank(

            self.get_proto_coact_U())

        span_norm       = (span_t / max_span).clamp(0.0, 1.0)

        proto_span_norm = torch.sigmoid(self.proto_span)

        score_span      = -(span_norm.unsqueeze(1) -

                             proto_span_norm.unsqueeze(0)).abs()

        return w_weight * score_w + w_coact * score_coact + w_span * score_span

    def compute_residual_score(self, coact_state, match_scores):

        beta    = F.softmax(match_scores, dim=-1)

        U       = self.get_proto_coact_U()

        B, C    = beta.shape

        K, R    = coact_state.K, U.size(2)

        U_recon      = torch.einsum("bc,cnr->bnr", beta, U)

        norm_Mhat    = (coact_state.topk_vals ** 2).sum(dim=1)

        idx          = coact_state.topk_idx

        batch_idx    = torch.arange(B, device=U.device).unsqueeze(1).expand(B, K)

        U_recon_topk = U_recon[batch_idx, idx, :]

        Au           = torch.einsum("bk,bkr->br",

                                    coact_state.topk_vals, U_recon_topk)

        cross        = (Au ** 2).sum(dim=1) / (K * R + 1e-9)

        norm_Mrecon  = (U_recon ** 2).sum(dim=(1, 2)) / (K * R + 1e-9)

        norm_Mhat_n  = norm_Mhat / (K + 1e-9)

        return (norm_Mhat_n - 2 * cross + norm_Mrecon).clamp(0.0, 1e4)

class ZeroShotHazardScorer(nn.Module):

    def __init__(self, num_atoms: int):

        super().__init__()

        self.num_atoms = num_atoms

        self.register_buffer("atom_hazard_prior", torch.ones(num_atoms))

    def set_hazard_prior(self, prior: torch.Tensor):

        assert prior.shape == (self.num_atoms,)

        self.atom_hazard_prior = F.relu(prior).clamp_min(1e-9)

    def forward(self, residual_norm_sq, coact_state):

        novelty = residual_norm_sq.clamp_min(0).sqrt()

        h       = self.atom_hazard_prior

        h_norm  = h / h.sum().clamp_min(1e-9)

        h_topk  = h_norm[coact_state.topk_idx]

        return novelty * (F.relu(coact_state.topk_vals) * h_topk).sum(dim=1)

class TRACE(nn.Module):

    def __init__(self, model_name_or_path: str,

                 num_atoms: int = 1024, temperature: float = 0.7,

                 num_attack_classes: int = 12, coact_rank: int = 16,

                 topk_atoms: int = 64, accumulator_mode: str = "attention",

                 decay_gamma: float = 0.9, window_size: int = 10,

                 proto_temp: float = 0.1):

        super().__init__()

        self.encoder = H_ACS_Encoder(model_name_or_path, num_atoms, temperature, routing_topk=topk_atoms)

        self.num_atoms = num_atoms

        self.accumulator    = TemporalAtomAccumulator(

            num_atoms, self.encoder.embed_dim,

            accumulator_mode, decay_gamma, window_size)

        self.coact_computer = CoActivationComputer(num_atoms, topk_atoms)

        self.prototype_bank = AttackPrototypeBank(

            num_attack_classes, num_atoms, coact_rank, proto_temp)

        self.zeroshot_scorer = ZeroShotHazardScorer(num_atoms)

        self.known_scale    = nn.Parameter(torch.tensor(2.0))

        self.zs_bonus_scale = nn.Parameter(torch.tensor(0.5))

        self.known_bias = nn.Parameter(torch.tensor(-1.0))

        self.risk_scale = nn.Parameter(torch.tensor(1.0))

        self.zs_scale   = nn.Parameter(torch.tensor(0.1))

    def forward_trajectory(self, input_ids_seq, attention_mask_seq,

                            traj_lengths=None):

        B, T, L = input_ids_seq.shape

        weights_list = []

        for t in range(T):

            enc = self.encoder._encode(input_ids_seq[:, t, :],

                                       attention_mask_seq[:, t, :])

            weights_list.append(enc["weights"])

        weights_seq = torch.stack(weights_list, dim=1)

        accumulated = self.accumulator(weights_seq, traj_lengths=traj_lengths)

        match_scores_list, zs_scores_list = [], []

        C = self.prototype_bank.num_classes

        for t in range(T):

            coact_state = self.coact_computer(accumulated, t=t)

            span_t      = torch.full((B,), float(t + 1),

                                     dtype=torch.float, device=accumulated.device)

            scores_t    = self.prototype_bank.match_score(coact_state, span_t)

            match_scores_list.append(scores_t)

            res_norm_sq = self.prototype_bank.compute_residual_score(

                coact_state, scores_t)

            zs_scores_list.append(

                self.zeroshot_scorer(res_norm_sq, coact_state))

        match_scores = torch.stack(match_scores_list, dim=1)

        zs_scores    = torch.stack(zs_scores_list,   dim=1)

        match_probs = F.softmax(match_scores, dim=-1)

        max_prob    = match_probs.max(dim=-1).values

        known_logit = (max_prob - 1.0 / C) * self.known_scale + self.known_bias

        zs_norm  = zs_scores / (zs_scores.detach().max().clamp_min(1e-6))

        zs_bonus = zs_norm * self.zs_bonus_scale

        step_scores = torch.sigmoid(known_logit + zs_bonus)

        if traj_lengths is not None:

            step_idx   = torch.arange(T, device=step_scores.device).unsqueeze(0)

            valid_mask = step_idx < traj_lengths.unsqueeze(1)

            risk_score = step_scores.masked_fill(

                ~valid_mask, float("-inf")).max(dim=1).values

        else:

            risk_score = step_scores.max(dim=1).values

        return {

            "weights_seq":    weights_seq,

            "accumulated":    accumulated,

            "match_scores":   match_scores,

            "zs_scores":      zs_scores,

            "known_scores":   torch.sigmoid(known_logit),

            "zs_scores_norm": torch.sigmoid(zs_bonus),

            "step_scores":    step_scores,

            "risk_score":     risk_score,

        }

    @torch.no_grad()

    def online_step(self, input_ids, attention_mask,

                    history_weights=None, history_accumulated=None,

                    concept_profile=None, steering_lambda=0.0):

        enc = self.encoder._encode(input_ids, attention_mask)

        w_t = enc["weights"]

        weights_seq = (torch.cat([history_weights, w_t.unsqueeze(1)], dim=1)

                       if history_weights is not None else w_t.unsqueeze(1))

        t_new = weights_seq.size(1) - 1

        if history_accumulated is not None and t_new > 0:

            acc_t       = self._incremental_accumulate(

                weights_seq, history_accumulated, t_new)

            accumulated = torch.cat(

                [history_accumulated, acc_t.unsqueeze(1)], dim=1)

        else:

            accumulated = self.accumulator(weights_seq)

        C           = self.prototype_bank.num_classes

        coact_state = self.coact_computer(accumulated, t=t_new)

        span_t      = torch.full((w_t.size(0),), float(t_new + 1),

                                 device=w_t.device)

        match_scores = self.prototype_bank.match_score(coact_state, span_t)

        res_norm_sq  = self.prototype_bank.compute_residual_score(

            coact_state, match_scores)

        zs_score_raw = self.zeroshot_scorer(res_norm_sq, coact_state)

        match_probs = F.softmax(match_scores, dim=-1)

        max_prob    = match_probs.max(dim=-1).values

        known_logit = (max_prob - 1.0 / C) * self.known_scale + self.known_bias

        zs_norm  = zs_score_raw / (zs_score_raw.detach().max().clamp_min(1e-6))

        zs_bonus = zs_norm * self.zs_bonus_scale

        final_score = torch.sigmoid(known_logit + zs_bonus)

        known_score = torch.sigmoid(known_logit)

        zs_score_n  = torch.sigmoid(zs_bonus)

        steered_vec = None

        if steering_lambda > 0 and concept_profile is not None:

            sw, _, _       = self.encoder.steer_weights(

                w_t, concept_profile, steering_lambda=steering_lambda)

            _, steered_vec = self.encoder._reconstruct(sw)

        return {

            "final_score":                 final_score,

            "known_score":                 known_score,

            "zs_score":                    zs_score_n,

            "match_scores":                match_scores,

            "A_t":                         accumulated[:, t_new, :],

            "topk_atoms":                  coact_state.topk_idx,

            "updated_history_weights":     weights_seq,

            "updated_history_accumulated": accumulated,

            "steered_vec":                 steered_vec,

        }

    def _incremental_accumulate(self, weights_seq, history_accumulated, t):

        if self.accumulator.mode == "decay":

            W      = self.accumulator.window_size

            start  = max(0, t - W + 1)

            window = weights_seq[:, start:t + 1, :]

            w_size = window.size(1)

            decay  = torch.tensor(

                [self.accumulator.decay_gamma ** (w_size - 1 - i)

                 for i in range(w_size)],

                device=weights_seq.device, dtype=weights_seq.dtype)

            return (window * (decay / decay.sum()).view(1, w_size, 1)).sum(1)

        else:

            W      = self.accumulator.window_size

            start  = max(0, t - W + 1)

            q_t    = self.accumulator.attn_q_proj(weights_seq[:, t:t + 1, :])

            k_hist = self.accumulator.attn_k_proj(weights_seq[:, start:t + 1, :])

            v_hist = weights_seq[:, start:t + 1, :]

            score  = torch.bmm(q_t, k_hist.transpose(1, 2))

            weight = torch.nan_to_num(

                F.softmax(score * self.accumulator.attn_scale, dim=-1), nan=0.0)

            return torch.bmm(weight, v_hist).squeeze(1)
