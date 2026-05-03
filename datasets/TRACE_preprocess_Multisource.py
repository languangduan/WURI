import os
import re
import json
import random
import argparse
from typing import List, Dict, Optional
from collections import defaultdict


# ════════════════════════════════════════════════════════════
# 统一标签空间
# ════════════════════════════════════════════════════════════

HAZARD_TYPE_NAMES = {
    0:  "data_theft",
    1:  "code_execution",
    2:  "persistence",
    3:  "supply_chain",
    4:  "destruction",
    5:  "reconnaissance",
    6:  "privilege_escalation",
    7:  "resource_abuse",
    8:  "info_theft",
    9:  "compliance_violation",
    10: "reasoning_failure",
    11: "tool_error",
}
NUM_HAZARD_TYPES = 12

AH_CATEGORY_TO_HAZARD: Dict[str, int] = {
    "Data_Exfiltration":           0,
    "System_RCE":                  1,
    "Persistence_Establishment":   2,
    "Persistence":                 2,
    "Supply_Chain_Poisoning":      3,
    "Destructive_Action":          4,
    "Internal_Reconnaissance":     5,
    "Privilege_Escalation":        6,
    "Resource_Exhaustion":         7,
    "Prompt_Intelligence_Theft":   8,
    "Prompt_Intel_Theft":          8,
    "Compliance_Bypass":           9,
    "Data_Exfil":                  0,
    "Data_Exfiltration_Attack":    0,
    "RCE":                         1,
    "Remote_Code_Execution":       1,
    "System_RCE_Attack":           1,
    "Supply_Chain":                3,
    "Supply_Chain_Attack":         3,
    "Destructive":                 4,
    "Destructive_Attack":          4,
    "Recon":                       5,
    "Reconnaissance":              5,
    "Internal_Recon":              5,
    "Privilege_Esc":               6,
    "Priv_Esc":                    6,
    "Resource_Exhaust":            7,
    "DoS":                         7,
    "Compliance":                  9,
    "Compliance_Violation":        9,
}

AT_FAILURE_TO_HAZARD: Dict[str, int] = {
    "unauthorized_disclosure":                                  0,
    "malicious_executables":                                    1,
    "insecure_interaction_or_execution":                        2,
    "choosing_malicious_tool":                                  3,
    "procedural_deviation_or_inaction":                         4,
    "unconfirmed_or_over_privileged_action":                    5,
    "over_privileged_action":                                   5,
    "tool_misuse_in_context":                                   6,
    "inefficient_or_wasteful_execution":                        7,
    "wasteful_execution":                                       7,
    "harmful_or_offensive_content":                             8,
    "harmful_or_illegal_instruction":                           9,
    "provide_inaccurate_misleading_or_unverified_information":  9,
    "provide_inaccurate_misleading_or_unverified":              9,
    "misleading_or_unverified_information":                     9,
    "flawed_planning_or_reasoning":                             10,
    "incorrect_tool_parameters":                                11,
    "failure_to_validate_tool_outputs":                         11,
    "instruction_for_harmful_illegal_activity":                 9,
    "tool_misuse_in_specific_context":                          6,
    "unauthorized_information_disclosure":                      0,
    "generation_of_malicious_executables":                      1,
    "generation_of_harmful_offensive_content":                  8,
}

JAILBREAK_TO_SURFACE: Dict[str, int] = {
    "Direct":                          0,
    "Recursive_Salami_Slicing":        1,
    "Security_Audit_Gaslighting":      2,
    "Dependency_Hook_Trigger":         3,
    "Pseudo_Diagnostic_Debugging":     4,
    "Contextual_Persona_Enforcement":  5,
    "Encoded_Payload_Smuggling":       6,
    "Implicit_Indirect_Injection":     7,
    "Logical_Dependency_Inversion":    8,
    "Legacy_Code_Maintenance":         9,
}
NUM_ATTACK_SURFACES = 10

AT_RISK_TO_SURFACE: Dict[str, int] = {
    "malicious_user_instruction_or_jailbreak": 0,
    "malicious_user":                          0,
    "jailbreak":                               0,
    "direct_prompt_injection":                 0,
    "indirect_prompt_injection":               7,
    "tool_description_injection":              7,
    "malicious_tool_execution":                3,
    "corrupted_tool_feedback":                 3,
    "unreliable_or_misinformation":            4,
    "inherent_agent_or_llm_failures":         -1,
    "inherent_agent_failures":                -1,
    "inherent_llm_failures":                  -1,
}

FAILURE_MODE_TO_ID: Dict[str, int] = {
    "unconfirmed_or_over_privileged_action":                    0,
    "flawed_planning_or_reasoning":                             1,
    "incorrect_tool_parameters":                                2,
    "choosing_malicious_tool":                                  3,
    "tool_misuse_in_context":                                   4,
    "failure_to_validate_tool_outputs":                         5,
    "insecure_interaction_or_execution":                        6,
    "procedural_deviation_or_inaction":                         7,
    "inefficient_or_wasteful_execution":                        8,
    "harmful_or_offensive_content":                             9,
    "harmful_or_illegal_instruction":                           10,
    "malicious_executables":                                    11,
    "unauthorized_disclosure":                                  12,
    "provide_inaccurate_misleading_or_unverified_information":  13,
    "provide_inaccurate_misleading_or_unverified":              13,
    "misleading_or_unverified_information":                     13,
    "over_privileged_action":                                   0,
    "wasteful_execution":                                       8,
    "instruction_for_harmful_illegal_activity":                 10,
    "tool_misuse_in_specific_context":                          4,
    "unauthorized_information_disclosure":                      12,
    "generation_of_malicious_executables":                      11,
    "generation_of_harmful_offensive_content":                  9,
}

# ── SafeDialBench 专用映射表 ──────────────────────────────────
SDB_METHOD_TO_SURFACE: Dict[str, int] = {
    # ── 论文 7 种攻击方法（全部收录）────────────────────────
    "Reference Attack":    7,   # 渐进引用 → Implicit_Indirect_Injection
    "Scence Construct":    8,   # ⚠️ 数据集原生拼写错误，必须保留此拼写
    "Scene Construct":     8,   # 正确拼写，兼容性保留
    "Scene Construction":  8,
    "Purpose Reverse":     8,   # 目的反转 → Logical_Dependency_Inversion
    "Purpose Reversal":    8,
    "Fallacy Attack":      2,   # 谬误攻击 → Security_Audit_Gaslighting
    "Fallacy":             2,
    "Probing Question":    4,   # 探测性提问 → Pseudo_Diagnostic_Debugging
    "Probing":             4,
    "Role Play":           5,   # 角色扮演 → Contextual_Persona_Enforcement
    "Role-play Attack":    5,
    "Topic Change":        1,   # 话题切换 → Recursive_Salami_Slicing
    "Topic Switch":        1,
    # ── 旧有条目保留 ─────────────────────────────────────────
    "Direct Attack":       0,
    "Direct":              0,
    "Roleplay":            5,
    "Hypothetical":        8,
    "Counterfactual":      8,
    "Authority":           2,
    "Encoded":             6,
    "Incremental":         1,
    "Crescendo":           1,
}

SDB_TASK_TO_HAZARD: Dict[str, int] = {
    # ── 论文 6 个安全维度（核心 task 名称）──────────────────
    "Fairness":    8,   # 公平性 → info_theft/harmful_content（偏见/歧视）
    "Legality":    9,   # 合法性 → compliance_violation（法律违规）
    "Morality":    9,   # 道德性 → compliance_violation（道德违规）
    "Aggression":  8,   # 攻击性 → info_theft（有害攻击性内容）
    "Ethics":      9,   # 伦理   → compliance_violation（自我伤害/虐待）
    "Privacy":     0,   # 隐私   → data_theft（隐私泄露）
    # ── 旧有条目保留 ─────────────────────────────────────────
    "Law and Morality":    9,
    "Social Norms":        9,
    "Personal Information": 0,
    "Cybersecurity":       1,
    "Hate Speech":         8,
    "Harassment":          8,
    "Violence":            9,
    "Self-harm":           9,
    "Financial Fraud":     0,
    "Fraud":               0,
    # ── 中文 task 名称（SDB_ZH）──────────────────────────────
    "公平性":  8,
    "合法性":  9,
    "道德":    9,
    "攻击性":  8,
    "伦理":    9,
    "隐私":    0,
}


UNKNOWN_ID = -1


# ════════════════════════════════════════════════════════════
# 工具函数
# ════════════════════════════════════════════════════════════

def load_jsonl(path: str) -> list:
    """读取 .jsonl 文件，每行一个 JSON 对象；也兼容整体 JSON 数组文件。"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    # 优先尝试整体 JSON（兼容 .json 和单体 .jsonl）
    if content.startswith("[") or content.startswith("{"):
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return list(data.values())
        except json.JSONDecodeError:
            pass
    # 逐行解析 JSONL
    for lineno, line in enumerate(content.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  ⚠️  JSONL parse error at line {lineno}: {e}")
    return records


def normalize_key(raw: str) -> str:
    s = raw.lower().strip()
    s = re.sub(r"[\s\-/&,]+", "_", s)
    s = re.sub(r"[^a-z0-9_]", "", s)
    return re.sub(r"_+", "_", s).strip("_")


def _edit_distance(a: str, b: str) -> int:
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for c2 in b:
        new_row = [row[0] + 1]
        for j, c1 in enumerate(a):
            new_row.append(min(row[j] + (c1 != c2), row[j + 1] + 1, new_row[-1] + 1))
        row = new_row
    return row[-1]


def lookup_id(mapping: Dict[str, int], raw: str) -> int:
    if not raw:
        return UNKNOWN_ID
    if raw in mapping:
        return mapping[raw]
    key = normalize_key(raw)
    if key in mapping:
        return mapping[key]
    best_key, best_dist = None, 999
    for k in mapping:
        d = _edit_distance(key, normalize_key(k))
        if d < best_dist:
            best_dist, best_key = d, k
    if best_dist <= 4 and best_key is not None:
        return mapping[best_key]
    return UNKNOWN_ID


def truncate(text: str, max_chars: int) -> str:
    text = str(text).strip()
    return text if len(text) <= max_chars else text[:max_chars] + "…"


def detect_injection(text: str) -> bool:
    markers = [
        "SYSTEM:", "SYSTEM NOTE:", "IGNORE PREVIOUS", "<!-- ", "{{",
        "[INST]", "ignore all previous instructions",
        "injected instruction", "injected_instruction",
    ]
    tl = text.lower()
    return any(m.lower() in tl for m in markers)


# ════════════════════════════════════════════════════════════
# Parsers
# ════════════════════════════════════════════════════════════

class AgentHazardParser:
    MAX_STEP = 300

    def parse(self, raw_list):
        records, stats = [], defaultdict(int)
        for i, item in enumerate(raw_list):
            rec = self._parse_one(item, i)
            if rec:
                records.append(rec)
                stats["parsed"] += 1
            else:
                stats["skipped_empty_steps"] += 1
        return records, dict(stats)

    def _parse_one(self, item, idx):
        steps_raw = item.get("decomposed_query") or item.get("decomposition") or []
        if not steps_raw:
            return None
        steps      = [f"[PLAN] {truncate(str(s), self.MAX_STEP)}" for s in steps_raw]
        cat_raw    = item.get("category", "")
        jb_raw     = item.get("jailbreak_method") or item.get("attack_strategy", "")
        hazard_id  = lookup_id(AH_CATEGORY_TO_HAZARD, cat_raw)
        surface_id = lookup_id(JAILBREAK_TO_SURFACE, jb_raw)
        raw_id     = item.get("id") or item.get("original_id")
        return {
            "id": f"ah_{raw_id if raw_id is not None else f'{idx:06d}'}",
            "source": "agenthazard",
            "steps": steps,
            "num_steps": len(steps),
            "labels": {
                "binary":            1,
                "hazard_type":       hazard_id,
                "attack_surface":    surface_id,
                "category_name":     cat_raw,
                "jailbreak_name":    jb_raw,
                "failure_mode":      UNKNOWN_ID,
                "failure_mode_name": "",
                "risk_source":       UNKNOWN_ID,
                "risk_source_name":  "",
            },
            "meta": {
                "is_benign":    0,
                "split":        None,
                "domain":       "plan",
                "has_injection": False,
                "original_id":  raw_id,
            },
        }


class ToolBenchParser:
    MAX_ARG  = 100
    MAX_RESP = 150
    MAX_MSG  = 150
    VALID_TOOL = re.compile(r'^[a-zA-Z0-9_-]{1,64}$')

    def __init__(self, min_steps=2, max_steps=20, tb_max=None):
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.tb_max    = tb_max   # 最多保留多少条 ToolBench 记录（None = 不限）

    def parse(self, raw_data):
        raw_list = list(raw_data.values()) if isinstance(raw_data, dict) else raw_data
        records, stats = [], defaultdict(int)
        for i, item in enumerate(raw_list):
            if self.tb_max is not None and len(records) >= self.tb_max:
                stats["truncated_by_tb_max"] += len(raw_list) - i
                break
            rec, reason = self._parse_one(item, i)
            if rec:
                records.append(rec)
                stats["parsed"] += 1
            else:
                stats[f"skipped_{reason}"] += 1
        return records, dict(stats)

    def _parse_one(self, item, idx):
        answers = item.get("answers", [])
        if not answers:
            return None, "no_answers"
        best  = self._best_idx(answers, item.get("preference", []))
        steps = self._extract_steps(answers[best], item.get("query", ""))
        if len(steps) < self.min_steps:
            return None, "too_few_steps"
        return {
            "id": f"tb_{idx:06d}",
            "source": "toolbench",
            "steps": steps[:self.max_steps],
            "num_steps": min(len(steps), self.max_steps),
            "labels": {
                "binary":            0,
                "hazard_type":       UNKNOWN_ID,
                "attack_surface":    UNKNOWN_ID,
                "category_name":     "",
                "jailbreak_name":    "",
                "failure_mode":      UNKNOWN_ID,
                "failure_mode_name": "",
                "risk_source":       UNKNOWN_ID,
                "risk_source_name":  "",
            },
            "meta": {
                "is_benign":    1,
                "split":        None,
                "domain":       "execution",
                "has_injection": False,
                "original_id":  None,
            },
        }, "ok"

    def _best_idx(self, answers, pref):
        n  = len(answers)
        vp = pref[:n] if len(pref) >= n else None
        if vp and len(vp) == n:
            mp    = max(vp)
            cands = [i for i, p in enumerate(vp) if p == mp]
            if cands:
                return min(cands, key=lambda i: answers[i].get("total_steps", 999))
        valid = [
            (i, a.get("total_steps", 999))
            for i, a in enumerate(answers)
            if not self._is_giveup(a.get("answer_details", []))
        ]
        return min(valid, key=lambda x: x[1])[0] if valid else 0

    @staticmethod
    def _is_giveup(details):
        for node in details:
            if isinstance(node.get("message", ""), str) and "give_up_and_restart" in node["message"]:
                return True
            if node.get("next") and ToolBenchParser._is_giveup(node["next"]):
                return True
        return False

    def _extract_steps(self, answer, query):
        steps = []
        if query:
            steps.append(f"[TASK] {truncate(query, 200)}")
        for node in answer.get("answer_details", []):
            self._dfs(node, steps)
        return steps

    def _dfs(self, node, steps):
        text = self._node_to_text(node.get("role", ""), node.get("message", ""))
        if text:
            steps.append(text)
        for child in node.get("next", []):
            self._dfs(child, steps)

    def _node_to_text(self, role, message):
        if not message or role == "system":
            return None
        if role == "tool":
            return self._parse_tool_msg(message)
        if role == "assistant":
            msg = str(message).strip()
            if re.search(r"does not match '\^\[a-zA-Z", msg) or not msg:
                return None
            return f"[THINK] {truncate(msg, self.MAX_MSG)}"
        return None

    def _parse_tool_msg(self, message):
        data = message if isinstance(message, dict) else None
        if data is None:
            try:
                data = json.loads(message)
            except Exception:
                try:
                    data = eval(message)
                except Exception:
                    data = self._regex_parse(message)
        if not data or not isinstance(data, dict):
            return None
        name = data.get("name", "")
        if name == "Finish" or not name:
            return None
        if not self.VALID_TOOL.match(name):
            return None
        args_raw = data.get("arguments", "{}")
        try:
            args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
        except Exception:
            args = {}
        resp = str(data.get("response", ""))
        if any(m in resp for m in ["<!DOCTYPE", "<!doctype", '"error": "Message error']):
            resp = "[API Error]"
        else:
            resp = truncate(resp, self.MAX_RESP)
        return f"[CALL] {name}({self._fmt(args)}) → {resp}"

    @staticmethod
    def _regex_parse(msg):
        result = {}
        for field in ["name", "arguments", "response"]:
            m = re.search(rf"'{field}':\s*'([^']*)'", msg)
            if m:
                result[field] = m.group(1)
        return result

    def _fmt(self, args):
        if not isinstance(args, dict) or not args:
            return ""
        return truncate(
            ", ".join(f"{k}={truncate(str(v), 35)}" for k, v in args.items()),
            self.MAX_ARG,
        )


class ATBenchParser:
    MAX_THOUGHT = 80
    MAX_ARG     = 120
    MAX_RESP    = 200
    MAX_QUERY   = 200

    def parse(self, raw_list):
        if isinstance(raw_list, dict):
            raw_list = list(raw_list.values())
        records, stats = [], defaultdict(int)
        for i, item in enumerate(raw_list):
            rec = self._parse_one(item, i)
            if rec:
                records.append(rec)
                stats["parsed"] += 1
                if rec["meta"]["has_injection"]:
                    stats["has_injection"] += 1
                if rec["labels"]["binary"] == 1:
                    stats["harmful"] += 1
                else:
                    stats["benign"] += 1
            else:
                stats["skipped_empty"] += 1
        return records, dict(stats)

    def _parse_one(self, item, idx):
        co = item.get("content", [])
        if not co:
            return None
        first = co[0]
        turns = first if isinstance(first, list) else co
        steps, has_inj = self._extract_steps(turns)
        if not steps:
            return None
        label      = int(item.get("label", 0))
        fm_raw     = item.get("failure_mode", "")
        rs_raw     = item.get("risk_source", "")
        hazard_id  = lookup_id(AT_FAILURE_TO_HAZARD, fm_raw) if label == 1 else UNKNOWN_ID
        surface_id = lookup_id(AT_RISK_TO_SURFACE, rs_raw)   if label == 1 else UNKNOWN_ID
        raw_id     = item.get("conv_id") or None
        return {
            "id": f"at_{raw_id if raw_id is not None else f'{idx:06d}'}",
            "source": "atbench",
            "steps": steps,
            "num_steps": len(steps),
            "labels": {
                "binary":            label,
                "hazard_type":       hazard_id,
                "attack_surface":    surface_id,
                "category_name":     "",
                "jailbreak_name":    rs_raw,
                "failure_mode":      lookup_id(FAILURE_MODE_TO_ID, fm_raw),
                "failure_mode_name": fm_raw,
                "risk_source":       lookup_id(
                    {k: v for k, v in AT_RISK_TO_SURFACE.items() if v >= 0}, rs_raw
                ),
                "risk_source_name":  rs_raw,
            },
            "meta": {
                "is_benign":    1 - label,
                "split":        None,
                "domain":       "execution",
                "has_injection": has_inj,
                "original_id":  raw_id,
                "tool_names":   [t.get("name", "") for t in item.get("tool_used", [])],
            },
        }

    def _extract_steps(self, turns):
        steps, has_inj = [], False
        i = 0
        while i < len(turns):
            turn = turns[i]
            role = turn.get("role", "")
            if role == "user":
                c = turn.get("content", "")
                if c:
                    steps.append(f"[TASK] {truncate(c, self.MAX_QUERY)}")
                i += 1
            elif role == "agent":
                env = (
                    turns[i + 1]
                    if i + 1 < len(turns) and turns[i + 1].get("role") == "environment"
                    else None
                )
                t, inj = self._parse_agent_env(turn, env)
                if t:
                    steps.append(t)
                if inj:
                    has_inj = True
                i += 2 if env else 1
            else:
                i += 1
        return steps, has_inj

    def _parse_agent_env(self, agent_turn, env_turn):
        thought       = agent_turn.get("thought", "").strip()
        name, args_str = self._parse_action(agent_turn.get("action", ""))
        if name is None:
            return None, False
        resp_str, inj = self._parse_env(env_turn) if env_turn else ("", False)
        parts = []
        if thought:
            parts.append(f"[THINK] {truncate(thought, self.MAX_THOUGHT)}")
        parts.append(f"[CALL] {name}({args_str})")
        if resp_str:
            parts.append(f"→ {resp_str}")
        return " ".join(parts), inj

    def _parse_action(self, action_raw):
        if not action_raw or action_raw.strip().startswith("Complete"):
            return None, ""
        try:
            data = json.loads(action_raw)
        except Exception:
            try:
                data = json.loads(action_raw.replace("'", '"'))
            except Exception:
                return None, ""
        name = data.get("name", "")
        if not name:
            return None, ""
        args = data.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return name, self._fmt(args)

    def _parse_env(self, env_turn):
        content = env_turn.get("content", "")
        if not content:
            return "", False
        try:
            data = json.loads(content) if isinstance(content, str) else content
            text = self._extract_text(data)
        except Exception:
            text = str(content)
        inj = detect_injection(text)
        return ("[INJECT] " if inj else "") + truncate(text, self.MAX_RESP), inj

    def _extract_text(self, data):
        if isinstance(data, str):
            return data
        if isinstance(data, dict):
            parts = []
            for key in ["notes", "_data_quality", "injected_instruction", "injected instruction"]:
                val = data.get(key)
                if val:
                    parts.append(str(val))
            result = data.get("result")
            if result:
                parts.append(
                    str(result.get("message", "")) if isinstance(result, dict) else str(result)
                )
            return " | ".join(filter(None, parts)) or str(data)
        return str(data)

    def _fmt(self, args):
        if not isinstance(args, dict) or not args:
            return ""
        return truncate(
            ", ".join(f"{k}={truncate(str(v), 35)}" for k, v in args.items()),
            self.MAX_ARG,
        )


class SafeDialBenchParser:
    """
    将 SafeDialBench 多轮对话数据转换为统一轨迹格式。

    步骤编码规则：
      [TASK]  → 第一轮 user 消息（任务起点）
      [USER]  → 后续 user 消息（攻击升级步骤）
      [RESP]  → bot 响应（模型行为记录）

    lang 参数：
      "en" → id 前缀 sdb_en_
      "zh" → id 前缀 sdb_zh_
    """

    MAX_USER  = 200
    MAX_BOT   = 150
    MIN_TURNS = 2

    def __init__(self, lang: str = "en"):
        assert lang in ("en", "zh"), f"lang must be 'en' or 'zh', got '{lang}'"
        self.lang = lang

    def parse(self, raw_list: list):
        if isinstance(raw_list, dict):
            raw_list = list(raw_list.values())
        records, stats = [], defaultdict(int)
        for i, item in enumerate(raw_list):
            rec = self._parse_one(item, i)
            if rec:
                records.append(rec)
                stats["parsed"] += 1
                stats[f"method_{item.get('method', 'unknown')}"] += 1
                stats[f"task_{item.get('task', 'unknown')}"]     += 1
            else:
                stats["skipped"] += 1
        return records, dict(stats)

    def _parse_one(self, item: dict, idx: int):
        history = item.get("history", [])
        if len(history) < self.MIN_TURNS:
            return None

        steps = self._extract_steps(history)
        if not steps:
            return None

        method  = item.get("method", "")
        task    = item.get("task", "")
        scene   = item.get("scene", "")
        raw_id  = item.get("id")

        # 优先用 scene 做 hazard 映射，其次用 task
        hazard_id = lookup_id(SDB_TASK_TO_HAZARD, scene)
        if hazard_id == UNKNOWN_ID:
            hazard_id = lookup_id(SDB_TASK_TO_HAZARD, task)

        surface_id = lookup_id(SDB_METHOD_TO_SURFACE, method)
        has_inj    = any(detect_injection(s) for s in steps)

        # id 前缀区分语言，避免 EN/ZH 之间 id 碰撞
        prefix = f"sdb_{self.lang}_"
        rec_id = f"{prefix}{raw_id if raw_id is not None else f'{idx:06d}'}"

        return {
            "id":       rec_id,
            "source":   "safedialbench",
            "steps":    steps,
            "num_steps": len(steps),
            "labels": {
                "binary":            1,
                "hazard_type":       hazard_id,
                "attack_surface":    surface_id,
                "category_name":     scene or task,
                "jailbreak_name":    method,
                "failure_mode":      UNKNOWN_ID,
                "failure_mode_name": "",
                "risk_source":       surface_id,
                "risk_source_name":  method,
            },
            "meta": {
                "is_benign":    0,
                "split":        None,
                "domain":       "dialogue",
                "has_injection": has_inj,
                "original_id":  raw_id,
                "model_type":   item.get("model_type", ""),
                "num_turns":    len(history),
                "task":         task,
                "scene":        scene,
                "lang":         self.lang,   # ← 语言标记
            },
        }

    def _extract_steps(self, history: list) -> List[str]:
        steps = []
        for i, turn in enumerate(history):
            user_msg = turn.get("user", "").strip()
            bot_msg  = turn.get("bot", "").strip()
            if user_msg:
                prefix = "[TASK]" if i == 0 else "[USER]"
                steps.append(f"{prefix} {truncate(user_msg, self.MAX_USER)}")
            if bot_msg:
                steps.append(f"[RESP] {truncate(bot_msg, self.MAX_BOT)}")
        return steps


# ════════════════════════════════════════════════════════════
# Split Manager（多数据源 + 双语 SDB）
# ════════════════════════════════════════════════════════════

class MultiSourceSplitManager:
    """
    Split 策略：
      train        : ATBench seen (harmful+benign)  +  SDB_EN train fold
      val          : ATBench seen val               +  SDB_EN val fold
      at_zeroshot  : ATBench harmful holdout（按 hazard_type）
      sdb_test_en  : SDB_EN test fold（按 method 分层 15%）
      sdb_test_zh  : SDB_ZH 全量（跨语言 zero-shot 测试，不参与训练）
      ah_test      : 全量 AgentHazard
      tb_test      : 全量 ToolBench
    """

    def __init__(
        self,
        zeroshot_hazard_types: List[int],
        val_ratio:      float = 0.10,
        sdb_test_ratio: float = 0.15,
        seed:           int   = 42,
    ):
        self.zs_types       = set(zeroshot_hazard_types)
        self.val_ratio      = val_ratio
        self.sdb_test_ratio = sdb_test_ratio
        random.seed(seed)

    def assign(
        self,
        ah_records,
        tb_records,
        at_records,
        sdb_en_records = None,
        sdb_zh_records = None,
    ):
        splits         = defaultdict(list)
        sdb_en_records = sdb_en_records or []
        sdb_zh_records = sdb_zh_records or []

        # ── ATBench ──────────────────────────────────────────
        at_seen, at_zs = [], []
        for r in at_records:
            hz      = r["labels"]["hazard_type"]
            is_harm = (r["labels"]["binary"] == 1)
            if is_harm and hz in self.zs_types:
                r["meta"]["split"] = "at_zeroshot"
                at_zs.append(r)
            else:
                at_seen.append(r)

        random.shuffle(at_seen)
        val_n    = max(1, int(len(at_seen) * self.val_ratio)) if at_seen else 0
        at_val   = at_seen[:val_n]
        at_train = at_seen[val_n:]

        for r in at_train: r["meta"]["split"] = "train"
        for r in at_val:   r["meta"]["split"] = "val"
        for r in at_zs:    r["meta"]["split"] = "at_zeroshot"

        splits["train"].extend(at_train)
        splits["val"].extend(at_val)
        splits["at_zeroshot"].extend(at_zs)

        # ── SDB_EN：按 method 分层 → train / val / sdb_test_en ──
        if sdb_en_records:
            sdb_by_method = defaultdict(list)
            for r in sdb_en_records:
                sdb_by_method[r["labels"]["jailbreak_name"]].append(r)

            en_train_pool, en_test_pool = [], []
            for method, recs in sdb_by_method.items():
                random.shuffle(recs)
                test_n = max(1, int(len(recs) * self.sdb_test_ratio))
                en_test_pool.extend(recs[:test_n])
                en_train_pool.extend(recs[test_n:])

            random.shuffle(en_train_pool)
            en_val_n   = max(1, int(len(en_train_pool) * self.val_ratio))
            en_val     = en_train_pool[:en_val_n]
            en_train   = en_train_pool[en_val_n:]

            for r in en_train:    r["meta"]["split"] = "train"
            for r in en_val:      r["meta"]["split"] = "val"
            for r in en_test_pool: r["meta"]["split"] = "sdb_test_en"

            splits["train"].extend(en_train)
            splits["val"].extend(en_val)
            splits["sdb_test_en"].extend(en_test_pool)

        # ── SDB_ZH：全量 → sdb_test_zh（不参与训练）────────────
        for r in sdb_zh_records:
            r["meta"]["split"] = "sdb_test_zh"
            splits["sdb_test_zh"].append(r)

        # ── AgentHazard / ToolBench ───────────────────────────
        for r in ah_records:
            r["meta"]["split"] = "ah_test"
            splits["ah_test"].append(r)

        for r in tb_records:
            r["meta"]["split"] = "tb_test"
            splits["tb_test"].append(r)

        # 最终打乱各 split
        for key in splits:
            random.shuffle(splits[key])

        return dict(splits)


# ════════════════════════════════════════════════════════════
# 统计与质量检查
# ════════════════════════════════════════════════════════════

def compute_stats(splits: dict) -> dict:
    stats = {}
    for name, records in splits.items():
        src_cnt    = defaultdict(int)
        bin_cnt    = defaultdict(int)
        haz_cnt    = defaultdict(int)
        dom_cnt    = defaultdict(int)
        srf_cnt    = defaultdict(int)
        step_lens  = []
        method_cnt = defaultdict(int)
        turn_lens  = []
        lang_cnt   = defaultdict(int)

        for r in records:
            src_cnt[r["source"]]              += 1
            bin_cnt[r["labels"]["binary"]]    += 1
            haz_cnt[r["labels"]["hazard_type"]] += 1
            dom_cnt[r["meta"]["domain"]]      += 1
            srf_cnt[r["labels"]["attack_surface"]] += 1
            step_lens.append(r["num_steps"])
            if r["source"] == "safedialbench":
                method_cnt[r["labels"]["jailbreak_name"]] += 1
                turn_lens.append(r["meta"].get("num_turns", 0))
                lang_cnt[r["meta"].get("lang", "unknown")]  += 1

        haz_named = {
            HAZARD_TYPE_NAMES.get(k, f"unknown({k})"): v
            for k, v in sorted(haz_cnt.items())
        }

        stats[name] = {
            "total":           len(records),
            "by_source":       dict(src_cnt),
            "binary":          {str(k): v for k, v in bin_cnt.items()},
            "by_hazard":       haz_named,
            "by_domain":       dict(dom_cnt),
            "by_surface":      dict(srf_cnt),
            "avg_steps":       round(sum(step_lens) / max(len(step_lens), 1), 2),
            "min_steps":       min(step_lens) if step_lens else 0,
            "max_steps":       max(step_lens) if step_lens else 0,
            "injection_count": sum(1 for r in records if r["meta"].get("has_injection")),
            "sdb_by_method":   dict(method_cnt),
            "sdb_avg_turns":   round(sum(turn_lens) / len(turn_lens), 2) if turn_lens else 0,
            "sdb_by_lang":     dict(lang_cnt),
        }
    return stats


def quality_check(records: list) -> list:
    issues, seen = [], set()
    for r in records:
        if r["id"] in seen:
            issues.append(f"Duplicate id: {r['id']}")
        seen.add(r["id"])
        if not r["steps"]:
            issues.append(f"Empty steps: {r['id']}")
        if r["source"] == "agenthazard" and r["labels"]["hazard_type"] == UNKNOWN_ID:
            issues.append(
                f"AH missing hazard_type: {r['id']} "
                f"(raw='{r['labels']['category_name']}')"
            )
        if (r["source"] == "atbench"
                and r["labels"]["binary"] == 1
                and r["labels"]["hazard_type"] == UNKNOWN_ID):
            issues.append(
                f"AT harmful missing hazard_type: {r['id']} "
                f"(failure_mode='{r['labels']['failure_mode_name']}')"
            )
        if r["source"] == "safedialbench":
            if r["meta"].get("num_turns", 0) < 2:
                issues.append(f"SDB too few turns: {r['id']}")
            if r["labels"]["attack_surface"] == UNKNOWN_ID:
                issues.append(
                    f"SDB unknown attack_surface: {r['id']} "
                    f"(method='{r['labels']['jailbreak_name']}')"
                )
    return issues


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="TRACE 预处理脚本 v5 - MultiSource (AT + SDB_EN + SDB_ZH)"
    )
    parser.add_argument("--agenthazard",       required=True,
                        help="AgentHazard JSON 文件路径")
    parser.add_argument("--toolbench",         required=True,
                        help="ToolBench JSON 文件路径")
    parser.add_argument("--atbench",           required=True,
                        help="ATBench JSON/JSONL 文件路径")
    parser.add_argument("--safedialbench_en",  default=None,
                        help="SafeDialBench 英文 JSONL 路径（参与训练）")
    parser.add_argument("--safedialbench_zh",  default=None,
                        help="SafeDialBench 中文 JSONL 路径（全量作为跨语言测试集）")
    parser.add_argument("--output_dir",        default="data/processed_multisource",
                        help="输出目录")
    parser.add_argument(
        "--zeroshot_hazard_types",
        nargs="+", type=int, default=[6, 7],
        help=f"AT 内部留出的 zero-shot hazard_type ID，可选: {HAZARD_TYPE_NAMES}",
    )
    parser.add_argument("--val_ratio",         type=float, default=0.10,
                        help="验证集比例（对 AT seen 和 SDB_EN train pool 均适用）")
    parser.add_argument("--sdb_test_ratio",    type=float, default=0.15,
                        help="SDB_EN 按 method 分层留出的测试比例")
    parser.add_argument("--min_steps",         type=int,   default=2)
    parser.add_argument("--max_steps",         type=int,   default=20)
    parser.add_argument("--tb_max",            type=int,   default=None,
                        help="ToolBench 最多保留条数（None = 不限，兼容旧版 --tb_max 50）")
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    # ── 创建输出目录 ──────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Loading raw data ...")

    # AgentHazard / ToolBench / ATBench（支持 JSON 和 JSONL）
    ah_raw = load_jsonl(args.agenthazard)
    tb_raw = load_jsonl(args.toolbench)
    at_raw = load_jsonl(args.atbench)
    print(f"  AgentHazard raw : {len(ah_raw)} items")
    print(f"  ToolBench   raw : {len(tb_raw)} items")
    print(f"  ATBench     raw : {len(at_raw)} items")

    # SafeDialBench EN（可选）
    sdb_en_raw = []
    if args.safedialbench_en and os.path.exists(args.safedialbench_en):
        sdb_en_raw = load_jsonl(args.safedialbench_en)
        print(f"  SDB_EN      raw : {len(sdb_en_raw)} items  ← {args.safedialbench_en}")
    else:
        print("  SDB_EN          : not provided, skipping")

    # SafeDialBench ZH（可选）
    sdb_zh_raw = []
    if args.safedialbench_zh and os.path.exists(args.safedialbench_zh):
        sdb_zh_raw = load_jsonl(args.safedialbench_zh)
        print(f"  SDB_ZH      raw : {len(sdb_zh_raw)} items  ← {args.safedialbench_zh}")
    else:
        print("  SDB_ZH          : not provided, skipping")

    # ── 解析 ──────────────────────────────────────────────────
    print("\nParsing ...")
    ah_records,    ah_stats    = AgentHazardParser().parse(ah_raw)
    tb_records,    tb_stats    = ToolBenchParser(
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        tb_max=args.tb_max,
    ).parse(tb_raw)
    at_records,    at_stats    = ATBenchParser().parse(at_raw)
    sdb_en_records, sdb_en_stats = [], {}
    sdb_zh_records, sdb_zh_stats = [], {}

    if sdb_en_raw:
        sdb_en_records, sdb_en_stats = SafeDialBenchParser(lang="en").parse(sdb_en_raw)
    if sdb_zh_raw:
        sdb_zh_records, sdb_zh_stats = SafeDialBenchParser(lang="zh").parse(sdb_zh_raw)

    print(f"  AgentHazard   : {ah_stats}")
    print(f"  ToolBench     : {tb_stats}")
    print(f"  ATBench       : {at_stats}")
    if sdb_en_stats:
        print(f"  SDB_EN        : {sdb_en_stats}")
    if sdb_zh_stats:
        print(f"  SDB_ZH        : {sdb_zh_stats}")

    # ── 质量检查 ──────────────────────────────────────────────
    print("\nQuality check ...")
    all_records = ah_records + tb_records + at_records + sdb_en_records + sdb_zh_records
    issues = quality_check(all_records)
    if issues:
        print(f"  ⚠️  {len(issues)} issues:")
        for iss in issues[:20]:
            print(f"    {iss}")
    else:
        print("  ✅ No issues")

    # ── 分割 ──────────────────────────────────────────────────
    zs_names = [HAZARD_TYPE_NAMES.get(i, str(i)) for i in args.zeroshot_hazard_types]
    print(f"\nAT zero-shot holdout: {list(zip(args.zeroshot_hazard_types, zs_names))}")

    splitter = MultiSourceSplitManager(
        zeroshot_hazard_types=args.zeroshot_hazard_types,
        val_ratio=args.val_ratio,
        sdb_test_ratio=args.sdb_test_ratio,
        seed=args.seed,
    )
    splits = splitter.assign(
        ah_records, tb_records, at_records,
        sdb_en_records=sdb_en_records,
        sdb_zh_records=sdb_zh_records,
    )

    # ── 写出 JSONL ────────────────────────────────────────────
    print("\nWriting JSONL ...")
    for split_name, records in splits.items():
        out_path = os.path.join(args.output_dir, f"{split_name}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {split_name:20s}: {len(records):5d} → {out_path}")

    # ── 统计 ──────────────────────────────────────────────────
    stats      = compute_stats(splits)
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n📊 stats.json     → {stats_path}")

    # ── label_map ─────────────────────────────────────────────
    label_map = {
        "dataset_mode":          "multisource",
        "version":               "v5",
        "num_hazard_types":      NUM_HAZARD_TYPES,
        "num_attack_surfaces":   NUM_ATTACK_SURFACES,
        "hazard_type":           HAZARD_TYPE_NAMES,
        "attack_surface":        {v: k for k, v in JAILBREAK_TO_SURFACE.items()},
        "zeroshot_hazard_types": args.zeroshot_hazard_types,
        "zeroshot_hazard_names": zs_names,
        "model_config": {
            "num_attack_classes": NUM_HAZARD_TYPES,
        },
        "domains":  ["plan", "execution", "dialogue"],
        "sources":  ["atbench", "safedialbench", "agenthazard", "toolbench"],
        "languages": ["en", "zh"],
        "splits": {
            "train":       "ATBench seen (harmful+benign) + SDB_EN train fold (85%)",
            "val":         "ATBench seen val (10%) + SDB_EN val fold (10% of 85%)",
            "at_zeroshot": "ATBench harmful holdout by hazard_type (zero-shot)",
            "sdb_test_en": "SDB_EN test fold (stratified by method, 15%)",
            "sdb_test_zh": "SDB_ZH full set (cross-lingual zero-shot test)",
            "ah_test":     "All AgentHazard samples (cross-dataset test)",
            "tb_test":     "ToolBench benign samples (FPR baseline)",
        },
    }
    lm_path = os.path.join(args.output_dir, "label_map.json")
    with open(lm_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print(f"📋 label_map.json → {lm_path}")

    # ── 汇总打印 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for sname, s in stats.items():
        pos = s["binary"].get("1", 0)
        neg = s["binary"].get("0", 0)
        if   neg == 0 and pos > 0: bal = "⚠️  NO NEGATIVES"
        elif pos == 0 and neg > 0: bal = "⚠️  NO POSITIVES"
        elif pos == 0 and neg == 0: bal = "⚠️  EMPTY"
        else:                       bal = "✅"

        print(f"\n[{sname}]")
        print(f"  total={s['total']} | pos={pos} neg={neg} {bal}")
        print(f"  sources   = {s['by_source']}")
        print(f"  domain    = {s['by_domain']}")
        print(f"  avg_steps = {s['avg_steps']} (min={s['min_steps']}, max={s['max_steps']})")
        if s.get("sdb_by_method"):
            print(f"  sdb_methods   = {s['sdb_by_method']}")
            print(f"  sdb_avg_turns = {s['sdb_avg_turns']}")
            print(f"  sdb_by_lang   = {s['sdb_by_lang']}")
        print(f"  hazard_type distribution:")
        for hname, cnt in s["by_hazard"].items():
            bar = "█" * min(cnt // 10, 40)
            print(f"    {hname:25s}: {cnt:4d} {bar}")
        if s["injection_count"]:
            print(f"  ⚠️  injections = {s['injection_count']}")


if __name__ == "__main__":
    main()