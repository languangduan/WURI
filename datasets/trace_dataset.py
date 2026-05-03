"""
TRACE 完整数据处理 Pipeline

数据来源：
  - AgentHazard : 攻击轨迹（有标签）
  - ToolBench   : 良性轨迹（无危害标签）

处理要点：
  1. ToolBench 树形轨迹 → 线性步骤序列
  2. 步骤文本构造（tool name + args + response 拼接）
  3. 质量过滤（失败轨迹、过短轨迹）
  4. 联合 Dataset，统一 collate
"""

import json
import random
import re
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import AutoTokenizer
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict


# ════════════════════════════════════════════════════════════
# 常量定义
# ════════════════════════════════════════════════════════════

CATEGORY_TO_ID = {
    "Data_Exfiltration":         0,
    "System_RCE":                1,
    "Persistence":               2,
    "Supply_Chain_Poisoning":    3,
    "Destructive_Action":        4,
    "Internal_Reconnaissance":   5,
    "Privilege_Escalation":      6,
    "Resource_Exhaustion":       7,
    "Prompt_Intelligence_Theft": 8,
    "Compliance_Bypass":         9,
}

JAILBREAK_TO_ID = {
    "Direct":                        0,
    "Recursive_Salami_Slicing":      1,
    "Security_Audit_Gaslighting":    2,
    "Dependency_Hook_Trigger":       3,
    "Pseudo_Diagnostic_Debugging":   4,
    "Contextual_Persona_Enforcement":5,
    "Encoded_Payload_Smuggling":     6,
    "Implicit_Indirect_Injection":   7,
    "Logical_Dependency_Inversion":  8,
    "Legacy_Code_Maintenance":       9,
}

BENIGN_LABEL = -1   # 良性轨迹的 category_label


# ════════════════════════════════════════════════════════════
# ToolBench 轨迹解析器
# ════════════════════════════════════════════════════════════

class ToolBenchParser:
    """
    将 ToolBench 的嵌套树形轨迹展平为线性步骤文本列表。

    步骤文本格式：
      - tool call : "Call {name}({args_summary}). Result: {resp_summary}"
      - assistant : "Reasoning: {message_truncated}"
      - user      : "Task: {query_truncated}"
    """

    # 工具调用步骤的文本模板
    TOOL_TEMPLATE   = "Call {name}({args}). Result: {response}"
    REASON_TEMPLATE = "Reasoning: {message}"
    TASK_TEMPLATE   = "Task: {message}"

    # 响应/参数的最大字符数（避免过长文本淹没语义）
    MAX_ARG_CHARS  = 120
    MAX_RESP_CHARS = 200
    MAX_MSG_CHARS  = 150

    def __init__(self, select_method: str = "best"):
        """
        Args:
            select_method:
              'best'   - 选 preference 最高的轨迹
              'first'  - 选第一条轨迹
              'all'    - 返回所有轨迹（数据增强）
        """
        self.select_method = select_method

    # ── 公共入口 ─────────────────────────────────────────
    def parse(self, instance: Dict) -> List[List[str]]:
        """
        解析单条 ToolBench 实例，返回步骤文本列表的列表。

        Returns:
            List[List[str]]：每个元素是一条轨迹的步骤文本列表
        """
        answers  = instance.get("answers", [])
        pref     = instance.get("preference", [])

        if not answers:
            return []

        if self.select_method == "best":
            idx = self._best_answer_idx(answers, pref)
            trajs = [self._extract_trajectory(answers[idx])]
        elif self.select_method == "first":
            trajs = [self._extract_trajectory(answers[0])]
        else:  # 'all'
            trajs = [self._extract_trajectory(a) for a in answers]

        # 过滤空轨迹
        return [t for t in trajs if t]

    # ── 选择最佳轨迹 ─────────────────────────────────────
    @staticmethod
    def _best_answer_idx(answers: List[Dict], preference: List[int]) -> int:
        """
        根据 preference 列表选择最佳答案。
        preference[i] 越大越好；若无偏好信息则选步骤最少的成功轨迹。
        """
        if preference and len(preference) == len(answers):
            return int(max(range(len(preference)), key=lambda i: preference[i]))

        # fallback：选有 final_answer 且步骤最少的
        successful = [
            (i, a) for i, a in enumerate(answers)
            if a.get("final_answer", "") and "give_up" not in a.get("final_answer", "")
        ]
        if successful:
            return min(successful, key=lambda x: x[1].get("total_steps", 999))[0]
        return 0

    # ── 树形 DFS 展平 ────────────────────────────────────
    def _extract_trajectory(self, answer: Dict) -> List[str]:
        """
        DFS 遍历 answer_details 树，按对话顺序提取步骤文本。
        跳过：system 消息、give_up_and_restart、空消息。
        """
        steps = []
        details = answer.get("answer_details", [])
        for node in details:
            self._dfs(node, steps)
        return steps

    def _dfs(self, node: Dict, steps: List[str]):
        role    = node.get("role", "")
        message = node.get("message", "")

        step_text = self._node_to_text(role, message)
        if step_text:
            steps.append(step_text)

        for child in node.get("next", []):
            self._dfs(child, steps)

    def _node_to_text(self, role: str, message: str) -> Optional[str]:
        """将单个节点转换为步骤文本，无效节点返回 None"""
        if not message or role == "system":
            return None

        # ── tool 节点：解析工具调用结构 ──────────────────
        if role == "tool":
            return self._parse_tool_message(message)

        # ── assistant 节点：推理文本 ──────────────────────
        if role == "assistant":
            msg = self._truncate(message, self.MAX_MSG_CHARS)
            # 过滤纯错误信息
            if self._is_error_only(msg):
                return None
            return self.REASON_TEMPLATE.format(message=msg)

        # ── user 节点：任务描述 ───────────────────────────
        if role == "user":
            msg = self._truncate(message, self.MAX_MSG_CHARS)
            return self.TASK_TEMPLATE.format(message=msg)

        return None

    def _parse_tool_message(self, message: str) -> Optional[str]:
        """
        解析 tool 节点的 message 字符串。

        ToolBench tool message 格式（字符串化的 dict）：
          "{'name': 'xxx', 'arguments': '...', 'response': '...'}"
        """
        try:
            # 尝试 eval 解析（ToolBench 使用单引号 dict 字符串）
            data = eval(message) if isinstance(message, str) else message
        except Exception:
            # fallback：正则提取
            data = self._regex_parse_tool(message)

        if not data:
            return None

        name = data.get("name", "unknown_tool")

        # 过滤 Finish 工具（不作为有效步骤）
        if name == "Finish":
            return None

        # 解析 arguments
        args_raw = data.get("arguments", "{}")
        args_str = self._summarize_args(args_raw)

        # 解析 response
        resp_raw = data.get("response", "")
        resp_str = self._summarize_response(resp_raw)

        return self.TOOL_TEMPLATE.format(
            name=name,
            args=args_str,
            response=resp_str,
        )

    @staticmethod
    def _regex_parse_tool(message: str) -> Dict:
        """正则 fallback 解析 tool message"""
        result = {}
        name_match = re.search(r"'name':\s*'([^']+)'", message)
        if name_match:
            result["name"] = name_match.group(1)
        args_match = re.search(r"'arguments':\s*'([^']*)'", message)
        if args_match:
            result["arguments"] = args_match.group(1)
        resp_match = re.search(r"'response':\s*'(.*?)'(?:,\s*'|\})", message, re.DOTALL)
        if resp_match:
            result["response"] = resp_match.group(1)
        return result

    def _summarize_args(self, args_raw: Any) -> str:
        """将参数 JSON 压缩为简短摘要"""
        if isinstance(args_raw, str):
            try:
                args_dict = json.loads(args_raw)
            except Exception:
                return self._truncate(str(args_raw), self.MAX_ARG_CHARS)
        else:
            args_dict = args_raw

        if not args_dict:
            return "()"

        # 只保留 key=value 对，截断长值
        parts = []
        for k, v in args_dict.items():
            v_str = self._truncate(str(v), 40)
            parts.append(f"{k}={v_str}")
        return ", ".join(parts)[:self.MAX_ARG_CHARS]

    def _summarize_response(self, resp_raw: Any) -> str:
        """将响应内容压缩为简短摘要"""
        resp_str = str(resp_raw) if not isinstance(resp_raw, str) else resp_raw

        # 过滤明显的错误响应
        if self._is_error_response(resp_str):
            return "[API Error]"

        return self._truncate(resp_str, self.MAX_RESP_CHARS)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        text = text.strip()
        return text if len(text) <= max_chars else text[:max_chars] + "..."

    @staticmethod
    def _is_error_only(text: str) -> bool:
        """判断 assistant 消息是否只包含错误信息（无实质内容）"""
        error_patterns = [
            r"does not match '\^\[a-zA-Z",   # 工具名格式错误
            r"give_up_and_restart",
        ]
        return any(re.search(p, text) for p in error_patterns)

    @staticmethod
    def _is_error_response(resp: str) -> bool:
        """判断 API 响应是否为错误"""
        return (
            '"error": "Message error' in resp or
            "<!DOCTYPE" in resp or          # HTML 错误页
            '"error": "Unauthorized"' in resp
        )


# ════════════════════════════════════════════════════════════
# ToolBench 质量过滤器
# ════════════════════════════════════════════════════════════

class ToolBenchFilter:
    """
    过滤低质量的 ToolBench 轨迹。

    过滤条件：
      1. 轨迹步骤数 < min_steps（过短，信息量不足）
      2. 所有答案均为 give_up_and_restart（任务失败）
      3. 工具名包含非 ASCII 字符且无法解析（编码问题）
      4. 步骤全为 [API Error]（API 不可用）
    """

    def __init__(
        self,
        min_steps: int = 2,
        max_steps: int = 20,
        min_tool_calls: int = 1,       # 至少包含1次真实工具调用
        allow_failed: bool = False,    # 是否保留失败轨迹
    ):
        self.min_steps     = min_steps
        self.max_steps     = max_steps
        self.min_tool_calls = min_tool_calls
        self.allow_failed  = allow_failed

    def is_valid(self, instance: Dict, steps: List[str]) -> bool:
        # 步骤数检查
        if len(steps) < self.min_steps or len(steps) > self.max_steps:
            return False

        # 工具调用数检查
        tool_calls = [s for s in steps if s.startswith("Call ")]
        if len(tool_calls) < self.min_tool_calls:
            return False

        # 全为 API Error 检查
        error_steps = [s for s in steps if "[API Error]" in s]
        if len(error_steps) == len(tool_calls):
            return False

        # 失败轨迹检查
        if not self.allow_failed:
            answers = instance.get("answers", [])
            all_failed = all(
                "give_up" in a.get("final_answer", "") or
                not a.get("final_answer", "")
                for a in answers
            )
            if all_failed:
                return False

        return True


# ════════════════════════════════════════════════════════════
# tokenize 工具函数（复用）
# ════════════════════════════════════════════════════════════

def tokenize_trajectory(
    steps: List[str],
    tokenizer,
    max_length: int = 128,
) -> Dict[str, torch.Tensor]:
    input_ids_list, mask_list = [], []
    for step in steps:
        enc = tokenizer(
            step,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids_list.append(enc["input_ids"].squeeze(0))
        mask_list.append(enc["attention_mask"].squeeze(0))
    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(mask_list),
    }


# ════════════════════════════════════════════════════════════
# ToolBench Dataset（良性轨迹）
# ════════════════════════════════════════════════════════════

class ToolBenchDataset(Dataset):
    """
    ToolBench 良性轨迹数据集。

    输出格式与 AgentHazardDataset 对齐，
    category_label = BENIGN_LABEL(-1)，用于区分良性/攻击样本。
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_step_length: int = 128,
        max_traj_length: int = 20,
        select_method: str = "best",
        min_steps: int = 2,
        max_instances: Optional[int] = None,   # 限制数量，与AgentHazard平衡
        seed: int = 42,
    ):
        super().__init__()
        random.seed(seed)

        self.tokenizer      = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_step_length = max_step_length
        self.max_traj_length = max_traj_length

        parser = ToolBenchParser(select_method=select_method)
        filt   = ToolBenchFilter(min_steps=min_steps)

        # 加载并解析
        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # ToolBench 可能是 list 或 dict
        if isinstance(raw, dict):
            raw = list(raw.values())

        self.data: List[Dict] = []   # {"steps": [...], "query": "..."}

        for inst in raw:
            traj_list = parser.parse(inst)
            for steps in traj_list:
                steps_trunc = steps[:max_traj_length]
                if filt.is_valid(inst, steps_trunc):
                    self.data.append({
                        "steps": steps_trunc,
                        "query": inst.get("query", ""),
                    })

        # 可选：限制数量（避免良性样本过多导致不平衡）
        if max_instances and len(self.data) > max_instances:
            random.shuffle(self.data)
            self.data = self.data[:max_instances]

        print(f"[ToolBenchDataset] Loaded {len(self.data)} valid benign trajectories")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item  = self.data[idx]
        steps = item["steps"]

        traj = tokenize_trajectory(
            steps, self.tokenizer, self.max_step_length
        )

        return {
            "pos_input_ids":      traj["input_ids"],        # [T, L]
            "pos_attention_mask": traj["attention_mask"],   # [T, L]
            # 良性样本无负样本（训练时跳过对比损失）
            "neg_input_ids":      torch.zeros_like(traj["input_ids"]),
            "neg_attention_mask": torch.zeros_like(traj["attention_mask"]),
            # 标签
            "category_label":  torch.tensor(BENIGN_LABEL, dtype=torch.long),
            "jailbreak_label": torch.tensor(BENIGN_LABEL, dtype=torch.long),
            "span":            torch.tensor(float(len(steps)), dtype=torch.float),
            "instance_id":     torch.tensor(idx, dtype=torch.long),
            "is_benign":       torch.tensor(1,   dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════
# AgentHazard Dataset（攻击轨迹，更新版）
# ════════════════════════════════════════════════════════════

class AgentHazardDataset(Dataset):
    """AgentHazard 攻击轨迹数据集（与 ToolBench 格式对齐）"""

    def __init__(
        self,
        data_path: str,
        tokenizer_name: str,
        max_step_length: int = 128,
        max_traj_length: int = 20,
        zeroshot_categories: Optional[List[str]] = None,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        super().__init__()
        random.seed(seed)

        self.tokenizer       = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_step_length = max_step_length
        self.max_traj_length = max_traj_length
        self.zeroshot_cats   = set(zeroshot_categories or [])

        with open(data_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        instances = self._normalize(raw)

        seen     = [i for i in instances if i["category"] not in self.zeroshot_cats]
        zeroshot = [i for i in instances if i["category"] in self.zeroshot_cats]

        random.shuffle(seen)
        val_size    = int(len(seen) * val_ratio)
        train_data  = seen[val_size:]
        val_data    = seen[:val_size]

        self.data = {
            "train":    train_data,
            "val":      val_data,
            "zeroshot": zeroshot,
        }[split]

        print(f"[AgentHazardDataset] split={split}, size={len(self.data)}")

    @staticmethod
    def _normalize(raw: List[Dict]) -> List[Dict]:
        result = []
        for item in raw:
            inst = {
                "id": item.get("id") or item.get("original_id", 0),
                "category": (
                    item.get("category") or
                    item.get("risk_category", "Unknown")
                ).replace(" ", "_"),
                "jailbreak_method": (
                    item.get("jailbreak_method") or
                    item.get("attack_strategy", "Unknown")
                ).replace(" ", "_"),
                "decomposed_query": (
                    item.get("decomposed_query") or
                    item.get("decomposition", [])
                ),
            }
            if inst["decomposed_query"]:
                result.append(inst)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        inst  = self.data[idx]
        steps = inst["decomposed_query"][:self.max_traj_length]

        traj = tokenize_trajectory(
            steps, self.tokenizer, self.max_step_length
        )

        cat_id = CATEGORY_TO_ID.get(inst["category"], -1)
        jb_id  = JAILBREAK_TO_ID.get(inst["jailbreak_method"], -1)

        return {
            "pos_input_ids":      traj["input_ids"],
            "pos_attention_mask": traj["attention_mask"],
            "neg_input_ids":      torch.zeros_like(traj["input_ids"]),
            "neg_attention_mask": torch.zeros_like(traj["attention_mask"]),
            "category_label":  torch.tensor(cat_id,        dtype=torch.long),
            "jailbreak_label": torch.tensor(jb_id,         dtype=torch.long),
            "span":            torch.tensor(float(len(steps)), dtype=torch.float),
            "instance_id":     torch.tensor(inst["id"],    dtype=torch.long),
            "is_benign":       torch.tensor(0,             dtype=torch.long),
        }


# ════════════════════════════════════════════════════════════
# Collate（处理变长轨迹，兼容两个数据集）
# ════════════════════════════════════════════════════════════

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    max_T = max(b["pos_input_ids"].size(0) for b in batch)
    L     = batch[0]["pos_input_ids"].size(1)

    def pad(t: torch.Tensor, target_T: int) -> torch.Tensor:
        T = t.size(0)
        if T < target_T:
            pad_block = torch.zeros(target_T - T, L, dtype=t.dtype)
            t = torch.cat([t, pad_block], dim=0)
        return t

    return {
        "pos_input_ids":      torch.stack([pad(b["pos_input_ids"],      max_T) for b in batch]),
        "pos_attention_mask": torch.stack([pad(b["pos_attention_mask"], max_T) for b in batch]),
        "category_label":     torch.stack([b["category_label"]  for b in batch]),
        "jailbreak_label":    torch.stack([b["jailbreak_label"] for b in batch]),
        "span":               torch.stack([b["span"]            for b in batch]),
        "is_benign":          torch.stack([b["is_benign"]       for b in batch]),
        "traj_lengths":       torch.tensor([b["pos_input_ids"].size(0) for b in batch]),
        "instance_id":        torch.stack([b["instance_id"]     for b in batch]),
    }


# ════════════════════════════════════════════════════════════
# 联合数据集构建工具
# ════════════════════════════════════════════════════════════

def build_joint_dataset(
    agenthazard_path: str,
    toolbench_path: str,
    tokenizer_name: str,
    zeroshot_categories: Optional[List[str]] = None,
    split: str = "train",
    balance_ratio: float = 1.0,    # ToolBench : AgentHazard 数量比
    **kwargs,
) -> ConcatDataset:
    """
    构建攻击+良性联合数据集。

    Args:
        balance_ratio: 良性样本数 = 攻击样本数 × balance_ratio
    """
    attack_set = AgentHazardDataset(
        agenthazard_path, tokenizer_name,
        zeroshot_categories=zeroshot_categories,
        split=split, **kwargs
    )
    max_benign = int(len(attack_set) * balance_ratio)

    benign_set = ToolBenchDataset(
        toolbench_path, tokenizer_name,
        max_instances=max_benign, **kwargs
    )

    print(f"\n[Joint Dataset] Attack={len(attack_set)}, Benign={len(benign_set)}, "
          f"Ratio={len(benign_set)/max(len(attack_set),1):.2f}")

    return ConcatDataset([attack_set, benign_set])