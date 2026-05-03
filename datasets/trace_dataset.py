import json

import random

import re

import torch

from torch.utils.data import Dataset, DataLoader, ConcatDataset

from transformers import AutoTokenizer

from typing import List, Dict, Optional, Tuple, Any

from collections import defaultdict

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

BENIGN_LABEL = -1

class ToolBenchParser:

    TOOL_TEMPLATE   = "Call {name}({args}). Result: {response}"

    REASON_TEMPLATE = "Reasoning: {message}"

    TASK_TEMPLATE   = "Task: {message}"

    MAX_ARG_CHARS  = 120

    MAX_RESP_CHARS = 200

    MAX_MSG_CHARS  = 150

    def __init__(self, select_method: str = "best"):

        self.select_method = select_method

    def parse(self, instance: Dict) -> List[List[str]]:

        answers  = instance.get("answers", [])

        pref     = instance.get("preference", [])

        if not answers:

            return []

        if self.select_method == "best":

            idx = self._best_answer_idx(answers, pref)

            trajs = [self._extract_trajectory(answers[idx])]

        elif self.select_method == "first":

            trajs = [self._extract_trajectory(answers[0])]

        else:

            trajs = [self._extract_trajectory(a) for a in answers]

        return [t for t in trajs if t]

    @staticmethod

    def _best_answer_idx(answers: List[Dict], preference: List[int]) -> int:

        if preference and len(preference) == len(answers):

            return int(max(range(len(preference)), key=lambda i: preference[i]))

        successful = [

            (i, a) for i, a in enumerate(answers)

            if a.get("final_answer", "") and "give_up" not in a.get("final_answer", "")

        ]

        if successful:

            return min(successful, key=lambda x: x[1].get("total_steps", 999))[0]

        return 0

    def _extract_trajectory(self, answer: Dict) -> List[str]:

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

        if not message or role == "system":

            return None

        if role == "tool":

            return self._parse_tool_message(message)

        if role == "assistant":

            msg = self._truncate(message, self.MAX_MSG_CHARS)

            if self._is_error_only(msg):

                return None

            return self.REASON_TEMPLATE.format(message=msg)

        if role == "user":

            msg = self._truncate(message, self.MAX_MSG_CHARS)

            return self.TASK_TEMPLATE.format(message=msg)

        return None

    def _parse_tool_message(self, message: str) -> Optional[str]:

        try:

            data = eval(message) if isinstance(message, str) else message

        except Exception:

            data = self._regex_parse_tool(message)

        if not data:

            return None

        name = data.get("name", "unknown_tool")

        if name == "Finish":

            return None

        args_raw = data.get("arguments", "{}")

        args_str = self._summarize_args(args_raw)

        resp_raw = data.get("response", "")

        resp_str = self._summarize_response(resp_raw)

        return self.TOOL_TEMPLATE.format(

            name=name,

            args=args_str,

            response=resp_str,

        )

    @staticmethod

    def _regex_parse_tool(message: str) -> Dict:

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

        if isinstance(args_raw, str):

            try:

                args_dict = json.loads(args_raw)

            except Exception:

                return self._truncate(str(args_raw), self.MAX_ARG_CHARS)

        else:

            args_dict = args_raw

        if not args_dict:

            return "()"

        parts = []

        for k, v in args_dict.items():

            v_str = self._truncate(str(v), 40)

            parts.append(f"{k}={v_str}")

        return ", ".join(parts)[:self.MAX_ARG_CHARS]

    def _summarize_response(self, resp_raw: Any) -> str:

        resp_str = str(resp_raw) if not isinstance(resp_raw, str) else resp_raw

        if self._is_error_response(resp_str):

            return "[API Error]"

        return self._truncate(resp_str, self.MAX_RESP_CHARS)

    @staticmethod

    def _truncate(text: str, max_chars: int) -> str:

        text = text.strip()

        return text if len(text) <= max_chars else text[:max_chars] + "..."

    @staticmethod

    def _is_error_only(text: str) -> bool:

        error_patterns = [

            r"does not match '\^\[a-zA-Z",

            r"give_up_and_restart",

        ]

        return any(re.search(p, text) for p in error_patterns)

    @staticmethod

    def _is_error_response(resp: str) -> bool:

        return (

            '"error": "Message error' in resp or

            "<!DOCTYPE" in resp or

            '"error": "Unauthorized"' in resp

        )

class ToolBenchFilter:

    def __init__(

        self,

        min_steps: int = 2,

        max_steps: int = 20,

        min_tool_calls: int = 1,

        allow_failed: bool = False,

    ):

        self.min_steps     = min_steps

        self.max_steps     = max_steps

        self.min_tool_calls = min_tool_calls

        self.allow_failed  = allow_failed

    def is_valid(self, instance: Dict, steps: List[str]) -> bool:

        if len(steps) < self.min_steps or len(steps) > self.max_steps:

            return False

        tool_calls = [s for s in steps if s.startswith("Call ")]

        if len(tool_calls) < self.min_tool_calls:

            return False

        error_steps = [s for s in steps if "[API Error]" in s]

        if len(error_steps) == len(tool_calls):

            return False

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

class ToolBenchDataset(Dataset):

    def __init__(

        self,

        data_path: str,

        tokenizer_name: str,

        max_step_length: int = 128,

        max_traj_length: int = 20,

        select_method: str = "best",

        min_steps: int = 2,

        max_instances: Optional[int] = None,

        seed: int = 42,

    ):

        super().__init__()

        random.seed(seed)

        self.tokenizer      = AutoTokenizer.from_pretrained(tokenizer_name)

        self.max_step_length = max_step_length

        self.max_traj_length = max_traj_length

        parser = ToolBenchParser(select_method=select_method)

        filt   = ToolBenchFilter(min_steps=min_steps)

        with open(data_path, "r", encoding="utf-8") as f:

            raw = json.load(f)

        if isinstance(raw, dict):

            raw = list(raw.values())

        self.data: List[Dict] = []

        for inst in raw:

            traj_list = parser.parse(inst)

            for steps in traj_list:

                steps_trunc = steps[:max_traj_length]

                if filt.is_valid(inst, steps_trunc):

                    self.data.append({

                        "steps": steps_trunc,

                        "query": inst.get("query", ""),

                    })

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

            "pos_input_ids":      traj["input_ids"],

            "pos_attention_mask": traj["attention_mask"],

            "neg_input_ids":      torch.zeros_like(traj["input_ids"]),

            "neg_attention_mask": torch.zeros_like(traj["attention_mask"]),

            "category_label":  torch.tensor(BENIGN_LABEL, dtype=torch.long),

            "jailbreak_label": torch.tensor(BENIGN_LABEL, dtype=torch.long),

            "span":            torch.tensor(float(len(steps)), dtype=torch.float),

            "instance_id":     torch.tensor(idx, dtype=torch.long),

            "is_benign":       torch.tensor(1,   dtype=torch.long),

        }

class AgentHazardDataset(Dataset):

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

def build_joint_dataset(

    agenthazard_path: str,

    toolbench_path: str,

    tokenizer_name: str,

    zeroshot_categories: Optional[List[str]] = None,

    split: str = "train",

    balance_ratio: float = 1.0,

    **kwargs,

) -> ConcatDataset:

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
