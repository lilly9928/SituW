import sys
sys.path.append('/data3/KJE/code/SituW/situW')

import argparse
import glob
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from utils.utils import OpenAIModel
from utils.gpt_pricing import get_text_prices_per_1m

try:
    import tiktoken
except Exception:
    tiktoken = None


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _canon_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = _norm_ws(str(s))
    if s == "":
        return None
    return s.lower()


def _strip_punct(s: str) -> str:
    return re.sub(r"[^\w\s\-]", "", s)


def norm_entity(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = _canon_text(s)
    if s is None:
        return None
    s = _strip_punct(s)
    s = _norm_ws(s)
    s = re.sub(r"^(the|a|an)\s+", "", s)
    return s if s else None


def norm_predicate(p: Optional[str]) -> Optional[str]:
    if p is None:
        return None
    p = _canon_text(p)
    if p is None:
        return None
    p = _strip_punct(p)
    p = _norm_ws(p)
    lemma_map = {
        "has": "have",
        "have": "have",
        "is": "be",
        "are": "be",
        "was": "be",
        "were": "be",
        "wants": "want",
        "want": "want",
        "doesnt": "do_not",
        "doesn't": "do_not",
        "dont": "do_not",
        "don't": "do_not"
    }
    return lemma_map.get(p, p)


def norm_object(o: Optional[str]) -> Optional[str]:
    if o is None:
        return None
    o = _canon_text(o)
    if o is None:
        return None
    o = _strip_punct(o)
    o = _norm_ws(o)
    o = re.sub(r"^(the|a|an)\s+", "", o)
    return o if o else None


def fact_key(s: str, p: str, o: Optional[str]) -> str:
    return json.dumps({"s": norm_entity(s) or "", "p": norm_predicate(p) or "", "o": norm_object(o) or ""}, sort_keys=True)


def is_variable(term: Optional[str]) -> bool:
    if term is None:
        return False
    t = norm_entity(term)
    if t is None:
        return False
    if len(t) == 1 and t.isalpha():
        return True
    if t in {"x", "y", "z", "someone", "somebody", "anyone", "person"}:
        return True
    return False


def load_stage1_models(stage1_path: str) -> Dict[str, Dict[str, Any]]:
    with open(stage1_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("id", ""))
        if pid == "__meta__":
            continue
        if pid:
            out[pid] = item
    return out


def load_raw_dataset(args) -> List[Dict[str, Any]]:
    # raw_data_file이 주어지면 그 파일을 그대로 로드
    if getattr(args, "raw_data_file", None):
        p = args.raw_data_file
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    # (호환) 기존 로직
    if 'd' in args.dataset_name:
        p = os.path.join(args.data_path, f'{args.dataset_name}_{args.split}.json')
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    if args.dataset_name == 'logiqa':
        p = os.path.join(args.data_path, f'{args.split}_new2.json')
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    p = os.path.join(args.data_path, f'{args.dataset_name.lower()}_{args.split}.json')
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_pid_and_conclusion(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    pid = item.get("story_id", None)
    if pid is None:
        pid = item.get("id", None)
    if pid is None:
        return None, None
    pid = str(pid)

    conclusion = None
    for k in ["conclusion", "hypothesis", "question", "query"]:
        if k in item and isinstance(item[k], str) and item[k].strip():
            conclusion = item[k].strip()
            break
    if conclusion is None:
        if "label" in item and "premises" in item and "conclusion" in item:
            conclusion = item.get("conclusion")
    return pid, conclusion


def split_top_level_or(text: str) -> List[str]:
    t = _norm_ws(text)
    t_low = t.lower()
    if "either " in t_low and " or " in t_low:
        m = re.match(r"(?i)^\s*either\s+(.*?)\s+or\s+(.*?)\s*$", t, flags=re.IGNORECASE)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
    parts = re.split(r"(?i)\s+or\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if len(parts) > 1 else [t]


def split_top_level_and(text: str) -> List[str]:
    t = _norm_ws(text)
    parts = re.split(r"(?i)\s+and\s+", t)
    parts = [p.strip() for p in parts if p.strip()]
    return parts if parts else [t]


def parse_literal_rule_based(s: str) -> Optional[Dict[str, Any]]:
    x = _norm_ws(s)
    x = re.sub(r"\.$", "", x).strip()

    m = re.match(r"(?i)^(.*?)\s+does(?:\s+)?n't\s+have\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "have", "object": obj, "polarity": False}
        return None

    m = re.match(r"(?i)^(.*?)\s+do(?:\s+)?n't\s+have\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "have", "object": obj, "polarity": False}
        return None

    m = re.match(r"(?i)^(.*?)\s+has\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "have", "object": obj, "polarity": True}
        return None

    m = re.match(r"(?i)^(.*?)\s+have\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "have", "object": obj, "polarity": True}
        return None

    m = re.match(r"(?i)^(.*?)\s+is\s+aware\s+that\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "aware", "object": obj, "polarity": True}
        return None

    m = re.match(r"(?i)^(.*?)\s+is\s+unaware\s+that\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "aware", "object": obj, "polarity": False}
        return None

    m = re.match(r"(?i)^(.*?)\s+is\s+not\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "be", "object": obj, "polarity": False}
        return None

    m = re.match(r"(?i)^(.*?)\s+is\s+(.*?)\s*$", x)
    if m:
        subj = norm_entity(m.group(1))
        obj = norm_object(m.group(2))
        if subj and obj:
            return {"type": "fact", "subject": subj, "predicate": "be", "object": obj, "polarity": True}
        return None

    return None


def parse_conclusion_rule_based(conclusion: str) -> Optional[List[List[Dict[str, Any]]]]:
    ors = split_top_level_or(conclusion)
    alternatives = []
    for part in ors:
        ands = split_top_level_and(part)
        lits = []
        ok = True
        for a in ands:
            lit = parse_literal_rule_based(a)
            if lit is None:
                ok = False
                break
            lits.append(lit)
        if ok and lits:
            alternatives.append(lits)
    if not alternatives:
        return None
    return alternatives


def first_json(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    s = str(text).strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    j = m.group(0)
    try:
        return json.loads(j)
    except Exception:
        j2 = re.sub(r"```json|```", "", j).strip()
        try:
            return json.loads(j2)
        except Exception:
            return None


def normalize_query_alternatives(alts: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
    out = []
    for conj in alts:
        cc = []
        for lit in conj:
            if lit.get("type") != "fact":
                continue
            s = norm_entity(lit.get("subject"))
            p = norm_predicate(lit.get("predicate"))
            o = norm_object(lit.get("object"))
            pol = bool(lit.get("polarity", True))
            if s and p:
                cc.append({"type": "fact", "subject": s, "predicate": p, "object": o, "polarity": pol})
        if cc:
            out.append(cc)
    return out


def build_fact_map(model: Dict[str, Any]) -> Dict[str, bool]:
    facts = {}
    for f in model.get("facts", []):
        s = f.get("subject")
        p = f.get("predicate")
        o = f.get("object")
        if s is None or p is None:
            continue
        k = fact_key(s, p, o)
        facts[k] = bool(f.get("polarity", True))
    return facts


def build_rule_list(model: Dict[str, Any]) -> List[Dict[str, Any]]:
    rules = []
    for r in model.get("rules", []):
        ant = r.get("antecedent", [])
        con = r.get("consequent", [])
        if not isinstance(ant, list) or not isinstance(con, list) or not ant or not con:
            continue
        ant2 = []
        con2 = []
        ok = True
        for a in ant:
            s = a.get("subject")
            p = a.get("predicate")
            o = a.get("object")
            if s is None or p is None:
                ok = False
                break
            ant2.append({
                "subject": norm_entity(s) or s,
                "predicate": norm_predicate(p) or p,
                "object": norm_object(o),
                "polarity": bool(a.get("polarity", True))
            })
        for c in con:
            s = c.get("subject")
            p = c.get("predicate")
            o = c.get("object")
            if s is None or p is None:
                ok = False
                break
            con2.append({
                "subject": norm_entity(s) or s,
                "predicate": norm_predicate(p) or p,
                "object": norm_object(o),
                "polarity": bool(c.get("polarity", True))
            })
        if ok:
            rules.append({"antecedent": ant2, "consequent": con2})
    return rules


def extract_constants_from_facts(facts: Dict[str, bool]) -> List[str]:
    consts = set()
    for k in facts.keys():
        obj = json.loads(k)
        s = obj.get("s", "")
        o = obj.get("o", "")
        if s:
            consts.add(s)
        if o and not is_variable(o):
            consts.add(o)
    return sorted(list(consts))


def forward_chain(
    facts: Dict[str, bool],
    rules: List[Dict[str, Any]],
    max_iter: int,
    seed_constants: Optional[List[str]] = None,
    max_constants: int = 50
) -> Dict[str, bool]:
    consts = seed_constants if seed_constants else extract_constants_from_facts(facts)
    consts = consts[:max_constants]

    changed = True
    it = 0
    while changed and it < max_iter:
        it += 1
        changed = False

        for r in rules:
            ant = r["antecedent"]
            con = r["consequent"]

            vars_in_rule = set()
            for a in ant:
                if is_variable(a["subject"]):
                    vars_in_rule.add(a["subject"])
                if is_variable(a.get("object")):
                    vars_in_rule.add(a["object"])
            for c in con:
                if is_variable(c["subject"]):
                    vars_in_rule.add(c["subject"])
                if is_variable(c.get("object")):
                    vars_in_rule.add(c.get("object"))

            if not vars_in_rule:
                all_ok = True
                for a in ant:
                    k = fact_key(a["subject"], a["predicate"], a.get("object"))
                    if k not in facts or facts[k] != bool(a.get("polarity", True)):
                        all_ok = False
                        break
                if not all_ok:
                    continue
                for c in con:
                    ck = fact_key(c["subject"], c["predicate"], c.get("object"))
                    if ck in facts:
                        continue
                    facts[ck] = bool(c.get("polarity", True))
                    changed = True
                continue

            if len(vars_in_rule) > 1:
                continue

            var = list(vars_in_rule)[0]
            for const in consts:
                all_ok = True
                for a in ant:
                    aa = dict(a)
                    if aa["subject"] == var:
                        aa["subject"] = const
                    if aa.get("object") == var:
                        aa["object"] = const
                    k = fact_key(aa["subject"], aa["predicate"], aa.get("object"))
                    if k not in facts or facts[k] != bool(aa.get("polarity", True)):
                        all_ok = False
                        break
                if not all_ok:
                    continue
                for c in con:
                    cc = dict(c)
                    if cc["subject"] == var:
                        cc["subject"] = const
                    if cc.get("object") == var:
                        cc["object"] = const
                    ck = fact_key(cc["subject"], cc["predicate"], cc.get("object"))
                    if ck in facts:
                        continue
                    facts[ck] = bool(cc.get("polarity", True))
                    changed = True

    return facts


def eval_literal(facts: Dict[str, bool], lit: Dict[str, Any]) -> str:
    k = fact_key(lit["subject"], lit["predicate"], lit.get("object"))
    if k not in facts:
        return "unknown"
    want = bool(lit.get("polarity", True))
    have = bool(facts[k])
    if have == want:
        return "true"
    return "false"


def eval_conjunction(facts: Dict[str, bool], conj: List[Dict[str, Any]]) -> str:
    any_unknown = False
    for lit in conj:
        r = eval_literal(facts, lit)
        if r == "false":
            return "false"
        if r == "unknown":
            any_unknown = True
    return "unknown" if any_unknown else "true"


def eval_disjunction(facts: Dict[str, bool], alts: List[List[Dict[str, Any]]]) -> str:
    any_unknown = False
    for conj in alts:
        r = eval_conjunction(facts, conj)
        if r == "true":
            return "true"
        if r == "unknown":
            any_unknown = True
    return "unknown" if any_unknown else "false"


def aggregate_world_results(world_results: List[str]) -> str:
    if all(r == "true" for r in world_results):
        return "entailed"
    if all(r == "false" for r in world_results):
        return "contradicted"
    return "unknown"


def seeds_from_query(alts: List[List[Dict[str, Any]]]) -> List[str]:
    s = set()
    for conj in alts:
        for lit in conj:
            ss = norm_entity(lit.get("subject"))
            oo = norm_object(lit.get("object"))
            if ss:
                s.add(ss)
            if oo and not is_variable(oo):
                s.add(oo)
    return sorted(list(s))


def relevant_constants_from_situation_model(complete_model: Dict[str, Any], query_seeds: List[str], topk_focus: int) -> List[str]:
    consts = set(query_seeds)
    focus = complete_model.get("focus_final", {}).get("entities", [])
    for f in focus[:topk_focus]:
        eid = f.get("entity_id", "")
        if isinstance(eid, str) and eid.startswith("ent:"):
            consts.add(norm_entity(eid.replace("ent:", "")) or eid.replace("ent:", ""))
    return sorted([c for c in consts if c])


def _expand_stage1_files(stage1_files: List[str]) -> List[str]:
    expanded: List[str] = []
    for p in stage1_files:
        if not p:
            continue
        if os.path.isfile(p) and p.lower().endswith(".txt"):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    expanded.extend(glob.glob(line) or [line])
        else:
            expanded.extend(glob.glob(p) or [p])

    out = []
    seen = set()
    for p in expanded:
        if not p:
            continue
        ap = os.path.abspath(p)
        if ap in seen:
            continue
        seen.add(ap)
        out.append(ap)
    return out


# =========================
# 평가용 유틸 (Uncertain 처리 포함)
# =========================
def gold_label_to_tri(label: Any) -> Optional[str]:
    """
    원본 gold label -> {entailed, contradicted, unknown}
    FOLIO: True / False / Uncertain
    """
    if label is None:
        return None
    s = _canon_text(str(label))
    if s is None:
        return None
    if s == "true":
        return "entailed"
    if s == "false":
        return "contradicted"
    if s == "uncertain":
        return "unknown"
    return None


def gold_label_to_bool(label: Any) -> Optional[bool]:
    """
    원본 gold label -> bool (True/False만)
    Uncertain은 None
    """
    tri = gold_label_to_tri(label)
    if tri == "entailed":
        return True
    if tri == "contradicted":
        return False
    return None  # unknown(=Uncertain) 또는 기타


def pred_label_to_tri(final_label: Any) -> Optional[str]:
    """
    예측 final_label -> {entailed, contradicted, unknown}
    - 3-class: entailed/contradicted/unknown
    - binary: true/false 도 지원
    """
    if final_label is None:
        return None
    s = _canon_text(str(final_label))
    if s is None:
        return None

    if s in {"entailed"}:
        return "entailed"
    if s in {"contradicted"}:
        return "contradicted"
    if s in {"unknown"}:
        return "unknown"

    # binary 출력 호환
    if s == "true":
        return "entailed"
    if s == "false":
        return "contradicted"

    return None


def pred_label_to_bool(final_label: Any) -> Optional[bool]:
    """
    요청사항 반영:
    - contradicted => False
    - entailed => True
    - unknown => None
    - binary true/false도 지원
    """
    tri = pred_label_to_tri(final_label)
    if tri == "entailed":
        return True
    if tri == "contradicted":
        return False
    return None


def compute_accuracy(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    1) 3-class 정확도(entailed/contradicted/unknown) : Uncertain 포함
    2) 2-class 정확도(True/False) : Uncertain 제외 (gold_bool None 제외)
       - unknown 예측은 overall에선 오답 처리 + covered accuracy도 함께 제공
    """
    # --- 3-class ---
    tri_total = tri_correct = tri_covered = 0
    tri_pred_unknown_or_none = 0
    tri_conf = {
        "entailed": {"entailed": 0, "contradicted": 0, "unknown": 0},
        "contradicted": {"entailed": 0, "contradicted": 0, "unknown": 0},
        "unknown": {"entailed": 0, "contradicted": 0, "unknown": 0},
    }

    # --- 2-class (True/False only) ---
    n_total = n_correct = n_covered = 0
    tp = tn = fp = fn = 0
    n_pred_unknown = 0

    for r in all_results:
        if r.get("id") == "__meta__":
            continue

        gold_tri = r.get("gold_tri", None)
        pred_tri = r.get("pred_tri", None)

        # 3-class 집계
        if gold_tri is not None:
            tri_total += 1
            if pred_tri is None:
                tri_pred_unknown_or_none += 1
            else:
                tri_covered += 1
                if pred_tri == gold_tri:
                    tri_correct += 1
                if gold_tri in tri_conf and pred_tri in tri_conf[gold_tri]:
                    tri_conf[gold_tri][pred_tri] += 1

        # 2-class 집계
        gold = r.get("gold_bool", None)
        pred = r.get("pred_bool", None)

        if gold is None:
            continue  # Uncertain 제외
        n_total += 1

        if pred is None:
            n_pred_unknown += 1
            continue  # overall에서는 오답 처리
        n_covered += 1

        if pred == gold:
            n_correct += 1

        if gold is True and pred is True:
            tp += 1
        elif gold is False and pred is False:
            tn += 1
        elif gold is False and pred is True:
            fp += 1
        elif gold is True and pred is False:
            fn += 1

    tri_acc_overall = (tri_correct / tri_total) if tri_total > 0 else None
    tri_acc_on_covered = (tri_correct / tri_covered) if tri_covered > 0 else None
    tri_coverage = (tri_covered / tri_total) if tri_total > 0 else None

    acc_overall = (n_correct / n_total) if n_total > 0 else None
    acc_on_covered = (n_correct / n_covered) if n_covered > 0 else None
    coverage = (n_covered / n_total) if n_total > 0 else None

    return {
        # 3-class
        "tri_eval_n_total": tri_total,
        "tri_eval_n_covered": tri_covered,
        "tri_eval_n_correct": tri_correct,
        "tri_eval_coverage": tri_coverage,
        "tri_eval_accuracy_overall": tri_acc_overall,
        "tri_eval_accuracy_on_covered": tri_acc_on_covered,
        "tri_eval_pred_none_count": tri_pred_unknown_or_none,
        "tri_eval_confusion": tri_conf,

        # 2-class (Uncertain 제외)
        "bin_eval_n_total": n_total,
        "bin_eval_n_covered": n_covered,
        "bin_eval_n_correct": n_correct,
        "bin_eval_coverage": coverage,
        "bin_eval_accuracy_overall": acc_overall,
        "bin_eval_accuracy_on_covered": acc_on_covered,
        "bin_eval_pred_unknown_count": n_pred_unknown,
        "bin_eval_confusion_tp": tp,
        "bin_eval_confusion_tn": tn,
        "bin_eval_confusion_fp": fp,
        "bin_eval_confusion_fn": fn,
    }


class Stage2Reasoner:
    def __init__(self, args):
        self.args = args
        self.stage1_files = _expand_stage1_files(args.stage1_files) if args.stage1_files else []
        self.stage1_dir = args.stage1_dir
        self.stage1_glob = args.stage1_glob
        self.save_path = args.save_path

        self.use_llm_conclusion = args.use_llm_conclusion
        self.prompt_dir = args.prompt_dir
        self.batch_size = args.batch_size
        self.max_new_tokens = args.max_new_tokens

        self.max_chain_iter = args.max_chain_iter
        self.max_ground_constants = args.max_ground_constants
        self.use_relevance = args.use_relevance
        self.topk_focus = args.topk_focus

        self.binary = args.binary
        self.binary_positive = args.binary_positive

        self.openai_api = None
        self.prompt_parse_conc = None
        self.prompt_validate_conc = None

        self.stats = {
            "total_api_time": 0.0,
            "batch_calls": 0,
            "single_calls": 0,
            "num_prompts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "total_wall_time_sec": 0.0,
            "total_forward_chain_time_sec": 0.0,
            "total_eval_time_sec": 0.0
        }

        if self.use_llm_conclusion:
            self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, self.max_new_tokens)
            with open(os.path.join(self.prompt_dir, "parse_conclusion_query.txt"), "r", encoding="utf-8") as f:
                self.prompt_parse_conc = f.read()
            with open(os.path.join(self.prompt_dir, "validate_conclusion_query.txt"), "r", encoding="utf-8") as f:
                self.prompt_validate_conc = f.read()

        os.makedirs(self.save_path, exist_ok=True)

    def _count_tokens(self, text: Optional[str]) -> int:
        if text is None:
            return 0
        text = str(text)
        if tiktoken is None:
            v = len(text) // 4
            return v if v > 0 else 1
        try:
            enc = tiktoken.encoding_for_model(self.args.model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def _update_batch_stats(self, prompts: List[str], outputs: List[str], elapsed: float):
        self.stats["total_api_time"] += float(elapsed)
        self.stats["batch_calls"] += 1
        self.stats["num_prompts"] += len(prompts)
        for p, o in zip(prompts, outputs):
            pt = self._count_tokens(p)
            ct = self._count_tokens(o)
            self.stats["prompt_tokens"] += pt
            self.stats["completion_tokens"] += ct
            self.stats["total_tokens"] += (pt + ct)

    def _compose_parse_prompt(self, conc: str) -> str:
        return self.prompt_parse_conc.replace("{{CONCLUSION}}", conc)

    def _compose_validate_prompt(self, conc: str, j: Dict[str, Any]) -> str:
        return self.prompt_validate_conc.replace("{{CONCLUSION}}", conc).replace("{{JSON}}", json.dumps(j, ensure_ascii=False))

    def _batch_generate(self, prompts: List[str]) -> List[str]:
        t0 = time.time()
        outs = self.openai_api.batch_generate(prompts)
        self._update_batch_stats(prompts, outs, time.time() - t0)
        return outs

    def parse_conclusions_llm(self, conclusions: List[str]) -> List[Optional[List[List[Dict[str, Any]]]]]:
        prompts = [self._compose_parse_prompt(c) for c in conclusions]
        outs = self._batch_generate(prompts)
        parsed = [first_json(o) for o in outs]

        val_prompts = []
        for c, j in zip(conclusions, parsed):
            if j is None:
                j = {"alternatives": []}
            val_prompts.append(self._compose_validate_prompt(c, j))
        val_outs = self._batch_generate(val_prompts)
        val_parsed = [first_json(o) for o in val_outs]

        results = []
        for c, j in zip(conclusions, val_parsed):
            if j is None or not isinstance(j, dict):
                results.append(None)
                continue
            alts = j.get("alternatives", None)
            if not isinstance(alts, list) or not alts:
                results.append(None)
                continue
            alts2 = []
            for conj in alts:
                if not isinstance(conj, list):
                    continue
                cc = []
                for lit in conj:
                    if not isinstance(lit, dict):
                        continue
                    if lit.get("type") != "fact":
                        continue
                    s = norm_entity(lit.get("subject"))
                    p = norm_predicate(lit.get("predicate"))
                    o = norm_object(lit.get("object"))
                    pol = bool(lit.get("polarity", True))
                    if s and p:
                        cc.append({"type": "fact", "subject": s, "predicate": p, "object": o, "polarity": pol})
                if cc:
                    alts2.append(cc)
            results.append(alts2 if alts2 else None)
        return results

    def predict_for_one(self, pid: str, stage1_item: Dict[str, Any], conclusion: str, parsed_query: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        complete_models = stage1_item.get("complete_models", [])
        if not isinstance(complete_models, list) or not complete_models:
            return {
                "id": pid,
                "conclusion": conclusion,
                "final_label": "unknown",
                "per_model": []
            }

        query_seeds = seeds_from_query(parsed_query)

        per_model = []
        world_results = []

        for cm in complete_models:
            facts = build_fact_map(cm)
            rules = build_rule_list(cm)

            seed_constants = query_seeds
            if self.use_relevance:
                seed_constants = relevant_constants_from_situation_model(cm, query_seeds, self.topk_focus)

            t_fc = time.time()
            facts_closed = forward_chain(
                facts=facts,
                rules=rules,
                max_iter=self.max_chain_iter,
                seed_constants=seed_constants,
                max_constants=self.max_ground_constants
            )
            self.stats["total_forward_chain_time_sec"] += float(time.time() - t_fc)

            t_ev = time.time()
            tr = eval_disjunction(facts_closed, parsed_query)
            self.stats["total_eval_time_sec"] += float(time.time() - t_ev)

            world_results.append(tr)

            per_model.append({
                "model_id": cm.get("model_id"),
                "score": cm.get("score", 0.0),
                "truth": tr
            })

        final_3 = aggregate_world_results(world_results)

        if self.binary:
            if self.binary_positive == "entailed":
                final_bin = "true" if final_3 == "entailed" else "false"
            elif self.binary_positive == "not_contradicted":
                final_bin = "false" if final_3 == "contradicted" else "true"
            else:
                final_bin = "true" if final_3 == "entailed" else "false"
            final_label = final_bin
        else:
            final_label = final_3

        return {
            "id": pid,
            "conclusion": conclusion,
            "final_label": final_label,
            "per_model": per_model
        }

    def _finalize_meta(self) -> Dict[str, Any]:
        n_prompts = self.stats["num_prompts"] if self.stats["num_prompts"] > 0 else 1

        avg_api_time = self.stats["total_api_time"] / n_prompts
        avg_total_tokens = self.stats["total_tokens"] / n_prompts
        avg_prompt_tokens = self.stats["prompt_tokens"] / n_prompts
        avg_completion_tokens = self.stats["completion_tokens"] / n_prompts

        price = get_text_prices_per_1m(self.args.model_name, tier="standard")
        if price["input_per_1m"] is None or price["output_per_1m"] is None:
            input_cost = None
            output_cost = None
            total_cost = None
        else:
            input_cost = (self.stats["prompt_tokens"] / 1000000.0) * float(price["input_per_1m"])
            output_cost = (self.stats["completion_tokens"] / 1000000.0) * float(price["output_per_1m"])
            total_cost = input_cost + output_cost

        meta = {
            "use_llm_conclusion": bool(self.use_llm_conclusion),
            "avg_api_time_per_prompt_sec": avg_api_time,
            "avg_total_tokens_per_prompt": avg_total_tokens,
            "avg_prompt_tokens_per_prompt": avg_prompt_tokens,
            "avg_completion_tokens_per_prompt": avg_completion_tokens,
            "total_api_time_sec": self.stats["total_api_time"],
            "total_wall_time_sec": self.stats["total_wall_time_sec"],
            "total_forward_chain_time_sec": self.stats["total_forward_chain_time_sec"],
            "total_eval_time_sec": self.stats["total_eval_time_sec"],
            "total_prompt_tokens": self.stats["prompt_tokens"],
            "total_completion_tokens": self.stats["completion_tokens"],
            "total_tokens": self.stats["total_tokens"],
            "batch_calls": self.stats["batch_calls"],
            "single_calls": self.stats["single_calls"],
            "num_prompts": self.stats["num_prompts"],
            "pricing_tier": price["tier"],
            "pricing_matched_key": price["matched"],
            "price_input_per_1m": price["input_per_1m"],
            "price_output_per_1m": price["output_per_1m"],
            "estimated_input_cost_usd": input_cost,
            "estimated_output_cost_usd": output_cost,
            "estimated_total_cost_usd": total_cost,
            "stage1_files_count": len(self.stage1_files) if self.stage1_files else 0,
            "stage1_files": self.stage1_files[:50],
            "raw_data_file": getattr(self.args, "raw_data_file", None)
        }
        return meta

    def run(self):
        t_wall = time.time()

        if self.stage1_files:
            stage1_files = [p for p in self.stage1_files if os.path.isfile(p)]
        else:
            stage1_files = sorted(glob.glob(os.path.join(self.stage1_dir, self.stage1_glob)))

        if not stage1_files:
            raise RuntimeError("No stage1 files found. Provide --stage1_files or --stage1_dir/--stage1_glob")

        # 원본 데이터 로드 (conclusion + gold label)
        raw = load_raw_dataset(self.args)

        pid_to_conc: Dict[str, str] = {}
        pid_to_gold_str: Dict[str, Any] = {}
        pid_to_gold_tri: Dict[str, Optional[str]] = {}
        pid_to_gold_bool: Dict[str, Optional[bool]] = {}

        for it in raw:
            pid, conc = get_pid_and_conclusion(it)
            if pid and conc:
                pid_to_conc[pid] = conc

            if pid:
                g = it.get("label", None)
                pid_to_gold_str[pid] = g
                pid_to_gold_tri[pid] = gold_label_to_tri(g)     # True/False/Uncertain -> ent/cont/unk
                pid_to_gold_bool[pid] = gold_label_to_bool(g)   # Uncertain -> None

        all_results: List[Dict[str, Any]] = []

        for fpath in stage1_files:
            stage1_map = load_stage1_models(fpath)
            pids = [pid for pid in stage1_map.keys() if pid in pid_to_conc]

            conclusions = [pid_to_conc[pid] for pid in pids]
            parsed_queries: List[Optional[List[List[Dict[str, Any]]]]] = [None] * len(pids)

            for i, c in enumerate(conclusions):
                q = parse_conclusion_rule_based(c)
                if q is not None:
                    parsed_queries[i] = normalize_query_alternatives(q)

            if self.use_llm_conclusion:
                need_idx = [i for i, q in enumerate(parsed_queries) if q is None]
                if need_idx:
                    batch_concs = [conclusions[i] for i in need_idx]
                    llm_qs = []
                    for j in range(0, len(batch_concs), self.batch_size):
                        llm_qs.extend(self.parse_conclusions_llm(batch_concs[j:j+self.batch_size]))
                    for idx, qq in zip(need_idx, llm_qs):
                        parsed_queries[idx] = normalize_query_alternatives(qq) if qq else None

            for pid, conc, q in tqdm(list(zip(pids, conclusions, parsed_queries)), desc=os.path.basename(fpath)):
                gold_str = pid_to_gold_str.get(pid, None)
                gold_tri = pid_to_gold_tri.get(pid, None)
                gold_bool = pid_to_gold_bool.get(pid, None)

                if not conc or q is None or not q:
                    res = {
                        "id": pid,
                        "conclusion": conc,
                        "final_label": "unknown",
                        "per_model": [],
                        "error": "failed_to_parse_conclusion",
                        "stage1_file": os.path.basename(fpath),

                        # gold/pred 추가
                        "gold_label": gold_str,
                        "gold_tri": gold_tri,
                        "gold_bool": gold_bool,
                        "pred_tri": "unknown",
                        "pred_bool": None,
                        "is_correct_tri": (gold_tri == "unknown") if gold_tri is not None else None,
                        "is_correct": False if gold_bool is not None else None,
                    }
                    all_results.append(res)
                    continue

                res = self.predict_for_one(pid, stage1_map[pid], conc, q)
                res["stage1_file"] = os.path.basename(fpath)

                # gold/pred 추가
                res["gold_label"] = gold_str
                res["gold_tri"] = gold_tri
                res["gold_bool"] = gold_bool

                res["pred_tri"] = pred_label_to_tri(res.get("final_label"))
                res["pred_bool"] = pred_label_to_bool(res.get("final_label"))

                if gold_tri is None:
                    res["is_correct_tri"] = None
                else:
                    res["is_correct_tri"] = (res["pred_tri"] is not None) and (res["pred_tri"] == gold_tri)

                if gold_bool is None:
                    res["is_correct"] = None
                else:
                    res["is_correct"] = (res["pred_bool"] is not None) and (res["pred_bool"] == gold_bool)

                all_results.append(res)

        self.stats["total_wall_time_sec"] = float(time.time() - t_wall)

        # meta + accuracy 추가
        meta = self._finalize_meta()
        meta.update(compute_accuracy(all_results))

        all_results.append({"id": "__meta__", "meta": meta})

        out_file = os.path.join(self.save_path, f"stage2_predictions_{int(time.time())}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(out_file)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--stage1_files", type=str, nargs="*", default=None)

    p.add_argument("--stage1_dir", type=str, default="/data3/KJE/code/SituW/situW/output/situation_memory_260101")
    p.add_argument("--stage1_glob", type=str, default="*.json")
    p.add_argument("--save_path", type=str, default="/data3/KJE/code/SituW/situW/output/stage2_predictions")

    # (호환용) 기존 옵션 유지
    p.add_argument("--data_path", type=str, default="./data/LogiQA2.0/logiqa2nli/DATA/QA2NLI")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, required=True)

    # 원본 raw json 직접 지정 (기본값: 요청하신 경로)
    p.add_argument(
        "--raw_data_file",
        type=str,
        default="/data3/KJE/code/WIL_DeepLearningProject_2/SituationMemory/data/ThinkAgent/FOLIO/folio_train.json",
        help="원본 데이터(json) 경로. 주어지면 data_path/dataset_name/split 대신 이 파일을 사용"
    )

    p.add_argument("--use_llm_conclusion", action="store_true")
    p.add_argument("--prompt_dir", type=str, default="/data3/KJE/code/SituW/situW/utils/prompt")
    p.add_argument("--batch_size", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=256)

    p.add_argument("--api_key", type=str, default=None)
    p.add_argument("--model_name", type=str, default=None)
    p.add_argument("--stop_words", type=str, default="------")

    p.add_argument("--max_chain_iter", type=int, default=10)
    p.add_argument("--max_ground_constants", type=int, default=50)
    p.add_argument("--use_relevance", action="store_true")
    p.add_argument("--topk_focus", type=int, default=10)

    p.add_argument("--binary", action="store_true")
    p.add_argument("--binary_positive", type=str, default="entailed",
                  choices=["entailed", "not_contradicted"])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = Stage2Reasoner(args)
    runner.run()
