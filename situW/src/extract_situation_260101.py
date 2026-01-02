import sys
sys.path.append('/data3/KJE/code/SituW/situW')

import argparse
import copy
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from nltk.tokenize import sent_tokenize

from utils.utils import OpenAIModel
from utils.gpt_pricing import get_text_prices_per_1m

try:
    import tiktoken
except Exception:
    tiktoken = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _canon(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = _norm_ws(str(s))
    if s == "":
        return None
    return s.lower()


def _is_subspan(span: Optional[str], text: str) -> bool:
    if span is None:
        return False
    span = str(span)
    return span in text


def _extract_first_json(text: str) -> Optional[Dict[str, Any]]:
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


def split_to_clauses(premises_text: str) -> List[str]:
    text = (premises_text or "").strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: List[str] = []
    if len(lines) > 1:
        chunks = lines
    else:
        sents = sent_tokenize(text)
        if not sents:
            chunks = [text]
        else:
            chunks = []
            for s in sents:
                parts = [p.strip() for p in re.split(r"\s*;\s*", s) if p.strip()]
                chunks.extend(parts if parts else [s.strip()])
    out = []
    for c in chunks:
        c2 = c.strip()
        if c2.endswith("."):
            c2 = c2[:-1].strip()
        if c2:
            out.append(c2)
    return out


def _sig_rule(antecedent: List[Dict[str, Any]], consequent: List[Dict[str, Any]]) -> str:
    def _nf(f):
        return (
            _canon(f.get("subject")) or "",
            _canon(f.get("predicate")) or "",
            _canon(f.get("object")) or "",
            bool(f.get("polarity", True))
        )
    a = sorted([_nf(x) for x in antecedent if isinstance(x, dict)])
    c = sorted([_nf(x) for x in consequent if isinstance(x, dict)])
    return json.dumps({"a": a, "c": c}, sort_keys=True)


def _fact_key(s: str, p: str, o: Optional[str]) -> str:
    return json.dumps({"s": _canon(s) or "", "p": _canon(p) or "", "o": _canon(o) or ""}, sort_keys=True)


def _tokenize_words(s: str) -> List[str]:
    s = _canon(s) or ""
    toks = re.findall(r"[a-z0-9]+", s)
    return toks


def _overlap_score(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    ta = set(_tokenize_words(a))
    tb = set(_tokenize_words(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union > 0 else 0.0


class EventCentricSituationModelBuilder:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.prompt_dir = args.prompt_dir

        self.enable_validation = args.enable_validation
        self.batch_size = args.batch_size

        self.max_models_per_story = args.max_models_per_story
        self.max_models_kept_by_score = args.max_models_kept_by_score

        self.focus_decay = args.focus_decay
        self.focus_boost_protagonist = args.focus_boost_protagonist
        self.focus_boost_entity = args.focus_boost_entity
        self.retrieval_topk_entities = args.retrieval_topk_entities
        self.retrieval_topk_events = args.retrieval_topk_events

        self.w_time = args.w_time
        self.w_location = args.w_location
        self.w_cause = args.w_cause
        self.w_intention = args.w_intention
        self.w_protagonist = args.w_protagonist

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)

        _ensure_dir(self.save_path)

        self.prompt_extract = _read_text(os.path.join(self.prompt_dir, "extract_event_logic_clause.txt"))
        self.prompt_validate = _read_text(os.path.join(self.prompt_dir, "validate_event_logic_extraction.txt"))

        self.stats = {
            "total_api_time": 0.0,
            "batch_calls": 0,
            "single_calls": 0,
            "num_prompts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

    def _count_tokens(self, text):
        if text is None:
            return 0
        text = str(text)
        if tiktoken is None:
            v = len(text) // 4
            return v if v > 0 else 1
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def _update_batch_stats(self, prompts, outputs, elapsed):
        self.stats["total_api_time"] += float(elapsed)
        self.stats["batch_calls"] += 1
        self.stats["num_prompts"] += len(prompts)
        for p, o in zip(prompts, outputs):
            pt = self._count_tokens(p)
            ct = self._count_tokens(o)
            self.stats["prompt_tokens"] += pt
            self.stats["completion_tokens"] += ct
            self.stats["total_tokens"] += (pt + ct)

    def _update_single_stats(self, prompt, output, elapsed):
        self.stats["total_api_time"] += float(elapsed)
        self.stats["single_calls"] += 1
        self.stats["num_prompts"] += 1
        pt = self._count_tokens(prompt)
        ct = self._count_tokens(output)
        self.stats["prompt_tokens"] += pt
        self.stats["completion_tokens"] += ct
        self.stats["total_tokens"] += (pt + ct)

    def load_raw_dataset(self, split):
        if 'd' in self.dataset_name:
            with open(os.path.join(self.data_path, f'{self.dataset_name}_{split}.json')) as f:
                raw_dataset = json.load(f)
        elif self.dataset_name == 'LogiQA2.0':
            with open(os.path.join(self.data_path, f'logiqa_{split}.json')) as f:
                raw_dataset = json.load(f)
        else:
            with open(os.path.join(self.data_path, f'{self.dataset_name.lower()}_{split}.json')) as f:
                raw_dataset = json.load(f)
        return raw_dataset

    def _compose_extract_prompt(self, clause_text: str) -> str:
        return self.prompt_extract.replace("{{CLAUSE}}", clause_text)

    def _compose_validate_prompt(self, clause_text: str, extraction_json: Dict[str, Any]) -> str:
        return (
            self.prompt_validate
            .replace("{{CLAUSE}}", clause_text)
            .replace("{{EXTRACTION_JSON}}", json.dumps(extraction_json, ensure_ascii=False))
        )

    def _batch_generate(self, prompts: List[str]) -> List[str]:
        t0 = time.time()
        outs = self.openai_api.batch_generate(prompts)
        self._update_batch_stats(prompts, outs, time.time() - t0)
        return outs

    def _parse_outputs(self, outputs: List[str]) -> List[Optional[Dict[str, Any]]]:
        return [_extract_first_json(o) for o in outputs]

    def _local_clean_extraction(self, clause_text: str, j: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if j is None or not isinstance(j, dict):
            return None

        out = {
            "event": j.get("event", {}),
            "entities": j.get("entities", []),
            "alternatives": j.get("alternatives", []),
            "confidence": j.get("confidence", 0.6)
        }

        if not isinstance(out["event"], dict):
            out["event"] = {}
        if not isinstance(out["entities"], list):
            out["entities"] = []
        if not isinstance(out["alternatives"], list):
            out["alternatives"] = []

        ev = out["event"]
        ev_norm = {}
        for k in ["time", "location", "cause", "intention"]:
            v = ev.get(k, None)
            ev_norm[k] = _canon(v) if v not in [None, "None", "none", "null", "NULL"] else None
        prot = ev.get("protagonist", None)
        if prot is None:
            ev_norm["protagonist"] = []
        elif isinstance(prot, list):
            ev_norm["protagonist"] = [(_canon(x) or "") for x in prot if _canon(x)]
        else:
            p = _canon(prot)
            ev_norm["protagonist"] = [p] if p else []

        ev_norm["evidence"] = ev.get("evidence", {})
        if not isinstance(ev_norm["evidence"], dict):
            ev_norm["evidence"] = {}

        for ek, evs in list(ev_norm["evidence"].items()):
            if evs is None:
                ev_norm["evidence"][ek] = None
            else:
                s = str(evs)
                ev_norm["evidence"][ek] = s if _is_subspan(s, clause_text) else None

        out["event"] = ev_norm

        ents_clean = []
        for e in out["entities"]:
            if not isinstance(e, dict):
                continue
            surface = e.get("surface", None)
            canonical = e.get("canonical", None)
            evidence = e.get("evidence", None)
            kind = e.get("kind", "unknown")
            ce = {
                "surface": _norm_ws(surface) if surface else None,
                "canonical": _canon(canonical) or _canon(surface),
                "kind": kind if kind else "unknown",
                "evidence": str(evidence) if evidence and _is_subspan(str(evidence), clause_text) else None
            }
            if ce["canonical"]:
                ents_clean.append(ce)
        out["entities"] = ents_clean

        alts_clean: List[List[Dict[str, Any]]] = []
        for alt in out["alternatives"]:
            if not isinstance(alt, list):
                continue
            alt_items = []
            for a in alt:
                if not isinstance(a, dict):
                    continue
                if a.get("type") == "fact":
                    s = _canon(a.get("subject"))
                    p = _canon(a.get("predicate"))
                    o = _canon(a.get("object"))
                    if not s or not p:
                        continue
                    pol = bool(a.get("polarity", True))
                    evs = a.get("evidence", None)
                    evs = str(evs) if evs and _is_subspan(str(evs), clause_text) else None
                    alt_items.append({
                        "type": "fact",
                        "subject": s,
                        "predicate": p,
                        "object": o,
                        "polarity": pol,
                        "quantifier": a.get("quantifier", "unknown"),
                        "modality": a.get("modality", "unknown"),
                        "evidence": evs,
                        "confidence": float(a.get("confidence", out.get("confidence", 0.6)))
                    })
                elif a.get("type") == "rule":
                    ant = a.get("antecedent", [])
                    con = a.get("consequent", [])
                    if not isinstance(ant, list) or not isinstance(con, list):
                        continue
                    ant2 = []
                    con2 = []
                    ok = True
                    for f in ant:
                        if not isinstance(f, dict):
                            ok = False
                            break
                        s = _canon(f.get("subject"))
                        p = _canon(f.get("predicate"))
                        o = _canon(f.get("object"))
                        if not s or not p:
                            ok = False
                            break
                        ant2.append({"subject": s, "predicate": p, "object": o, "polarity": bool(f.get("polarity", True))})
                    for f in con:
                        if not isinstance(f, dict):
                            ok = False
                            break
                        s = _canon(f.get("subject"))
                        p = _canon(f.get("predicate"))
                        o = _canon(f.get("object"))
                        if not s or not p:
                            ok = False
                            break
                        con2.append({"subject": s, "predicate": p, "object": o, "polarity": bool(f.get("polarity", True))})
                    if not ok:
                        continue
                    evs = a.get("evidence", None)
                    evs = str(evs) if evs and _is_subspan(str(evs), clause_text) else None
                    alt_items.append({
                        "type": "rule",
                        "antecedent": ant2,
                        "consequent": con2,
                        "evidence": evs,
                        "confidence": float(a.get("confidence", out.get("confidence", 0.6)))
                    })
            alts_clean.append(alt_items)
        if not alts_clean:
            alts_clean = [[]]
        out["alternatives"] = alts_clean

        try:
            out["confidence"] = float(out.get("confidence", 0.6))
        except Exception:
            out["confidence"] = 0.6

        return out

    def _validate_batch_llm(self, clause_texts: List[str], cleaned_jsons: List[Optional[Dict[str, Any]]]) -> List[Optional[Dict[str, Any]]]:
        if not self.enable_validation:
            return cleaned_jsons
        prompts = []
        for ct, cj in zip(clause_texts, cleaned_jsons):
            if cj is None:
                cj = {"event": {"time": None, "location": None, "cause": None, "intention": None, "protagonist": [], "evidence": {}},
                      "entities": [], "alternatives": [[]], "confidence": 0.0}
            prompts.append(self._compose_validate_prompt(ct, cj))
        outs = self._batch_generate(prompts)
        parsed = self._parse_outputs(outs)
        out = []
        for ct, pj in zip(clause_texts, parsed):
            out.append(self._local_clean_extraction(ct, pj))
        return out

    def _make_empty_model(self) -> Dict[str, Any]:
        mid = str(uuid.uuid4())
        return {
            "model_id": mid,
            "score": 0.0,
            "entities": {},
            "facts": {},
            "rules": {},
            "events": [],
            "focus": {"entities": {}, "goals": [], "last_event_id": None},
            "indices": {"entity_to_events": {}, "time_to_events": {}, "location_to_events": {}},
            "graph": {"nodes": [], "edges": []},
            "trace": {"applied_clauses": 0, "contradictions": 0, "branches": 0}
        }

    def _graph_has_node(self, model: Dict[str, Any], nid: str) -> bool:
        return any(n.get("id") == nid for n in model["graph"]["nodes"])

    def _add_graph_node(self, model: Dict[str, Any], node: Dict[str, Any]) -> None:
        nid = node.get("id")
        if nid and not self._graph_has_node(model, nid):
            model["graph"]["nodes"].append(node)

    def _add_graph_edge(self, model: Dict[str, Any], edge: Dict[str, Any]) -> None:
        model["graph"]["edges"].append(edge)

    def _add_entity(self, model: Dict[str, Any], ent: Dict[str, Any], clause_id: str) -> str:
        canonical = _canon(ent.get("canonical") or ent.get("surface")) or "unknown"
        surface = _norm_ws(ent.get("surface") or "")
        kind = ent.get("kind", "unknown")
        evidence = ent.get("evidence", None)

        if canonical not in model["entities"]:
            eid = f"ent:{canonical}"
            model["entities"][canonical] = {
                "entity_id": eid,
                "canonical": canonical,
                "surface_forms": [surface] if surface else [],
                "kind": kind,
                "evidence": [evidence] if evidence else [],
                "sources": [clause_id]
            }
            self._add_graph_node(model, {"id": eid, "type": "entity", "canonical": canonical, "kind": kind})
        else:
            eid = model["entities"][canonical]["entity_id"]
            if surface and surface not in model["entities"][canonical]["surface_forms"]:
                model["entities"][canonical]["surface_forms"].append(surface)
                model["entities"][canonical]["surface_forms"].sort()
            if evidence and evidence not in model["entities"][canonical]["evidence"]:
                model["entities"][canonical]["evidence"].append(evidence)
            if clause_id not in model["entities"][canonical]["sources"]:
                model["entities"][canonical]["sources"].append(clause_id)
        return eid

    def _add_fact(self, model: Dict[str, Any], fact: Dict[str, Any], clause_id: str, base_conf: float) -> bool:
        s = _canon(fact.get("subject"))
        p = _canon(fact.get("predicate"))
        o = _canon(fact.get("object"))
        if not s or not p:
            return True
        pol = bool(fact.get("polarity", True))
        key = _fact_key(s, p, o)
        if key in model["facts"]:
            if bool(model["facts"][key]["polarity"]) != pol:
                model["trace"]["contradictions"] += 1
                return False
            evs = fact.get("evidence", None)
            if evs and evs not in model["facts"][key]["evidence"]:
                model["facts"][key]["evidence"].append(evs)
            if clause_id not in model["facts"][key]["sources"]:
                model["facts"][key]["sources"].append(clause_id)
            model["facts"][key]["confidence"] = max(float(model["facts"][key]["confidence"]), float(fact.get("confidence", base_conf)))
            return True

        fid = f"fact:{uuid.uuid4().hex}"
        model["facts"][key] = {
            "fact_id": fid,
            "subject": s,
            "predicate": p,
            "object": o,
            "polarity": pol,
            "quantifier": fact.get("quantifier", "unknown"),
            "modality": fact.get("modality", "unknown"),
            "evidence": [fact.get("evidence")] if fact.get("evidence") else [],
            "sources": [clause_id],
            "confidence": float(fact.get("confidence", base_conf))
        }
        self._add_graph_node(model, {"id": fid, "type": "fact", "subject": s, "predicate": p, "object": o, "polarity": pol})
        return True

    def _add_rule(self, model: Dict[str, Any], rule: Dict[str, Any], clause_id: str, base_conf: float) -> bool:
        ant = rule.get("antecedent", [])
        con = rule.get("consequent", [])
        if not isinstance(ant, list) or not isinstance(con, list):
            return True
        rsig = _sig_rule(ant, con)
        if rsig in model["rules"]:
            evs = rule.get("evidence", None)
            if evs and evs not in model["rules"][rsig]["evidence"]:
                model["rules"][rsig]["evidence"].append(evs)
            if clause_id not in model["rules"][rsig]["sources"]:
                model["rules"][rsig]["sources"].append(clause_id)
            model["rules"][rsig]["confidence"] = max(float(model["rules"][rsig]["confidence"]), float(rule.get("confidence", base_conf)))
            return True

        rid = f"rule:{uuid.uuid4().hex}"
        model["rules"][rsig] = {
            "rule_id": rid,
            "antecedent": ant,
            "consequent": con,
            "evidence": [rule.get("evidence")] if rule.get("evidence") else [],
            "sources": [clause_id],
            "confidence": float(rule.get("confidence", base_conf))
        }
        self._add_graph_node(model, {"id": rid, "type": "rule"})
        return True

    def _ground_rule_with_fact(self, rule_facts: List[Dict[str, Any]], fact: Dict[str, Any]) -> Optional[Dict[str, str]]:
        sub_map = {}
        fs = _canon(fact["subject"])
        fp = _canon(fact["predicate"])
        fo = _canon(fact.get("object"))
        fpol = bool(fact["polarity"])
        for rf in rule_facts:
            rs = _canon(rf.get("subject"))
            rp = _canon(rf.get("predicate"))
            ro = _canon(rf.get("object"))
            rpol = bool(rf.get("polarity", True))
            if rp != fp or ro != fo or rpol != fpol:
                continue
            if rs and len(rs) == 1 and rs.isalpha():
                sub_map[rs] = fs
                return sub_map
        return None

    def _apply_subst(self, f: Dict[str, Any], sub: Dict[str, str]) -> Dict[str, Any]:
        s = _canon(f.get("subject"))
        p = _canon(f.get("predicate"))
        o = _canon(f.get("object"))
        if s in sub:
            s = sub[s]
        if o in sub:
            o = sub[o]
        return {"subject": s, "predicate": p, "object": o, "polarity": bool(f.get("polarity", True))}

    def _forward_chain_limited(self, model: Dict[str, Any], clause_id: str) -> bool:
        changed = True
        it = 0
        while changed and it < 20:
            it += 1
            changed = False
            facts_list = list(model["facts"].values())
            for rsig, r in list(model["rules"].items()):
                ant = r.get("antecedent", [])
                con = r.get("consequent", [])
                if not ant or not con:
                    continue
                subs = []
                for fact in facts_list:
                    sub = self._ground_rule_with_fact(ant, fact)
                    if sub:
                        subs.append(sub)
                if not subs:
                    continue
                for sub in subs:
                    all_ok = True
                    for af in ant:
                        af2 = self._apply_subst(af, sub)
                        k = _fact_key(af2["subject"], af2["predicate"], af2.get("object"))
                        if k not in model["facts"] or bool(model["facts"][k]["polarity"]) != bool(af2["polarity"]):
                            all_ok = False
                            break
                    if not all_ok:
                        continue
                    for cf in con:
                        cf2 = self._apply_subst(cf, sub)
                        derived_fact = {
                            "subject": cf2["subject"],
                            "predicate": cf2["predicate"],
                            "object": cf2.get("object"),
                            "polarity": bool(cf2["polarity"]),
                            "quantifier": "unknown",
                            "modality": "necessary",
                            "evidence": None,
                            "confidence": float(r.get("confidence", 0.5))
                        }
                        ok = self._add_fact(model, derived_fact, clause_id, float(r.get("confidence", 0.5)))
                        if not ok:
                            return False
                        k = _fact_key(derived_fact["subject"], derived_fact["predicate"], derived_fact.get("object"))
                        if k in model["facts"] and clause_id in model["facts"][k]["sources"] and len(model["facts"][k]["sources"]) == 1:
                            changed = True
        return True

    def _update_focus(self, model: Dict[str, Any], protagonist_eids: List[str], mentioned_eids: List[str], intention: Optional[str]) -> None:
        fe = model["focus"]["entities"]
        for k in list(fe.keys()):
            fe[k] = float(fe[k]) * float(self.focus_decay)
            if fe[k] < 1e-4:
                del fe[k]
        for eid in mentioned_eids:
            fe[eid] = float(fe.get(eid, 0.0)) + float(self.focus_boost_entity)
        for eid in protagonist_eids:
            fe[eid] = float(fe.get(eid, 0.0)) + float(self.focus_boost_protagonist)
        if intention:
            goals = model["focus"]["goals"]
            if intention not in goals:
                goals.append(intention)
                if len(goals) > 20:
                    goals.pop(0)

    def _retrieve_relevant_events(self, model: Dict[str, Any], current_event: Dict[str, Any]) -> List[str]:
        fe = model["focus"]["entities"]
        top_entities = sorted(fe.items(), key=lambda x: x[1], reverse=True)[: int(self.retrieval_topk_entities)]
        candidate_event_ids = set()
        for eid, _a in top_entities:
            for ev_id in model["indices"]["entity_to_events"].get(eid, []):
                candidate_event_ids.add(ev_id)
        lt = model["focus"].get("last_event_id", None)
        if lt:
            candidate_event_ids.add(lt)
        cur_time = current_event.get("time", None)
        cur_loc = current_event.get("location", None)
        if cur_time:
            for ev_id in model["indices"]["time_to_events"].get(cur_time, []):
                candidate_event_ids.add(ev_id)
        if cur_loc:
            for ev_id in model["indices"]["location_to_events"].get(cur_loc, []):
                candidate_event_ids.add(ev_id)

        cand = list(candidate_event_ids)
        if not cand:
            return []
        recent_order = {e["event_id"]: e["order"] for e in model["events"]}
        cand.sort(key=lambda x: recent_order.get(x, -1), reverse=True)
        return cand[: int(self.retrieval_topk_events)]

    def _dimension_shared(self, ev_a: Dict[str, Any], ev_b: Dict[str, Any]) -> Dict[str, float]:
        shared = {}

        if ev_a.get("time") and ev_b.get("time") and ev_a["time"] == ev_b["time"]:
            shared["time"] = 1.0
        if ev_a.get("location") and ev_b.get("location") and ev_a["location"] == ev_b["location"]:
            shared["location"] = 1.0

        pa = set(ev_a.get("protagonist", []))
        pb = set(ev_b.get("protagonist", []))
        if pa and pb:
            inter = len(pa & pb)
            if inter > 0:
                shared["protagonist"] = inter / max(len(pa), len(pb))

        sc = _overlap_score(ev_a.get("cause"), ev_b.get("text"))
        if sc > 0.0:
            shared["cause"] = sc

        si = _overlap_score(ev_a.get("intention"), ev_b.get("text"))
        if si > 0.0:
            shared["intention"] = si

        return shared

    def _link_event_by_dimensions(self, model: Dict[str, Any], cur_ev: Dict[str, Any], prev_ev: Dict[str, Any]) -> float:
        shared = self._dimension_shared(cur_ev, prev_ev)
        if not shared:
            return 0.0

        added = 0.0
        cur_id = cur_ev["event_id"]
        prev_id = prev_ev["event_id"]

        if "time" in shared:
            self._add_graph_edge(model, {"type": "shared_time", "from": cur_id, "to": prev_id, "strength": shared["time"]})
            added += float(self.w_time) * float(shared["time"])
        if "location" in shared:
            self._add_graph_edge(model, {"type": "shared_location", "from": cur_id, "to": prev_id, "strength": shared["location"]})
            added += float(self.w_location) * float(shared["location"])
        if "protagonist" in shared:
            self._add_graph_edge(model, {"type": "shared_protagonist", "from": cur_id, "to": prev_id, "strength": shared["protagonist"]})
            added += float(self.w_protagonist) * float(shared["protagonist"])
        if "cause" in shared:
            self._add_graph_edge(model, {"type": "causal_relevance", "from": cur_id, "to": prev_id, "strength": shared["cause"]})
            added += float(self.w_cause) * float(shared["cause"])
        if "intention" in shared:
            self._add_graph_edge(model, {"type": "goal_relevance", "from": cur_id, "to": prev_id, "strength": shared["intention"]})
            added += float(self.w_intention) * float(shared["intention"])

        return added

    def _apply_current_model_to_integrated(
        self,
        base_model: Dict[str, Any],
        clause_id: str,
        clause_order: int,
        clause_text: str,
        cur_json: Dict[str, Any],
        alt_assertions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        model = copy.deepcopy(base_model)
        model["trace"]["applied_clauses"] += 1

        clause_node = f"clause:{clause_id}"
        self._add_graph_node(model, {"id": clause_node, "type": "clause", "order": clause_order})

        cur_event = cur_json.get("event", {})
        entities = cur_json.get("entities", [])
        base_conf = float(cur_json.get("confidence", 0.6))

        mentioned_eids = []
        for ent in entities:
            eid = self._add_entity(model, ent, clause_id)
            mentioned_eids.append(eid)

        prot_names = cur_event.get("protagonist", [])
        prot_eids = []
        for pn in prot_names:
            if pn:
                eid = self._add_entity(model, {"canonical": pn, "surface": pn, "kind": "unknown", "evidence": None}, clause_id)
                prot_eids.append(eid)
                if eid not in mentioned_eids:
                    mentioned_eids.append(eid)

        ev_id = f"ev:{uuid.uuid4().hex}"
        ev_node = {"id": ev_id, "type": "event", "clause_id": clause_id, "order": clause_order}
        self._add_graph_node(model, ev_node)
        self._add_graph_edge(model, {"type": "denoted_by", "from": ev_id, "to": clause_node})

        time_v = cur_event.get("time", None)
        loc_v = cur_event.get("location", None)
        cause_v = cur_event.get("cause", None)
        intent_v = cur_event.get("intention", None)

        if time_v:
            tn = f"dim:time:{time_v}"
            self._add_graph_node(model, {"id": tn, "type": "dim_time", "value": time_v})
            self._add_graph_edge(model, {"type": "at_time", "from": ev_id, "to": tn})
        if loc_v:
            ln = f"dim:location:{loc_v}"
            self._add_graph_node(model, {"id": ln, "type": "dim_location", "value": loc_v})
            self._add_graph_edge(model, {"type": "at_location", "from": ev_id, "to": ln})
        if cause_v:
            cn = f"dim:cause:{uuid.uuid4().hex}"
            self._add_graph_node(model, {"id": cn, "type": "dim_cause", "value": cause_v})
            self._add_graph_edge(model, {"type": "has_cause", "from": ev_id, "to": cn})
        if intent_v:
            inn = f"dim:intention:{uuid.uuid4().hex}"
            self._add_graph_node(model, {"id": inn, "type": "dim_intention", "value": intent_v})
            self._add_graph_edge(model, {"type": "has_intention", "from": ev_id, "to": inn})

        for pe in prot_eids:
            self._add_graph_edge(model, {"type": "has_protagonist", "from": ev_id, "to": pe})

        event_obj = {
            "event_id": ev_id,
            "clause_id": clause_id,
            "order": clause_order,
            "text": clause_text,
            "time": time_v,
            "location": loc_v,
            "cause": cause_v,
            "intention": intent_v,
            "protagonist": prot_names,
            "protagonist_entity_ids": prot_eids,
            "mentioned_entity_ids": mentioned_eids,
            "confidence": base_conf
        }

        relevant_ids = self._retrieve_relevant_events(model, event_obj)
        added_link_score = 0.0
        if relevant_ids:
            prev_map = {e["event_id"]: e for e in model["events"]}
            for rid in relevant_ids:
                if rid in prev_map:
                    added_link_score += self._link_event_by_dimensions(model, event_obj, prev_map[rid])

        model["events"].append(event_obj)
        model["focus"]["last_event_id"] = ev_id

        for eid in mentioned_eids:
            model["indices"]["entity_to_events"].setdefault(eid, []).append(ev_id)
        if time_v:
            model["indices"]["time_to_events"].setdefault(time_v, []).append(ev_id)
        if loc_v:
            model["indices"]["location_to_events"].setdefault(loc_v, []).append(ev_id)

        for a in alt_assertions:
            if not isinstance(a, dict):
                continue
            if a.get("type") == "fact":
                ok = self._add_fact(model, a, clause_id, base_conf)
                if not ok:
                    return None
                fid_key = _fact_key(a["subject"], a["predicate"], a.get("object"))
                if fid_key in model["facts"]:
                    self._add_graph_edge(model, {"type": "event_asserts_fact", "from": ev_id, "to": model["facts"][fid_key]["fact_id"]})
            elif a.get("type") == "rule":
                ok = self._add_rule(model, a, clause_id, base_conf)
                if not ok:
                    return None
                rsig = _sig_rule(a.get("antecedent", []), a.get("consequent", []))
                if rsig in model["rules"]:
                    self._add_graph_edge(model, {"type": "event_states_rule", "from": ev_id, "to": model["rules"][rsig]["rule_id"]})

        ok_chain = self._forward_chain_limited(model, clause_id)
        if not ok_chain:
            return None

        self._update_focus(model, prot_eids, mentioned_eids, intent_v)

        model["score"] += float(base_conf) + float(added_link_score)
        return model

    def _model_signature(self, model: Dict[str, Any]) -> str:
        fact_items = []
        for k, v in model["facts"].items():
            fact_items.append((k, bool(v["polarity"])))
        fact_items.sort()
        rule_sigs = sorted(list(model["rules"].keys()))
        event_dim = []
        for e in model["events"]:
            event_dim.append((_canon(e.get("time")) or "", _canon(e.get("location")) or "", tuple(sorted([_canon(x) or "" for x in e.get("protagonist", [])]))))
        return json.dumps({"facts": fact_items, "rules": rule_sigs, "ev": event_dim}, sort_keys=True)

    def _dedup_and_prune_models(self, models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = {}
        for m in models:
            sig = self._model_signature(m)
            if sig not in seen:
                seen[sig] = m
            else:
                if m["score"] > seen[sig]["score"]:
                    seen[sig] = m
        uniq = list(seen.values())
        uniq.sort(key=lambda x: x["score"], reverse=True)
        if self.max_models_kept_by_score and self.max_models_kept_by_score > 0:
            uniq = uniq[: self.max_models_kept_by_score]
        if self.max_models_per_story and self.max_models_per_story > 0:
            uniq = uniq[: self.max_models_per_story]
        return uniq

    def build_clause_items(self, raw_dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        clause_items: List[Dict[str, Any]] = []
        for item in raw_dataset:
            pid = item.get("story_id", None)
            if pid is None:
                pid = item.get("id", None)
            if pid is None:
                continue
            pid = str(pid)
            if pid in seen:
                continue
            seen.add(pid)

            premises_text = item.get("premises", item.get("premise", item.get("context", "")))
            clauses = split_to_clauses(premises_text)
            for i, c in enumerate(clauses):
                clause_id = f"{pid}:{i}"
                clause_items.append({"parent_id": pid, "clause_id": clause_id, "order": i, "text": c})
        clause_items.sort(key=lambda x: (x["parent_id"], x["order"]))
        return clause_items

    def run(self):
        raw_dataset = self.load_raw_dataset(self.split)
        clause_items = self.build_clause_items(raw_dataset)

        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        print(f"Built {len(clause_items)} clauses.")

        models_by_pid: Dict[str, List[Dict[str, Any]]] = {}
        current_by_pid: Dict[str, List[Dict[str, Any]]] = {}

        chunks = [clause_items[i:i + self.batch_size] for i in range(0, len(clause_items), self.batch_size)]
        for chunk in tqdm(chunks):
            clause_texts = [x["text"] for x in chunk]
            clause_ids = [x["clause_id"] for x in chunk]
            clause_orders = [x["order"] for x in chunk]
            pids = [x["parent_id"] for x in chunk]

            try:
                prompts = [self._compose_extract_prompt(ct) for ct in clause_texts]
                outs = self._batch_generate(prompts)
                parsed = self._parse_outputs(outs)

                cleaned_local = [self._local_clean_extraction(ct, pj) for ct, pj in zip(clause_texts, parsed)]
                cleaned = self._validate_batch_llm(clause_texts, cleaned_local)

                for pid, clause_id, order, ct, cj in zip(pids, clause_ids, clause_orders, clause_texts, cleaned):
                    if pid not in models_by_pid:
                        models_by_pid[pid] = [self._make_empty_model()]
                        current_by_pid[pid] = []

                    if cj is None:
                        current_by_pid[pid].append({"clause_id": clause_id, "order": order, "text": ct, "current_model": None})
                        continue

                    current_by_pid[pid].append({"clause_id": clause_id, "order": order, "text": ct, "current_model": cj})

                    alternatives = cj.get("alternatives", [[]])
                    if not isinstance(alternatives, list) or len(alternatives) == 0:
                        alternatives = [[]]

                    next_models = []
                    for m in models_by_pid[pid]:
                        for alt in alternatives:
                            nm = self._apply_current_model_to_integrated(m, clause_id, order, ct, cj, alt)
                            if nm is not None:
                                next_models.append(nm)

                    if len(alternatives) > 1:
                        for _m in next_models:
                            _m["trace"]["branches"] += 1

                    models_by_pid[pid] = self._dedup_and_prune_models(next_models)

            except Exception as e:
                print("Error in batch generation: ", e)

                # 배치 실패 시: 개별 처리로 폴백
                for pid, clause_id, order, ct in zip(pids, clause_ids, clause_orders, clause_texts):
                    try:
                        if pid not in models_by_pid:
                            models_by_pid[pid] = [self._make_empty_model()]
                            current_by_pid[pid] = []

                        prompt = self._compose_extract_prompt(ct)

                        # --- 단건 생성(fallback) ---
                        # 1) _generate 메서드가 있으면 사용
                        if hasattr(self, "_generate") and callable(getattr(self, "_generate")):
                            out = self._generate(prompt)
                        # 2) openai_api.generate가 있으면 사용 (첫 예시 스타일)
                        elif hasattr(self, "openai_api") and hasattr(self.openai_api, "generate"):
                            out, _ = self.openai_api.generate(prompt)
                        # 3) 그것도 없으면 batch_generate를 1개짜리로 호출
                        else:
                            out = self._batch_generate([prompt])[0]
                        # --------------------------

                        parsed_one = self._parse_outputs([out])[0]
                        cleaned_local_one = self._local_clean_extraction(ct, parsed_one)
                        cj = self._validate_batch_llm([ct], [cleaned_local_one])[0]

                        if cj is None:
                            current_by_pid[pid].append({"clause_id": clause_id, "order": order, "text": ct, "current_model": None})
                            continue

                        current_by_pid[pid].append({"clause_id": clause_id, "order": order, "text": ct, "current_model": cj})

                        alternatives = cj.get("alternatives", [[]])
                        if not isinstance(alternatives, list) or len(alternatives) == 0:
                            alternatives = [[]]

                        next_models = []
                        for m in models_by_pid[pid]:
                            for alt in alternatives:
                                nm = self._apply_current_model_to_integrated(m, clause_id, order, ct, cj, alt)
                                if nm is not None:
                                    next_models.append(nm)

                        if len(alternatives) > 1:
                            for _m in next_models:
                                _m["trace"]["branches"] += 1

                        models_by_pid[pid] = self._dedup_and_prune_models(next_models)

                    except Exception as inner_e:
                        print("Error in generating clause:", pid, clause_id, inner_e)

                        # 실패한 항목은 current_model=None으로 기록 (파이프라인 계속 진행)
                        if pid not in models_by_pid:
                            models_by_pid[pid] = [self._make_empty_model()]
                            current_by_pid[pid] = []
                        current_by_pid[pid].append({"clause_id": clause_id, "order": order, "text": ct, "current_model": None})

            # break

        final_outputs = []
        for pid, models in models_by_pid.items():
            complete_models = []
            for m in models:
                focus_sorted = sorted(m["focus"]["entities"].items(), key=lambda x: x[1], reverse=True)
                focus_snapshot = [{"entity_id": k, "activation": v} for k, v in focus_sorted[:25]]
                complete_models.append({
                    "model_id": m["model_id"],
                    "score": m["score"],
                    "entities": list(m["entities"].values()),
                    "facts": list(m["facts"].values()),
                    "rules": list(m["rules"].values()),
                    "events": m["events"],
                    "focus_final": {"entities": focus_snapshot, "goals": m["focus"]["goals"], "last_event_id": m["focus"]["last_event_id"]},
                    "graph": m["graph"],
                    "trace": m["trace"]
                })

            final_outputs.append({
                "id": pid,
                "current_models": current_by_pid.get(pid, []),
                "complete_models": complete_models
            })

        n_prompts = self.stats["num_prompts"] if self.stats["num_prompts"] > 0 else 1
        avg_time = self.stats["total_api_time"] / n_prompts
        avg_total_tokens = self.stats["total_tokens"] / n_prompts
        avg_prompt_tokens = self.stats["prompt_tokens"] / n_prompts
        avg_completion_tokens = self.stats["completion_tokens"] / n_prompts

        price = get_text_prices_per_1m(self.model_name, tier="standard")
        if price["input_per_1m"] is None or price["output_per_1m"] is None:
            input_cost = None
            output_cost = None
            total_cost = None
        else:
            input_cost = (self.stats["prompt_tokens"] / 1000000.0) * float(price["input_per_1m"])
            output_cost = (self.stats["completion_tokens"] / 1000000.0) * float(price["output_per_1m"])
            total_cost = input_cost + output_cost

        meta = {
            "avg_time_per_prompt_sec": avg_time,
            "avg_total_tokens_per_prompt": avg_total_tokens,
            "avg_prompt_tokens_per_prompt": avg_prompt_tokens,
            "avg_completion_tokens_per_prompt": avg_completion_tokens,
            "total_api_time_sec": self.stats["total_api_time"],
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
            "enable_validation": self.enable_validation,
            "focus_decay": self.focus_decay,
            "retrieval_topk_entities": self.retrieval_topk_entities,
            "retrieval_topk_events": self.retrieval_topk_events,
            "weights": {
                "time": self.w_time,
                "location": self.w_location,
                "cause": self.w_cause,
                "intention": self.w_intention,
                "protagonist": self.w_protagonist
            }
        }

        final_outputs.append({"id": "__meta__", "meta": meta})

        save_file = os.path.join(
            self.save_path,
            f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_event_situation_models.json'
        )
        with open(save_file, 'w', encoding="utf-8") as f:
            json.dump(final_outputs, f, indent=2, ensure_ascii=False)

        print(f"Saved event-centric situation models to {save_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/LogiQA2.0/logiqa2nli/DATA/QA2NLI')
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='/data3/KJE/code/SituW/situW/output/situation_memory_260101')
    parser.add_argument('--api_key', type=str, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str, default='situ_model')
    parser.add_argument('--max_new_tokens', default=512, type=int)

    parser.add_argument('--prompt_dir', type=str, default='/data3/KJE/code/SituW/situW/utils/prompt')
    parser.add_argument('--enable_validation', action='store_true')
    parser.add_argument('--batch_size', type=int, default=30)

    parser.add_argument('--max_models_per_story', type=int, default=8)
    parser.add_argument('--max_models_kept_by_score', type=int, default=32)

    parser.add_argument('--focus_decay', type=float, default=0.90)
    parser.add_argument('--focus_boost_protagonist', type=float, default=1.0)
    parser.add_argument('--focus_boost_entity', type=float, default=0.5)
    parser.add_argument('--retrieval_topk_entities', type=int, default=5)
    parser.add_argument('--retrieval_topk_events', type=int, default=8)

    parser.add_argument('--w_time', type=float, default=1.0)
    parser.add_argument('--w_location', type=float, default=1.0)
    parser.add_argument('--w_cause', type=float, default=1.0)
    parser.add_argument('--w_intention', type=float, default=1.0)
    parser.add_argument('--w_protagonist', type=float, default=2.0)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    builder = EventCentricSituationModelBuilder(args)
    builder.run()
