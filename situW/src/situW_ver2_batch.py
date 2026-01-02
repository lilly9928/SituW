import sys
sys.path.append('/data3/KJE/code/SituW/situW')

import json
import os
import re
import time
import argparse
import uuid
from tqdm import tqdm
from utils.utils import OpenAIModel

# optional imports for token + pricing
try:
    import tiktoken
except Exception:
    tiktoken = None

try:
    from utils.gpt_pricing import get_text_prices_per_1m
except Exception:
    get_text_prices_per_1m = None


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _extract_first_json(text):
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


def split_sentences(context: str):
    # (원하면 공백 없는 마침표 케이스까지 커버하는 split로 바꿀 수 있음)
    context = (context or "").strip()
    if not context:
        return []
    sents = re.split(r"(?<=[\.\?\!])\s+", context)
    return [_norm_ws(s) for s in sents if _norm_ws(s)]


def ordered_union(a, b):
    seen = set(a)
    out = list(a)
    for x in b:
        x = _norm_ws(str(x))
        if x and x not in seen:
            out.append(x)
            seen.add(x)
    return out


def fmt_list_py(l):
    return repr(l)


def _make_run_id():
    return time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]


def _unique_path(path: str) -> str:
    """If path exists, append _vN before extension."""
    if not os.path.exists(path):
        return path
    base, ext = os.path.splitext(path)
    for i in range(1, 10000):
        cand = f"{base}_v{i}{ext}"
        if not os.path.exists(cand):
            return cand
    # fallback
    return f"{base}_{uuid.uuid4().hex}{ext}"


class GPT3_Incremental_Memo_Distiller:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode

        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)

        self.memory_step_prompt = self._read_text(args.memory_step_prompt_path)
        self.final_reason_prompt = self._read_text(args.final_reason_prompt_path)

        os.makedirs(self.save_path, exist_ok=True)

        self.batch_size = args.batch_size
        self.max_steps = args.max_steps
        self.save_all = args.save_all

        # ----- run id & unique filenames -----
        self.run_id = args.run_id if args.run_id else _make_run_id()
        base_name = f"{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_{self.run_id}"

        self.distill_out_path = _unique_path(os.path.join(self.save_path, f"{base_name}_distill_correct.jsonl"))
        self.all_out_path = _unique_path(os.path.join(self.save_path, f"{base_name}_all.json"))
        self.meta_out_path = _unique_path(os.path.join(self.save_path, f"{base_name}_meta.json"))

        # ----- cost stats -----
        self.stats = {
            "run_id": self.run_id,
            "total_api_time_sec": 0.0,
            "batch_calls": 0,
            "num_prompts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "model_name": self.model_name,
            "pricing": None,
            "estimated_input_cost_usd": None,
            "estimated_output_cost_usd": None,
            "estimated_total_cost_usd": None,
        }

        self.pricing = None
        if get_text_prices_per_1m is not None:
            try:
                self.pricing = get_text_prices_per_1m(self.model_name, tier="standard")
                self.stats["pricing"] = self.pricing
            except Exception:
                self.pricing = None

    def _read_text(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _count_tokens(self, text: str) -> int:
        if text is None:
            return 0
        text = str(text)
        if tiktoken is None:
            # rough estimate
            v = len(text) // 4
            return v if v > 0 else 1
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))

    def _update_global_cost(self, prompts, outputs, elapsed_sec):
        self.stats["total_api_time_sec"] += float(elapsed_sec)
        self.stats["batch_calls"] += 1
        self.stats["num_prompts"] += len(prompts)

        for p, o in zip(prompts, outputs):
            pt = self._count_tokens(p)
            ct = self._count_tokens(o)
            self.stats["prompt_tokens"] += pt
            self.stats["completion_tokens"] += ct
            self.stats["total_tokens"] += (pt + ct)

    def _estimate_cost_usd(self, prompt_tokens: int, completion_tokens: int):
        if not self.pricing:
            return None, None, None
        inp = self.pricing.get("input_per_1m", None)
        out = self.pricing.get("output_per_1m", None)
        if inp is None or out is None:
            return None, None, None
        input_cost = (prompt_tokens / 1_000_000.0) * float(inp)
        output_cost = (completion_tokens / 1_000_000.0) * float(out)
        return input_cost, output_cost, input_cost + output_cost

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

    def _get_fields(self, sample):
        if self.dataset_name == "logiqa":
            context = sample.get("premise", "")
            question = (sample.get("hypothesis", "") or "").strip()
            gold = sample.get("label", None)
            options = sample.get("options", None)
            sid = sample.get("id", None)
            return sid, context, question, options, gold
        else:
            context = sample.get("premises", "")
            question = (sample.get("conclusion", "") or "").strip()
            gold = sample.get("label", None)
            options = ['true', 'false', 'uncertain']
            sid = sample.get("story_id", None)
            return sid, context, question, options, gold

    def _compose_step_prompt(self, sentence, mem_statements, mem_protagonists):
        p = self.memory_step_prompt
        p = p.replace("[[SENTENCE]]", sentence)
        p = p.replace("[[MEM_STATEMENTS]]", json.dumps(mem_statements, ensure_ascii=False))
        p = p.replace("[[MEM_PROTAGONISTS]]", json.dumps(mem_protagonists, ensure_ascii=False))
        return p

    def _compose_final_prompt(self, question, options, statements, reading):
        p = self.final_reason_prompt
        p = p.replace("[[QUESTION]]", question)
        p = p.replace("[[OPTIONS]]", json.dumps(options, ensure_ascii=False) if options else "none")
        p = p.replace("[[STATEMENTS]]", json.dumps(statements, ensure_ascii=False))
        p = p.replace("[[READING]]", reading)
        return p

    def _safe_parse_step(self, sentence, out_text):
        j = _extract_first_json(out_text)
        if not isinstance(j, dict):
            return {
                "time": ["none"],
                "space": ["none"],
                "intention": ["none"],
                "proposition": _norm_ws(sentence),
                "protagonists_in_sentence": [],
                "confidence": 0.0,
            }

        def _as_list(x):
            if x is None:
                return ["none"]
            if isinstance(x, list) and len(x) > 0:
                return [str(v) for v in x]
            if isinstance(x, str) and x.strip():
                return [x.strip()]
            return ["none"]

        time_v = _as_list(j.get("time"))
        space_v = _as_list(j.get("space"))
        intention_v = _as_list(j.get("intention"))

        prop = j.get("proposition", None)
        prop = _norm_ws(str(prop)) if prop else _norm_ws(sentence)

        prots = j.get("protagonists_in_sentence", [])
        if not isinstance(prots, list):
            prots = []
        prots = [str(x) for x in prots if _norm_ws(str(x))]

        conf = j.get("confidence", 0.0)
        try:
            conf = float(conf)
        except Exception:
            conf = 0.0

        return {
            "time": time_v,
            "space": space_v,
            "intention": intention_v,
            "proposition": prop,
            "protagonists_in_sentence": prots,
            "confidence": conf,
        }

    def post_process_final_answer(self, reasoning_text):
        m = re.search(r"Final Answer:\s*(.*)", str(reasoning_text), re.IGNORECASE)
        if m:
            return m.group(1).strip()
        return "No final answer found."

    def normalize_gold_pred(self, gold, pred):
        g = str(gold).strip().lower() if gold is not None else ""
        p = str(pred).strip().lower()
        mapping = {
            "true": "true",
            "false": "false",
            "uncertain": "uncertain",
            "unknown": "uncertain",
            "not sure": "uncertain",
        }
        g = mapping.get(g, g)
        p = mapping.get(p, p)
        return g, p

    def run(self):
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        print(f"run_id={self.run_id}")
        print(f"distill_out={self.distill_out_path}")
        print(f"all_out={self.all_out_path}")
        print(f"meta_out={self.meta_out_path}")

        outputs_all = []
        correct_written = 0

        distill_f = open(self.distill_out_path, "w", encoding="utf-8")

        chunks = [raw_dataset[i:i + self.batch_size] for i in range(0, len(raw_dataset), self.batch_size)]
        for chunk in tqdm(chunks):
            chunk_start = time.time()

            states = []
            for sample in chunk:
                sid, context, question, options, gold = self._get_fields(sample)
                sents = split_sentences(context)
                if self.max_steps and self.max_steps > 0:
                    sents = sents[: self.max_steps]

                states.append({
                    "sid": sid,
                    "sample": sample,
                    "context": context,
                    "question": question,
                    "options": options,
                    "gold": gold,
                    "sentences": sents,
                    "mem_statements": [],
                    "mem_protagonists": [],
                    "reading_lines": [],
                    "step_json": [],
                    # per-sample cost accumulator
                    "cost": {
                        "api_time_sec": 0.0,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "num_prompts": 0,
                        "estimated_input_cost_usd": None,
                        "estimated_output_cost_usd": None,
                        "estimated_total_cost_usd": None,
                    }
                })

            # -------- STEP extraction loop --------
            max_len = max((len(st["sentences"]) for st in states), default=0)
            for t in range(max_len):
                prompts = []
                idx_map = []
                for i, st in enumerate(states):
                    if t >= len(st["sentences"]):
                        continue
                    sent = st["sentences"][t]
                    prompt = self._compose_step_prompt(sent, st["mem_statements"], st["mem_protagonists"])
                    prompts.append(prompt)
                    idx_map.append(i)

                if not prompts:
                    break

                t0 = time.time()
                outs = self.openai_api.batch_generate(prompts)
                elapsed = time.time() - t0

                # global stats
                self._update_global_cost(prompts, outs, elapsed)

                # distribute time equally across prompts (simple attribution)
                per_prompt_time = elapsed / max(1, len(prompts))

                for prompt_text, out_text, i in zip(prompts, outs, idx_map):
                    st = states[i]
                    sent = st["sentences"][t]
                    step = self._safe_parse_step(sent, out_text)

                    # per-sample token stats
                    pt = self._count_tokens(prompt_text)
                    ct = self._count_tokens(out_text)
                    st["cost"]["api_time_sec"] += per_prompt_time
                    st["cost"]["prompt_tokens"] += pt
                    st["cost"]["completion_tokens"] += ct
                    st["cost"]["total_tokens"] += (pt + ct)
                    st["cost"]["num_prompts"] += 1

                    # update memory
                    st["mem_statements"].append(step["proposition"])
                    st["mem_protagonists"] = ordered_union(st["mem_protagonists"], step["protagonists_in_sentence"])

                    causality = list(st["mem_statements"])
                    protagonist = list(st["mem_protagonists"])

                    line = (
                        f"{sent}. "
                        f"{{time: {fmt_list_py(step['time'])}, space:{fmt_list_py(step['space'])}, "
                        f"causality: {fmt_list_py(causality)}, intention:{fmt_list_py(step['intention'])}, "
                        f"protagonist: {fmt_list_py(protagonist)}}}"
                    )
                    st["reading_lines"].append(line)
                    st["step_json"].append({
                        "sentence": sent,
                        "step": step,
                        "cumulative": {"causality": causality, "protagonist": protagonist}
                    })

            # -------- FINAL reasoning batch --------
            final_prompts = []
            for st in states:
                rtxt = "Let's read step by step.\n" + "\n\n".join(st["reading_lines"])
                final_prompts.append(self._compose_final_prompt(st["question"], st["options"], st["mem_statements"], rtxt))

            if final_prompts:
                t0 = time.time()
                final_outs = self.openai_api.batch_generate(final_prompts)
                elapsed = time.time() - t0

                self._update_global_cost(final_prompts, final_outs, elapsed)
                per_prompt_time = elapsed / max(1, len(final_prompts))

                # per-sample final stats
                for st, ptxt, otxt in zip(states, final_prompts, final_outs):
                    pt = self._count_tokens(ptxt)
                    ct = self._count_tokens(otxt)
                    st["cost"]["api_time_sec"] += per_prompt_time
                    st["cost"]["prompt_tokens"] += pt
                    st["cost"]["completion_tokens"] += ct
                    st["cost"]["total_tokens"] += (pt + ct)
                    st["cost"]["num_prompts"] += 1
            else:
                final_outs = []

            chunk_time_cost = time.time() - chunk_start

            # -------- Pack outputs --------
            for st, final_reasoning in zip(states, final_outs):
                pred = self.post_process_final_answer(final_reasoning)
                gold, pred_norm = self.normalize_gold_pred(st["gold"], pred)
                is_correct = (gold == pred_norm)

                # estimate USD for this sample
                inp, outp, tot = self._estimate_cost_usd(st["cost"]["prompt_tokens"], st["cost"]["completion_tokens"])
                st["cost"]["estimated_input_cost_usd"] = inp
                st["cost"]["estimated_output_cost_usd"] = outp
                st["cost"]["estimated_total_cost_usd"] = tot

                outputs_all.append({
                    "id": st["sid"],
                    "question": st["question"],
                    "context": st["context"],
                    "reading": "Let's read step by step.\n" + "\n\n".join(st["reading_lines"]),
                    "memory": {
                        "statements": st["mem_statements"],
                        "protagonists": st["mem_protagonists"],
                        "steps": st["step_json"]
                    },
                    "final_reasoning": final_reasoning,
                    "predicted": pred_norm,
                    "gold": gold,
                    "is_correct": is_correct,
                    "chunk_time_cost_sec": chunk_time_cost,
                    "cost": st["cost"],
                })

                if is_correct:
                    user_prompt = (
                        "Task Description: Given a logical statement problem, analyze and structure it step by step.\n"
                        "For each statement, extract: time, space, causality (accumulated), intention, protagonist (accumulated).\n"
                        "----\n"
                        f"Problem:\n{st['context']}\n"
                        "----\n"
                        f"Question:\n{st['question']}\n"
                        "----\n"
                        "Reading:\n"
                    )
                    assistant_target = (
                        "Let's read step by step.\n"
                        + "\n\n".join(st["reading_lines"])
                        + "\n\nFinal Answer: "
                        + pred_norm
                    )

                    distill_record = {
                        "id": st["sid"],
                        "prompt": user_prompt,
                        "completion": assistant_target,
                        "gold": gold,
                        "predicted": pred_norm,
                        "cost": st["cost"],  
                    }
                    distill_f.write(json.dumps(distill_record, ensure_ascii=False) + "\n")
                    correct_written += 1


            if self.save_all:
                with open(self.all_out_path, "w", encoding="utf-8") as f:
                    json.dump(outputs_all, f, indent=2, ensure_ascii=False)

        distill_f.close()

        # -------- finalize global cost estimate --------
        ginp, gout, gtot = self._estimate_cost_usd(self.stats["prompt_tokens"], self.stats["completion_tokens"])
        self.stats["estimated_input_cost_usd"] = ginp
        self.stats["estimated_output_cost_usd"] = gout
        self.stats["estimated_total_cost_usd"] = gtot
        self.stats["num_correct_written"] = correct_written

        with open(self.meta_out_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False)

        print(f"Saved correct-only distillation jsonl: {self.distill_out_path} (n={correct_written})")
        if self.save_all:
            print(f"Saved all outputs: {self.all_out_path}")
        print(f"Saved meta(cost) json: {self.meta_out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--split", type=str, required=True)
    p.add_argument("--save_path", type=str, required=True)
    p.add_argument("--api_key", type=str, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--stop_words", type=str, default="------")
    p.add_argument("--mode", type=str, default="memo_distill")
    p.add_argument("--max_new_tokens", type=int, default=512)

    p.add_argument("--memory_step_prompt_path", type=str, default="./utils/prompt/memory_step.txt")
    p.add_argument("--final_reason_prompt_path", type=str, default="./utils/prompt/final_reasoning_from_memory.txt")

    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=0, help="0 means no limit")
    p.add_argument("--save_all", action="store_true")

    p.add_argument("--run_id", type=str, default="")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    runner = GPT3_Incremental_Memo_Distiller(args)
    runner.run()