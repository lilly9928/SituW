import os
import json
import sys
import argparse
import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

sys.path.append('/data3/KJE/code/SituW')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str)
    parser.add_argument('--save_path', type=str, default='/data3/KJE/code/SituW/situW/output')

    parser.add_argument('--model_name', type=str, default='mistral_7b_instruct')

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['folio_val', 'prontoqa_val', 'proofwriter_val', 'logiqa_val']
    )

    parser.add_argument('--max_new_tokens', type=int, default=16)

    # ✅ 배치 크기
    parser.add_argument('--batch_size', type=int, default=8)

    # chain-of-thought 옵션
    parser.add_argument('--cot', type=str, default='none')

    # generation options
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)

    args = parser.parse_args()
    return args


class HG_Model:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.model_name = args.model_name
        self.mode = args.mode
        self.model_path = ''
        self.cot = args.cot

        # save path: cot 여부에 따라 하위 폴더 변경
        if self.cot == 'cot':
            self.save_path = os.path.join(args.save_path, "zero_shot_cot")
        else:
            self.save_path = os.path.join(args.save_path, "zero_shot")

        # -------------------------
        # model registry
        # -------------------------
        ### LLAMA ###
        if self.model_name == 'llama_33_70b_instruct':
            self.model_path = "meta-llama/Llama-3.3-70B-Instruct"
        elif self.model_name == 'llama_31_70b_instruct':
            self.model_path = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        elif self.model_name == 'llama_31_8b_instruct':
            self.model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"

        ### MISTRAL ###
        elif self.model_name == 'mistral_7b_instruct':
            self.model_path = "mistralai/Mistral-7B-Instruct-v0.3"
        elif self.model_name == 'mistral_large_instruct':
            self.model_path = "mistralai/Mistral-Large-Instruct-2407"

        ### QWEN ###
        elif self.model_name == 'qwen_72b_instruct':
            self.model_path = 'Qwen/Qwen2.5-72B-Instruct'

        ### DEEPSEEK ###
        elif self.model_name == 'deepseek_2.5':
            self.model_path = 'deepseek-ai/DeepSeek-V2.5'

        else:
            raise ValueError(f"Unknown model_name: {self.model_name}")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            cache_dir='/data3/hg_weight/hg_weight'
        )
        # ✅ decoder-only 배치 생성에서는 left padding 권장
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir="/data3/hg_weight/hg_weight",
            quantization_config=bnb_config,
            device_map="auto"
        )
        self.model.eval()

        # pad token safety
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # chat template availability
        self.use_chat_template = hasattr(self.tokenizer, "apply_chat_template") and (self.tokenizer.chat_template is not None)

    # -------------------------
    # dataset loader
    # -------------------------
    def load_raw_dataset(self):
        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_dataset = json.load(f)
        if not isinstance(raw_dataset, list):
            raise ValueError("Dataset json should be a list of dicts.")
        return raw_dataset

    # -------------------------
    # label mapping (mode-specific)
    # -------------------------
    def get_ground_truth_choice(self, item):
        if self.mode == "folio_val":
            label_str = item.get("label", None)
            if label_str is None:
                return None
            s = str(label_str).strip().lower()
            if s == "true":
                return "A"
            if s == "false":
                return "B"
            if s in ["unknown", "uncertain", "undetermined"]:
                return "C"
            return None

        elif self.mode == "prontoqa_val":
            ans = item.get("answer", None)
            if ans is None:
                return None
            s = str(ans).strip().upper()
            if s in ["A", "B"]:
                return s
            return None

        elif self.mode == "proofwriter_val":
            ans = item.get("answer", None)
            if ans is None:
                return None
            s = str(ans).strip().lower()
            if s == "true":
                return "A"
            if s == "false":
                return "B"
            if s in ["unknown", "uncertain", "undetermined"]:
                return "C"
            return None

        elif self.mode == "logiqa_val":
            lab = item.get("label", None)
            if lab is None:
                return None
            s = str(lab).strip().lower()
            if s == "entailment":
                return "A"
            if s in ["not-entailment", "not entailment", "not_entailment", "non-entailment", "nonentailment"]:
                return "B"
            return None

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    # -------------------------
    # build prompt messages (mode-specific)
    # -------------------------
    def build_prompt_messages(self, item):
        if self.mode == "folio_val":
            premises = item.get("premises", "").strip()
            conclusion = item.get("conclusion", "").strip()

            context = premises.replace("\n", " ")
            question = (
                "Based on the above information, is the following statement true, false, or uncertain? "
                + conclusion
            )
            options = ["A) True", "B) False", "C) Uncertain"]

            if self.cot == "cot":
                user_content = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n'
                    f'- {options[2]}\n\n'
                    "Let's think step by step:"
                )
            else:
                user_content = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n'
                    f'- {options[2]}\n\n'
                    "Answer with only one letter among A, B, C."
                )

        elif self.mode == "prontoqa_val":
            context = str(item.get("context", "")).strip()
            question = str(item.get("question", "")).strip()
            options_list = item.get("options", [])

            options_lines = []
            if isinstance(options_list, list) and len(options_list) > 0:
                for opt in options_list:
                    options_lines.append(f"- {str(opt).strip()}")
            else:
                options_lines = ["- A) True", "- B) False"]

            if self.cot == "cot":
                user_content = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n' + "\n".join(options_lines) + "\n\n"
                    "Let's think step by step:"
                )
            else:
                user_content = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n' + "\n".join(options_lines) + "\n\n"
                    "Answer with only one letter among A, B."
                )

        elif self.mode == "proofwriter_val":
            theory = str(item.get("theory", "")).strip()
            q = str(item.get("question", "")).strip()

            context = theory
            question = f"Based on the above theory, is the following statement true, false, or unknown? {q}"
            options = ["A) True", "B) False", "C) Unknown"]

            if self.cot == "cot":
                question += " Let's think step by step."

            # ✅ ProofWriter는 CoT를 써도 최종 출력은 letter만 요구하도록 유지
            user_content = (
                f'context: "{context}"\n'
                f'question: "{question}"\n'
                f'options:\n'
                f'- {options[0]}\n'
                f'- {options[1]}\n'
                f'- {options[2]}\n\n'
                "Answer with only one letter among A, B, C."
            )

        elif self.mode == "logiqa_val":
            premise = str(item.get("premise", "")).strip()
            hypothesis = str(item.get("hypothesis", "")).strip()

            question = "Does the premise entail the hypothesis?"
            options = ["A) entailment", "B) not-entailment"]

            if self.cot == "cot":
                question += " Let's think step by step."

            # ✅ LogiQA도 CoT를 써도 최종 출력은 letter만 요구하도록 유지
            user_content = (
                f'premise: "{premise}"\n'
                f'hypothesis: "{hypothesis}"\n'
                f'question: "{question}"\n'
                f'options:\n'
                f'- {options[0]}\n'
                f'- {options[1]}\n\n'
                "Answer with only one letter among A, B."
            )

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        messages = [
            {"role": "system", "content": "You are a careful reasoner. Follow the format strictly. Output only the option letter."},
            {"role": "user", "content": user_content},
        ]
        return messages

    # -------------------------
    # output parsing (mode-specific)
    # -------------------------
    def parse_choice(self, text):
        if text is None:
            return None
        t = text.strip()

        # prontoqa/logiqa -> A/B
        if self.mode in ["prontoqa_val", "logiqa_val"]:
            m = re.search(r"\b([AB])\b", t, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()

            tl = t.lower()

            if self.mode == "prontoqa_val":
                if "true" in tl:
                    return "A"
                if "false" in tl:
                    return "B"
                return None

            # logiqa_val fallback (⚠️ not-entailment 먼저)
            if "not-entailment" in tl or "not entailment" in tl or "non-entailment" in tl:
                return "B"
            if "entailment" in tl or "entailed" in tl or "entails" in tl:
                return "A"
            return None

        # folio/proofwriter -> A/B/C
        m = re.search(r"\b([ABC])\b", t, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        tl = t.lower()
        if "true" in tl:
            return "A"
        if "false" in tl:
            return "B"
        if "uncertain" in tl or "unknown" in tl:
            return "C"
        return None

    # -------------------------
    # prompt -> string
    # -------------------------
    def messages_to_prompt(self, messages):
        if self.use_chat_template:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return messages[-1]["content"]

    # -------------------------
    # file save utils
    # -------------------------
    def _atomic_json_dump(self, obj, path):
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)

    def _make_payload(self, results, correct_cnt, total_cnt):
        acc = (correct_cnt / total_cnt) * 100.0 if total_cnt > 0 else 0.0
        return {"accuracy": acc, "results": results}

    # -------------------------
    # batched generation
    # -------------------------
    @torch.no_grad()
    def run_batch(self, batch_items):
        batch_prompts = []
        gt_choices = []
        ids = []
        story_ids = []

        for item in batch_items:
            if self.mode == "folio_val":
                uid = item.get("example_id", None)
                story_id = item.get("story_id", None)
            else:
                uid = item.get("id", None)
                story_id = None

            messages = self.build_prompt_messages(item)
            prompt = self.messages_to_prompt(messages)

            batch_prompts.append(prompt)
            gt_choices.append(self.get_ground_truth_choice(item))
            ids.append(uid)
            story_ids.append(story_id)

        inputs = self.tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.args.max_new_tokens,
            do_sample=self.args.do_sample,
            temperature=self.args.temperature if self.args.do_sample else 0.0,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        results = []
        for i in range(len(batch_items)):
            gen_ids = outputs[i][input_lens[i]:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            pred = self.parse_choice(gen_text)
            gt = gt_choices[i]
            is_correct = (pred == gt) if (pred is not None and gt is not None) else False

            results.append({
                "id": ids[i],
                "story_id": story_ids[i],
                "prediction": pred,
                "ground_truth": gt,
                "correct": is_correct,
                "raw_output": gen_text,
            })

        return results

    # -------------------------
    # generation loop
    # - 처음 2 배치까지만 중간 저장(덮어쓰기)해서 생성 형태 확인
    # - 이후에는 저장 안 하다가 마지막에 최종 저장
    # -------------------------
    def batch_generation(self):
        raw_dataset = self.load_raw_dataset()

        results = []
        correct_cnt = 0
        total_cnt = 0

        bs = max(1, int(self.args.batch_size))
        buffer = []

        os.makedirs(self.save_path, exist_ok=True)

        if self.cot == "cot":
            out_file = os.path.join(self.save_path, f"{self.model_name}_{self.mode}_cot_result.json")
        else:
            out_file = os.path.join(self.save_path, f"{self.model_name}_{self.mode}_result.json")

        saved_batches = 0  # ✅ 몇 배치 저장했는지
        save_first_n_batches = 2  # ✅ 처음 N 배치만 저장

        for item in tqdm(raw_dataset, desc=f"Evaluating ({self.mode})", total=len(raw_dataset)):
            buffer.append(item)
            if len(buffer) < bs:
                continue

            batch_results = self.run_batch(buffer)
            buffer = []

            for r in batch_results:
                results.append(r)
                total_cnt += 1
                correct_cnt += int(r["correct"])

            # ✅ 처음 2 배치까지만 중간 저장
            if saved_batches < save_first_n_batches:
                payload = self._make_payload(results, correct_cnt, total_cnt)
                self._atomic_json_dump(payload, out_file)
                saved_batches += 1

        # leftover
        if len(buffer) > 0:
            batch_results = self.run_batch(buffer)
            for r in batch_results:
                results.append(r)
                total_cnt += 1
                correct_cnt += int(r["correct"])

            # leftover도 "배치"로 간주해서, 아직 2번 저장 안 했으면 저장
            if saved_batches < save_first_n_batches:
                payload = self._make_payload(results, correct_cnt, total_cnt)
                self._atomic_json_dump(payload, out_file)
                saved_batches += 1

        # ✅ 최종 결과 저장(전체 다 돌고 난 뒤)
        payload = self._make_payload(results, correct_cnt, total_cnt)
        self._atomic_json_dump(payload, out_file)

        acc = payload["accuracy"]
        print(f"Done. ACC: {correct_cnt}/{total_cnt} = {acc:.2f}%")
        print(f"Results saved to: {out_file}")
        return results


if __name__ == '__main__':
    args = parse_args()
    hg_model = HG_Model(args)
    hg_model.batch_generation()
