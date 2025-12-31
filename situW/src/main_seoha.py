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

    # ✅ few-shot 옵션: none / two
    parser.add_argument('--shot', type=str, default='none', choices=['none', 'two'])

    # generation options
    parser.add_argument('--do_sample', action='store_true')
    parser.add_argument('--temperature', type=float, default=0.0)

    parser.add_argument('--cache_dir', type=str, default='/data3/hg_weight/hg_weight')

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
        self.shot = args.shot
        self.cache_dir = args.cache_dir

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
            cache_dir=self.cache_dir
        )
        # ✅ decoder-only 배치 생성에서는 left padding 권장
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir=self.cache_dir,
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
        # few-shot examples (2-shot)
        # - two-shot + cot: reasoning 포함 예시
        # - two-shot (cot 아님): reasoning 없이 Answer만 있는 예시
        # -------------------------
        self.fewshot_map = self._build_fewshot_map()

    def _build_fewshot_map(self):
        """
        Returns a dict: mode -> dict{"cot": str, "plain": str}
        Used only when args.shot == "two".
        """
        # -------------------------
        # proofwriter
        # -------------------------
        proofwriter_two_cot = (
            'context: Gary is furry. Gary is nice. Gary is red. Gary is rough. Gary is not smart. Gary is white. Gary is young. '
            'If Gary is nice and Gary is not white then Gary is red. If someone is white then they are red. All young people are furry. '
            'If someone is white and not red then they are furry. Smart, red people are rough. '
            'If Gary is not red and Gary is not furry then Gary is not smart. If Gary is white then Gary is not smart. '
            'If someone is rough and not white then they are not smart.\n'
            'question: Based on the above theory, is the following statement true, false, or unknown? Gary is white.\n'
            'options:\n'
            '- A) True\n'
            '- B) False\n'
            '- C) Unknown\n\n'
            "Let's think step by step:\n\n"
            'Gary is described as furry, nice, red, rough, not smart, white, and young. The question asks if "Gary is white" is true, false, or unknown. '
            'This fact is stated directly, so it is assumed true unless it creates a contradiction. The rule that white implies red is satisfied, since Gary is both. '
            'The rule that if Gary is white, he is not smart also matches the facts. The rule that all young people are furry is consistent as well. '
            'No conditional contradicts Gary being white. Therefore, the statement "Gary is white" is true.\n'
            'Answer: A) True\n\n'
            'context: The bald eagle chases the bear. The bald eagle eats the bear. The bear chases the bald eagle. The bear eats the bald eagle. '
            'The bear is green. The bear is nice. The bear likes the bald eagle. '
            'If the bear likes the bald eagle and the bald eagle eats the bear then the bald eagle is nice. '
            'If someone is cold and nice then they like the bald eagle.\n'
            'question: Based on the above theory, is the following statement true, false, or unknown? The bear is not nice.\n'
            'options:\n'
            '- A) True\n'
            '- B) False\n'
            '- C) Unknown\n\n'
            "Let's think step by step:\n\n"
            'The context states that the bear is nice. This is a direct fact. The question asks whether "The bear is not nice" is true, false, or unknown. '
            'Since the opposite is explicitly stated, the claim that the bear is not nice contradicts the given fact. '
            'No rule overrides or conflicts with the statement that the bear is nice. Therefore, the statement "The bear is not nice" must be false.\n'
            'Answer: B) False\n\n'
        )

        # ✅ 요청하신: two-shot인데 cot 아닐 때는 reasoning 없이 Answer만
        proofwriter_two_plain = (
            'context: Gary is furry. Gary is nice. Gary is red. Gary is rough. Gary is not smart. Gary is white. Gary is young. '
            'If Gary is nice and Gary is not white then Gary is red. If someone is white then they are red. All young people are furry. '
            'If someone is white and not red then they are furry. Smart, red people are rough. '
            'If Gary is not red and Gary is not furry then Gary is not smart. If Gary is white then Gary is not smart. '
            'If someone is rough and not white then they are not smart.\n'
            'question: Based on the above theory, is the following statement true, false, or unknown? Gary is white.\n'
            'options:\n'
            '- A) True\n'
            '- B) False\n'
            '- C) Unknown\n\n'
            'Answer: A) True\n\n'
            'context: The bald eagle chases the bear. The bald eagle eats the bear. The bear chases the bald eagle. The bear eats the bald eagle. '
            'The bear is green. The bear is nice. The bear likes the bald eagle. '
            'If the bear likes the bald eagle and the bald eagle eats the bear then the bald eagle is nice. '
            'If someone is cold and nice then they like the bald eagle.\n'
            'question: Based on the above theory, is the following statement true, false, or unknown? The bear is not nice.\n'
            'options:\n'
            '- A) True\n'
            '- B) False\n'
            '- C) Unknown\n\n'
            'Answer: B) False\n\n'
        )

        # -------------------------
        # folio
        # -------------------------
        folio_two_cot = (
            "context: All people who regularly drink coffee are dependent on caffeine.\n"
            "People regularly drink coffee, or they don't want to be addicted to caffeine, or both.\n"
            "No one who doesn't want to be addicted to caffeine is unaware that caffeine is a drug.\n"
            "Rina is either a student who is unaware that caffeine is a drug, or she is not a student and is she aware that caffeine is a drug.\n"
            "Rina is either a student who is dependent on caffeine, or she is not a student and not dependent on caffeine.\n"
            "question: Based on the above information, is the following statement true, false, or uncertain? Rina doesn't want to be addicted to caffeine or is unaware that caffeine is a drug.\n"
            "options:\n"
            "- A) True\n"
            "- B) False\n"
            "- C) Uncertain\n\n"
            "Let's think step by step:\n\n"
            "From the given information, Rina must fall into one of two cases. Either she is a student who is unaware that caffeine is a drug and dependent on caffeine, "
            "or she is not a student, is aware that caffeine is a drug, and not dependent on caffeine. In the first case, Rina is unaware that caffeine is a drug, "
            "so the statement “Rina doesn't want to be addicted to caffeine or is unaware that caffeine is a drug” is true because the second part holds. "
            "In the second case, Rina is not dependent on caffeine and does not regularly drink coffee. Since everyone either drinks coffee or does not want to be addicted to caffeine, "
            "she must not want to be addicted. This makes the first part of the statement true.\n"
            "Because the statement is true in all possible cases, the correct answer is A) True.\n"
            "Answer: A) True\n\n"
            "context: The Blake McFall Company Building is a building added to the National Register of Historic Places in 1990.\n"
            "The Emmet Building is a five-story building in Portland, Oregon.\n"
            "The Emmet Building was built in 1915.\n"
            "The Emmet Building is another name for the Blake McFall Company Building.\n"
            "John works at the Emmet Building.\n"
            "question: Based on the above information, is the following statement true, false, or uncertain? John started his current job in 1990.\n"
            "options:\n"
            "- A) True\n"
            "- B) False\n"
            "- C) Uncertain\n\n"
            "Let's think step by step:\n\n"
            "The information states that the Blake McFall Company Building was added to the National Register of Historic Places in 1990, and that the Emmet Building is another name for the same building. "
            "We are also told that John works at the Emmet Building. However, there is no information about when John started working there, nor any rule that links the year a building was added to the register "
            "with the start date of someone’s job. Therefore, John’s job start year cannot be determined from the given facts.\n\n"
            "Answer: C) Uncertain\n\n"
        )

        folio_two_plain = (
            "context: All people who regularly drink coffee are dependent on caffeine.\n"
            "People regularly drink coffee, or they don't want to be addicted to caffeine, or both.\n"
            "No one who doesn't want to be addicted to caffeine is unaware that caffeine is a drug.\n"
            "Rina is either a student who is unaware that caffeine is a drug, or she is not a student and is she aware that caffeine is a drug.\n"
            "Rina is either a student who is dependent on caffeine, or she is not a student and not dependent on caffeine.\n"
            "question: Based on the above information, is the following statement true, false, or uncertain? Rina doesn't want to be addicted to caffeine or is unaware that caffeine is a drug.\n"
            "options:\n"
            "- A) True\n"
            "- B) False\n"
            "- C) Uncertain\n\n"
            "Answer: A) True\n\n"
            "context: The Blake McFall Company Building is a building added to the National Register of Historic Places in 1990.\n"
            "The Emmet Building is a five-story building in Portland, Oregon.\n"
            "The Emmet Building was built in 1915.\n"
            "The Emmet Building is another name for the Blake McFall Company Building.\n"
            "John works at the Emmet Building.\n"
            "question: Based on the above information, is the following statement true, false, or uncertain? John started his current job in 1990.\n"
            "options:\n"
            "- A) True\n"
            "- B) False\n"
            "- C) Uncertain\n\n"
            "Answer: C) Uncertain\n\n"
        )

        # -------------------------
        # logiqa
        # -------------------------
        logiqa_two_cot = (
            "premise: Screenwriter moviegoers are those who don't mind being spoiled by spoilers and even inquire about plot introductions and review all kinds of movies in advance. "
            "This kind of moviegoers pursue the feeling of controlling the development of the plot and don't like surprises.\n"
            "hypothesis: Xiao Li belongs to the screenwriter moviegoers according to the above definition, because he is fond of suspense movies, enjoys brain-burning plots, and assumes the role of a detective when watching movies.\n"
            "question: Does the premise entail the hypothesis?\n"
            "options:\n"
            "- A) entailment\n"
            "- B) not-entailment\n\n"
            "Let's think step by step:\n\n"
            "The premise defines screenwriter moviegoers as people who do not mind spoilers, actively look up plot introductions, review movies in advance, seek control over the plot development, and dislike surprises. "
            "The hypothesis claims that Xiao Li belongs to this group because he likes suspense movies, enjoys complex plots, and imagines himself as a detective while watching films. "
            "These traits are different from the defining characteristics in the premise and do not imply that he likes spoilers, avoids surprises, or checks plots in advance. "
            "Therefore, the premise does not logically support the hypothesis.\n\n"
            "Answer: B) not-entailment\n\n"
            "premise: All foreign students from China live on campus; All students living on campus must participate in the sports meeting; Some Chinese students have joined the student union; "
            "Some students majoring in psychology have also joined the student union; None of the psychology majors took part in the sports meeting.\n"
            "hypothesis: Some Chinese students majored in psychology cannot be drawn as a conclusion.\n"
            "question: Does the premise entail the hypothesis?\n"
            "options:\n"
            "- A) entailment\n"
            "- B) not-entailment\n\n"
            "Let's think step by step:\n\n"
            "The premises state that all foreign students from China live on campus, and all students living on campus must participate in the sports meeting. "
            "Some Chinese students have joined the student union, and some psychology majors have also joined the student union. None of the psychology majors took part in the sports meeting. "
            "This tells us that psychology majors did not live on campus, so they cannot be foreign students from China. While some Chinese students and some psychology majors are in the student union, "
            "there is no direct information linking any Chinese students to the psychology major. Therefore, we cannot conclude that any Chinese student majored in psychology based on the given information.\n\n"
            "Answer: A) entailment\n\n"
        )

        logiqa_two_plain = (
            "premise: Screenwriter moviegoers are those who don't mind being spoiled by spoilers and even inquire about plot introductions and review all kinds of movies in advance. "
            "This kind of moviegoers pursue the feeling of controlling the development of the plot and don't like surprises.\n"
            "hypothesis: Xiao Li belongs to the screenwriter moviegoers according to the above definition, because he is fond of suspense movies, enjoys brain-burning plots, and assumes the role of a detective when watching movies.\n"
            "question: Does the premise entail the hypothesis?\n"
            "options:\n"
            "- A) entailment\n"
            "- B) not-entailment\n\n"
            "Answer: B) not-entailment\n\n"
            "premise: All foreign students from China live on campus; All students living on campus must participate in the sports meeting; Some Chinese students have joined the student union; "
            "Some students majoring in psychology have also joined the student union; None of the psychology majors took part in the sports meeting.\n"
            "hypothesis: Some Chinese students majored in psychology cannot be drawn as a conclusion.\n"
            "question: Does the premise entail the hypothesis?\n"
            "options:\n"
            "- A) entailment\n"
            "- B) not-entailment\n\n"
            "Answer: A) entailment\n\n"
        )

        # prontoqa: 예시 없음
        prontoqa_two_cot = ""
        prontoqa_two_plain = ""

        return {
            "proofwriter_val": {"cot": proofwriter_two_cot, "plain": proofwriter_two_plain},
            "folio_val": {"cot": folio_two_cot, "plain": folio_two_plain},
            "logiqa_val": {"cot": logiqa_two_cot, "plain": logiqa_two_plain},
            "prontoqa_val": {"cot": prontoqa_two_cot, "plain": prontoqa_two_plain},
        }

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
        # few-shot prefix (2-shot) if requested
        fewshot_prefix = ""
        if self.shot == "two":
            mode_pack = self.fewshot_map.get(self.mode, {"cot": "", "plain": ""})
            if self.cot == "cot":
                fewshot_prefix = mode_pack.get("cot", "")
            else:
                fewshot_prefix = mode_pack.get("plain", "")

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
                cur = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n'
                    f'- {options[2]}\n\n'
                    "Let's think step by step:"
                )
            else:
                cur = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n'
                    f'- {options[2]}\n\n'
                    "Answer with only one letter among A, B, C."
                )

            user_content = (fewshot_prefix + cur) if fewshot_prefix else cur

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
                cur = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n' + "\n".join(options_lines) + "\n\n"
                    "Let's think step by step:"
                )
            else:
                cur = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n' + "\n".join(options_lines) + "\n\n"
                    "Answer with only one letter among A, B."
                )

            user_content = (fewshot_prefix + cur) if fewshot_prefix else cur

        elif self.mode == "proofwriter_val":
            theory = str(item.get("theory", "")).strip()
            q = str(item.get("question", "")).strip()

            context = theory
            question = f"Based on the above theory, is the following statement true, false, or unknown? {q}"
            options = ["A) True", "B) False", "C) Unknown"]

            if self.cot == "cot":
                cur = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n'
                    f'- {options[2]}\n\n'
                    "Let's think step by step:"
                )
            else:
                cur = (
                    f'context: "{context}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n'
                    f'- {options[2]}\n\n'
                    "Answer with only one letter among A, B, C."
                )

            user_content = (fewshot_prefix + cur) if fewshot_prefix else cur

        elif self.mode == "logiqa_val":
            premise = str(item.get("premise", "")).strip()
            hypothesis = str(item.get("hypothesis", "")).strip()

            question = "Does the premise entail the hypothesis?"
            options = ["A) entailment", "B) not-entailment"]

            if self.cot == "cot":
                cur = (
                    f'premise: "{premise}"\n'
                    f'hypothesis: "{hypothesis}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n\n'
                    "Let's think step by step:"
                )
            else:
                cur = (
                    f'premise: "{premise}"\n'
                    f'hypothesis: "{hypothesis}"\n'
                    f'question: "{question}"\n'
                    f'options:\n'
                    f'- {options[0]}\n'
                    f'- {options[1]}\n\n'
                    "Answer with only one letter among A, B."
                )

            user_content = (fewshot_prefix + cur) if fewshot_prefix else cur

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
        tail = t[-200:]  # ✅ 끝부분 위주로 파싱 (프롬프트 에코 방지)

        # prontoqa/logiqa -> A/B
        if self.mode in ["prontoqa_val", "logiqa_val"]:
            m = re.search(r"([AB])\s*\)", tail, flags=re.IGNORECASE)
            if m:
                return m.group(1).upper()

            ms = re.findall(r"\b([AB])\b", tail, flags=re.IGNORECASE)
            if ms:
                return ms[-1].upper()

            tl = tail.lower()

            if self.mode == "prontoqa_val":
                if "true" in tl:
                    return "A"
                if "false" in tl:
                    return "B"
                return None

            if "not-entailment" in tl or "not entailment" in tl or "non-entailment" in tl:
                return "B"
            if "entailment" in tl or "entailed" in tl or "entails" in tl:
                return "A"
            return None

        # folio/proofwriter -> A/B/C
        m = re.search(r"([ABC])\s*\)", tail, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper()

        ms = re.findall(r"\b([ABC])\b", tail, flags=re.IGNORECASE)
        if ms:
            return ms[-1].upper()

        tl = tail.lower()
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

        input_len = inputs["input_ids"].shape[1]

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
            gen_ids = outputs[i][input_len:]
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

        shot_tag = "" if self.shot == "none" else f"_{self.shot}shot"

        if self.cot == "cot":
            out_file = os.path.join(self.save_path, f"{self.model_name}_{self.mode}{shot_tag}_cot_result.json")
        else:
            out_file = os.path.join(self.save_path, f"{self.model_name}_{self.mode}{shot_tag}_result.json")

        saved_batches = 0
        save_first_n_batches = 2

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

            if saved_batches < save_first_n_batches:
                payload = self._make_payload(results, correct_cnt, total_cnt)
                self._atomic_json_dump(payload, out_file)
                saved_batches += 1

        if len(buffer) > 0:
            batch_results = self.run_batch(buffer)
            for r in batch_results:
                results.append(r)
                total_cnt += 1
                correct_cnt += int(r["correct"])

            if saved_batches < save_first_n_batches:
                payload = self._make_payload(results, correct_cnt, total_cnt)
                self._atomic_json_dump(payload, out_file)
                saved_batches += 1

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
