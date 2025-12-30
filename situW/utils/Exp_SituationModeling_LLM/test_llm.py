import os 
import json
import tqdm
import sys
sys.path.append('/data3/KJE/code/SituW')
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import re


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/data3/KJE/code/SituW/situW/utils/Exp_SituationModeling_LLM/data')
    parser.add_argument('--save_path', type=str, default='/data3/KJE/code/SituW/situW/utils/Exp_SituationModeling_LLM/outputs')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='llama_70b_instruct')
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    args = parser.parse_args()
    return args


class HG_Model:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.api_key = args
        self.model_path = ''

        ### LLAMA
        if self.model_name =='llama_33_70b_instruct':
            self.model_path = "meta-llama/Llama-3.3-70B-Instruct"
        
        elif self.model_name == 'llama_31_70b_instruct':
            self.model_path = 'meta-llama/Meta-Llama-3.1-70B-Instruct'
            
        elif self.model_name =='llama_31_8b_instruct':
            self.model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
        
        ### MISTRAL
        elif self.model_name == 'mistral_7b_instruct':
            self.model_path = 'mistralai/Mistral-7B-Instruct-v0.3'
        
        elif self.model_name == 'mistral_large_instruct':
            self.model_path = 'mistralai/Mistral-Large-Instruct-2407'


        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
                

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, cache_dir ='/data3/hg_weight/hg_weight')
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            cache_dir="/data3/hg_weight/hg_weight",
            quantization_config=bnb_config,
            device_map="auto"
        )


    def load_raw_dataset(self):
        with open(os.path.join(self.data_path, f'questions_shuffled.json')) as f:
                raw_dataset = json.load(f)

        return raw_dataset
    
    def format_memory(self, memory_dict):
        return (
            f"Time: {memory_dict.get('time', 'N/A')}\n"
            f"Location: {memory_dict.get('location', 'N/A')}\n"
            f"Protagonist: {memory_dict.get('protagonist', 'N/A')}"
        )

    def batch_generation(self, batch_size=10):
        raw_dataset = self.load_raw_dataset()

        labels_alpha = ['a', 'b', 'c', 'd']
        labels_display = ['(a)', '(b)', '(c)', '(d)']
        answer_pattern = re.compile(r'\b[a-d]\b')

        base_prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
            "A confusable pair refers to two expressions that are similar in meaning "
            "or used in similar contexts, making them easy for learners to confuse.\n"
            "They are not completely synonymous, but their meanings or usages partially overlap.\n"
            "Among the options (a), (b), (c), and (d) below, choose the confusable pair. Answer in short answer. "
            "Format: The confusable pair is <option> and <option>.\n\n"
            "<|start_header_id|>assistant<|end_header_id|>"
        )


        # base_prompt = (
        #     "<|begin_of_text|><|start_header_id|>user<|end_header_id|>"
        #     "A confusable pair refers to two expressions that are similar in meaning "
        #     "or used in similar contexts, making them easy for learners to confuse.\n"
        #     "They are not completely synonymous, but their meanings or usages partially overlap.\n"
        #     "Among the options (a), (b), (c), and (d) below, choose the confusable pair. Think step by step.\n\n"
        #     "<|start_header_id|>assistant<|end_header_id|>"
        # )

        
        results = []

        cnt_here = 1
        sum= 0

        for item in raw_dataset:
            pid = item['question_id']
            options = item['options']

            
            gt_indices = [
                idx for idx, opt in enumerate(options)
                if opt['label'] == 0
            ]
            gt_answer = sorted(labels_alpha[i] for i in gt_indices)

            
            prompt = base_prompt
            for label, option in zip(labels_display, options):
                prompt += f"{label} {option['text']}\n"


            inputs = self.tokenizer(prompt,return_tensors="pt").to(self.model.device)
            input_len = inputs["input_ids"].shape[1]
            # breakpoint()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=15,
                do_sample=False,
                temperature=0.0
            )

            generated_ids = outputs[0][input_len:]

            generated_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            )
            
            # breakpoint()

            pred_answer = sorted(
                answer_pattern.findall(generated_text.lower())
            )

            is_correct = (pred_answer == gt_answer)

            sum += is_correct

            print(f"ACC {sum/cnt_here*100}")
            
            results.append({
                "question_id": pid,
                "prediction": pred_answer,
                "ground_truth": gt_answer,
                "correct": is_correct,
                "raw_output": generated_text,
            })
            
            cnt_here += 1


        os.makedirs(self.save_path, exist_ok=True)
        out_file = os.path.join(self.save_path, f"{self.model_name}_result.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {out_file}")

        return results



if __name__ == '__main__':
    args = parse_args()
    hg_model = HG_Model(args)
    # gpt3_problem_reduction.reasoning_graph_generation()
    hg_model.batch_generation(batch_size=10)