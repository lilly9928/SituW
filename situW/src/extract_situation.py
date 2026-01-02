import sys
sys.path.append('/data3/KJE/code/SituW/situW')

import json
import os
from tqdm import tqdm
from utils.utils import OpenAIModel
import argparse
import re
import sys
import time
from nltk.tokenize import sent_tokenize
from utils.gpt_pricing import get_text_prices_per_1m

try:
    import tiktoken
except Exception:
    tiktoken = None

class GPT3_Reasoning_Graph_Baseline:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)

        self.stats = {
            "total_api_time": 0.0,
            "batch_calls": 0,
            "single_calls": 0,
            "num_prompts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }

        self.prompt = {
            'time':       'Extract the time: When does this occur? Reply with a short phrase or None.',
            'location':   'Extract the location: Where does this occur? Reply with a short phrase or None.',
            'cause':      'Extract the cause: What triggered this? Reply with a short phrase or None.',
            'intention':  'Extract the intention: What is the purpose? Reply with a short phrase or None.',
            'protagonist':'Extract the protagonist: Who is involved? Reply with a short phrase or None.'
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
        elif self.dataset_name == 'logiqa':
           with open(os.path.join(self.data_path, f'{split}_new2.json')) as f:
                raw_dataset = json.load(f)
        else:
            with open(os.path.join(self.data_path, f'{self.dataset_name.lower()}_{split}.json')) as f:
                raw_dataset = json.load(f)

        return raw_dataset
    
    def format_memory(self, memory_dict):
        return (
            f"Time: {memory_dict.get('time', 'N/A')}\n"
            f"Location: {memory_dict.get('location', 'N/A')}\n"
            f"Protagonist: {memory_dict.get('protagonist', 'N/A')}"
        )

    def batch_reasoning_graph_generation(self, batch_size=10):
        memory_bank = {}
        raw_dataset = self.load_raw_dataset(self.split)
        
        seen = set()
        sentence_items = []
        for item in raw_dataset:
            pid = item['story_id']
            if pid in seen:
                continue
            seen.add(pid)

            for sent in [s.strip() for s in item['premises'].split('.') if s.strip()]:
                sentence_items.append({'parent_id': pid, 'premises': sent})

        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        dataset_chunks = [sentence_items[i:i + batch_size] for i in range(0, len(sentence_items), batch_size)]
        for chunk in tqdm(dataset_chunks):
            try:
                prompts_time = [f"{example['premises']} {self.prompt['time']}" for example in chunk]
                t0 = time.time()
                batch_time = self.openai_api.batch_generate(prompts_time)
                # breakpoint()
                self._update_batch_stats(prompts_time, batch_time, time.time() - t0)

                prompts_location = [f"{example['premises']} {self.prompt['location']}" for example in chunk]
                t0 = time.time()
                batch_location = self.openai_api.batch_generate(prompts_location)
                self._update_batch_stats(prompts_location, batch_location, time.time() - t0)

                prompts_protagonist = [f"{example['premises']} {self.prompt['protagonist']}" for example in chunk]
                t0 = time.time()
                batch_protagonist = self.openai_api.batch_generate(prompts_protagonist)
                self._update_batch_stats(prompts_protagonist, batch_protagonist, time.time() - t0)

                for i, example in enumerate(chunk):
                    pid = example['parent_id']
                    if pid not in memory_bank:
                        memory_bank[pid] = {'time':set(),
                                            'location':set(),
                                            'protagonist':set(),
                                            'cause':set(),
                                            'intention':set()}
                    memory_bank[pid]['time'].add(batch_time[i])
                    memory_bank[pid]['location'].add(batch_location[i])
                    memory_bank[pid]['protagonist'].add(batch_protagonist[i])

                prompts_cause = [
                    f"{example['premises']} {self.format_memory(memory_bank[example['parent_id']])} {self.prompt['cause']}"
                    for example in chunk
                ]
                t0 = time.time()
                batch_cause = self.openai_api.batch_generate(prompts_cause)
                self._update_batch_stats(prompts_cause, batch_cause, time.time() - t0)

                prompts_intention = [
                    f"{example['premises']} {self.format_memory(memory_bank[example['parent_id']])} {self.prompt['intention']}"
                    for example in chunk
                ]
                t0 = time.time()
                batch_intention = self.openai_api.batch_generate(prompts_intention)
                self._update_batch_stats(prompts_intention, batch_intention, time.time() - t0)

                for i, example in enumerate(chunk):
                    pid = example['parent_id']
                    memory_bank[pid]['cause'].add(batch_cause[i])
                    memory_bank[pid]['intention'].add(batch_intention[i])

            except Exception as e:
                print("Error in batch generation: ", e)
                for sample in chunk:
                    try:
                        pid = sample['parent_id']
                        if pid not in memory_bank:
                            memory_bank[pid] = {'time':set(),
                                                'location':set(),
                                                'protagonist':set(),
                                                'cause':set(),
                                                'intention':set()}

                        prompt_time = f"{sample['premises']} {self.prompt['time']}"
                        t0 = time.time()
                        time_r, _ = self.openai_api.generate(prompt_time)
                        self._update_single_stats(prompt_time, time_r, time.time() - t0)

                        prompt_location = f"{sample['premises'] } {self.prompt['location']}"
                        t0 = time.time()
                        location_r, _ = self.openai_api.generate(prompt_location)
                        self._update_single_stats(prompt_location, location_r, time.time() - t0)

                        prompt_protagonist = f"{sample['premises'] } {self.prompt['protagonist']}"
                        t0 = time.time()
                        protagonist_r, _ = self.openai_api.generate(prompt_protagonist)
                        self._update_single_stats(prompt_protagonist, protagonist_r, time.time() - t0)
                        
                        memory_bank[pid]['time'].add(time_r)
                        memory_bank[pid]['location'].add(location_r)
                        memory_bank[pid]['protagonist'].add(protagonist_r)

                        prompt_cause = f"{sample['premises']} {self.format_memory(memory_bank[pid])} {self.prompt['cause']}"
                        t0 = time.time()
                        cause_r, _ = self.openai_api.generate(prompt_cause)
                        self._update_single_stats(prompt_cause, cause_r, time.time() - t0)

                        prompt_intention = f"{sample['premises']} {self.format_memory(memory_bank[pid])} {self.prompt['intention']}"
                        t0 = time.time()
                        intention_r, _ = self.openai_api.generate(prompt_intention)
                        self._update_single_stats(prompt_intention, intention_r, time.time() - t0)

                        memory_bank[pid]['cause'].add(cause_r)
                        memory_bank[pid]['intention'].add(intention_r)

                    except Exception as inner_e:
                        print('Error in generating example:', sample.get('parent_id', 'N/A'), inner_e)
            
        final_memory_outputs = [
        {
            'id': pid,
            'memory': {
                k: list(v) for k, v in memory.items()
            }
        }
        for pid, memory in memory_bank.items()
        ]

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
            "estimated_total_cost_usd": total_cost
        }

        final_memory_outputs.append({"id": "__meta__", "meta": meta})

        save_file = os.path.join(
            self.save_path,
            f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_memory_only.json'
        )
        with open(save_file, 'w') as f:
            json.dump(final_memory_outputs, f, indent=2, ensure_ascii=False)

        print(f"Saved parent-level memory outputs to {save_file}")

        
    def update_answer(self,sample, reading, reasoning, time_cost):
        final_answer = self.post_process_c(reasoning)
        final_choice = self.final_process_logiqa(final_answer)

        if self.dataset_name=='logiqa':
            dict_output = {'id': sample['id'],
                    'questtion': sample['hypothesis'],
                    'original_context': sample['premise'],
                    'reading': reading,
                    'reasoning': reasoning,
                    'predicted_answer': final_answer, 
                    'answer': sample['label'],
                    'predicted_choice': final_choice,
                    'time_cost': time_cost}
        
        else:
            dict_output = {'id': sample['id'],
                    'questtion': sample['question'],
                    'original_context': sample['context'],
                    'reading': reading,
                    'reasoning': reasoning,
                    'predicted_answer': final_answer, 
                    'answer': sample['answer'],
                    'predicted_choice': final_choice,
                    'time_cost': time_cost}
        return dict_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/LogiQA2.0/logiqa2nli/DATA/QA2NLI')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--save_path', type=str, default='/data3/KJE/code/SituW/situW/output/situation_memory')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', default=20,type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    gpt3_problem_reduction.batch_reasoning_graph_generation(batch_size=10)
