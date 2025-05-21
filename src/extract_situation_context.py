import sys
sys.path.append('/data3/KJE/code/WIL_DeepLearningProject_2/NS_Parser/SITUM_EMNLP')

import json
import os
from tqdm import tqdm
from utils.utils import OpenAIModel
import argparse
import re
import sys
import time
from nltk.tokenize import sent_tokenize

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

        self.prompt = {
            'time':       'Extract the time: When does this occur? Reply with a short phrase or None.',
            'location':   'Extract the location: Where does this occur? Reply with a short phrase or None.',
            'cause':      'Extract the cause: What triggered this? Reply with a short phrase or None.',
            'intention':  'Extract the intention: What is the purpose? Reply with a short phrase or None.',
            'protagonist':'Extract the protagonist: Who is involved? Reply with a short phrase or None.'
        }


    def load_raw_dataset(self, split):
        if 'd' in self.dataset_name: 
             with open(os.path.join(self.data_path, f'{self.dataset_name}_{split}.json')) as f:
                raw_dataset = json.load(f)
        elif self.dataset_name == 'logiqa':
           with open(os.path.join(self.data_path, f'{split}_new2.json')) as f:
                raw_dataset = json.load(f)
        else:
            with open(os.path.join(self.data_path, self.dataset_name, f'{split}.json')) as f:
                raw_dataset = json.load(f)

        return raw_dataset
    
    def format_memory(self, memory_dict):
        return (
            f"Time: {memory_dict.get('time', 'N/A')}\n"
            f"Location: {memory_dict.get('location', 'N/A')}\n"
            f"Protagonist: {memory_dict.get('protagonist', 'N/A')}"
        )

    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        memory_bank = {}
        raw_dataset = self.load_raw_dataset(self.split)
        
        seen = set()
        sentence_items = []
        for item in raw_dataset:
            pid = item['id']
            if pid in seen:
                continue
            seen.add(pid)

            for sent in [s.strip() for s in item['premise'].split('.') if s.strip()]:
                sentence_items.append({'parent_id': pid, 'premise': sent})

        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # split dataset into chunks
        dataset_chunks = [sentence_items[i:i + batch_size] for i in range(0, len(sentence_items), batch_size)]
        for chunk in tqdm(dataset_chunks):
            try:
                start_time = time.time()
                # import pdb;pdb.set_trace()
                batch_time = self.openai_api.batch_generate([
                    f"{example['premise']} {self.prompt['time']}" for example in chunk
                ])
                batch_location = self.openai_api.batch_generate([
                    f"{example['premise']} {self.prompt['location']}" for example in chunk
                ])
                batch_protagonist = self.openai_api.batch_generate([
                    f"{example['premise']} {self.prompt['protagonist']}" for example in chunk
                ])

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

                batch_cause = self.openai_api.batch_generate([
                    f"{example['premise']} {self.format_memory(memory_bank[example['parent_id']])} {self.prompt['cause']}"
                    for example in chunk
                ])
                batch_intention = self.openai_api.batch_generate([
                    f"{example['premise']} {self.format_memory(memory_bank[example['parent_id']])} {self.prompt['intention']}"
                    for example in chunk
                ])

                for i, example in enumerate(chunk):
                    pid = example['parent_id']
                    memory_bank[pid]['cause'].add(batch_cause[i])
                    memory_bank[pid]['intention'].add(batch_intention[i])

                # break

            except Exception as e:
                print("Error in batch generation: ", e)
                for sample in chunk:
                    try:
                        pid = sample['parent_id']
                        start_time = time.time()

                        # 개별 추론 처리
                        prompt_time = f"{sample['premise']} {self.prompt['time']}"
                        time_r, _ = self.openai_api.generate(prompt_time)

                        prompt_location = f"{sample['premise'] } {self.prompt['location']}"
                        location_r, _ = self.openai_api.generate(prompt_location)

                        prompt_protagonist = f"{sample['premise'] } {self.prompt['protagonist']}"
                        protagonist_r, _ = self.openai_api.generate(prompt_protagonist)
                        
                        memory_bank[pid]['time'].add(time_r)
                        memory_bank[pid]['location'].add(location_r)
                        memory_bank[pid]['protagonist'].add(protagonist_r)

                        prompt_cause = f"{sample['premise']} {self.format_memory(memory_bank[pid])} {self.prompt['cause']}"
                        cause_r, _ = self.openai_api.generate(prompt_cause)

                        prompt_intention = f"{sample['premise']} {self.format_memory(memory_bank[pid])} {self.prompt['intention']}"
                        intention_r, _ = self.openai_api.generate(prompt_intention)

                        memory_bank[pid]['cause'].add(cause_r)
                        memory_bank[pid]['intention'].add(intention_r)
                        
                        # break

                    except Exception as inner_e:
                        print('Error in generating example:', sample.get('parent_id', 'N/A'), inner_e)


   
        # 최종 저장 - parent_id 기준 memory만 저장
        final_memory_outputs = [
        {
            'id': pid,
            'memory': {
                k: list(v) for k, v in memory.items()  # <- set을 list로 변환
            }
        }
        for pid, memory in memory_bank.items()
        ]
        save_file = os.path.join(
            self.save_path,
            f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_memory_only.json'
        )
        with open(save_file, 'w') as f:
            json.dump(final_memory_outputs, f, indent=2, ensure_ascii=False)

        print(f"Saved parent-level memory outputs to {save_file}")

        
    def update_answer(self,sample, reading, reasoning, time_cost):
        # import pdb;pdb.set_trace()
        final_answer = self.post_process_c(reasoning)
        final_choice = self.final_process_logiqa(final_answer)

        if self.dataset_name=='logiqa':
            # sample = json.loads(sample)
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
    parser.add_argument('--save_path', type=str, default='./outputs')
    parser.add_argument('--demonstration_path', type=str, default='./icl_examples')
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--stop_words', type=str, default='------')
    parser.add_argument('--mode', type=str)
    parser.add_argument('--max_new_tokens', type=int)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    gpt3_problem_reduction = GPT3_Reasoning_Graph_Baseline(args)
    # gpt3_problem_reduction.reasoning_graph_generation()
    gpt3_problem_reduction.batch_reasoning_graph_generation(batch_size=10)
