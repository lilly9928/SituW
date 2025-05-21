import json
import os
from tqdm import tqdm
from utils.utils import OpenAIModel
import argparse
import re
import sys
import time


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


    def load_in_context_examples_reading(self):
        file_path = os.path.join('./utils/prompt', 'reading.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    # /data3/KJE/code/WIL_DeepLearningProject_2/NS_Parser/SITUM_EMNLP/utils/prompt/LogicQA_CoT.txt
    def load_inference_CoT(self):
        file_path = os.path.join('./utils/prompt','LogicQA_CoT.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_inference_direct(self):
        file_path = os.path.join('./utils/prompt','direct.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
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

    
    def index_context(self, context):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', context)
        formatted_context = enumerate(sentences, start=1)
        indexed_sentences = '\n'.join([f"{index}: {sentence}" for index, sentence in formatted_context])
        return str(indexed_sentences)

    def construct_prompt_reading(self, record, in_context_examples_reading):
        full_prompt = in_context_examples_reading
        
        if self.dataset_name=='logiqa':
            # record = json.loads(record)

            context = record['premise']
            question = record['hypothesis'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
        
        else:
            context = record['context']
            question = record['question'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
       
        return full_prompt
    
    def construct_prompt_CoT(self, record, response_r, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        if self.dataset_name=='logiqa':
            # record = json.loads(record)
            
            context = record['premise']
            question = record['hypothesis'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[READING]]', response_r)
        
        else:
            context = record['context']
            question = record['question'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[READING]]', response_r)
        return full_prompt
    

    def construct_prompt_direct(self, record, response,in_context_examples_trans):
        full_prompt = in_context_examples_trans
        if self.dataset_name=='logiqa':
            # record = json.loads(record)
            
            context = record['premise']
            question = record['hypothesis'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', response)
            full_prompt = full_prompt.replace('[[CONCLUSION]]', question)

        return full_prompt

    def post_process_a(self, response_a):
        response_a = str(response_a)
        context_start = response_a.find('"context":') + 10
        context_end = response_a.find('",\n"Question"')
        context = response_a[context_start:context_end].strip()
        question_start = response_a.find('"Question":') + 11
        question_end = response_a[question_start:].find('"}') + question_start
        question = response_a[question_start:question_end].strip()
        return context, question
    
    def post_process_c(self, response_c):
        # 패턴: "Final Answer:" 이후의 텍스트 전부 추출 (줄 끝까지)
        pattern_final_answer = r"Final Answer:\s*(.*)"
        match = re.search(pattern_final_answer, response_c, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        return "No final answer found in the text."

    
    def final_process(self, final_answer):
        final_answer = final_answer.lower()
        if final_answer == "true":
            final_answer = 'A'
        elif final_answer == "false":
            final_answer = 'B'
        elif final_answer == "unknown":
            final_answer = 'C'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer
    

    def final_process_logiqa(self, final_answer):
        final_answer = final_answer.lower()
        if "A" in final_answer:
            final_answer = 'Entailment'
        elif "B" in final_answer:
            final_answer = 'Not-entailment'
        else:
            final_answer = "No final answer found in the text."  
        return final_answer
    

    def batch_reasoning_graph_generation(self, batch_size=10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        # raw_dataset = raw_dataset[860:]
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        # load in-context examples
        in_context_examples_reading = self.load_in_context_examples_reading()
        in_context_examples_CoT = self.load_inference_CoT()
        in_context_examples_direct = self.load_inference_direct()
        # import pdb;pdb.set_trace()         
        outputs = []
        
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            try:
                # import pdb;pdb.set_trace()
                start_time = time.time()
                print("Reading...")
                batch_reading = self.openai_api.batch_generate([self.construct_prompt_reading(example, in_context_examples_reading) for example in chunk])

                print("Reasoning...")
                batch_reasoning = self.openai_api.batch_generate([self.construct_prompt_CoT(example,responses_r, in_context_examples_CoT) for example, responses_r in zip(chunk, batch_reading)])
                
                end_time = time.time()
                time_cost = end_time - start_time
                for sample, reading, reasoning in zip(chunk,batch_reading,batch_reasoning):
                    # import pdb;pdb.set_trace()
               
                    dict_output = self.update_answer(sample,reading, reasoning, time_cost)
                    outputs.append(dict_output)
    
                # with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_missing.json'), 'w') as f:
                #     json.dump(outputs, f, indent=2, ensure_ascii=False)
                with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_reading.json'), 'w') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)
                
                # print(outputs)
                # break

            except Exception as e:
                print("Error in batch generation: ", e)
                for sample in chunk:
                    try:
                        start_time = time.time()
                        print("Reading...")
                        # import pdb;pdb.set_trace()
                        prompts_reading = self.construct_prompt_reading(sample, in_context_examples_reading)
                        reading, _ = self.openai_api.generate(prompts_reading)

                        print("Reasoning...")
                        prompts_reasoning = self.construct_prompt_CoT(sample, reading, in_context_examples_CoT)
                        reasoning, _ = self.openai_api.generate(prompts_reasoning)
                        
                        end_time = time.time()
                        time_cost = end_time - start_time

                        dict_output = self.update_answer(sample,reading, reasoning, time_cost)
                        outputs.append(dict_output)

                        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_reading.json'), 'w') as f:
                            json.dump(outputs, f, indent=2, ensure_ascii=False)

                        # with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_missing.json'), 'w') as f:
                        #     json.dump(outputs, f, indent=2, ensure_ascii=False)
                        
                        # print(outputs)
                        # break
                            
                    except Exception as e:
                        print('Error in generating example: ', sample['id'] , e)
                        
                            

        # with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
        #     json.dump(outputs, f, indent=2, ensure_ascii=False)
    
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


