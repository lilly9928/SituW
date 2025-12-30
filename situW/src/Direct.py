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


    def load_inference_CoT(self):
        file_path = os.path.join('./utils/prompt','LogicQA_CoT_real.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    
    def load_inference(self):
        file_path = os.path.join('./utils/prompt','direct.txt')
        with open(file_path) as f:
            in_context_examples = f.read()
        return in_context_examples
    
    def load_raw_dataset(self, split):
        if 'd' in self.dataset_name: 
             with open(os.path.join(self.data_path, f'{self.dataset_name}_{split}.json')) as f:
                raw_dataset = json.load(f)
        elif self.dataset_name == 'logiqa':
            with open('/data3/KJE/code/WIL_DeepLearningProject_2/NS_Parser/SITUM_EMNLP/data/LogiQA2.0/logiqa2nli/DATA/QA2NLI/test_new_ours_2.json') as f:
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

    
    def construct_prompt_CoT(self, record, in_context_examples_trans):
        full_prompt = in_context_examples_trans
        if self.dataset_name=='logiqa':
            # record = json.loads(record)
            
            context = record['premise']
            question = record['hypothesis'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[READING]]', context)
        
        else:
            context = record['context']
            question = record['question'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)
            full_prompt = full_prompt.replace('[[READING]]', context)
        return full_prompt


    def construct_prompt_inference(self, record, response,inference):
        full_prompt = inference
        if self.dataset_name=='logiqa':
            # record = json.loads(record)
            
            context = record['premise']
            question = record['hypothesis'].strip()
            full_prompt = full_prompt.replace('[[response]]', response)
            full_prompt = full_prompt.replace('[[QUESTION]]', question)

        return full_prompt
    

    def construct_prompt_direct(self, record,inference):
        full_prompt = inference
        if self.dataset_name=='logiqa':
            # record = json.loads(record)
            context = record['premise']
            question = record['hypothesis'].strip()
            full_prompt = full_prompt.replace('[[CONTEXT]]', context)
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
        # in_context_examples_reading = self.load_in_context_examples_reading()
        in_context_examples_CoT = self.load_inference_CoT()
        in_context_examples_infer = self.load_inference()
        # import pdb;pdb.set_trace()         
        outputs = []
        
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]
        for chunk in tqdm(dataset_chunks):
            try:
                # import pdb;pdb.set_trace()
                start_time = time.time()
    
                print("Reasoning...")
                # import pdb;pdb.set_trace()
                # batch_reasoning = self.openai_api.batch_generate([self.construct_prompt_direct(example, in_context_examples_infer) for example in chunk])

                batch_inference = self.openai_api.batch_generate([self.construct_prompt_direct(example, in_context_examples_infer) for  example in chunk])
                
                end_time = time.time()
                time_cost = end_time - start_time
                for sample,answer in zip(chunk,batch_inference):
                    # import pdb;pdb.set_trace()
               
                    dict_output = self.update_answer(sample,answer, time_cost)
                    outputs.append(dict_output)
    
                # with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_missing.json'), 'w') as f:
                #     json.dump(outputs, f, indent=2, ensure_ascii=False)
                with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_Direct.json'), 'w') as f:
                    json.dump(outputs, f, indent=2, ensure_ascii=False)
                
                # print(outputs)
                # break

            except Exception as e:
                print("Error in batch generation: ", e)
                for sample in chunk:
                    try:
                        start_time = time.time()
          
                        print("Reasoning...")
                        prompts_reasoning = self.construct_prompt_direct(sample, in_context_examples_infer)
                        inference, _ = self.openai_api.generate(prompts_reasoning)
                        
                        end_time = time.time()
                        time_cost = end_time - start_time

                        dict_output = self.update_answer(sample,inference, time_cost)
                        outputs.append(dict_output)

                        with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_Direct.json'), 'w') as f:
                            json.dump(outputs, f, indent=2, ensure_ascii=False)

                        # with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}_missing.json'), 'w') as f:
                        #     json.dump(outputs, f, indent=2, ensure_ascii=False)
                        
                        # print(outputs)
                        # break
                            
                    except Exception as e:
                        print('Error in generating example: ', sample['id'] , e)
                            

        # with open(os.path.join(self.save_path, f'{self.mode}_{self.dataset_name}_{self.split}_{self.model_name}.json'), 'w') as f:
        #     json.dump(outputs, f, indent=2, ensure_ascii=False)
    
    def update_answer(self,sample,answer, time_cost):
        # import pdb;pdb.set_trace()
        # final_answer = self.post_process_c(reasoning)
        # final_choice = self.final_process_logiqa(final_answer)

        if self.dataset_name=='logiqa':
            # sample = json.loads(sample)
            dict_output = {'id': sample['id'],
                       'questtion': sample['hypothesis'],
                       'original_context': sample['premise'],
                    #    'reasoning': reasoning,
                       "predicted_answer":answer,
                    #    'predicted_answer': final_answer, 
                       'answer': sample['label'],
                    #    'predicted_choice': final_choice,
                       'time_cost': time_cost}
        
        else:
            dict_output = {'id': sample['id'],
                       'questtion': sample['question'],
                       'original_context': sample['context'],
                    #    'reasoning': reasoning,
                    #    'reasoning': reasoning,
                       "predicted_answer":answer,
                    #    'predicted_answer': final_answer, 
                       'answer': sample['answer'],
                    #    'predicted_choice': final_choice,
                       'time_cost': time_cost}
        return dict_output


