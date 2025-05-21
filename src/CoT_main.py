import sys
sys.path.append('/data3/KJE/code/WIL_DeepLearningProject_2/NS_Parser/SITUM_EMNLP')
import argparse
from CoT import GPT3_Reasoning_Graph_Baseline

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


