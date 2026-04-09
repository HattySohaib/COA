import os
import sys
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_exact_match
from vanilla import run_vanilla
from coa import run_coa

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct"
]

TASK_REQUIREMENT = "Please complete the code given below."

def build_vanilla_prompt(sample):
    context = sample['context']
    code_input = sample['input']
    gold_answer = sample['answers']
    prompt = f"""{TASK_REQUIREMENT}
 
{context}{code_input}Next line of code:
"""
    return prompt, gold_answer

def get_context(sample):
    return sample['context'] + "\n" + sample['input'], sample['answers']

def build_worker_prompt(sample, chunk, previous_msg):
    return f"""Worker W_i:
{chunk}
Here is the summary of the previous source text: {previous_msg}
You need to read the current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to generate a summary for the whole text. Thus, your generated summary should be relatively long."""

def build_manager_prompt(sample, final_worker_msg):
    return f"""Manager M:
{TASK_REQUIREMENT}
The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary:
{final_worker_msg}
Answer:"""

def main():
    print("Loading repobench-p dataset from THUDM/LongBench...")
    dataset = load_dataset("THUDM/LongBench", "repobench-p", split="test")
    dataset = dataset.select(range(min(2, len(dataset))))

    for model_id in MODELS:
        print(f"\n{'='*50}\nTesting Model: {model_id}\n{'='*50}")
        model, tokenizer = load_model(model_id)

        run_vanilla(model, tokenizer, dataset, "RepoBench-P", build_vanilla_prompt, compute_exact_match, model_id)
        run_coa(model, tokenizer, dataset, "RepoBench-P", get_context, build_worker_prompt, build_manager_prompt, compute_exact_match, model_id)
        
        cleanup_memory(model, tokenizer)

if __name__ == "__main__":
    main()
