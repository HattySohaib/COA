import os
import sys
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_f1
from vanilla import run_vanilla
from coa import run_coa

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-4-e4b-it",
    "Qwen/Qwen2.5-7B-Instruct"
]

TASK_REQUIREMENT = "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation."

def build_vanilla_prompt(sample):
    context = sample['context']
    question = sample['input']
    gold_answer = sample['answers']
    prompt = f"""{TASK_REQUIREMENT}

Story: {context}

Now, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.

Question: {question}

Answer:"""
    return prompt, gold_answer

def get_context(sample):
    return sample['context'], sample['answers']

def build_worker_prompt(sample, chunk, previous_msg):
    return f"""Worker Wi:
{chunk}
Here is the summary of the previous source text: {previous_msg}
Question: {sample['input']}
You need to read current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to answer the Query, if any. So please write the summary that can include the evidence for answering the Query:"""

def build_manager_prompt(sample, final_worker_msg):
    return f"""Manager M:
{TASK_REQUIREMENT}
The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary:
{final_worker_msg}
Question: {sample['input']}
Answer:"""

def main():
    print("Loading narrativeqa dataset from THUDM/LongBench...")
    dataset = load_dataset("THUDM/LongBench", "narrativeqa", split="test")
    # dataset = dataset.select(range(...)) # Removed truncation for full eval

    for model_id in MODELS:
        print(f"\n{'='*50}\nTesting Model: {model_id}\n{'='*50}")
        model, tokenizer = load_model(model_id)

        run_vanilla(model, tokenizer, dataset, "NarrativeQA", build_vanilla_prompt, compute_f1, model_id)
            # run_coa(model, tokenizer, dataset, "NarrativeQA", get_context, build_worker_prompt, build_manager_prompt, compute_f1, model_id)
        
        cleanup_memory(model, tokenizer)
        del model
        del tokenizer
        import gc
        gc.collect()
        import torch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
