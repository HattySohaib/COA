import os
import sys
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_exact_match
from vanilla import run_vanilla
from coa import run_coa

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-4-e4b-it",
    "Qwen/Qwen2.5-7B-Instruct"
]

TASK_REQUIREMENT = "Answer the multiple choice question based strictly on the provided text by providing the correct option."

def build_vanilla_prompt(sample):
    context = sample['input']
    gold_answer = sample['output']
    prompt = f"""{TASK_REQUIREMENT}

{context}
Answer:"""
    return prompt, gold_answer

def get_context(sample):
    return sample['input'], sample['output']

def build_worker_prompt(sample, chunk, previous_msg):
    return f"""Worker W_i:
{chunk}
Here is the summary of the previous source text: {previous_msg}
Question: {sample['input'][-2000:]}
You need to read current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to answer the Query, if any. So please write the summary that can include the evidence for answering the Query:"""

def build_manager_prompt(sample, final_worker_msg):
    return f"""Manager M:
{TASK_REQUIREMENT}
The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary:
{final_worker_msg}
Question: {sample['input'][-2000:]}
Answer:"""

def main():
    print("Loading quality dataset from tau/scrolls...")
    dataset = load_dataset("tau/scrolls", "quality", split="validation", trust_remote_code=True)
    # dataset = dataset.select(range(...)) # Removed truncation for full eval

    for model_id in MODELS:
        print(f"\n{'='*50}\nTesting Model: {model_id}\n{'='*50}")
        model, tokenizer = load_model(model_id)

        run_vanilla(model, tokenizer, dataset, "QuALITY", build_vanilla_prompt, compute_exact_match, model_id)
            # run_coa(model, tokenizer, dataset, "QuALITY", get_context, build_worker_prompt, build_manager_prompt, compute_exact_match, model_id)
        
        cleanup_memory(model, tokenizer)

if __name__ == "__main__":
    main()
