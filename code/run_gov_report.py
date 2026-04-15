import os
import sys
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_rouge
from vanilla import run_vanilla
from coa import run_coa

MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "google/gemma-2-9b-it",
    "Qwen/Qwen2.5-7B-Instruct"
]

TASK_REQUIREMENT = "You are given a report by a government agency. Write a one-page summary of the report."

def build_vanilla_prompt(sample):
    context = sample['context']
    gold_answer = sample['answers'][0]
    prompt = f"""{TASK_REQUIREMENT}

Report:
{context}

Now, write a one-page summary of the report.

Summary:"""
    return prompt, gold_answer

def get_context(sample):
    return sample['context'], sample['answers'][0]

def build_worker_prompt(sample, chunk, previous_msg):
    return f"""Worker Wi:
{chunk}
Here is the summary of the previous source text: {previous_msg}
You need to read the current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to generate a summary for the whole text. Thus, your generated summary should be relatively long."""

def build_manager_prompt(sample, final_worker_msg):
    return f"""Manager M:
{TASK_REQUIREMENT}
The following are given passages. However, the source text is too long and has been summarized. You need to answer based on the summary:
{final_worker_msg}
Answer:"""

