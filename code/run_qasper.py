import os
import sys
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_f1

TASK_REQUIREMENT = "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation."

def build_vanilla_prompt(sample):
    context = sample['context']
    question = sample['input']
    gold_answer = sample['answers']
    prompt = f"""{TASK_REQUIREMENT}

Article: {context}

Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.

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
The following is the summarized article:
{final_worker_msg}
Question: {sample['input']}
Answer:"""