import sys
sys.path.insert(0, "code")
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_f1
from vanilla import run_vanilla
from coa import run_coa
import os
os.chdir("d:/COA")

# Test only Gemma COA on 5 qasper samples
MODELS = ["google/gemma-2-9b-it"]

TASK_REQUIREMENT = 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.'

def build_worker_prompt(sample, chunk, previous_msg):
    return f"""Worker W_i:
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

def get_context(sample):
    return sample['context'], sample['answers']

print("Loading qasper...")
dataset = load_dataset("THUDM/LongBench", "qasper", split="test", trust_remote_code=True)
dataset = dataset.select(range(5))

for model_id in MODELS:
    print(f"\n{'='*50}\nTesting Model: {model_id}\n{'='*50}")
    try:
        model, tokenizer = load_model(model_id)
        run_coa(model, tokenizer, dataset, "Qasper", get_context, build_worker_prompt, build_manager_prompt, compute_f1, model_id)
        cleanup_memory(model, tokenizer)
    except Exception as e:
        print(f"ERROR for {model_id}: {e}")
        import traceback; traceback.print_exc()
