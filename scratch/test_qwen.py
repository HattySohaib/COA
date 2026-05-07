import sys
sys.path.insert(0, 'code')
import argparse
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_f1, compute_rouge
from ours import run_ours
from run_quality import compute_exact_match_letter

# We need the task specific imports only for get_context
from run_hotpotqa import get_context as hc
from run_narrativeqa import get_context as nc
from run_gov_report import get_context as gc_ctx
from run_quality import get_context as qc
from run_qasper import get_context as qaspc

TASKS = [
    {"name": "HotpotQA", "ds": "hotpotqa", "metric": compute_f1, "ctx": hc},
    {"name": "NarrativeQA", "ds": "narrativeqa", "metric": compute_f1, "ctx": nc},
    {"name": "GovReport", "ds": "gov_report", "metric": compute_rouge, "ctx": gc_ctx},
    {"name": "QuALITY", "ds": "quality", "metric": compute_exact_match_letter, "ctx": qc},
    {"name": "Qasper", "ds": "qasper", "metric": compute_f1, "ctx": qaspc}
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    args = parser.parse_args()

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nEvaluating Model: {model_id} on All Datasets ({args.mode} mode)")
    model, tokenizer = load_model(model_id)

    try:
        for task in TASKS:
            print(f"\n\n--- Running {task['name']} ---")
            try:
                if task['name'] == 'QuALITY':
                    dataset = load_dataset("tau/zero_scrolls", "quality", split="validation")
                else:
                    dataset = load_dataset("THUDM/LongBench", task['ds'], split="test", trust_remote_code=True)
                if args.mode == "test":
                    # Select a tiny slice for quick pipeline testing
                    dataset = dataset.select(range(min(5, len(dataset))))

                run_ours(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    dataset_name=task['name'],
                    get_context_fn=task['ctx'],
                    build_worker_prompt_fn=None,  # Ours custom prompts defined natively
                    build_manager_prompt_fn=None,
                    metric_fn=task['metric'],
                    model_id=model_id
                )
                print(f"--- Finished {task['name']} ---")
            except Exception as e:
                print(f"--- Failed {task['name']}: {e}")
                continue
    finally:
        cleanup_memory(model, tokenizer)

if __name__ == "__main__":
    main()
