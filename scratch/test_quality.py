import sys
sys.path.insert(0, 'code')
import argparse
from datasets import load_dataset
from utils import load_model, cleanup_memory
from vanilla import run_vanilla
from coa import run_coa
from run_quality import build_vanilla_prompt, get_context, build_worker_prompt, build_manager_prompt, compute_exact_match_letter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    args = parser.parse_args()

    print("Loading QuALITY (tau/zero_scrolls)...")
    dataset = load_dataset("tau/zero_scrolls", "quality", split="validation", trust_remote_code=True)
    if args.mode == "test":
        dataset = dataset.select(range(5))

    MODELS = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-4-e4b-it",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    for model_id in MODELS:
        print(f"\nEvaluating Model: {model_id} on QuALITY ({args.mode} mode)")
        model, tokenizer = load_model(model_id)

        try:
            print("\n--- 1. VANILLA Pipeline ---")
            run_vanilla(model, tokenizer, dataset, "QuALITY", build_vanilla_prompt, compute_exact_match_letter, model_id)
            
            print("\n--- 2. COA Pipeline ---")
            run_coa(model, tokenizer, dataset, "QuALITY", get_context, build_worker_prompt, build_manager_prompt, compute_exact_match_letter, model_id)
        except Exception as e:
            print(f"Failed: {e}")
        finally:
            cleanup_memory(model, tokenizer)
            del model
            del tokenizer
            import gc
            gc.collect()
            import torch
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
