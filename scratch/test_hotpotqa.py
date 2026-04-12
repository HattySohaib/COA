import sys
sys.path.insert(0, 'code')
import argparse
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_f1
from vanilla import run_vanilla
from coa import run_coa
from rag import run_rag
from run_hotpotqa import build_vanilla_prompt, get_context, build_worker_prompt, build_manager_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    args = parser.parse_args()

    print("Loading HotpotQA...")
    dataset = load_dataset("THUDM/LongBench", "hotpotqa", split="test", trust_remote_code=True)
    if args.mode == "test":
        # Selecting the NEXT 5 samples (indices 10 through 14) to avoid the previous ones
        dataset = dataset.select(range(10, min(15, len(dataset))))

    MODELS = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "google/gemma-4-e4b-it",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    for model_id in MODELS:
        print(f"\nEvaluating Model: {model_id} on HotpotQA ({args.mode} mode)")
        model, tokenizer = load_model(model_id)

        try:
            print("\n--- 1. VANILLA Pipeline ---")
            run_vanilla(model, tokenizer, dataset, "HotpotQA", build_vanilla_prompt, compute_f1, model_id)
            
            print("\n--- 2. COA Pipeline ---")
            run_coa(model, tokenizer, dataset, "HotpotQA", get_context, build_worker_prompt, build_manager_prompt, compute_f1, model_id)
            
            print("\n--- 3. RAG Pipeline ---")
            run_rag(model, tokenizer, dataset, "HotpotQA", build_vanilla_prompt, compute_f1, model_id)
        except Exception as e:
            print(f"Failed: {e}")
        finally:
            cleanup_memory(model, tokenizer)

if __name__ == "__main__":
    main()
