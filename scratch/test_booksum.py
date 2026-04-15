import sys
sys.path.insert(0, 'code')
import argparse
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_rouge
from vanilla import run_vanilla
from coa import run_coa
from rag import run_rag
from run_booksum import build_vanilla_prompt, get_context, build_worker_prompt, build_manager_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    args = parser.parse_args()

    print("Loading BookSum...")
    dataset = load_dataset("THUDM/LongBench", "booksum", split="test", trust_remote_code=True)
    if args.mode == "test":
        # Selecting the NEXT 5 samples (indices 10 through 14) to avoid the previous ones
        dataset = dataset.select(range(10, min(15, len(dataset))))

    MODELS = [
        "google/gemma-4-e4b-it",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    for model_id in MODELS:
        print(f"\nEvaluating Model: {model_id} on BookSum ({args.mode} mode)")
        model, tokenizer = load_model(model_id)

        try:
            print("\n--- 1. VANILLA Pipeline ---")
            run_vanilla(model, tokenizer, dataset, "BookSum", build_vanilla_prompt, compute_rouge, model_id)
            
            print("\n--- 2. COA Pipeline ---")
            run_coa(model, tokenizer, dataset, "BookSum", get_context, build_worker_prompt, build_manager_prompt, compute_rouge, model_id)
            
            print("\n--- 3. RAG Pipeline ---")
            run_rag(model, tokenizer, dataset, "BookSum", build_vanilla_prompt, compute_rouge, model_id)
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
