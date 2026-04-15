import sys
sys.path.insert(0, 'code')
import argparse
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_f1
from vanilla import run_vanilla
from coa import run_coa
from rag import run_rag
from run_narrativeqa import build_vanilla_prompt, get_context, build_worker_prompt, build_manager_prompt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["test", "full"], default="test")
    args = parser.parse_args()

    print("Loading NarrativeQA...")
    dataset = load_dataset("THUDM/LongBench", "narrativeqa", split="test", trust_remote_code=True)
    if args.mode == "test":
        dataset = dataset.select(range(min(5, len(dataset))))

    MODELS = [
        "google/gemma-4-e4b-it",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
    
    for model_id in MODELS:
        print(f"\nEvaluating Model: {model_id} on NarrativeQA ({args.mode} mode)")
        model, tokenizer = load_model(model_id)

        try:
            print("\n--- 1. VANILLA Pipeline ---")
            run_vanilla(model, tokenizer, dataset, "NarrativeQA", build_vanilla_prompt, compute_f1, model_id)
            
            print("\n--- 2. COA Pipeline ---")
            run_coa(model, tokenizer, dataset, "NarrativeQA", get_context, build_worker_prompt, build_manager_prompt, compute_f1, model_id)
            
            print("\n--- 3. RAG Pipeline ---")
            run_rag(model, tokenizer, dataset, "NarrativeQA", build_vanilla_prompt, compute_f1, model_id)
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
