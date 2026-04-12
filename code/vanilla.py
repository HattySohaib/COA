import os
import json
import time
from utils import generate_answer, parse_final_answer

def run_vanilla(model, tokenizer, dataset, dataset_name, build_prompt_fn, metric_fn, model_id):
    print(f"\n[{dataset_name}] Running VANILLA Pipeline...")
    total_score = 0
    total_tokens = 0
    total_latency = 0
    results = []

    for i, sample in enumerate(dataset):
        # Apply Middle-Truncation to the context document so instructions aren't right-truncated
        # We cap the context string itself to ~6500 tokens to perfectly fit the rigid 8192 budget.
        truncated_sample = {}
        for k, v in sample.items():
            if isinstance(v, str) and len(v.split()) > 1000:
                tokens = tokenizer.encode(v, add_special_tokens=False)
                if len(tokens) > 6500:
                    half = 3250
                    truncated_tokens = tokens[:half] + tokens[-half:]
                    v = tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            truncated_sample[k] = v

        prompt, gold_answer = build_prompt_fn(truncated_sample)
        
        start_time = time.time()
        response, tokens = generate_answer(model, tokenizer, prompt, max_new_tokens=2048)
        latency = time.time() - start_time
        
        predicted = parse_final_answer(response)
        
        score = metric_fn(gold_answer, predicted)
        total_score += score
        total_tokens += tokens
        total_latency += latency
        
        results.append({
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "score": score,
            "tokens": tokens,
            "latency_seconds": latency
        })
        
        # Determine current average scaling. E.g. rouge vs f1 vs exact match
        try:
            from utils import compute_f1
            is_f1 = (metric_fn == compute_f1)
        except ImportError:
            is_f1 = False
            
        current_avg_scale = (total_score / (i + 1)) * 100 if not is_f1 else (total_score / (i + 1)) * 100
        
        print(f"Sample {i+1}/{len(dataset)} | Cur. Score: {score:.3f} | Avg. Score: {current_avg_scale:.2f} | Latency: {latency:.2f}s")

    n = len(dataset)
    avg_score = (total_score / n) * 100 if n > 0 else 0
    avg_tokens = total_tokens / n if n > 0 else 0
    avg_latency = total_latency / n if n > 0 else 0
    print(f"[{dataset_name}] VANILLA Result: {avg_score:.2f} | Avg Tokens: {avg_tokens:.1f} | Avg Latency: {avg_latency:.2f}s")
    
    # Save results
    safe_model_id = model_id.replace("/", "-")
    out_dir = os.path.join("results", safe_model_id, "vanilla")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{dataset_name}.json")
    
    output_dict = {
        "dataset": dataset_name,
        "model": model_id,
        "pipeline": "vanilla",
        "avg_score": avg_score,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "samples": results
    }
    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=4)
        
    return avg_score
