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
        prompt, gold_answer = build_prompt_fn(sample)
        
        start_time = time.time()
        response, tokens = generate_answer(model, tokenizer, prompt, max_new_tokens=250)
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
        
        print(f"Sample {i+1} | Score: {score:.3f} | Latency: {latency:.2f}s | Pred: {str(predicted)[:30]}")

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
