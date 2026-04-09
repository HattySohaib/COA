import os
import json
import time
import torch
from utils import generate_answer, parse_final_answer, chunk_text

def run_coa(model, tokenizer, dataset, dataset_name, get_context_fn, build_worker_prompt_fn, build_manager_prompt_fn, metric_fn, model_id):
    print(f"\n[{dataset_name}] Running COA Pipeline...")
    total_score = 0
    total_tokens = 0
    total_latency = 0
    results = []

    for i, sample in enumerate(dataset):
        context, gold_answer = get_context_fn(sample)
        chunks = chunk_text(context)
        
        worker_messages = []
        previous_msg = ""
        sample_tokens = 0
        
        start_time = time.time()
        
        # Worker sequence
        for j, chunk in enumerate(chunks):
            worker_prompt = build_worker_prompt_fn(sample, chunk, previous_msg)
            inputs = tokenizer(worker_prompt, return_tensors="pt", truncation=True, max_length=2000).to(model.device)
            input_length = inputs.input_ids.shape[1]
            sample_tokens += input_length

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    max_length=None,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = outputs[0][input_length:]
            sample_tokens += len(generated_tokens)
            generated_msg = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            worker_messages.append(generated_msg)
            previous_msg = generated_msg

        # Manager sequence
        manager_prompt = build_manager_prompt_fn(sample, previous_msg) # The paper passes only CU_l to the manager!
        response, manager_tokens = generate_answer(model, tokenizer, manager_prompt, max_new_tokens=250)
        sample_tokens += manager_tokens
        
        latency = time.time() - start_time
        predicted = parse_final_answer(response)

        score = metric_fn(gold_answer, predicted)
        total_score += score
        total_tokens += sample_tokens
        total_latency += latency
        
        results.append({
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "score": score,
            "tokens": sample_tokens,
            "latency_seconds": latency
        })
        
        print(f"Sample {i+1} | Score: {score:.3f} | Latency: {latency:.2f}s | Pred: {str(predicted)[:30]}")

    n = len(dataset)
    avg_score = (total_score / n) * 100 if n > 0 else 0
    avg_tokens = total_tokens / n if n > 0 else 0
    avg_latency = total_latency / n if n > 0 else 0
    print(f"[{dataset_name}] COA Result: {avg_score:.2f} | Avg Tokens: {avg_tokens:.1f} | Avg Latency: {avg_latency:.2f}s")
    
    # Save results
    safe_model_id = model_id.replace("/", "-")
    out_dir = os.path.join("results", safe_model_id, "coa")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{dataset_name}.json")
    
    output_dict = {
        "dataset": dataset_name,
        "model": model_id,
        "pipeline": "coa",
        "avg_score": avg_score,
        "avg_tokens": avg_tokens,
        "avg_latency": avg_latency,
        "samples": results
    }
    with open(out_path, "w") as f:
        json.dump(output_dict, f, indent=4)
        
    return avg_score
