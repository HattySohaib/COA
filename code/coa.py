import os
import json
import time
import torch
from utils import generate_answer, parse_final_answer

# Paper config: 8k token budget per agent
# Workers generate up to 1024 tokens (evidence summary)
# Manager generates up to 2048 tokens (final answer)
WORKER_MAX_INPUT_TOKENS = 6912   # 8192 - 1024 (new tokens) - ~256 prompt overhead
WORKER_MAX_NEW_TOKENS = 1024
MANAGER_MAX_NEW_TOKENS = 2048
# Adjust chunk size so that chunk + previous_msg (1024 tokens) + prompt overhead fits in 6912 without right truncating
CHUNK_SIZE = 5500 

def chunk_by_tokens(text, tokenizer, max_tokens=CHUNK_SIZE):
    """
    Split text into chunks where each chunk fits within max_tokens.
    This matches the paper's approach of fitting context within model limits.
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i + max_tokens]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
    return chunks

def run_coa(model, tokenizer, dataset, dataset_name, get_context_fn, build_worker_prompt_fn, build_manager_prompt_fn, metric_fn, model_id):
    print(f"\n[{dataset_name}] Running COA Pipeline...")
    total_score = 0
    total_tokens = 0
    total_latency = 0
    results = []

    safe_model_id = model_id.replace("/", "-")
    out_dir = os.path.join("results", safe_model_id, "coa")
    os.makedirs(out_dir, exist_ok=True)
    
    # Protect full benchmarks from being overwritten by small test runs
    suffix = "_TEST" if len(dataset) < 20 else ""
    out_path = os.path.join(out_dir, f"results_{dataset_name}{suffix}.json")

    for i, sample in enumerate(dataset):
        context, gold_answer = get_context_fn(sample)

        # Chunk context by tokens to fit within 8k window
        chunks = chunk_by_tokens(context, tokenizer)
        print(f"  Sample {i+1}: {len(chunks)} chunk(s), context length: {len(tokenizer.encode(context))} tokens")

        previous_msg = ""
        sample_tokens = 0
        start_time = time.time()

        # === Worker sequence (COA paper Algorithm 1) ===
        # W_i reads chunk c_i + CU_{i-1} and produces CU_i
        for j, chunk in enumerate(chunks):
            worker_prompt = build_worker_prompt_fn(sample, chunk, previous_msg)
            
            # Tune the prompt for instruction models to eliminate conversational padding
            worker_prompt += "\n\nImportant: Write the summary directly. Do not include any conversational filler, such as 'Here is the summary', 'Based on the text', or 'Sure'."

            messages = [{"role": "user", "content": worker_prompt}]
            chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # Truncate to fit within WORKER_MAX_INPUT_TOKENS
            inputs = tokenizer(
                chat_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=WORKER_MAX_INPUT_TOKENS
            ).to(model.device)
            input_length = inputs.input_ids.shape[1]
            sample_tokens += input_length

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=WORKER_MAX_NEW_TOKENS,
                    do_sample=False,            # Greedy decoding per paper
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated_tokens = outputs[0][input_length:]
            sample_tokens += len(generated_tokens)
            previous_msg = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            print(f"  [Worker {j+1}/{len(chunks)}] out: {previous_msg[:100]}...")

        # === Manager sequence ===
        # M reads task requirement + CU_l (last worker message) + question
        manager_prompt = build_manager_prompt_fn(sample, previous_msg)
        response, manager_tokens = generate_answer(model, tokenizer, manager_prompt, max_new_tokens=MANAGER_MAX_NEW_TOKENS)
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

        current_avg_scale = (total_score / (i + 1)) * 100
        print(f"Sample {i+1}/{len(dataset)} | Cur. Score: {score:.3f} | Avg. Score: {current_avg_scale:.2f} | Latency: {latency:.2f}s")
        
        # Incremental save
        output_dict = {
            "dataset": dataset_name,
            "model": model_id,
            "pipeline": "coa",
            "avg_score": (total_score / (i + 1)) * 100,
            "avg_tokens": (total_tokens / (i + 1)),
            "avg_latency": (total_latency / (i + 1)),
            "samples": results
        }
        with open(out_path, "w") as f:
            json.dump(output_dict, f, indent=4)

    n = len(dataset)
    avg_score = (total_score / n) * 100 if n > 0 else 0
    avg_tokens = total_tokens / n if n > 0 else 0
    avg_latency = total_latency / n if n > 0 else 0
    print(f"\n[{dataset_name}] COA Result: {avg_score:.2f}% | Avg Tokens: {avg_tokens:.0f} | Avg Latency: {avg_latency:.2f}s")


    return avg_score
