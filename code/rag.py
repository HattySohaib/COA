import os
import json
import time
from utils import generate_answer, parse_final_answer
import torch
from sentence_transformers import SentenceTransformer
import numpy as np

def run_rag(model, tokenizer, dataset, dataset_name, build_prompt_fn, metric_fn, model_id):
    print(f"\n[{dataset_name}] Running RAG Pipeline...")
    total_score = 0
    total_tokens = 0
    total_latency = 0
    results = []

    safe_model_id = model_id.replace('/', '-')
    out_dir = os.path.join('results', safe_model_id, 'rag')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"results_{dataset_name}.json")

    # Load previously saved results
    if os.path.exists(out_path):
        try:
            with open(out_path, "r") as f:
                output_dict = json.load(f)
                results = output_dict.get('samples', [])
                for r in results:
                    total_score += r['score']
                    total_tokens += r['tokens']
                    total_latency += r['latency_seconds']
                print(f"Resumed from sample {len(results)}")
        except Exception as e:
            print(f"Error loading {out_path}: {e}")

    # Load Retriever model
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu')

    def chunk_text(text, chunk_size=300):
        words = text.split()
        return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

    safe_model_id = model_id.replace("/", "-")
    out_dir = os.path.join("results", safe_model_id, "rag")
    os.makedirs(out_dir, exist_ok=True)
    
    # Protect full benchmarks from being overwritten by small test runs
    suffix = "_TEST" if len(dataset) < 20 else ""
    out_path = os.path.join(out_dir, f"results_{dataset_name}{suffix}.json")

    for i, sample in enumerate(dataset):
        if i < len(results):
            continue

        context = sample.get('context', sample.get('chapter', sample.get('input', '')))
        question = sample.get('input', '')
        
        # Handle LongBench answers vs scrolls/booksum outputs
        if 'answers' in sample:
            gold_answer = sample['answers']
        elif 'output' in sample:
            gold_answer = [sample['output']]
        elif 'summary_text' in sample:
            gold_answer = [sample['summary_text']]
        else:
            gold_answer = []
            
        # Implement pseudo-query fallback for empty queries (e.g. GovReport)
        if not question or str(question).strip() == "":
            question = "Summarize the main points and key details of the document."
        
        # 1. Chunk and encode
        chunks = chunk_text(context)
        chunk_embeddings = retriever_model.encode(chunks, convert_to_tensor=True)
        question_embedding = retriever_model.encode(question, convert_to_tensor=True)
        
        # 2. Score and retrieve Top K
        cos_scores = torch.nn.functional.cosine_similarity(question_embedding, chunk_embeddings)
        top_k = min(15, len(chunks)) # Retrieve ~4500 words total to match ~6900 Token limit!
        top_results = torch.topk(cos_scores, k=top_k)
        
        # 3. Restore chronological order
        top_indices = top_results[1].tolist()
        top_indices.sort()
        
        retrieved_context = "\n...\n".join([chunks[idx] for idx in top_indices])
        
        prompt, _ = build_prompt_fn({**sample, 'context': retrieved_context})

        start_time = time.time()
        response, tokens = generate_answer(model, tokenizer, prompt, max_new_tokens=1024)
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
        
        current_avg_scale = (total_score / (i + 1)) * 100
        print(f"Sample {i+1}/{len(dataset)} | Cur. Score: {score:.3f} | Avg. Score: {current_avg_scale:.2f} | Latency: {latency:.2f}s")
        
        # Incremental save
        output_dict = {
            "dataset": dataset_name,
            "model": model_id,
            "pipeline": "rag",
            "avg_score": (total_score / (i + 1)) * 100,
            "avg_tokens": (total_tokens / (i + 1)),
            "avg_latency": (total_latency / (i + 1)),
            "samples": results
        }
        with open(out_path, "w") as f:
            json.dump(output_dict, f, indent=4)

    import gc
    del retriever_model
    gc.collect()
    torch.cuda.empty_cache()

    return (total_score / len(dataset)) * 100 if len(dataset) > 0 else 0
