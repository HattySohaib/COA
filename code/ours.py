import os
import json
import time
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_answer, parse_final_answer

# Paper config for COA module scaling
WORKER_MAX_INPUT_TOKENS = 6912   
WORKER_MAX_NEW_TOKENS = 1024
MANAGER_MAX_NEW_TOKENS = 2048
CHUNK_SIZE = 5500 

def get_top_30_percent_sentences(query, context, tokenizer):
    """
    RAG Step: Extracts the top 30% of sentences from the context 
    that are most semantically similar to the query using TF-IDF (vectorless but with strong IDF weights).
    """
    sentences = nltk.sent_tokenize(context)
    if not sentences:
        return context
        
    k = max(1, int(len(sentences) * 0.30))
    if k == len(sentences):
        return context

    try:
        # TF-IDF gives us much "stronger semantic weighing" natively than pure term frequency
        # Stop-words are removed to weight the rare/important terms cleanly
        vectorizer = TfidfVectorizer(stop_words='english')
        docs = sentences + [query]
        tfidf_matrix = vectorizer.fit_transform(docs)
        
        query_vec = tfidf_matrix[-1]
        sentence_vecs = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vec, sentence_vecs).flatten()
    except Exception as e:
        print(f"TF-IDF Semantic Scoring failed ({e}), falling back to all sentences.")
        return context

    # Retrieve top k indices
    top_indices = similarities.argsort()[-k:][::-1]
    
    # Sort chronologically to preserve the narrative/document flow for the COA workers
    top_indices = sorted(top_indices)
    
    reduced_context = " ".join([sentences[i] for i in top_indices])
    return reduced_context

def run_ours(model, tokenizer, dataset, dataset_name, get_context_fn, 
             build_worker_prompt_fn, build_manager_prompt_fn, metric_fn, model_id):
    
    print(f"\n[{dataset_name}] Running OURS (Vectorless Semantic RAG 30% -> COA) Pipeline...")
    total_score = 0
    total_tokens = 0
    total_latency = 0
    results = []

    safe_model_id = model_id.replace("/", "-")
    out_dir = os.path.join("results", safe_model_id, "ours")
    os.makedirs(out_dir, exist_ok=True)
    
    suffix = "_TEST" if len(dataset) < 20 else ""
    out_path = os.path.join(out_dir, f"results_{dataset_name}{suffix}.json")

    for i, sample in enumerate(dataset):
        context_text, gold_answer = get_context_fn(sample)
        query = sample.get('input', '')
        
        # --- 1. RAG STEP ---
        # Reduce the context dynamically to the top 30% semantic sentences
        reduced_context = get_top_30_percent_sentences(query, context_text, tokenizer)
        
        # --- 2. COA STEP ---
        tokens = tokenizer.encode(reduced_context, add_special_tokens=False)
        chunks = []
        for j in range(0, len(tokens), CHUNK_SIZE):
            chunk_tokens = tokens[j:j+CHUNK_SIZE]
            chunks.append(tokenizer.decode(chunk_tokens, skip_special_tokens=True))

        worker_summaries = []
        previous_msg = "None"
        chunk_tokens_count = 0
        chunk_latency = 0

        for chunk_idx, chunk in enumerate(chunks):
            worker_prompt = build_worker_prompt_fn(sample, chunk, previous_msg)
            start_t = time.time()
            summary, t_cnt = generate_answer(model, tokenizer, worker_prompt, max_new_tokens=WORKER_MAX_NEW_TOKENS)
            chunk_latency += (time.time() - start_t)
            chunk_tokens_count += t_cnt
            previous_msg = summary
            worker_summaries.append(summary)

        final_worker_msg = worker_summaries[-1] if worker_summaries else ""
        manager_prompt = build_manager_prompt_fn(sample, final_worker_msg)
        
        start_t = time.time()
        manager_response, m_cnt = generate_answer(model, tokenizer, manager_prompt, max_new_tokens=MANAGER_MAX_NEW_TOKENS)
        latency = chunk_latency + (time.time() - start_t)
        tokens_used = chunk_tokens_count + m_cnt
        
        predicted = parse_final_answer(manager_response)
        score_val = metric_fn(gold_answer, predicted)
        
        total_score += score_val
        total_tokens += tokens_used
        total_latency += latency
        
        results.append({
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "score": score_val,
            "tokens": tokens_used,
            "latency_seconds": latency,
            "chunks_processed": len(chunks)
        })
        
        # Determine scaling for display
        try:
            from utils import compute_f1
            is_f1 = (metric_fn == compute_f1)
        except ImportError:
            is_f1 = False
            
        current_avg_scale = (total_score / (i + 1)) * 100 if not is_f1 else (total_score / (i + 1)) * 100
        
        print(f"Sample {i+1}/{len(dataset)} | Cur. Score: {score_val:.3f} | Avg. Score: {current_avg_scale:.2f} | Latency: {latency:.2f}s | Chunks: {len(chunks)}")
        
        # Incremental save
        output_dict = {
            "dataset": dataset_name,
            "model": model_id,
            "pipeline": "ours",
            "avg_score": current_avg_scale,
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
    print(f"[{dataset_name}] OURS Result: {avg_score:.2f} | Avg Tokens: {avg_tokens:.1f} | Avg Latency: {avg_latency:.2f}s")
    
    return avg_score
