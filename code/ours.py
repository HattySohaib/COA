import os
import json
import time
import math
import collections
import numpy as np

from utils import generate_answer, parse_final_answer

# ── Global defaults (overridden per-task by TASK_CONFIGS) ──
WORKER_MAX_INPUT_TOKENS = 6912
WORKER_EVIDENCE_MAX_TOKENS = 700
BLOCK_SIZE = 500          # Smaller blocks → more precise BM25 retrieval
CHUNK_TOKEN_LIMIT = 5800

# ── Task-adaptive configuration ──
# Each task gets its own retrieval aggressiveness, chunk budget, and generation limits.
TASK_CONFIGS = {
    # QA: very focused retrieval, minimal COA overhead
    "HotpotQA":    {"top_ratio": 0.20, "max_blocks": 8,  "add_neighbors": False, "max_chunks": 2, "worker_tokens": 256, "manager_tokens": 512,  "manager_evidence": 3000, "type": "qa"},
    "Qasper":      {"top_ratio": 0.20, "max_blocks": 8,  "add_neighbors": False, "max_chunks": 2, "worker_tokens": 256, "manager_tokens": 512,  "manager_evidence": 3000, "type": "qa"},
    # Narrative QA: moderate retrieval
    "NarrativeQA": {"top_ratio": 0.15, "max_blocks": 10, "add_neighbors": True,  "max_chunks": 3, "worker_tokens": 320, "manager_tokens": 640,  "manager_evidence": 3600, "type": "qa"},
    # Summarization: broad coverage
    "GovReport":   {"top_ratio": 0.30, "max_blocks": 14, "add_neighbors": True,  "max_chunks": 4, "worker_tokens": 512, "manager_tokens": 1024, "manager_evidence": 4000, "type": "summary"},
    # MCQ
    "QuALITY":     {"top_ratio": 0.25, "max_blocks": 12, "add_neighbors": True,  "max_chunks": 3, "worker_tokens": 384, "manager_tokens": 512,  "manager_evidence": 3600, "type": "mcq"},
}

# Fallback config for unknown datasets
DEFAULT_CONFIG = {"top_ratio": 0.25, "max_blocks": 10, "add_neighbors": True, "max_chunks": 3, "worker_tokens": 320, "manager_tokens": 640, "manager_evidence": 3600, "type": "qa"}

TASK_REQUIREMENTS = {
    "HotpotQA": "Answer the question based on the given passages. Only give me the answer and do not output any other words.",
    "NarrativeQA": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question as concisely as you can, using a single phrase if possible. Do not provide any explanation.",
    "GovReport": "You are given a report by a government agency. Write a one-page summary of the report.",
    "QuALITY": "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D).",
    "Qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation."
}

# ═══════════════════════  BM25  ═══════════════════════

class BM25Okapi:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.avgdl = sum(float(len(doc)) for doc in corpus) / max(1, self.corpus_size)
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []

        nd = {}
        for document in corpus:
            self.doc_len.append(len(document))
            frequencies = collections.Counter(document)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                nd[word] = nd.get(word, 0) + 1
        
        for word, freq in nd.items():
            self.idf[word] = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5) + 1.0

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        if not query:
            return scores
        for q in query:
            q_idf = self.idf.get(q, 0)
            if q_idf <= 0:
                continue
            for i, document in enumerate(self.doc_freqs):
                freq = document.get(q, 0)
                if freq == 0: continue
                numerator = freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                scores[i] += q_idf * (numerator / denominator)
        return scores

# ═══════════════════════  Helpers  ═══════════════════════

def tokenize_basic(text):
    return text.lower().split()

def clip_by_tokens(text, tokenizer, max_tokens):
    if not text:
        return ""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[-max_tokens:], skip_special_tokens=True)

def normalize_query(raw_query, max_chars=2000):
    """Extract the actual question from potentially templated prompts."""
    if not raw_query:
        return ""

    q = str(raw_query).strip()
    markers = [
        "\n\nQuestion and Possible Answers:\n",
        "\n\nQuestion:\n",
        "\nQuestion:\n",
        "Question and Possible Answers:",
        "Question:"
    ]
    for marker in markers:
        if marker in q:
            q = q.split(marker, 1)[1].strip()
            break

    if len(q) > max_chars:
        q = q[-max_chars:]
    return q

# ═══════════════════════  Chunk building  ═══════════════════════

def build_chunks_from_selected_indices(blocks, selected_indices, tokenizer, chunk_token_limit, max_chunks):
    """Pack selected blocks into at most max_chunks chunks."""
    if not selected_indices:
        return [" ".join(blocks)] if blocks else []

    selected = sorted(set(idx for idx in selected_indices if 0 <= idx < len(blocks)))
    chunks = []
    current = []
    current_len = 0

    for idx in selected:
        block_text = blocks[idx]
        block_len = len(tokenizer.encode(block_text, add_special_tokens=False))
        if current and current_len + block_len > chunk_token_limit:
            chunks.append(" ".join(current))
            current = [block_text]
            current_len = block_len
        else:
            current.append(block_text)
            current_len += block_len

    if current:
        chunks.append(" ".join(current))

    # Merge down to max_chunks if we exceeded
    if len(chunks) > max_chunks:
        merged = []
        group_size = int(math.ceil(len(chunks) / float(max_chunks)))
        for i in range(0, len(chunks), group_size):
            merged.append(" ".join(chunks[i:i + group_size]))
        return merged

    return chunks

def build_retrieved_chunks(query, context_text, tokenizer, config):
    """BM25-based retrieval with task-adaptive filtering."""
    tokens = tokenizer.encode(context_text, add_special_tokens=False)
    blocks = []
    for i in range(0, len(tokens), BLOCK_SIZE):
        block_tokens = tokens[i:i + BLOCK_SIZE]
        blocks.append(tokenizer.decode(block_tokens, skip_special_tokens=True))

    if not blocks:
        return [context_text]

    top_ratio = config["top_ratio"]
    max_blocks = config["max_blocks"]
    add_neighbors = config["add_neighbors"]
    max_chunks = config["max_chunks"]

    tokenized_blocks = [tokenize_basic(b) for b in blocks]
    bm25 = BM25Okapi(tokenized_blocks)

    if query:
        scores = bm25.get_scores(tokenize_basic(query))
        ranked_indices = list(np.argsort(scores)[::-1])
        
        # ── FIX Bug 1: Only keep blocks with positive BM25 score ──
        ranked_indices = [idx for idx in ranked_indices if scores[idx] > 0]
        
        keep_k = max(1, int(len(blocks) * top_ratio))
        keep_k = min(keep_k, max_blocks, len(ranked_indices))
        seed_indices = ranked_indices[:keep_k]
    else:
        # Query-agnostic fallback: evenly spaced blocks for broad coverage
        scores = np.zeros(len(blocks))
        keep_k = min(max(4, int(len(blocks) * top_ratio)), max_blocks)
        if keep_k >= len(blocks):
            seed_indices = list(range(len(blocks)))
        else:
            step = (len(blocks) - 1) / float(keep_k - 1)
            seed_indices = sorted(set(int(round(i * step)) for i in range(keep_k)))

    # ── Conditional neighbor expansion ──
    selected = set(seed_indices)
    if add_neighbors:
        for idx in list(seed_indices):
            if idx - 1 >= 0:
                selected.add(idx - 1)
            if idx + 1 < len(blocks):
                selected.add(idx + 1)

    # ── FIX Bug 1: Hard cap AFTER neighbor expansion ──
    if len(selected) > max_blocks:
        selected = sorted(selected, key=lambda i: scores[i], reverse=True)[:max_blocks]

    return build_chunks_from_selected_indices(blocks, selected, tokenizer, CHUNK_TOKEN_LIMIT, max_chunks)

# ═══════════════════════  Prompts  ═══════════════════════

def custom_worker_prompt(query, task_req, chunk, previous_msg):
    anti_filler = "Write plain text only. No headings, no markdown, no filler phrases."
    normalized_query = query if query else "No explicit question provided."
    previous_msg = previous_msg if previous_msg and previous_msg != "None" else "None"

    return f"""Task Requirement: {task_req}

Question (if available):
{normalized_query}

Current Document Chunk:
{chunk}

Previously Extracted Evidence:
{previous_msg}

Update the evidence using this chunk. Keep concrete entities, dates, numbers, causal links, and key findings. Preserve only high-signal details useful for completing the task requirement. Use at most 8 concise bullet points. {anti_filler}"""

def custom_manager_prompt(query, task_req, evidence_text, task_type="qa"):
    """Task-aware manager prompt — no more MCQ instructions bleeding into QA tasks."""
    normalized_query = query if query else "No explicit question provided."
    
    # ── FIX Bug 5: Task-specific output format instructions ──
    if task_type == "mcq":
        format_instr = "Output ONLY the single letter (A, B, C, or D) of the best answer. No explanation."
    elif task_type == "summary":
        format_instr = "Write a comprehensive summary based on the consolidated evidence. No filler."
    else:
        format_instr = "Answer directly and concisely. Do not explain your reasoning."

    return f"""Task Requirement: {task_req}

Question (if available):
{normalized_query}

Consolidated Evidence:
{evidence_text}

{format_instr}"""

# ═══════════════════════  Evidence aggregation  ═══════════════════════

def aggregate_worker_summaries(worker_summaries, tokenizer, max_tokens):
    """Merge worker outputs into a single evidence string for the manager."""
    if not worker_summaries:
        return ""
    if len(worker_summaries) == 1:
        return clip_by_tokens(worker_summaries[0].strip(), tokenizer, max_tokens)
    merged = []
    for i, summary in enumerate(worker_summaries, start=1):
        if summary and summary.strip():
            merged.append(f"[Evidence {i}] {summary.strip()}")
    evidence = "\n\n".join(merged)
    return clip_by_tokens(evidence, tokenizer, max_tokens)

# ═══════════════════════  Main pipeline  ═══════════════════════

def run_ours(model, tokenizer, dataset, dataset_name, get_context_fn, 
             build_worker_prompt_fn, build_manager_prompt_fn, metric_fn, model_id):
    
    config = TASK_CONFIGS.get(dataset_name, DEFAULT_CONFIG)
    task_type = config["type"]
    worker_max_new = config["worker_tokens"]
    manager_max_new = config["manager_tokens"]
    manager_evidence_max = config["manager_evidence"]
    
    print(f"\n[{dataset_name}] Running OURS (BM25 + Adaptive COA) Pipeline...")
    print(f"  Config: top_ratio={config['top_ratio']}, max_blocks={config['max_blocks']}, "
          f"neighbors={config['add_neighbors']}, max_chunks={config['max_chunks']}, type={task_type}")
    
    total_score = 0
    total_tokens = 0
    total_latency = 0
    results = []

    safe_model_id = model_id.replace("/", "-")
    out_dir = os.path.join("results", safe_model_id, "ours")
    os.makedirs(out_dir, exist_ok=True)
    
    suffix = "_TEST" if len(dataset) < 20 else ""
    out_path = os.path.join(out_dir, f"results_{dataset_name}{suffix}.json")
    
    task_req = TASK_REQUIREMENTS.get(dataset_name, "Answer the question based on the document.")

    for i, sample in enumerate(dataset):
        context_text, gold_answer = get_context_fn(sample)
        query = normalize_query(sample.get('input', ''))
        raw_token_count = len(tokenizer.encode(context_text, add_special_tokens=False)) if context_text else 0
        raw_block_count = int(math.ceil(raw_token_count / float(BLOCK_SIZE))) if context_text else 0

        adaptive_chunks = build_retrieved_chunks(query, context_text, tokenizer, config)

        start_time = time.time()
        tokens_used = 0

        # ── FIX Bug 3: Single-pass shortcut — skip COA when retrieval fits in one call ──
        if len(adaptive_chunks) <= 1:
            context_for_manager = adaptive_chunks[0] if adaptive_chunks else ""
            evidence = clip_by_tokens(context_for_manager, tokenizer, manager_evidence_max)
            manager_prompt = custom_manager_prompt(query, task_req, evidence, task_type)
            manager_response, m_cnt = generate_answer(model, tokenizer, manager_prompt, max_new_tokens=manager_max_new)
            tokens_used = m_cnt
            predicted = parse_final_answer(manager_response)
        else:
            # ── Multi-chunk: run worker chain ──
            worker_summaries = []
            previous_msg = "None"

            for chunk_idx, chunk in enumerate(adaptive_chunks):
                previous_for_worker = clip_by_tokens(previous_msg, tokenizer, WORKER_EVIDENCE_MAX_TOKENS)
                worker_prompt = custom_worker_prompt(query, task_req, chunk, previous_for_worker)
                summary, t_cnt = generate_answer(model, tokenizer, worker_prompt, max_new_tokens=worker_max_new)
                tokens_used += t_cnt
                previous_msg = summary
                worker_summaries.append(summary)

            aggregated_evidence = aggregate_worker_summaries(worker_summaries, tokenizer, manager_evidence_max)
            manager_prompt = custom_manager_prompt(query, task_req, aggregated_evidence, task_type)
            manager_response, m_cnt = generate_answer(model, tokenizer, manager_prompt, max_new_tokens=manager_max_new)
            tokens_used += m_cnt
            predicted = parse_final_answer(manager_response)

        latency = time.time() - start_time
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
            "chunks_processed": len(adaptive_chunks),
            "raw_blocks": raw_block_count,
            "query_chars": len(query)
        })
        
        current_avg_scale = (total_score / (i + 1)) * 100
        
        print(f"Sample {i+1}/{len(dataset)} | Score: {score_val:.3f} | Avg: {current_avg_scale:.2f} | "
              f"Lat: {latency:.1f}s | Blocks: {raw_block_count} | Chunks: {len(adaptive_chunks)} | Tok: {tokens_used}")
        
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
