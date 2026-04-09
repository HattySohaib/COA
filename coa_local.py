"""
Chain of Agents (COA) — Local Implementation
=============================================
Adapted from Kaggle notebook for local Windows workstation.
Supports three methods: Vanilla, RAG, and Chain-of-Agents (COA).

Usage:
    python coa_local.py --method vanilla --dataset qasper --num_samples 5
    python coa_local.py --method vanilla --dataset narrativeqa --num_samples 5
"""

import argparse
import re
import string
import collections
import json
import torch
from tqdm.auto import tqdm


# ============================================================
# 1. MODEL LOADING
# ============================================================

def load_quantized_model(model_id="meta-llama/Meta-Llama-3-8B-Instruct"):
    """Load model with 4-bit quantization via bitsandbytes."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    print(f"Loading {model_id} in 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    mem_gb = model.get_memory_footprint() / 1e9
    print(f"Model loaded successfully! Memory footprint: {mem_gb:.2f} GB")
    return model, tokenizer


# ============================================================
# 2. DATASET LOADING
# ============================================================

def load_and_prepare_dataset(name="qasper", num_samples=50):
    """Load Qasper or NarrativeQA set and sample."""
    from datasets import load_dataset

    if name == "qasper":
        print(f"Downloading the Qasper validation set...")
        dataset = load_dataset(
            "allenai/qasper",
            split="validation",
            trust_remote_code=True,
        )
    elif name == "narrativeqa":
        print(f"Downloading NarrativeQA from LongBench...")
        dataset = load_dataset(
            "THUDM/LongBench",
            "narrativeqa",
            split="test",
            trust_remote_code=True,
        )
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    if num_samples is not None and num_samples > 0 and num_samples < len(dataset):
        sandbox_dataset = dataset.shuffle(seed=42).select(range(num_samples))
    else:
        sandbox_dataset = dataset.shuffle(seed=42)
    print(f"Successfully loaded {len(sandbox_dataset)} samples.\n")
    return sandbox_dataset


def format_context(sample):
    """Flatten structured data into a readable string context."""
    if "full_text" in sample:  # Qasper format
        title = sample.get("title", "Unknown Title")
        abstract = sample.get("abstract", "No abstract provided.")

        formatted_text = f"Document Title: {title}\n"
        formatted_text += f"Abstract: {abstract}\n\n"

        full_text = sample.get("full_text", {})
        if full_text and "section_name" in full_text and "paragraphs" in full_text:
            for sec_name, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
                if not sec_name:
                    sec_name = "Unnamed Section"
                formatted_text += f"--- Section: {sec_name} ---\n"
                if isinstance(paragraphs, list):
                    formatted_text += "\n".join(paragraphs) + "\n\n"

        return formatted_text.strip()
    elif "context" in sample:  # LongBench format
        return sample["context"].strip()
    return ""


def extract_gold_answer(sample, question_idx=0):
    """Extract question and gold answer."""
    if "qas" in sample:  # Qasper
        question = sample["qas"]["question"][question_idx]
        answer_data = sample["qas"]["answers"][question_idx]["answer"][0]

        if answer_data["free_form_answer"]:
            gold = answer_data["free_form_answer"]
        elif answer_data["extractive_spans"]:
            gold = ", ".join(answer_data["extractive_spans"])
        else:
            gold = str(answer_data["yes_no"])
        return question, gold
    elif "input" in sample and "answers" in sample:  # LongBench NarrativeQA
        return sample["input"], sample["answers"][0]
    return "", ""


# ============================================================
# 3. PROMPT TEMPLATES
# ============================================================

def build_vanilla_prompt(context, question):
    """Vanilla full-context prompt with CoT."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI assistant. Answer the question based strictly on the provided text context.
Output your response as a valid JSON object with exactly two keys: "reasoning" and "final_answer".

CRITICAL INSTRUCTIONS:
1. Provide step-by-step "reasoning", citing specific evidence from the text.
2. Provide the "final_answer". This MUST be concise (a short phrase or entity).
3. If it is a yes/no question, "final_answer" MUST be strictly "True" or "False".
4. If the answer is not found in the text, "final_answer" MUST be exactly "None".
5. Do NOT output any text outside the JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>
Document Text:
{context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{
  "reasoning": \""""


def build_rag_prompt(rag_context, question):
    """RAG prompt with retrieved excerpts and CoT."""
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert AI assistant. Answer the question based strictly on the provided text excerpts.
Output your response as a valid JSON object with exactly two keys: "reasoning" and "final_answer".

CRITICAL INSTRUCTIONS:
1. Provide step-by-step "reasoning", citing specific evidence from the excerpts.
2. Provide the "final_answer". This MUST be concise (a short phrase or entity).
3. If it is a yes/no question, "final_answer" MUST be strictly "True" or "False".
4. If the answer is not found in the text, "final_answer" MUST be exactly "None".
5. Do NOT output any text outside the JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>
Text Excerpts:
{rag_context}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{
  "reasoning": \""""


def build_coa_worker_prompt(chunk, question, previous_worker_msg):
    """COA worker agent prompt."""
    msg_section = (
        f"Message from previous worker:\n{previous_worker_msg}\n"
        if previous_worker_msg
        else "You are the first worker. There is no previous message.\n"
    )
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a Worker Agent in a team analyzing a long document. Your task is to help answer a question by extracting relevant information from your assigned text chunk.
You will receive a message from the previous worker and your text chunk. 
Analyze the chunk and output a concise message containing any clues relevant to the question. If the chunk does not contain relevant information, state that.
Pass along the most important findings from the previous worker's message as well, synthesizing them with your findings.<|eot_id|><|start_header_id|>user<|end_header_id|>
Question to eventually answer: {question}

{msg_section}
Your Text Chunk:
{chunk}

Output your explicit message for the next agent.<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def build_coa_manager_prompt(worker_messages, question):
    """COA manager agent prompt."""
    messages_str = "\n".join([f"Worker {k+1}: {msg}" for k, msg in enumerate(worker_messages)])
    return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are the Manager Agent. Your team of workers has processed a long document chunk-by-chunk and gathered the following messages to help you answer a question.
Synthesize the workers' messages to answer the question.
Output your response as a valid JSON object with exactly two keys: "reasoning" and "final_answer".
1. "reasoning" MUST provide step-by-step logic based on the worker messages.
2. "final_answer" MUST be the exact, concise phrase. 
3. If it is a yes/no question, output strictly "True" or "False". 
4. If the answer is not found, output exactly "None".
Do not include any other text outside the JSON.<|eot_id|><|start_header_id|>user<|end_header_id|>
Workers' Messages:
{messages_str}

Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{{
  "reasoning": \""""


# ============================================================
# 4. METRICS & PARSING
# ============================================================

def normalize_answer(s):
    """Normalize answer string for F1 computation."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))


def compute_f1(a_gold, a_pred):
    """Compute token-level F1 between gold and predicted answers."""
    gold_toks = normalize_answer(a_gold).split()
    pred_toks = normalize_answer(a_pred).split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    return (2 * precision * recall) / (precision + recall)


def extract_json_from_response(text):
    """Extract JSON from model response text."""
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return json.loads(match.group(0)) if match else json.loads(text)
    except json.JSONDecodeError:
        return {"final_answer": ""}


def truncate_context(tokenizer, context, max_context_tokens=3200):
    """Truncate context text to fit within token budget, preserving prompt structure.
    
    We reserve ~800 tokens for system instructions, question, and generation,
    and give the rest to the context.
    """
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    if len(context_ids) > max_context_tokens:
        context_ids = context_ids[:max_context_tokens]
        context = tokenizer.decode(context_ids, skip_special_tokens=True)
    return context


def generate_answer(model, tokenizer, prompt, max_new_tokens=80):
    """Run inference and return raw response text and input token count."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=8192
    ).to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            max_length=None,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][input_length:]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True), input_length


def parse_final_answer(response_text, is_coa=False):
    """Parse model response into a final answer string.
    
    The prompt is structured so the model continues after either '{\\n  "reasoning": "' 
    or '{"final_answer": "' so we assemble and try JSON parsing first.
    """
    if is_coa:
        full_response = '{"final_answer": "' + response_text
    else:
        full_response = '{\n  "reasoning": "' + response_text
        
    parsed = extract_json_from_response(full_response)
    answer = parsed.get("final_answer", "")
    
    # If JSON parsing failed, extract the answer directly from the raw text via regex
    if not answer:
        match = re.search(r'"final_answer"\s*:\s*([^,}]+)', response_text, re.IGNORECASE)
        if match:
            raw_val = match.group(1).strip()
            cleaned = re.sub(r'^["\']|["\']$', '', raw_val).strip()
            cleaned = re.sub(r'["\s}]+$', '', cleaned)
            if cleaned:
                answer = cleaned
        elif is_coa:
            cleaned = response_text.split('}')[0].strip()
            cleaned = re.sub(r'^["\']|["\']$', '', cleaned).strip()
            if cleaned:
                answer = cleaned
    
    # Standardize booleans
    if str(answer).lower() == "true":
        answer = "True"
    elif str(answer).lower() == "false":
        answer = "False"
        
    return str(answer) if answer is not None else ""


# ============================================================
# 5. PIPELINE: VANILLA
# ============================================================

def run_vanilla(model, tokenizer, dataset):
    """Vanilla baseline — full context truncated to model max_length."""
    print(f"\n{'='*50}")
    print(f"VANILLA BASELINE (N={len(dataset)})")
    print(f"{'='*50}")

    total_f1 = 0
    total_tokens = 0
    results = []

    for i, sample in enumerate(tqdm(dataset, desc="Vanilla")):
        question, gold_answer = extract_gold_answer(sample)
        context = format_context(sample)
        context = truncate_context(tokenizer, context, max_context_tokens=3200)
        prompt = build_vanilla_prompt(context, question)

        response, token_count = generate_answer(model, tokenizer, prompt, max_new_tokens=300)
        total_tokens += token_count
        predicted = parse_final_answer(response)

        f1 = compute_f1(gold_answer, predicted)
        total_f1 += f1

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "f1": f1,
            "prompt_tokens": token_count,
        })

        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"  Q: {question}")
            print(f"  Gold: {gold_answer}")
            print(f"  Pred: {predicted}")
            print(f"  F1: {f1:.2f}")
            print(f"  Tokens: {token_count}")
            print(f"  RAW: {response[:150]}")

    avg_f1 = (total_f1 / len(dataset)) * 100
    avg_tokens = total_tokens / len(dataset)
    print(f"\n{'='*50}")
    print(f"VANILLA RESULTS: F1 = {avg_f1:.2f}% | Avg Tokens/Sample: {avg_tokens:.1f}")
    print(f"{'='*50}")
    return results, avg_f1, avg_tokens


# ============================================================
# 6. PIPELINE: RAG
# ============================================================

def load_retriever():
    """Load the BGE-Small sentence-transformer retriever."""
    from sentence_transformers import SentenceTransformer
    print("Loading BGE-Small retriever model...")
    retriever = SentenceTransformer("BAAI/bge-small-en-v1.5")
    print("Retriever loaded.\n")
    return retriever


def retrieve_top_k_chunks(retriever, context_string, question, k=3):
    """Retrieve top-k relevant chunks using cosine similarity."""
    from sentence_transformers import util

    raw_chunks = context_string.split("\n\n")
    chunks = [c.strip() for c in raw_chunks if len(c.strip()) > 50]

    if len(chunks) <= k:
        return context_string

    question_embedding = retriever.encode(question, convert_to_tensor=True)
    chunk_embeddings = retriever.encode(chunks, convert_to_tensor=True)

    cos_scores = util.cos_sim(question_embedding, chunk_embeddings)[0]
    top_results = torch.topk(cos_scores, k=k)

    retrieved = "--- RETRIEVED EXCERPTS ---\n\n"
    for idx in top_results[1]:
        retrieved += chunks[idx] + "\n\n"
    return retrieved.strip()


def run_rag(model, tokenizer, dataset):
    """RAG baseline — retrieve top-3 chunks, then answer."""
    retriever = load_retriever()

    print(f"\n{'='*50}")
    print(f"RAG BASELINE (Top-3) (N={len(dataset)})")
    print(f"{'='*50}")

    total_f1 = 0
    total_tokens = 0
    results = []

    for i, sample in enumerate(tqdm(dataset, desc="RAG")):
        question, gold_answer = extract_gold_answer(sample)
        context = format_context(sample)

        rag_context = retrieve_top_k_chunks(retriever, context, question, k=3)
        prompt = build_rag_prompt(rag_context, question)

        response, token_count = generate_answer(model, tokenizer, prompt, max_new_tokens=300)
        total_tokens += token_count
        predicted = parse_final_answer(response)

        f1 = compute_f1(gold_answer, predicted)
        total_f1 += f1

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "f1": f1,
            "prompt_tokens": token_count,
        })

        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"  Q: {question}")
            print(f"  Gold: {gold_answer}")
            print(f"  Pred: {predicted}")
            print(f"  F1: {f1:.2f}")
            print(f"  Tokens: {token_count}")

    avg_f1 = (total_f1 / len(dataset)) * 100
    avg_tokens = total_tokens / len(dataset)
    print(f"\n{'='*50}")
    print(f"RAG RESULTS: F1 = {avg_f1:.2f}% | Avg Tokens/Sample: {avg_tokens:.1f}")
    print(f"{'='*50}")
    return results, avg_f1, avg_tokens


# ============================================================
# 7. PIPELINE: CHAIN OF AGENTS (COA)
# ============================================================

def run_coa(model, tokenizer, dataset):
    """Chain of Agents — sequential workers + manager."""
    print(f"\n{'='*50}")
    print(f"CHAIN OF AGENTS (N={len(dataset)})")
    print(f"{'='*50}")

    total_f1 = 0
    total_tokens = 0
    results = []

    for i, sample in enumerate(tqdm(dataset, desc="COA")):
        question, gold_answer = extract_gold_answer(sample)
        context = format_context(sample)

        # 1. Chunk the paper (~1200 words per chunk)
        raw_paragraphs = context.split("\n\n")
        chunks = []
        current_chunk = ""
        for p in raw_paragraphs:
            if len(current_chunk.split()) + len(p.split()) < 1200:
                current_chunk += p + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = p + "\n\n"
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # 2. Worker agent loop — sequential note accumulation
        worker_messages = []
        previous_msg = ""
        sample_tokens = 0
        print(f"\nProcessing Sample {i+1} with {len(chunks)} chunks...")
        for j, chunk in enumerate(chunks):
            print(f"  -> Worker {j+1}/{len(chunks)} processing chunk...", end="\r")
            worker_prompt = build_coa_worker_prompt(chunk, question, previous_msg)
            inputs = tokenizer(
                worker_prompt, return_tensors="pt", truncation=True, max_length=2000
            ).to(model.device)
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
            generated_msg = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            worker_messages.append(generated_msg)
            previous_msg = generated_msg

        # 3. Manager agent — synthesize final answer
        manager_prompt = build_coa_manager_prompt(worker_messages, question)
        response, manager_tokens = generate_answer(model, tokenizer, manager_prompt, max_new_tokens=250)
        sample_tokens += manager_tokens
        total_tokens += sample_tokens

        predicted = parse_final_answer(response, is_coa=False)

        f1 = compute_f1(gold_answer, predicted)
        total_f1 += f1

        results.append({
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "f1": f1,
            "prompt_tokens": sample_tokens,
        })

        if i < 3:
            print(f"\n--- Example {i+1} ---")
            print(f"  Q: {question}")
            print(f"  Gold: {gold_answer}")
            print(f"  Pred: {predicted}")
            print(f"  F1: {f1:.2f}")
            print(f"  Tokens: {sample_tokens}")
            print(f"  Final Worker Msg snippet: {previous_msg[:200]}...")

    avg_f1 = (total_f1 / len(dataset)) * 100
    avg_tokens = total_tokens / len(dataset)
    print(f"\n{'='*50}")
    print(f"COA RESULTS: F1 = {avg_f1:.2f}% | Avg Tokens/Sample: {avg_tokens:.1f}")
    print(f"{'='*50}")
    return results, avg_f1, avg_tokens


# ============================================================
# 8. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Chain of Agents — Local Runner")
    parser.add_argument(
        "--method",
        type=str,
        default="vanilla",
        choices=["vanilla", "rag", "coa", "all"],
        help="Which pipeline to run (default: vanilla)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="qasper",
        choices=["qasper", "narrativeqa"],
        help="Dataset to evaluate (default: qasper)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of samples to evaluate (default: 5)",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="HuggingFace model ID",
    )
    args = parser.parse_args()

    # Load model
    model, tokenizer = load_quantized_model(args.model_id)

    # Load dataset
    dataset = load_and_prepare_dataset(args.dataset, args.num_samples)

    # Run selected method(s)
    all_results = {}

    if args.method in ("vanilla", "all"):
        results, f1, avg_toks = run_vanilla(model, tokenizer, dataset)
        all_results["vanilla"] = {"results": results, "avg_f1": f1, "avg_tokens": avg_toks}
        torch.cuda.empty_cache()

    if args.method in ("rag", "all"):
        results, f1, avg_toks = run_rag(model, tokenizer, dataset)
        all_results["rag"] = {"results": results, "avg_f1": f1, "avg_tokens": avg_toks}
        torch.cuda.empty_cache()

    if args.method in ("coa", "all"):
        results, f1, avg_toks = run_coa(model, tokenizer, dataset)
        all_results["coa"] = {"results": results, "avg_f1": f1, "avg_tokens": avg_toks}
        torch.cuda.empty_cache()

    # Summary
    if len(all_results) > 0:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for method, data in all_results.items():
            print(f"  {method.upper():>10}: F1 = {data['avg_f1']:.2f}% | Avg Tokens: {data['avg_tokens']:.1f}")

    # Save results
    output_file = f"results_{args.dataset}_{args.method}_n{args.num_samples}.json"
    with open(output_file, "w") as f:
        # Convert results to serializable format
        save_data = {}
        for method, data in all_results.items():
            save_data[method] = {
                "avg_f1": data["avg_f1"],
                "results": data["results"],
            }
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
