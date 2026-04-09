import torch
import gc
import re
import json
import string
import collections
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import nltk
from rouge_score import rouge_scorer

try:
    nltk.download('punkt')
except:
    pass

def load_model(model_name):
    print(f"\n[utils] Loading model: {model_name}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"[utils] Model loaded successfully! Memory footprint: {model.get_memory_footprint() / 1e9:.2f} GB")
    return model, tokenizer

def cleanup_memory(model, tokenizer):
    print("\n[utils] Cleaning up memory...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("[utils] Memory cleaned.")

def generate_answer(model, tokenizer, prompt_text, max_new_tokens=150):
    messages = [{"role": "user", "content": prompt_text}]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = tokenizer(chat_prompt, return_tensors="pt", truncation=True, max_length=12000).to(model.device)
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
    return tokenizer.decode(generated_tokens, skip_special_tokens=True), input_length + len(generated_tokens)

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    return white_space_fix(remove_articles(remove_punc(str(s).lower())))

def compute_f1(a_gold, a_pred):
    if not isinstance(a_gold, list):
        a_gold = [a_gold]
    best_f1 = 0
    for gold in a_gold:
        gold_toks = normalize_answer(gold).split()
        pred_toks = normalize_answer(a_pred).split()
        common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
        num_same = sum(common.values())
        if len(gold_toks) == 0 or len(pred_toks) == 0:
            f1 = int(gold_toks == pred_toks)
        elif num_same == 0:
            f1 = 0
        else:
            precision = 1.0 * num_same / len(pred_toks)
            recall = 1.0 * num_same / len(gold_toks)
            f1 = (2 * precision * recall) / (precision + recall)
        best_f1 = max(best_f1, f1)
    return best_f1

def compute_rouge(a_gold, a_pred):
    if not isinstance(a_gold, list):
        a_gold = [a_gold]
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    best_rouge = 0
    for gold in a_gold:
        scores = scorer.score(gold, str(a_pred))
        best_rouge = max(best_rouge, scores['rougeL'].fmeasure)
    return best_rouge

def compute_exact_match(a_gold, a_pred):
    if not isinstance(a_gold, list):
        a_gold = [a_gold]
    best_em = 0
    for gold in a_gold:
        if normalize_answer(gold) == normalize_answer(a_pred):
            best_em = 1
    return float(best_em)

def extract_json_from_response(text):
    return text.strip()

def parse_final_answer(response):
    return response.strip()

def chunk_text(text, chunk_size=1200):
    raw_paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""
    for p in raw_paragraphs:
        if len(current_chunk.split()) + len(p.split()) < chunk_size:
            current_chunk += p + "\n\n"
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = p + "\n\n"
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks
