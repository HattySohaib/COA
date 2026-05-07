import os
import sys
# Make sure to append the 'code' dir for running directly
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from datasets import load_dataset
from utils import load_model, cleanup_memory, compute_rouge
from vanilla import run_vanilla
from coa import run_coa
import re

# Source: Extracted directly from the official SCROLLS zero-shot templates (tau/zero_scrolls huggingface dataset builder code).
TASK_REQUIREMENT = "You are provided a story and a multiple-choice question with 4 possible answers (marked by A, B, C, D). Choose the best answer by writing its corresponding letter (either A, B, C, or D)."

def compute_exact_match_letter(gold, predicted):
    """Quality outputs a single letter (A, B, C, or D)."""
    if gold is None or predicted is None:
        return 0.0
    
    gold = gold.strip().upper()
    pred_clean = predicted.strip()
    
    # 1. Look for highly specific indicators and structured answers
    patterns = [
        r'(?i)answer is\s*\**\(*([A-D])\)*\**',
        r'(?i)correct answer is\s*\**\(*([A-D])\)*\**',
        r'(?i)option\s+\**\(*([A-D])\)*\**',
        r'\*\*([A-D])\*\*',  # Bolded letter
        r'\(([A-D])\)',      # Letter in parentheses
        r'(?i)^answer:\s*([A-D])\b', # Start with "Answer: A"
        r'^([A-D])\b'        # Starts exactly with the letter "A"
    ]
    
    for p in patterns:
        match = re.search(p, pred_clean)
        if match:
            return 1.0 if match.group(1).upper() == gold else 0.0
            
    # 2. If no specific indicators found, but response is somewhat short, fallback to the first isolated letter
    # This prevents diluting the score on massive rambling responses where "A" might start a random sentence
    if len(pred_clean) < 100:
        match = re.search(r'\b([A-D])\b', pred_clean.replace(".", "").upper())
        if match:
            return 1.0 if match.group(1) == gold else 0.0
            
    return 0.0

def build_vanilla_prompt(sample):
    # tau/zero_scrolls has exactly formatted the entire input strictly
    # matching the zero-shot prompt including the question and choices.
    prompt = sample['input'] 
    if "Answer:" not in prompt[-20:]: 
        prompt += "\nAnswer:"
    gold_answer = sample['output']
    return prompt, gold_answer

def _split_zero_scrolls_format(input_str):
    # Format is:
    # <Instruction>
    # Story:
    # <Text>
    # Question and Possible Answers:
    # <Question>
    try:
        story_split = input_str.split("\n\nStory:\n", 1)
        rest = story_split[1]
        
        q_split = rest.split("\n\nQuestion and Possible Answers:\n", 1)
        context_text = q_split[0].strip()
        question_block = q_split[1].strip()
        return context_text, question_block
    except IndexError:
        return input_str, "Choose the best letter A, B, C, or D."

def get_context(sample):
    context_text, _ = _split_zero_scrolls_format(sample['input'])
    return context_text, sample['output']

def build_worker_prompt(sample, chunk, previous_msg):
    _, question_block = _split_zero_scrolls_format(sample['input'])
    return f"""Worker Wi:
{chunk}
Here is the summary of the previous source text: {previous_msg}
Question: {question_block}
You need to read current source text and summary of previous source text (if any) and generate a summary to include them both. Later, this summary will be used for other agents to answer the Query, if any. So please write the summary that can include the evidence for answering the Query:"""

def build_manager_prompt(sample, final_worker_msg):
    _, question_block = _split_zero_scrolls_format(sample['input'])
    return f"""Manager M:
{TASK_REQUIREMENT}
The following is an abridged summary of the story:
{final_worker_msg}

Question and Possible Answers:
{question_block}

Answer:"""

