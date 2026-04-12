import sys
import os

# Ensure datasets can be loaded
try:
    from datasets import load_dataset
except ImportError:
    print("datasets library not installed.")
    sys.exit(1)

datasets_to_check = [
    ("THUDM/LongBench", "qasper"),
    ("THUDM/LongBench", "narrativeqa"),
    ("THUDM/LongBench", "hotpotqa"),
    ("THUDM/LongBench", "musique"),
    ("THUDM/LongBench", "qmsum"),
    ("THUDM/LongBench", "gov_report"),
    ("THUDM/LongBench", "repobench-p"),
    ("tau/scrolls", "quality"),
    ("tau/scrolls", "booksum")
]

for repo, name in datasets_to_check:
    print(f"\n{'='*40}\nDataset: {repo} / {name}\n{'='*40}")
    try:
        if repo == "tau/scrolls":
            ds = load_dataset(repo, name, trust_remote_code=True, split="validation")
        else:
            ds = load_dataset(repo, name, split="test")
        
        sample = ds[0]
        print(f"Keys: {list(sample.keys())}")
        
        if 'input' in sample:
            print(f"Input snippet: {str(sample['input'])[:100]}...")
        if 'context' in sample:
            print(f"Context snippet: {str(sample['context'])[:100]}...")
        if 'answers' in sample:
            print(f"Answers: {sample['answers']}")
        elif 'output' in sample:
            print(f"Output snippet: {str(sample['output'])[:100]}...")
    except Exception as e:
        print(f"Failed to load {name}: {e}")
