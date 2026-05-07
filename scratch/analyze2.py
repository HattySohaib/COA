import json, os, glob

for base in ["results/Qwen-Qwen2.5-7B-Instruct 2", "results/meta-llama-Meta-Llama-3-8B-Instruct 2", "results/google-gemma-4-e4b-it 2"]:
    print(f"\n=== {base} ===")
    for path in glob.glob(os.path.join(base, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(path))
            model = d.get("model","?")
            pipe = d.get("pipeline","?")
            ds = d.get("dataset","?")
            score = d.get("avg_score", 0)
            tokens = d.get("avg_tokens", 0)
            n = len(d.get("samples",[]))
            print(f"  {pipe:8s} | {ds:15s} | Score: {score:7.2f} | Tokens: {tokens:8.0f} | N={n}")
        except Exception as e:
            print(f"  Error: {path}: {e}")
