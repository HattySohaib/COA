import json, os, glob

def scan_dir(base):
    for path in glob.glob(os.path.join(base, "**", "*.json"), recursive=True):
        try:
            d = json.load(open(path))
            model = d.get("model","?")
            pipe = d.get("pipeline","?")
            ds = d.get("dataset","?")
            score = d.get("avg_score", 0)
            tokens = d.get("avg_tokens", 0)
            latency = d.get("avg_latency", 0)
            n = len(d.get("samples",[]))
            print(f"{model:45s} | {pipe:8s} | {ds:15s} | Score: {score:7.2f} | Tokens: {tokens:8.0f} | Lat: {latency:7.1f}s | N={n}")
            
            # For "ours" pipeline, show per-sample detail
            if pipe == "ours":
                for i, s in enumerate(d.get("samples",[])):
                    pred = str(s.get("predicted_answer",""))[:120]
                    gold = str(s.get("gold_answer",""))[:120]
                    print(f"   S{i+1}: score={s['score']:.3f} tok={s['tokens']} chunks={s.get('chunks_processed','?')} blocks={s.get('raw_blocks','?')}")
                    print(f"      PRED: {pred}")
                    print(f"      GOLD: {gold}")
        except Exception as e:
            print(f"Error reading {path}: {e}")

print("=" * 120)
print("FINAL RESULTS (completed benchmarks)")
print("=" * 120)
scan_dir("final_results")

print("\n" + "=" * 120)
print("RESULTS (in-progress / partial)")
print("=" * 120)
scan_dir("results")
