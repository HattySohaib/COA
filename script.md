Ran terminal command: cat requirements.txt

Ran terminal command: find code -name "_.py" -exec grep -Eh "^import |^from " {} + | sort | uniq
Completed with input: find code -name "_.py" -exec grep -Eh "^import |^from " {} + | sort | uniq

Ran terminal command: Select-String -Pattern "^import |^from " -Path code\*.py | Select-Object -ExpandProperty Line | Sort-Object | Get-Unique

Ran terminal command: & "D:\COA\.venv\Scripts\python.exe" -m pip show transformers sentence-transformers rouge_score numpy nltk

Ran terminal command: Add-Content requirements.txt "`nrouge_score>=0.1.2`nnltk>=3.9.0`nnumpy>=2.4.0"

I've updated the requirements.txt file to explicitly include the text analysis dependencies necessary for computing ROUGE and F1 scores (`rouge_score`, `nltk`, and `numpy`).

To correctly set everything up and run the models on a dataset on your other PCs, here are the step-by-step terminal commands you'll need:

### 1. Environment Setup (Run Once Per PC)

After cloning the repository to the new PC, set up the virtual environment:

```powershell
python -m venv .venv
# On Windows:
.venv\Scripts\Activate.ps1
# On Linux/Mac:
# source .venv/bin/activate

# Install PyTorch for CUDA 12.4 (as configured in your file)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install the rest of the dependencies
pip install -r requirements.txt

# Download the NLTK tokenization data needed for Rouge math
python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"
```

### 2. Execution Prompts

Since we packaged the entire pipeline (Vanilla, Chain of Agents, and RAG) into streamlined dataset wrappers, the commands are completely self-contained.

To run the complete benchmark (all 3 models over the 3 modes natively), run the target script from the root directory with the `--mode full` argument.

**For example, to run NarrativeQA:**

```powershell
python scratch/test_narrativeqa.py --mode full
```

**Here are the specific commands for all the datasets you generated:**

```powershell
python scratch/test_booksum.py --mode full
python scratch/test_gov_report.py --mode full
python scratch/test_hotpotqa.py --mode full
python scratch/test_musique.py --mode full
python scratch/test_qasper.py --mode full
python scratch/test_qmsum.py --mode full
python scratch/test_quality.py --mode full
python scratch/test_repobench.py --mode full
```

If you ever want to simply verify the pipeline on the new hardware without starting a true multi-hour benchmark, run any script as `--mode test`, which intelligently isolates a chunk of 5 samples:

```powershell
python scratch/test_booksum.py --mode test
```
