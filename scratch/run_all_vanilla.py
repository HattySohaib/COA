import os
import sys
import glob
sys.path.insert(0, 'code')
import importlib

def main():
    # Find all run_*.py files in code/
    code_dir = 'code'
    scripts = [f for f in os.listdir(code_dir) if f.startswith('run_') and f.endswith('.py')]
    
    for script in scripts:
        module_name = script[:-3]
        print(f"\n=============================================")
        print(f"Running {module_name}...")
        print(f"=============================================")
        
        try:
            module = importlib.import_module(module_name)
            # Run the main function
            if hasattr(module, 'main'):
                module.main()
            else:
                print(f"No main() found in {module_name}")
        except Exception as e:
            print(f"Failed to run {module_name}: {e}")

if __name__ == '__main__':
    main()
