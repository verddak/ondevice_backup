# tests/run_all_checkpoint_tests.py
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = Path(PROJECT_ROOT) / "checkpoints"  # ✅ Path() 추가

def find_checkpoints():
    """Find all .pt files"""
    checkpoints = []
    for model_dir in CHECKPOINT_DIR.iterdir():
        if model_dir.is_dir():
            checkpoints.extend(model_dir.glob("*.pt"))
    return checkpoints

def main():
    print("="*60)
    print("🔍 Running Checkpoint Tests")
    print("="*60)
    
    # Find checkpoints
    checkpoints = find_checkpoints()
    print(f"\n📁 Found {len(checkpoints)} checkpoints")
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        sys.exit(1)
    
    # Test scripts to run
    tests = [
        "test1_checkpoint_info.py",
        "test2_checkpoint_param.py",
        "test3_checkpoint_load.py",
    ]
    
    failed = []
    
    for checkpoint in checkpoints:
        print(f"\n{'='*60}")
        print(f"Testing: {checkpoint.name}")
        print('='*60)
        
        args_file = checkpoint.parent / f"{checkpoint.stem}_args.json"
        
        for test in tests:
            print(f"\n▶ Running {test}...")
            
            # Run subprocess
            result = subprocess.run(
                [sys.executable, test, str(checkpoint), str(args_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ {test} passed")
                print(result.stdout)
            else:
                print(f"❌ {test} failed")
                print(result.stderr)
                failed.append(f"{checkpoint.name} - {test}")
    
    # Summary
    print("\n" + "="*60)
    if not failed:
        print("🎉 All tests passed!")
    else:
        print(f"❌ {len(failed)} test(s) failed:")
        for f in failed:
            print(f"  - {f}")
        sys.exit(1)

if __name__ == "__main__":
    main()