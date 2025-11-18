# run_all_tests.py
import subprocess
import sys

tests = [
    "test1_simple_cnn.py",
    "test2_simple_cnn_npy_hbm.py"
]

print("=" * 50)
print("Running Model Tests")
print("=" * 50)

failed = []
for test in tests:
    print(f"\n▶ Running {test}...")
    result = subprocess.run([sys.executable, test], capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ {test} passed")
        print(result.stdout)
    else:
        print(f"❌ {test} failed")
        print(result.stderr)
        failed.append(test)

print("\n" + "=" * 50)
if not failed:
    print("🎉 All tests passed!")
else:
    print(f"❌ {len(failed)} test(s) failed:")
    for t in failed:
        print(f"  - {t}")
    sys.exit(1)