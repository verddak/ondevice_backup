# run_all_tests.py
import subprocess
import sys

tests = [
    "test1_import.py",
    "test2_dataset.py", 
    "test3_dataloader.py",
    "test4_reproducibility.py",
    # "test5_ucr_data.py",  # 실제 데이터 있을 때만, tsai가 깔려 있어야 함
    "test6_load_npy_hbm.py",
    "test6-1_load_npy_2mm#1.py",
    "test6-2_load_npy_5mm#2.py",
    # "test7_data_leakage.py" # train / valid 각각의 data leakage가 없는지 확인
]

print("=" * 50)
print("Running DataLoader Tests")
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