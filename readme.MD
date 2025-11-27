## 환경 설정

```
conda create -n ondevice python=3.12
conda activate ondevice
pip install -r requirement.txt
pip install -e .
```

폴더 설명

```
augmentation: 각 증강기법 클래스가 들어 있습니다.
checkpoint: 모델 체크포인트가 들어 있습니다.
data: 데이터폴더들 (X_train.npy, X_valid.npy, y_train.npy, y_valid.npy)과 dataloader들이 들어 있습니다.
scripts/bash: 여기 training bash script가 있습니다.
search: 증강기법을 search
test_scripts: 코드짜면서 사전 디버깅하는 테스트케이스를 돌려볼수 있습니다.
training:
utils: 부가적 함수들이 들어있습니다.
```