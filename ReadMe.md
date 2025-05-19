# 딥러닝 템플릿 프로젝트

이 프로젝트는 다양한 딥러닝 모델을 개발하고 실험하기 위한 템플릿 구조를 제공합니다.

## 🔧 구성

```
project_name/
├── backbone/                 # 커스텀 백본 (ex. ResNet)
├── tasks/                   # 태스크 별 모델 구조 (classification 등)
├── trainers/                # 학습 로직
├── data/                    # 데이터 로딩 코드 (datasets 디렉터리에서 불러옴)
├── configs/                 # YAML 설정 파일
├── utils/                   # 유틸리티 함수 (로깅 등)
├── datasets/                # 실제 데이터셋 위치 (gitignore로 제외됨)
├── experiments/             # 실험 결과 저장
├── main.py                  # 학습 실행 진입점
├── requirements.txt         # 의존성 목록
└── README.md                # 프로젝트 설명서
```

## 📦 설치
```bash
pip install -r requirements.txt
```

## 🚀 실행
```bash
python main.py
```

## ⚙️ 설정 변경
- `configs/backbone.yaml`: 백본 이름, 학습률, 에폭 수 설정
- `configs/dataset.yaml`: 배치 크기, 클래스 수, 경로 설정

## 🧪 출력
- 학습 로그: `experiments/exp001/train.log`
- config 백업: `experiments/exp001/config.yaml`
- 모델 저장: `experiments/exp001/latest.pth`, `best.pth` (추후 확장)

---

이 템플릿은 확장 가능하며, 다양한 태스크와 모델을 쉽게 통합할 수 있도록 설계되어 있습니다.
