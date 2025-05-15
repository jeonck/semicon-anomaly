# 반도체 공정 이상치 탐지 시스템 (MOMENT 모델 적용)

이 프로젝트는 MOMENT 시계열 파운데이션 모델을 활용하여 반도체 공정 데이터에서 이상치를 탐지하는 웹 애플리케이션입니다. 실시간으로 데이터를 분석하고, 시계열 기반의 이상 패턴을 감지하여 반도체 제조 공정의 품질을 모니터링합니다.

## 주요 기능

- **고급 이상치 탐지**: MOMENT 시계열 파운데이션 모델을 사용한 심층적인 이상치 탐지
- **실시간 시각화**: 시계열 데이터, 분포, 상관관계, 이상치 점수 등 다양한 시각화 제공
- **그룹별 분석**: 배치 번호, 장비 ID 등 범주형 변수에 따른 이상치 패턴 분석
- **모델 비교**: MOMENT 모델과 전통적인 IsolationForest 모델의 성능 비교
- **유연한 설정**: 사용자가 분석할 특성과 이상치 비율을 조정 가능
- **반응형 웹 인터페이스**: 모바일과 데스크톱 모두에서 최적화된 사용자 경험

## 기술 스택

- **백엔드**: Python, Starlette, Uvicorn
- **프론트엔드**: HTML, CSS, JavaScript, Bootstrap 5
- **데이터 처리**: Pandas, NumPy
- **이상치 탐지 모델**: MOMENT 파운데이션 모델, IsolationForest
- **시각화**: Plotly.js
- **환경 관리**: UV (빠른 패키지 매니저, 권장), Python venv, Conda
- **컨테이너화**: Docker, Docker Compose
- **클라우드 배포**: Kubernetes

## 이상치 유형

이 애플리케이션에서 탐지할 수 있는 이상치 유형:

1. **포인트 이상(Point Anomalies)**: 개별 데이터 포인트의 이상
2. **컨텍스트 이상(Contextual Anomalies)**: 특정 컨텍스트에서만 이상으로 간주되는 패턴
3. **집단 이상(Collective Anomalies)**: 개별 데이터 포인트는 정상이지만 집단으로 볼 때 이상인 패턴

## 설치 및 실행 방법

### 환경 설정 (권장: UV 패키지 매니저)

UV는 Rust로 작성된 최신 고성능 Python 패키지 매니저로, 기존 pip보다 10-100배 빠른 설치 속도를 제공합니다. 이 프로젝트에서는 UV를 사용한 가상 환경 설정을 적극 권장합니다.

```bash
# Linux/macOS - UV 패키지 매니저로 환경 설정 및 실행 (강력 권장)
$ ./uv_run.sh

# Windows
$ uv_run.bat
```

#### UV 장점

- **농촉한 설치 속도**: pip보다 최대 100배 빠른 설치 속도
- **재현 가능한 빌드**: 동일한 환경을 정확히 재현할 수 있는 잠금 파일 시스템 지원
- **필요한 의존성만 정확하게 설치**: 의존성 해결 알고리즘 최적화
- **통합 환경 관리**: venv와 pip을 하나로 통합한 사용자 경험

자세한 UV 사용법은 `UV_GUIDE.md` 파일을 참조하세요.

#### 대체 방법 1: Python 가상 환경 (venv)

```bash
# Linux/macOS
$ ./setup.sh

# Windows
$ setup.bat
```

#### 대체 방법 2: Conda/Mamba 환경

```bash
# Conda 환경 생성 및 의존성 설치
$ conda env create -f environment.yml

# 환경 활성화
$ conda activate anomaly_detection
```

### 애플리케이션 실행

```bash
# UV 패키지 매니저로 환경 설정 및 실행 (강력 권장)
$ ./uv_run.sh

# Windows
$ uv_run.bat
```

위 명령어는 UV 패키지 매니저를 사용하여 다음 작업을 자동으로 수행합니다:
1. UV가 설치되지 않은 경우 필요한 설치 자동 실행
2. Python 가상 환경 생성 및 활성화
3. 프로젝트 의존성 패키지 설치 (매우 빠른 속도로 진행)
4. 애플리케이션 실행

영상 전처리 및 이상치 탐지 모델 준비가 완료되면 브라우저에서 다음 주소로 접속할 수 있습니다:

```
http://localhost:5000
```

#### 대체 실행 방법

```bash
# 기존 venv 환경에서 실행
$ ./run_app.sh

# Windows venv 환경
$ run_app.bat

# 또는 직접 명령 실행
$ python app.py
```

### Docker 컨테이너 실행

```bash
# 기본 Docker 이미지 빌드 및 실행
$ docker-compose up

# 멀티스테이지 빌드 (더 작은 이미지)
$ docker build -t anomaly_app:multistage -f Dockerfile.multistage .
$ docker run -p 5000:5000 anomaly_app:multistage
```

자세한 컨테이너 사용법은 `CONTAINER_GUIDE.md` 파일을 참조하세요.

### Kubernetes 배포

```bash
$ kubectl apply -f k8s/deploy-svc.yaml
```

## 웹 인터페이스 접속

애플리케이션 실행 후, 웹 브라우저에서 다음 URL로 접속할 수 있습니다:

```
http://localhost:5000
```

## 성능 평가

이상치 탐지 성능은 다음과 같은 지표로 평가됩니다:

- **정밀도(Precision)**: 이상치로 탐지된 것 중 실제 이상치의 비율
- **재현율(Recall)**: 실제 이상치 중 이상치로 탐지된 비율
- **F1 점수**: 정밀도와 재현율의 조화 평균
- **처리 시간**: 이상치 탐지에 소요된 시간

## MOMENT 모델 소개

MOMENT(Multivariate dOmain generalization with Many-to-many rEconstrucTion)는 시계열 데이터를 위한 최신 파운데이션 모델로, 다양한 도메인의 시계열 데이터에 대해 전이 학습이 가능합니다. 반도체 공정과 같은 복잡한 산업 데이터에서 이상치를 효과적으로 탐지하는 데 탁월한 성능을 보입니다.

### MOMENT 모델의 주요 특징

- **복잡한 시계열 패턴 인식**: 딥러닝 기반으로 복잡한 비선형 패턴과 시간적 의존성을 학습
- **다변량 데이터 처리 능력**: 여러 센서와 측정값 간의 상관관계를 고려한 통합 분석
- **도메인 일반화 능력**: 서로 다른 생산 라인이나 공정 간에도 일관된 성능 유지
- **재구성 기반의 이상치 탐지**: 정상 패턴을 학습하고 재구성 오차를 통해 이상치 감지
- **빠른 적응력**: 적은 양의 데이터로도 특정 도메인에 빠르게 적응 가능

### 기존 방법과의 비교

| 특성 | MOMENT 모델 | 전통적 방법(IsolationForest 등) |
|------|------------|----------------------------|
| 복잡한 패턴 인식 | 우수함 | 제한적 |
| 다변량 상관관계 분석 | 자동 처리 | 제한적/수동 처리 필요 |
| GPU 가속 | 지원 | 제한적 지원 |
| 파라미터 최적화 | 자동 | 수동 튜닝 필요 |
| 실시간 처리 | 학습 후 빠른 추론 | 비교적 빠름 |

이 애플리케이션에서는 GPU가 있는 경우 자동으로 GPU를 활용하여 MOMENT 모델의 성능을 최대화합니다. GPU가 없는 환경에서도 작동하지만, 처리 속도가 다소 느려질 수 있습니다.

## 폴더 구조

```
anomaly_detection_app/
├── app.py                    # 메인 Starlette 애플리케이션
├── data_generator.py         # 가상 반도체 공정 데이터 생성기
├── moment_detector.py        # MOMENT 모델 이상치 탐지 구현
├── pyproject.toml            # 프로젝트 메타데이터 및 의존성 정의
├── requirements.txt          # 필요 패키지 목록
├── environment.yml           # conda/mamba 환경 설정 파일
├── setup.sh                  # Linux/macOS용 환경 설정 스크립트
├── setup.bat                 # Windows용 환경 설정 스크립트
├── run_app.sh                # Linux/macOS용 실행 스크립트
├── run_app.bat               # Windows용 실행 스크립트
├── uv_run.sh                 # UV를 사용한 Linux/macOS용 실행 스크립트
├── uv_run.bat                # UV를 사용한 Windows용 실행 스크립트
├── Dockerfile                # Docker 이미지 빌드 파일
├── Dockerfile.multistage     # 멀티스테이지 Docker 빌드 파일
├── docker-compose.yml        # Docker Compose 설정 파일
├── UV_GUIDE.md               # UV 패키지 관리 가이드
├── CONTAINER_GUIDE.md        # 컨테이너 환경 가이드
├── data/                     # 데이터 파일 저장 폴더
│   └── semiconductor_process_data.csv
├── static/                   # 정적 파일 폴더
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── main.js
├── templates/                # HTML 템플릿 폴더
│   └── index.html
└── k8s/                      # Kubernetes 배포 파일
    └── deploy-svc.yaml
```

## 코드 설명

### 주요 컴포넌트

1. **app.py**: 웹 애플리케이션 메인 진입점. Starlette 프레임워크를 사용하여 API 엔드포인트와 웹 인터페이스를 제공합니다.

2. **moment_detector.py**: MOMENT 모델을 활용한 이상치 탐지 구현. PyTorch 기반의 MOMENT 모델과 함께 IsolationForest 모델도 대안으로 제공합니다.

3. **data_generator.py**: 테스트용 반도체 공정 데이터를 생성합니다. 실제 애플리케이션에서는 실제 데이터로 대체될 수 있습니다.

### API 엔드포인트

- `/`: 메인 웹 인터페이스
- `/detect`: 이상치 탐지 API (POST)
- `/plot/{feature}`: 특정 특성에 대한 시각화 API
- `/group_analysis`: 그룹별 이상치 분석 API (POST)
- `/model_compare`: MOMENT와 IsolationForest 모델 비교 API (POST)

## 주의사항

MOMENT 모델은 계산 집약적이므로, 고성능 하드웨어(가급적 GPU)에서 실행하는 것이 권장됩니다. GPU가 없는 시스템에서는 자동으로 CPU 모드로 실행되지만 처리 속도가 느려질 수 있습니다.

## UV 패키지 매니저 관련 정보

- `uv_run.sh` 및 `uv_run.bat` 스크립트는 UV를 사용한 프로젝트 설정 및 실행을 위한 스크립트입니다.
- UV가 설치되지 않은 경우 자동으로 설치하는 과정이 포함되어 있습니다.
- 첫 번째 실행시 UV는 필요한 모든 패키지를 자동으로 설치하므로, 추가 설정이 필요하지 않습니다.
- 기존 pip을 사용할 때보다 훨씬 빠른 속도로 환경을 설정할 수 있습니다.

## 라이센스

MIT 라이센스

## 기여 방법

1. 이 저장소를 포크하세요
2. 새 브랜치를 생성하세요 (`git checkout -b feature/amazing-feature`)
3. 변경사항을 커밋하세요 (`git commit -m 'Add some amazing feature'`)
4. 브랜치에 푸시하세요 (`git push origin feature/amazing-feature`)
5. Pull Request를 생성하세요


