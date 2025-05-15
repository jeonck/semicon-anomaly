# 컨테이너 환경 개발 및 배포 가이드

이 문서는 반도체 공정 이상치 탐지 애플리케이션을 컨테이너 환경(Docker, Kubernetes)에서 개발 및 배포하는 과정을 설명합니다.

## 1. 로컬 개발 환경 vs 컨테이너 환경

### 로컬 개발 환경
- Python 3.11 및 UV 패키지 관리자 사용
- 직접 가상환경 생성 및 의존성 설치
- 로컬 파일 시스템에 데이터 저장
- `python app.py` 또는 `./run_app.sh`로 실행

### 컨테이너 환경
- Docker 또는 Kubernetes 기반 실행
- 이미지 내에 모든 의존성 패키징
- 볼륨 마운트를 통한 데이터 유지
- 이미지 빌드 후 컨테이너로 실행

## 2. Docker 이미지 빌드 가이드

### 기본 빌드
```bash
# 기본 이미지 빌드
docker build -t anomaly-detection-app .

# 빌드 테스트 실행
docker run -p 5000:5000 anomaly-detection-app
```

### 최적화된 멀티 스테이지 빌드
보다 최적화된 이미지를 위해 멀티 스테이지 빌드를 사용할 수 있습니다. 이를 위해 Dockerfile.multistage 파일을 생성하세요:

```dockerfile
# Build stage
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -sSf https://install.ultramarine.tools | sh \
    && ln -s ~/.cargo/bin/uv /usr/local/bin/uv

# Copy requirements
COPY requirements.txt .

# Install dependencies into a virtual environment
RUN uv venv .venv
RUN uv pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application files
COPY . .

# Create a non-root user
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Run the application
CMD ["python", "app.py"]
```

빌드 명령:
```bash
docker build -t anomaly-detection-app:optimized -f Dockerfile.multistage .
```

### GPU 지원 빌드
GPU 지원을 위한 빌드는 다음과 같이 수행합니다:

```bash
# NVIDIA CUDA 기반 이미지 사용
docker build -t anomaly-detection-app:gpu -f Dockerfile.gpu .

# GPU 지원 실행
docker run --gpus all -p 5000:5000 anomaly-detection-app:gpu
```

## 3. Kubernetes 배포 가이드

### 사전 준비
- 동작하는 Kubernetes 클러스터
- kubectl 설정
- (GPU 사용 시) NVIDIA GPU Operator 설치

### 배포 실행
```bash
# ConfigMap 설정 업데이트 (필요시)
kubectl create configmap semicon-anomaly-config \
  --from-literal=DEVICE=cpu \
  --from-literal=LOG_LEVEL=INFO \
  -n semicon-anomaly --dry-run=client -o yaml | kubectl apply -f -

# 배포 실행
kubectl apply -f k8s/deploy-svc.yaml

# 배포 상태 확인
kubectl get pods -n semicon-anomaly
```

### GPU 사용 설정
GPU를 사용하려면 배포 파일에서 다음 부분을 추가/수정해야 합니다:

```yaml
spec:
  containers:
  - name: semicon-anomaly
    resources:
      limits:
        nvidia.com/gpu: 1  # 사용할 GPU 개수
```

## 4. 모니터링 및 디버깅

### 컨테이너 로그 확인
```bash
# Docker 로그
docker logs -f <container_id>

# Kubernetes 로그
kubectl logs -f deployment/semicon-anomaly -n semicon-anomaly
```

### 컨테이너 내부 접속
```bash
# Docker 컨테이너 접속
docker exec -it <container_id> /bin/bash

# Kubernetes Pod 접속
kubectl exec -it <pod_name> -n semicon-anomaly -- /bin/bash
```

### 리소스 모니터링
```bash
# Docker 컨테이너 리소스 사용량
docker stats <container_id>

# Kubernetes Pod 리소스 사용량
kubectl top pod <pod_name> -n semicon-anomaly
```

## 5. 배포 자동화 (CI/CD)

### GitHub Actions 예시
`.github/workflows/docker-build.yml` 파일을 생성하여 CI/CD 파이프라인을 설정할 수 있습니다:

```yaml
name: Docker Build and Push

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Log in to container registry
      uses: docker/login-action@v2
      with:
        registry: your-registry.io
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}
    
    - name: Build and push image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: your-registry.io/anomaly-detection-app:latest
        
    - name: Deploy to Kubernetes
      if: github.ref == 'refs/heads/main'
      uses: steebchen/kubectl@v2
      with:
        config: ${{ secrets.KUBE_CONFIG_DATA }}
        command: apply -f k8s/deploy-svc.yaml
```

## 6. 문제 해결

### 일반적인 문제 및 해결 방법

1. **momentfm 패키지 설치 실패**
   ```
   ERROR: Could not find a version that satisfies the requirement momentfm>=0.2.0
   ```
   - 해결책: requirements.txt에서 버전을 `momentfm>=0.1.4`로 변경

2. **GPU 감지 실패**
   - 해결책: CUDA 설치 확인 및 `nvidia-smi` 명령어로 GPU 상태 확인

3. **메모리 부족 오류**
   - 해결책: 컨테이너 메모리 제한 증가 또는 배치 크기 감소

4. **컨테이너 시작 실패**
   - 해결책: 로그 확인 및 환경 변수 설정 검증

5. **Kubernetes 볼륨 마운트 실패**
   - 해결책: PVC 상태 확인 및 스토리지 클래스 존재 여부 확인
