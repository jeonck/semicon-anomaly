# UV 패키지 관리 가이드

이 문서는 UV 패키지 관리 도구를 사용하여 Python 환경을 효율적으로 관리하는 방법을 설명합니다.

## UV란?

UV는 Rust로 작성된 빠르고 신뢰할 수 있는 Python 패키지 인스톨러이자 환경 관리 도구입니다. 기존의 pip와 venv(또는 virtualenv)를 대체하는 통합 솔루션을 제공합니다.

주요 특징:
- 빠른 성능 (pip보다 10-100배 빠름)
- 재현 가능한 빌드
- 보안 기능 내장
- 선택적 잠금 파일
- 통합된 가상환경 관리

## 설치 방법

### macOS / Linux

```bash
curl -sSf https://install.ultramarine.tools | sh
```

### Windows

```bash
pip install uv
```

### Python에서 설치

```bash
pip install uv
```

## 기본 사용법

### 가상환경 생성

```bash
uv venv --python 3.11
```

이 명령은 현재 디렉토리에 `.venv` 폴더를 생성합니다.

### 가상환경 활성화

**Linux/macOS**:
```bash
source .venv/bin/activate
```

**Windows**:
```
.venv\Scripts\activate
```

### 패키지 설치

```bash
# requirements.txt 파일에서 설치
uv pip install -r requirements.txt

# 특정 패키지 설치
uv pip install numpy pandas

# 특정 버전 설치
uv pip install pandas==2.0.0
```

### 패키지 업그레이드

```bash
uv pip install --upgrade pandas
```

### 설치된 패키지 목록 확인

```bash
uv pip list
```

### 의존성 관리

UV는 pip의 `requirements.txt`와 호환됩니다. 추가로 더 상세한 의존성 잠금을 제공하는 잠금 파일 기능을 지원합니다.

#### 잠금 파일 생성

```bash
uv pip compile requirements.txt -o requirements.lock
```

#### 잠금 파일에서 설치

```bash
uv pip install -r requirements.lock
```

## UV와 Python 3.11

UV는 Python 3.11에서 최적으로 작동하며, Python 3.11의 향상된 성능과 기능을 최대한 활용할 수 있습니다:

- 빠른 패키지 설치 속도
- 효율적인 의존성 해결
- 안정적인 가상환경 관리

## 자주 발생하는 문제 해결

### UV 설치 오류

```bash
# 권한 문제가 발생할 경우
sudo curl -sSf https://install.ultramarine.tools | sh

# 또는 인터넷 연결 문제 시 pip로 설치
pip install uv
```

### 패키지 호환성 문제

```bash
# 호환 가능한 정확한 버전 지정
uv pip install package==x.y.z

# 호환성 충돌 확인
uv pip check
```

### 가상환경 활성화 오류

```bash
# 가상환경 재생성
rm -rf .venv
uv venv
```

## UV와 conda/mamba 비교

UV는 conda/mamba와 다른 접근 방식을 취합니다:

- conda/mamba: 바이너리 패키지 관리자로, Python뿐만 아니라 여러 프로그래밍 언어 및 시스템 라이브러리를 관리
- UV: Python 특화 패키지 관리자로, PyPI 저장소를 사용하며 pip의 직접적인 대체재

UV는 가볍고 빠른 반면, conda는 더 많은 기능과 교차 언어 지원을 제공합니다. 프로젝트 요구 사항에 따라 선택하세요.

## 추가 자료

- [UV 공식 문서](https://github.com/astral-sh/uv)
- [UV vs pip 성능 비교](https://astral.sh/blog/uv)
