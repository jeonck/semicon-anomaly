#!/bin/bash
# uv_run.sh - UV를 사용한 간편 실행 스크립트

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}UV를 사용한 반도체 공정 이상치 탐지 웹 애플리케이션 실행${NC}"
echo "====================================================="

# UV 설치 확인
if ! command -v uv &>/dev/null; then
    echo -e "${YELLOW}UV가 설치되어 있지 않습니다. 설치를 시도합니다...${NC}"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # PATH에 UV 추가
    export PATH="$HOME/.uv/bin:$PATH"
    
    # 설치 확인
    if ! command -v uv &>/dev/null; then
        echo -e "${RED}UV 설치에 실패했습니다. 수동으로 설치해주세요:${NC}"
        echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
        exit 1
    fi
fi

# UV 버전 확인
UV_VERSION=$(uv --version)
echo -e "${GREEN}UV 버전: $UV_VERSION${NC}"

# pyproject.toml 파일 확인
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}pyproject.toml 파일이 없습니다.${NC}"
    exit 1
fi

# 의존성 설치 (필요한 경우)
echo -e "${YELLOW}의존성 확인 및 설치...${NC}"
uv pip sync --python=3.11 pyproject.toml

# 애플리케이션 실행
echo -e "${GREEN}애플리케이션을 시작합니다...${NC}"
uv run python app.py

# 실행 종료 메시지
echo -e "${GREEN}애플리케이션이 종료되었습니다.${NC}"
