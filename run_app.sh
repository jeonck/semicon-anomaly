#!/bin/bash
# run_app.sh - 애플리케이션 실행 스크립트

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 가상환경 확인
if [ ! -d ".venv" ]; then
    echo -e "${RED}가상환경이 발견되지 않았습니다.${NC}"
    echo -e "${YELLOW}환경 설정을 먼저 실행해주세요:${NC}"
    echo "./setup.sh"
    exit 1
fi

# 가상환경 활성화
echo -e "${GREEN}가상환경을 활성화합니다...${NC}"
source .venv/bin/activate

# 애플리케이션 실행
echo -e "${GREEN}반도체 공정 이상치 탐지 웹 애플리케이션을 시작합니다...${NC}"
python app.py

# 실행 종료 메시지
echo -e "${GREEN}애플리케이션이 종료되었습니다.${NC}"
