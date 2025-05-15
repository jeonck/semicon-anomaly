#!/bin/bash
# setup.sh - Python 3.11 및 UV 환경 설정 스크립트

# 색상 설정
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}반도체 공정 이상치 탐지 웹 애플리케이션 환경 설정 스크립트${NC}"
echo "====================================================="

# 운영체제 확인
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
    # Linux 배포판 확인
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$NAME
    else
        DISTRO="Unknown Linux"
    fi
else
    OS="Unknown"
fi

echo -e "${YELLOW}운영체제: $OS${NC}"
if [[ "$OS" == "Linux" ]]; then
    echo -e "${YELLOW}배포판: $DISTRO${NC}"
fi

# Python 3.11 설치 확인 또는 설치
INSTALL_PYTHON=false

check_python() {
    if command -v python3.11 &>/dev/null; then
        PYTHON_PATH=$(which python3.11)
        PYTHON_VERSION=$(python3.11 --version)
        echo -e "${GREEN}Python 3.11이 설치되어 있습니다: $PYTHON_VERSION${NC}"
        return 0
    else
        return 1
    fi
}

if ! check_python; then
    echo -e "${YELLOW}Python 3.11이 설치되어 있지 않습니다.${NC}"
    echo -e "${YELLOW}Python 3.11을 설치하시겠습니까? (y/n) ${NC}"
    read -p "선택: " choice
    
    if [[ $choice == "y" || $choice == "Y" ]]; then
        INSTALL_PYTHON=true
    else
        echo -e "${RED}Python 3.11이 없으면 애플리케이션을 실행할 수 없습니다.${NC}"
        echo -e "${YELLOW}스크립트를 종료합니다.${NC}"
        exit 1
    fi
fi

if [ "$INSTALL_PYTHON" = true ]; then
    if [[ "$OS" == "Linux" ]]; then
        if [[ "$DISTRO" == *"Ubuntu"* || "$DISTRO" == *"Debian"* ]]; then
            echo -e "${YELLOW}Ubuntu/Debian에 Python 3.11을 설치합니다...${NC}"
            sudo apt-get update
            sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
        elif [[ "$DISTRO" == *"Fedora"* || "$DISTRO" == *"Red Hat"* || "$DISTRO" == *"CentOS"* ]]; then
            echo -e "${YELLOW}RHEL/Fedora/CentOS에 Python 3.11을 설치합니다...${NC}"
            sudo dnf install -y python3.11 python3.11-devel
        else
            echo -e "${RED}이 Linux 배포판에 대한 자동 설치 스크립트가 없습니다.${NC}"
            echo -e "${YELLOW}Python 3.11을 직접 설치한 후 다시 시도해주세요.${NC}"
            exit 1
        fi
    elif [[ "$OS" == "macOS" ]]; then
        echo -e "${YELLOW}macOS에는 Homebrew를 사용하여 Python 3.11을 설치하는 것을 권장합니다:${NC}"
        echo -e "${YELLOW}Homebrew를 설치하시겠습니까? (y/n) ${NC}"
        read -p "선택: " brew_choice
        
        if [[ $brew_choice == "y" || $brew_choice == "Y" ]]; then
            if ! command -v brew &>/dev/null; then
                echo -e "${YELLOW}Homebrew 설치 중...${NC}"
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            else
                echo -e "${GREEN}Homebrew가 이미 설치되어 있습니다.${NC}"
            fi
            
            echo -e "${YELLOW}Python 3.11 설치 중...${NC}"
            brew install python@3.11
        else
            echo -e "${RED}Python 3.11 설치를 건너뜁니다.${NC}"
            echo -e "${YELLOW}Python 3.11을 직접 설치한 후 다시 시도해주세요.${NC}"
            exit 1
        fi
    else
        echo -e "${RED}지원되지 않는 운영체제입니다.${NC}"
        echo -e "${YELLOW}Python 3.11을 직접 설치한 후 다시 시도해주세요.${NC}"
        exit 1
    fi
    
    # Python 설치 확인
    if ! check_python; then
        echo -e "${RED}Python 3.11 설치에 실패했습니다.${NC}"
        exit 1
    fi
fi

# UV 설치 확인 및 설치
if command -v uv &>/dev/null; then
    UV_VERSION=$(uv --version)
    echo -e "${GREEN}UV가 설치되어 있습니다: $UV_VERSION${NC}"
else
    echo -e "${YELLOW}UV가 설치되어 있지 않습니다. 설치를 시작합니다...${NC}"
    if [[ "$OS" == "macOS" || "$OS" == "Linux" ]]; then
        echo "curl -sSf https://install.ultramarine.tools | sh"
        curl -sSf https://install.ultramarine.tools | sh
    else
        echo -e "${RED}자동 설치가 지원되지 않는 운영체제입니다. 수동으로 UV를 설치해주세요:${NC}"
        echo "pip install uv"
        exit 1
    fi
fi

# 가상환경 생성
echo -e "${YELLOW}가상환경을 생성합니다...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}기존 가상환경이 발견되었습니다. 새로 생성하시겠습니까? (y/n)${NC}"
    read -p "선택: " choice
    if [[ $choice == "y" || $choice == "Y" ]]; then
        rm -rf .venv
        python3.11 -m uv venv
    else
        echo -e "${GREEN}기존 가상환경을 사용합니다.${NC}"
    fi
else
    python3.11 -m uv venv
fi

# 의존성 패키지 설치
echo -e "${YELLOW}의존성 패키지를 설치합니다...${NC}"
source .venv/bin/activate
uv pip install -r requirements.txt

# MOMENT 패키지 설치 확인
if python -c "import momentfm" &>/dev/null; then
    MOMENT_VERSION=$(python -c "import momentfm; print(momentfm.__version__)")
    echo -e "${GREEN}MOMENT 패키지가 설치되어 있습니다: $MOMENT_VERSION${NC}"
else
    echo -e "${YELLOW}MOMENT 패키지 설치를 시도합니다...${NC}"
    uv pip install momentfm
    
    if python -c "import momentfm" &>/dev/null; then
        MOMENT_VERSION=$(python -c "import momentfm; print(momentfm.__version__)")
        echo -e "${GREEN}MOMENT 패키지가 설치되었습니다: $MOMENT_VERSION${NC}"
    else
        echo -e "${RED}MOMENT 패키지 설치에 실패했습니다. 대체 모델(IsolationForest)이 사용됩니다.${NC}"
    fi
fi

echo -e "${GREEN}설정이 완료되었습니다!${NC}"
echo -e "${YELLOW}애플리케이션 실행 방법:${NC}"
echo "source .venv/bin/activate  # 가상환경 활성화"
echo "python app.py              # 애플리케이션 실행"
echo ""
echo -e "${GREEN}또는 간편 실행 스크립트 사용:${NC}"
echo "chmod +x run_app.sh"
echo "./run_app.sh"
echo ""
echo -e "${GREEN}웹 브라우저에서 http://localhost:5000 으로 접속하세요.${NC}"
