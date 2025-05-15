@echo off
REM uv_run.bat - UV를 사용한 간편 실행 스크립트 (Windows용)

echo UV를 사용한 반도체 공정 이상치 탐지 웹 애플리케이션 실행
echo =====================================================

REM UV 설치 확인
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo UV가 설치되어 있지 않습니다. 설치를 시도합니다...
    pip install uv
    if %ERRORLEVEL% neq 0 (
        echo UV 설치에 실패했습니다. 수동으로 설치해주세요:
        echo pip install uv
        pause
        exit /b 1
    )
)

REM UV 버전 확인
for /f "tokens=*" %%i in ('uv --version') do set UV_VERSION=%%i
echo UV 버전: %UV_VERSION%

REM pyproject.toml 파일 확인
if not exist pyproject.toml (
    echo pyproject.toml 파일이 없습니다.
    pause
    exit /b 1
)

REM 의존성 설치 (필요한 경우)
echo 의존성 확인 및 설치...
uv pip sync --python=3.11 pyproject.toml

REM 애플리케이션 실행
echo 애플리케이션을 시작합니다...
uv run python app.py

REM 실행 종료 메시지
echo 애플리케이션이 종료되었습니다.
pause
