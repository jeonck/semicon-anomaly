@echo off
REM run_app.bat - 애플리케이션 실행 스크립트 (Windows용)

REM 가상환경 확인
if not exist .venv (
    echo 가상환경이 발견되지 않았습니다.
    echo 환경 설정을 먼저 실행해주세요:
    echo setup.bat
    exit /b 1
)

REM 가상환경 활성화
echo 가상환경을 활성화합니다...
call .venv\Scripts\activate

REM 애플리케이션 실행
echo 반도체 공정 이상치 탐지 웹 애플리케이션을 시작합니다...
python app.py

REM 실행 종료 메시지
echo 애플리케이션이 종료되었습니다.
