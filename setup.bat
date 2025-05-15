@echo off
REM setup.bat - Python 3.11 및 UV 환경 설정 스크립트 (Windows용)

echo 반도체 공정 이상치 탐지 웹 애플리케이션 환경 설정 스크립트
echo =====================================================

REM Python 버전 확인
echo Python 확인 중...
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python이 설치되어 있지 않습니다.
    
    echo Python 3.11을 설치하시겠습니까? (y/n)
    set /p install_choice=선택: 
    
    if /i "%install_choice%"=="y" (
        echo 웹 브라우저로 Python 3.11 다운로드 페이지를 엽니다...
        start https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
        echo 설치가 완료되면 이 스크립트를 다시 실행해주세요.
        pause
        exit /b 1
    ) else (
        echo Python 3.11 설치를 취소했습니다.
        echo Python 3.11이 없으면 애플리케이션을 실행할 수 없습니다.
        pause
        exit /b 1
    )
)

python -c "import sys; exit(1 if not (sys.version_info.major == 3 and sys.version_info.minor >= 11) else 0)" 2>nul
if %ERRORLEVEL% neq 0 (
    echo 현재 설치된 Python 버전은 3.11 이하입니다.
    
    echo Python 3.11을 설치하시겠습니까? (y/n)
    set /p install_choice=선택: 
    
    if /i "%install_choice%"=="y" (
        echo 웹 브라우저로 Python 3.11 다운로드 페이지를 엽니다...
        start https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe
        echo 설치가 완료되면 이 스크립트를 다시 실행해주세요.
        pause
        exit /b 1
    ) else (
        echo Python 3.11 설치를 취소했습니다.
        echo Python 3.11이 없으면 애플리케이션을 실행할 수 없습니다.
        pause
        exit /b 1
    )
) else (
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version)"') do set PYTHON_VERSION=%%i
    echo Python 버전: %PYTHON_VERSION%
)

REM UV 설치 확인 및 설치
echo UV 확인 중...
where uv >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo UV가 설치되어 있지 않습니다. 설치를 시작합니다...
    pip install uv
    if %ERRORLEVEL% neq 0 (
        echo UV 설치에 실패했습니다.
        pause
        exit /b 1
    )
) else (
    for /f "tokens=*" %%i in ('uv --version') do set UV_VERSION=%%i
    echo UV가 설치되어 있습니다: %UV_VERSION%
)

REM 가상환경 생성
echo 가상환경을 생성합니다...
if exist .venv (
    echo 기존 가상환경이 발견되었습니다. 새로 생성하시겠습니까? (y/n)
    set /p choice=선택: 
    if /i "%choice%"=="y" (
        rmdir /s /q .venv
        uv venv
    ) else (
        echo 기존 가상환경을 사용합니다.
    )
) else (
    uv venv
)

REM 의존성 패키지 설치
echo 의존성 패키지를 설치합니다...
call .venv\Scripts\activate
uv pip install -r requirements.txt

REM MOMENT 패키지 설치 확인
python -c "import momentfm" >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo MOMENT 패키지 설치를 시도합니다...
    uv pip install momentfm
    
    python -c "import momentfm" >nul 2>nul
    if %ERRORLEVEL% neq 0 (
        echo MOMENT 패키지 설치에 실패했습니다. 대체 모델(IsolationForest)이 사용됩니다.
    ) else (
        for /f "tokens=*" %%i in ('python -c "import momentfm; print(momentfm.__version__)"') do set MOMENT_VERSION=%%i
        echo MOMENT 패키지가 설치되었습니다: %MOMENT_VERSION%
    )
) else (
    for /f "tokens=*" %%i in ('python -c "import momentfm; print(momentfm.__version__)"') do set MOMENT_VERSION=%%i
    echo MOMENT 패키지가 설치되어 있습니다: %MOMENT_VERSION%
)

echo 설정이 완료되었습니다!
echo 애플리케이션 실행 방법:
echo call .venv\Scripts\activate  REM 가상환경 활성화
echo python app.py                REM 애플리케이션 실행
echo.
echo 또는 간편 실행 스크립트 사용:
echo run_app.bat
echo.
echo 웹 브라우저에서 http://localhost:5000 으로 접속하세요.
pause
