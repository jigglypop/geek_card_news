@echo off
echo UV 환경 확인 중...
where uv >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo UV가 설치되어 있지 않습니다. 설치 중...
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
)

echo 패키지 설치 중...
uv sync

echo Playwright 브라우저 설치 중...
uv run playwright install chromium

echo 긱뉴스 카드뉴스 생성 중...
uv run python main.py

pause 