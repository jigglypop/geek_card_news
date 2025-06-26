#!/bin/bash

echo "UV 환경 확인 중..."
if ! command -v uv &> /dev/null; then
    echo "UV가 설치되어 있지 않습니다. 설치 중..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "패키지 설치 중..."
uv sync

echo "Playwright 브라우저 설치 중..."
uv run playwright install chromium

echo "긱뉴스 카드뉴스 생성 중..."
uv run python geek_news.py 