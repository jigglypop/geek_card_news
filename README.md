# geek_card_news

GeekNews(news.hada.io)의 최신 뉴스를 가져와서 한 페이지 카드뉴스로 만들어주는 도구입니다.

## UV 설치

```bash
# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# macOS/Linux  
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 패키지 설치

```bash
uv sync
```

## Playwright 브라우저 설치

```bash
uv run playwright install chromium
```

## 실행

```bash
uv run python geek_news.py
```

## 출력물

- `output/geek_news.html` - HTML 파일
- `output/geek_news_[timestamp].pdf` - PDF 파일
- `output/images/geek_news_[timestamp].png` - PNG 이미지 