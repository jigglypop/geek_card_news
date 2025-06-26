import os
import asyncio
import base64
from datetime import datetime
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from config import *
from dotenv import load_dotenv
import httpx
from playwright.async_api import async_playwright

load_dotenv()

class GeekNewsCardGenerator:
    def __init__(self):
        self.output_dir = PATH_CONFIG["output_dir"]
        self.template_dir = PATH_CONFIG["template_dir"]
        self.style_file = PATH_CONFIG["style_file"]
        self.font_dir = PATH_CONFIG["font_dir"]
        
        print("한국어 요약 모델 로딩 중...")
        # .env 파일에서 Hugging Face 토큰 읽기
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            'lcw99/t5-large-korean-text-summary',
            token=hf_token,
            trust_remote_code=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            'lcw99/t5-large-korean-text-summary',
            token=hf_token
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print("모델 로딩 완료")
        
    def summarize_text(self, text):
        if not text or len(text.strip()) < 50:
            print(f"  [요약 건너뜀] 텍스트가 너무 짧습니다: {text[:100]}")
            return ""
        
        print(f"  [요약 원문] {text[:150]}...")
        try:
            prompt = f"""다음 내용을 3개의 완전한 문장으로 요약하되, 부드럽고 친근한 문체로 작성해 주세요.

요약할 내용: {text}

요약 (반드시 3개 문장, 부드러운 문체):
첫 번째 문장: 
두 번째 문장: 
세 번째 문장:"""
            inputs = self.tokenizer(
                prompt,
                max_length=768, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=300,
                min_length=100,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(f"  [요약 성공] {summary.strip()}")
            return summary.strip()
        except Exception as e:
            print(f"  [요약 실패] 오류: {e}")
            return "요약 생성에 실패했습니다."
    
    async def fetch_article_content(self, url):
        if not url:
            return ""
        try:
            print(f"  [디버그] 원문 기사 접근: {url}")
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True)
                soup = BeautifulSoup(response.text, 'lxml')
                if "github.com" in url:
                    content = soup.find('article')
                elif "youtube.com" in url or "youtu.be" in url:
                    return "" 
                else:
                    content = soup.find('body')
                if content:
                    content_text = content.get_text(separator=' ', strip=True)
                    print(f"  [디버그] 원문 내용 길이: {len(content_text)}")
                    return content_text
                return ""
        except Exception as e:
            print(f"  [오류] 기사 본문 가져오기 실패: {url}, {e}")
            return ""
    async def fetch_geek_news_detail(self, topic_id):
        try:
            async with httpx.AsyncClient() as client:
                url = f'https://news.hada.io/topic?id={topic_id}'
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'lxml')
                # 긱뉴스 본문 내용 찾기 (class="topic_contents" 내부)
                contents_elem = soup.find('div', class_='topic_contents')
                if contents_elem:
                    return contents_elem.get_text(separator=' ', strip=True)
                # 대안으로 topic_desc 클래스도 확인
                desc_elem = soup.find('div', class_='topic_desc')
                if desc_elem:
                    return desc_elem.get_text(strip=True)
                return ""
        except Exception as e:
            print(f"긱뉴스 상세 정보 가져오기 실패: {e}")
            return ""
    
    async def fetch_geek_news(self):
        async with httpx.AsyncClient() as client:
            response = await client.get('https://news.hada.io/')
            soup = BeautifulSoup(response.text, 'lxml')
            
            news_items = []
            topics = soup.find_all('div', class_='topic_row')[:5]
            

            
            for topic in topics:
                title_elem = topic.find('div', class_='topictitle')
                if title_elem:
                    title_link = title_elem.find('a')
                    if title_link:
                        title = title_link.text.strip()
                        original_link = title_link.get('href', '')
                        
                        # 긱뉴스 토픽 페이지 링크 찾기
                        topic_id = None
                        
                        # 댓글 링크나 다른 내부 링크에서 topic ID 찾기
                        all_links = topic.find_all('a')
                        for link in all_links:
                            href = link.get('href', '')
                            if 'topic?id=' in href:
                                topic_id = href.split('id=')[-1].split('&')[0]
                                break

                        # 긱뉴스 페이지에서 설명 가져오기
                        desc_elem = topic.find('span', class_='topicdesc')
                        desc = desc_elem.text.strip() if desc_elem else ''
                        
                        # 긱뉴스 토픽 페이지에서 본문 가져오기
                        if topic_id:
                            detailed_desc = await self.fetch_geek_news_detail(topic_id)
                            if detailed_desc and len(detailed_desc) > len(desc):
                                desc = detailed_desc

                        # 설명 요약
                        summarized_desc = self.summarize_text(desc)
                        
                        news_items.append({
                            'title': title,
                            'description': summarized_desc,
                            'link': original_link,
                            'topic_id': topic_id
                        })
            
            return news_items
    
    def encode_font_to_base64(self, font_path):
        try:
            with open(font_path, "rb") as font_file:
                return base64.b64encode(font_file.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"폰트 파일을 찾을 수 없습니다: {font_path}")
            return None
    
    def generate_embedded_font_css(self):
        font_css = ""
        pretendard_regular = self.encode_font_to_base64(os.path.join(self.font_dir, FONT_CONFIG["pretendard_regular"]))
        pretendard_medium = self.encode_font_to_base64(os.path.join(self.font_dir, FONT_CONFIG["pretendard_medium"]))
        pretendard_bold = self.encode_font_to_base64(os.path.join(self.font_dir, FONT_CONFIG["pretendard_bold"]))
        
        if pretendard_regular:
            font_css += f"""
@font-face {{
    font-family: 'Pretendard';
    font-weight: normal;
    src: url(data:font/otf;base64,{pretendard_regular}) format('opentype');
    font-display: swap;
}}"""
        if pretendard_medium:
            font_css += f"""
@font-face {{
    font-family: 'Pretendard';
    font-weight: 500;
    src: url(data:font/otf;base64,{pretendard_medium}) format('opentype');
    font-display: swap;
}}"""
        if pretendard_bold:
            font_css += f"""
@font-face {{
    font-family: 'Pretendard';
    font-weight: 700;
    src: url(data:font/otf;base64,{pretendard_bold}) format('opentype');
    font-display: swap;
}}"""
        
        blackhan = self.encode_font_to_base64(os.path.join(self.font_dir, FONT_CONFIG["blackhan"]))
        if blackhan:
            font_css += f"""
@font-face {{
    font-family: 'Black Han Sans';
    font-weight: normal;
    src: url(data:font/ttf;base64,{blackhan}) format('truetype');
    font-display: swap;
}}"""
        return font_css
    
    def get_available_characters(self):
        characters = {}
        character_dir = PATH_CONFIG["character_dir"]
        
        if not os.path.exists(character_dir):
            return characters
        
        for file in os.listdir(character_dir):
            name, ext = os.path.splitext(file)
            if ext.lower() in [e.lower() for e in IMAGE_CONFIG["character_extensions"]]:
                characters[name] = os.path.join(character_dir, file)
        
        return characters
    
    def encode_image_to_base64(self, image_path):
        if not image_path or not os.path.exists(image_path):
            return None
        
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"이미지 인코딩 실패: {image_path}, 오류: {e}")
            return None
    
    async def create_geek_news_html(self, news_items):
        html_pages = ""
        
        available_characters = self.get_available_characters()
        main_character_path = available_characters.get(IMAGE_CONFIG["main_character"])
        main_character_base64 = self.encode_image_to_base64(main_character_path) if main_character_path else None
        
        page_characters = []
        for name, path in available_characters.items():
            if name != IMAGE_CONFIG["main_character"] and name != IMAGE_CONFIG["all_characters"]:
                character_base64 = self.encode_image_to_base64(path)
                if character_base64:
                    page_characters.append(character_base64)
        
        html_pages += f"""
        <div class="page cover">
            <div class="card-container">
                <div class="cover-content">
                    <div class="text-section">
                        <p class="title-sub">오늘의 주요 뉴스</p>
                        <h1 class="title-main">긱뉴스</h1>
                    </div>
                    <div class="character-section">
                        {f'<img src="data:image/png;base64,{main_character_base64}" class="character main-character" alt="캐릭터" />' if main_character_base64 else ''}
                    </div>
                    <div class="decorative-elements">
                        <div class="speech-bubble">
                            💬
                            <span class="bubble-text">뉴-스!</span>
                        </div>
                        <div class="lightbulb">💡</div>
                        <div class="star star1">⭐</div>
                        <div class="star star2">⭐</div>
                        <div class="star star3">⭐</div>
                    </div>
                </div>
            </div>
        </div>
        """
        
        for page_index, news in enumerate(news_items):
            character_base64 = page_characters[page_index] if page_index < len(page_characters) else None
            # 도메인 라벨 생성
            domain_label = "링크"
            if news['link']:
                if "github.com" in news['link']:
                    domain_label = "GitHub"
                elif "youtube.com" in news['link'] or "youtu.be" in news['link']:
                    domain_label = "YouTube"
                elif "blog" in news['link'] or "medium.com" in news['link']:
                    domain_label = "블로그"
                elif "news" in news['link']:
                    domain_label = "뉴스"
                else:
                    domain_label = "원문"
            
            geek_news_link = f"https://news.hada.io/topic?id={news['topic_id']}" if news['topic_id'] else ""
            
            # 주제별 카테고리 표시 (텍스트)
            topic_category = "기술"
            if "github" in news['link'].lower() or "git" in news['title'].lower():
                topic_category = "개발"
            elif "youtube" in news['link'].lower():
                topic_category = "영상"
            elif "blog" in news['link'].lower():
                topic_category = "블로그"
            elif "ai" in news['title'].lower() or "gemini" in news['title'].lower():
                topic_category = "AI"
            elif "데이터" in news['title'] or "data" in news['title'].lower():
                topic_category = "데이터"
            
            html_pages += f"""
            <div class="page news-page">
                <div class="card-container">
                    <div class="news-header-section">
                        <div class="topic-category">{topic_category}</div>
                        <h2 class="category-title">GeekNews #{page_index + 1}</h2>
                    </div>
                    <div class="news-content-section">
                        <div class="news-title">{news['title']}</div>
                        <div class="news-description">{news['description']}</div>
                        <div class="links-section">
                            {f'<div class="link-item"><span class="link-label">토론:</span> <a href="{geek_news_link}" target="_blank">{geek_news_link}</a></div>' if geek_news_link else ''}
                            {f'<div class="link-item"><span class="link-label">{domain_label}:</span> <a href="{news["link"]}" target="_blank">{news["link"]}</a></div>' if news['link'] else ''}
                        </div>
                    </div>
                    {f'<img src="data:image/png;base64,{character_base64}" class="page-character" alt="캐릭터" />' if character_base64 else ''}
                </div>
            </div>
            """
        
        # 마지막 요약 페이지 생성
        from datetime import datetime
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        summary_items = ""
        for index, news in enumerate(news_items, 1):
            # 카테고리 설정
            topic_category = "기술"
            if "github" in news['link'].lower() or "git" in news['title'].lower():
                topic_category = "개발"
            elif "ai" in news['title'].lower() or "gemini" in news['title'].lower():
                topic_category = "AI"
            elif "데이터" in news['title'] or "data" in news['title'].lower():
                topic_category = "데이터"
            
            summary_items += f"""
            <div class="summary-item">
                <div class="summary-header">
                    <span class="summary-number">{index}</span>
                    <span class="summary-category">{topic_category}</span>
                </div>
                <div class="summary-title">{news['title']}</div>
            </div>
            """
        
        html_pages += f"""
        <div class="page summary-page">
            <div class="card-container">
                <div class="summary-header-section">
                    <h1 class="summary-main-title">GeekNews 요약</h1>
                    <p class="summary-date">{current_date}</p>
                </div>
                <div class="summary-content">
                    <h2 class="summary-subtitle">오늘의 주요 뉴스</h2>
                    <div class="summary-list">
                        {summary_items}
                    </div>
                    <div class="summary-footer">
                        <p>총 {len(news_items)}개의 뉴스를 확인했어요</p>
                        <p class="summary-source">출처: GeekNews (news.hada.io)</p>
                    </div>
                </div>
            </div>
        </div>
        """
        
        return html_pages
    
    def create_geek_news_styles(self):
        return ""
    
    async def generate_pdf_and_png(self, html_content, css_content):
        embedded_fonts = self.generate_embedded_font_css()
        style_content = self.read_file(os.path.join(self.template_dir, self.style_file))
        
        final_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>오늘의 긱뉴스</title>
            <style>
            {embedded_fonts}
            {style_content}
            {css_content}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """
        
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        pdf_filename = f"geek_news_{timestamp}.pdf"
        
        html_output_path = os.path.join(self.output_dir, "geek_news.html")
        pdf_output_path = os.path.join(self.output_dir, pdf_filename)
        
        with open(html_output_path, "w", encoding="utf-8") as f:
            f.write(final_html)
        
        print(f"'{html_output_path}' 파일 생성 완료")
        
        html_path_url = f'file://{os.path.abspath(html_output_path)}'
        
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_viewport_size({"width": 1080, "height": 1080})
            await page.goto(html_path_url, wait_until='networkidle')
            await page.wait_for_timeout(2000)
            
            await page.pdf(
                path=pdf_output_path, 
                width="1080px", 
                height="1080px", 
                print_background=True,
                margin={'top': '0px', 'right': '0px', 'bottom': '0px', 'left': '0px'}
            )
            
            print(f"'{pdf_output_path}' 파일 생성 완료")
            
            pages = await page.query_selector_all('.page')
            print(f"총 {len(pages)} 페이지 PNG/JPG 생성 중...")
            for i, page_element in enumerate(pages, 1):
                # PNG 생성
                png_filename = f"geek_page_{i:02d}.png"
                png_output_path = os.path.join(PATH_CONFIG["image_dir"], png_filename)
                await page_element.screenshot(
                    path=png_output_path,
                    type='png',
                    omit_background=False
                )
                print(f"'{png_filename}' 생성 완료")
                
                # JPG 생성 (배경 포함)
                jpg_filename = f"geek_page_{i:02d}.jpg"
                jpg_output_path = os.path.join(PATH_CONFIG["image_dir"], jpg_filename)
                await page_element.screenshot(
                    path=jpg_output_path,
                    type='jpeg',
                    quality=95,
                    omit_background=False
                )
                print(f"'{jpg_filename}' 생성 완료")
            
            await browser.close()
    
    def read_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"오류: 파일 '{filepath}'을(를) 찾을 수 없습니다.")
            return ""
    
    def create_output_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(PATH_CONFIG["image_dir"], exist_ok=True)
    
    async def generate(self):
        print("=== 긱뉴스 카드뉴스 생성 시작 ===")
        self.create_output_directory()
        
        print("긱뉴스 데이터 가져오는 중...")
        news_items = await self.fetch_geek_news()

        if not news_items:
            print("뉴스를 가져올 수 없습니다.")
            return

        print(f"{len(news_items)}개의 뉴스를 가져왔습니다.")

        html_content = await self.create_geek_news_html(news_items)
        css_content = self.create_geek_news_styles()
        
        await self.generate_pdf_and_png(html_content, css_content)
        
        print("=== 긱뉴스 카드뉴스 생성 완료 ===")

async def main():
    generator = GeekNewsCardGenerator()
    await generator.generate()

if __name__ == "__main__":
    asyncio.run(main()) 