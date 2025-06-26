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
        print("한국어 요약 모델 로딩 중...")
        # .env 파일에서 Hugging Face 토큰 읽기
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.tokenizer = AutoTokenizer.from_pretrained(
            AI_CONFIG["model_name"],
            token=hf_token,
            trust_remote_code=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            AI_CONFIG["model_name"],
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
            prompt_template = self.load_template(PATH_CONFIG["summary_prompt"])
            prompt = prompt_template.format(text=text)
            
            inputs = self.tokenizer(
                prompt,
                max_length=AI_CONFIG["max_input_length"], 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=AI_CONFIG["max_output_length"],
                min_length=AI_CONFIG["min_output_length"],
                length_penalty=AI_CONFIG["length_penalty"],
                num_beams=AI_CONFIG["num_beams"],
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            print(f"  [요약 성공] {summary.strip()}")
            return summary.strip()
        except Exception as e:
            print(f"  [요약 실패] 오류: {e}")
            return "요약 생성에 실패했습니다."
    
    async def get_content(self, url):
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
    async def get_detail(self, topic_id):
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
        async with httpx.AsyncClient(timeout=CRAWLING_CONFIG["timeout"]) as client:
            response = await client.get(CRAWLING_CONFIG["base_url"])
            soup = BeautifulSoup(response.text, 'lxml')
            news_items = []
            topics = soup.find_all('div', class_='topic_row')[:CRAWLING_CONFIG["news_count"]]
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
                            detailed_desc = await self.get_detail(topic_id)
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
    
    def generate_embedded_font_css(self):
        return """
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Black+Han+Sans&display=swap');
"""
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
    def load_template(self, template_name):
        template_path = os.path.join(self.template_dir, template_name)
        return self.read_file(template_path)
    async def create_html(self, news_items):
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
        
        # QR 코드 로드 (루트 디렉토리에서)
        qr_path = "QR.png"
        qr_base64 = self.encode_image_to_base64(qr_path) if os.path.exists(qr_path) else None
        
        # 커버 템플릿 로드
        cover_template = self.load_template(PATH_CONFIG["cover_template"])
        
        character_html = f'<img src="data:image/png;base64,{main_character_base64}" class="character main-character" alt="캐릭터" />' if main_character_base64 else ''
        qr_html = f'<div class="qr-section"><img src="data:image/png;base64,{qr_base64}" class="qr-code" alt="QR코드" /></div>' if qr_base64 else ''
        
        html_pages += cover_template.format(
            cover_subtitle=TEXT_CONFIG["cover_subtitle"],
            cover_title=TEXT_CONFIG["cover_title"],
            character_image=character_html,
            qr_section=qr_html
        )
        
        # 뉴스 템플릿 로드
        news_template = self.load_template(PATH_CONFIG["news_template"])
        
        for page_index, news in enumerate(news_items):
            character_base64 = page_characters[page_index] if page_index < len(page_characters) else None
            
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
            
            # 링크 HTML 생성
            links_html = ""
            if geek_news_link:
                links_html += f'<div class="link-item"><span class="link-label">토론:</span> <a href="{geek_news_link}" target="_blank">{geek_news_link}</a></div>'
            if news['link']:
                links_html += f'<div class="link-item"><span class="link-label">{domain_label}:</span> <a href="{news["link"]}" target="_blank">{news["link"]}</a></div>'
            
            character_html = f'<img src="data:image/png;base64,{character_base64}" class="page-character" alt="캐릭터" />' if character_base64 else ''
            
            html_pages += news_template.format(
                topic_category=topic_category,
                page_number=page_index + 1,
                news_title=news['title'],
                news_description=news['description'],
                links=links_html,
                character_image=character_html
            )
        
        from datetime import datetime
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        
        # 요약 아이템 템플릿 로드
        summary_item_template = self.load_template(PATH_CONFIG["summary_item_template"])
        
        summary_items = ""
        for index, news in enumerate(news_items, 1):
            topic_category = "기술"
            if "github" in news['link'].lower() or "git" in news['title'].lower():
                topic_category = "개발"
            elif "ai" in news['title'].lower() or "gemini" in news['title'].lower():
                topic_category = "AI"
            elif "데이터" in news['title'] or "data" in news['title'].lower():
                topic_category = "데이터"
            
            summary_items += summary_item_template.format(
                number=index,
                category=topic_category,
                title=news['title']
            )
        
        # 요약 페이지 템플릿 로드
        summary_template = self.load_template(PATH_CONFIG["summary_template"])
        
        html_pages += summary_template.format(
            summary_title=TEXT_CONFIG["summary_title"],
            summary_date=current_date,
            summary_subtitle=TEXT_CONFIG["summary_subtitle"],
            summary_items=summary_items,
            summary_footer=TEXT_CONFIG["summary_footer_text"].format(count=len(news_items)),
            summary_source=TEXT_CONFIG["summary_source"]
        )
        
        return html_pages
    def create_styles(self):
        base_css = self.load_template(PATH_CONFIG["style_file"])
        
        # config.py의 설정으로 CSS 커스터마이징
        customized_css = base_css.replace("{{cover_background}}", COLOR_CONFIG["cover_background"])
        customized_css = customized_css.replace("{{news_background}}", COLOR_CONFIG["news_background"])
        customized_css = customized_css.replace("{{summary_background}}", COLOR_CONFIG["summary_background"])
        customized_css = customized_css.replace("{{end_background}}", COLOR_CONFIG["end_background"])
        
        customized_css = customized_css.replace("{{cover_title_size}}", FONT_CONFIG["cover_title"])
        customized_css = customized_css.replace("{{cover_subtitle_size}}", FONT_CONFIG["cover_subtitle"])
        customized_css = customized_css.replace("{{news_title_size}}", FONT_CONFIG["news_title"])
        customized_css = customized_css.replace("{{news_description_size}}", FONT_CONFIG["news_description"])
        customized_css = customized_css.replace("{{news_category_size}}", FONT_CONFIG["news_category"])
        customized_css = customized_css.replace("{{news_number_size}}", FONT_CONFIG["news_number"])
        customized_css = customized_css.replace("{{link_text_size}}", FONT_CONFIG["link_text"])
        customized_css = customized_css.replace("{{summary_title_size}}", FONT_CONFIG["summary_title"])
        customized_css = customized_css.replace("{{summary_subtitle_size}}", FONT_CONFIG["summary_subtitle"])
        customized_css = customized_css.replace("{{summary_item_title_size}}", FONT_CONFIG["summary_item_title"])
        
        return customized_css
    
    async def generate_all(self, html_content, css_content):
        embedded_fonts = self.generate_embedded_font_css()
        
        final_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>오늘의 긱뉴스</title>
            <style>
            {embedded_fonts}
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
            await page.set_viewport_size({"width": OUTPUT_CONFIG["page_width"], "height": OUTPUT_CONFIG["page_height"]})
            await page.goto(html_path_url, wait_until='networkidle')
            await page.wait_for_timeout(2000)
            if OUTPUT_CONFIG["generate_pdf"]:
                await page.pdf(
                    path=pdf_output_path, 
                    width=f"{OUTPUT_CONFIG['page_width']}px", 
                    height=f"{OUTPUT_CONFIG['page_height']}px", 
                    print_background=True,
                    margin=OUTPUT_CONFIG["pdf_margin"]
                )
                print(f"'{pdf_output_path}' 파일 생성 완료")
            pages = await page.query_selector_all('.page')
            print(f"총 {len(pages)} 페이지 이미지 생성 중...")
            for i, page_element in enumerate(pages, 1):
                if OUTPUT_CONFIG["generate_png"]:
                    png_filename = f"geek_page_{i:02d}.png"
                    png_output_path = os.path.join(PATH_CONFIG["image_dir"], png_filename)
                    await page_element.screenshot(
                        path=png_output_path,
                        type='png',
                        omit_background=False
                    )
                    print(f"'{png_filename}' 생성 완료")
                    
                if OUTPUT_CONFIG["generate_jpg"]:
                    jpg_filename = f"geek_page_{i:02d}.jpg"
                    jpg_output_path = os.path.join(PATH_CONFIG["image_dir"], jpg_filename)
                    await page_element.screenshot(
                        path=jpg_output_path,
                        type='jpeg',
                        quality=OUTPUT_CONFIG["image_quality"],
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
        news_items = await self.fetch_geek_news()
        if not news_items:
            print("뉴스를 가져올 수 없습니다.")
            return
        print(f"{len(news_items)}개의 뉴스를 가져왔습니다.")
        html_content = await self.create_html(news_items)
        css_content = self.create_styles()
        await self.generate_all(html_content, css_content)
        print("=== 긱뉴스 카드뉴스 생성 완료 ===")
async def main():
    generator = GeekNewsCardGenerator()
    await generator.generate()

if __name__ == "__main__":
    asyncio.run(main()) 