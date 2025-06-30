import os
import asyncio
from datetime import datetime
from bs4 import BeautifulSoup
from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch
from config import *
from dotenv import load_dotenv
import httpx
from playwright.async_api import async_playwright
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
import uvicorn
from contextlib import asynccontextmanager
from PIL import Image
import io
import logging
from PIL import ImageDraw

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

class GeekNewsCardGenerator:
    def __init__(self):
        self.output_dir = PATH_CONFIG["output_dir"]
        self.template_dir = PATH_CONFIG["template_dir"]
        self.generated_image_path = os.path.join(self.output_dir, "geek_news.png")
        self._initialize_model()

    def _initialize_model(self):
        logging.info("한국어 요약 모델 로딩 중...")
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.tokenizer = AutoTokenizer.from_pretrained(
            AI_CONFIG["model_name"], token=hf_token, trust_remote_code=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            AI_CONFIG["model_name"], token=hf_token
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        logging.info("모델 로딩 완료")

    def summarize_text(self, text):
        if not text or len(text.strip()) < 50:
            logging.info(f"  [요약 건너뜀] 텍스트가 너무 짧습니다: {text[:100]}")
            return ""
        logging.info(f"  [요약 원문] {text[:150]}...")
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
            logging.info(f"  [요약 성공] {summary.strip()}")
            return summary.strip()
        except Exception as e:
            logging.error(f"  [요약 실패] 오류: {e}")
            return "요약 생성에 실패했습니다."

    async def get_detail(self, topic_id):
        try:
            async with httpx.AsyncClient() as client:
                url = f'https://news.hada.io/topic?id={topic_id}'
                response = await client.get(url)
                soup = BeautifulSoup(response.text, 'lxml')
                contents_elem = soup.find('div', class_='topic_contents')
                if contents_elem:
                    return contents_elem.get_text(separator=' ', strip=True)
                desc_elem = soup.find('div', class_='topic_desc')
                if desc_elem:
                    return desc_elem.get_text(strip=True)
                return ""
        except Exception as e:
            logging.error(f"긱뉴스 상세 정보 가져오기 실패: {e}")
            return ""

    async def fetch_news(self):
        async with httpx.AsyncClient(timeout=CRAWLING_CONFIG["timeout"]) as client:
            response = await client.get(CRAWLING_CONFIG["base_url"])
            soup = BeautifulSoup(response.text, 'lxml')
            news_items = []
            topics = soup.find_all('div', class_='topic_row')[:CRAWLING_CONFIG["news_count"]]
            for topic in topics:
                title_elem = topic.find('div', class_='topictitle')
                if not title_elem: continue
                title_link = title_elem.find('a')
                if not title_link: continue

                title = title_link.text.strip()
                original_link = title_link.get('href', '')

                topic_id = None
                all_links = topic.find_all('a')
                for link in all_links:
                    href = link.get('href', '')
                    if 'topic?id=' in href:
                        try:
                            topic_id = href.split('id=')[-1].split('&')[0]
                            break
                        except:
                            pass
                
                desc_elem = topic.find('span', class_='topicdesc')
                desc = desc_elem.text.strip() if desc_elem else ''

                if topic_id:
                    detailed_desc = await self.get_detail(topic_id)
                    if detailed_desc and len(detailed_desc) > len(desc):
                        desc = detailed_desc
                
                summarized_desc = self.summarize_text(desc)
                news_items.append({
                    'title': title,
                    'description': summarized_desc if summarized_desc else "요약 정보가 없습니다.",
                    'link': original_link,
                    'topic_id': topic_id
                })
            return news_items

    def get_character_sources(self):
        characters = {}
        base_url = S3_CONFIG["base_url"]
        prefix = S3_CONFIG["character_prefix"]
        for name in IMAGE_CONFIG["character_names"]:
            characters[name] = f"{base_url}{prefix}{name}.png"
        return characters

    def load_template(self, template_name):
        template_path = os.path.join(self.template_dir, template_name)
        return self.read_file(template_path)

    def _render_cover_page(self, character_src, qr_src):
        cover_template = self.load_template(PATH_CONFIG["cover_template"])
        character_html = f'<img src="{character_src}" class="character main-character" alt="캐릭터" />' if character_src else ''
        qr_html = f'<div class="qr-section"><img src="{qr_src}" class="qr-code" alt="QR코드" /></div>' if qr_src else ''
        
        return cover_template.format(
            cover_subtitle=TEXT_CONFIG["cover_subtitle"],
            cover_title=TEXT_CONFIG["cover_title"],
            character_image=character_html,
            qr_section=qr_html
        )
    
    def _render_news_pages(self, news_items, page_characters):
        news_template = self.load_template(PATH_CONFIG["news_template"])
        pages_html = ""

        for i in range(0, len(news_items), 2):
            page_news_items = news_items[i:i+2]
            news_html_parts = []

            for news_item in page_news_items:
                domain_label = "원문"
                link = news_item.get('link')
                if link:
                    if "github.com" in link: domain_label = "GitHub"
                    elif "youtube.com" in link or "youtu.be" in link: domain_label = "YouTube"
                    elif "blog" in link or "medium.com" in link: domain_label = "블로그"
                    elif "news" in link: domain_label = "뉴스"
                
                geek_news_link = f"https://news.hada.io/topic?id={news_item['topic_id']}" if news_item.get('topic_id') else ""
                
                topic_category = "기술"
                link_lower = link.lower() if link else ''
                title_lower = news_item.get('title', '').lower()
                if "github" in link_lower or "git" in title_lower: topic_category = "개발"
                elif "youtube" in link_lower: topic_category = "영상"
                elif "blog" in link_lower: topic_category = "블로그"
                elif "ai" in title_lower or "gemini" in title_lower: topic_category = "AI"
                elif "데이터" in title_lower or "data" in title_lower: topic_category = "데이터"

                links_html = ""
                if geek_news_link:
                    links_html += f'<div class="link-item"><span class="link-label">토론:</span><a href="{geek_news_link}" target="_blank">{geek_news_link}</a></div>'
                if link:
                    links_html += f'<div class="link-item"><span class="link-label">{domain_label}:</span><a href="{link}" target="_blank">{link}</a></div>'

                news_html_parts.append(f"""
                    <div class="news-item">
                        <div class="news-header"><span class="news-category">{topic_category}</span></div>
                        <h2 class="news-title">{news_item['title']}</h2>
                        <p class="news-description">{news_item['description']}</p>
                        <div class="links">{links_html}</div>
                    </div>
                """)
            
            news_content_for_page = f'<div class="news-container">{"".join(news_html_parts)}</div>'
            
            character_src = page_characters[(i // 2) % len(page_characters)] if page_characters else None
            character_html = f'<img src="{character_src}" class="page-character" alt="캐릭터" />' if character_src else ''

            pages_html += news_template.format(
                page_number=(i // 2) + 1,
                news_content=news_content_for_page,
                character_image=character_html
            )
        return pages_html

    def _render_summary_page(self, news_items):
        current_date = datetime.now().strftime("%Y년 %m월 %d일")
        summary_item_template = self.load_template(PATH_CONFIG["summary_item_template"])
        summary_items_html = ""
        
        for index, news in enumerate(news_items, 1):
            topic_category = "기술"
            link_lower = news.get('link', '').lower()
            title_lower = news.get('title', '').lower()
            if "github" in link_lower or "git" in title_lower: topic_category = "개발"
            elif "ai" in title_lower or "gemini" in title_lower: topic_category = "AI"
            elif "데이터" in title_lower or "data" in title_lower: topic_category = "데이터"
            
            summary_items_html += summary_item_template.format(
                number=index,
                category=topic_category,
                title=news['title']
            )

        summary_template = self.load_template(PATH_CONFIG["summary_template"])
        return summary_template.format(
            summary_title=TEXT_CONFIG["summary_title"],
            summary_date=current_date,
            summary_subtitle=TEXT_CONFIG["summary_subtitle"],
            summary_items=summary_items_html,
            summary_footer=TEXT_CONFIG["summary_footer_text"].format(count=len(news_items)),
            summary_source=TEXT_CONFIG["summary_source"]
        )

    async def create_html(self, news_items):
        available_characters = self.get_character_sources()
        main_character_src = available_characters.get(IMAGE_CONFIG["main_character"])
        page_characters = [
            src for name, src in available_characters.items() 
            if name != IMAGE_CONFIG["main_character"] and name != IMAGE_CONFIG["all_characters"]
        ]
        qr_src = f"{S3_CONFIG['base_url']}{S3_CONFIG['qr_code_key']}"
        cover_html = self._render_cover_page(main_character_src, qr_src)
        news_html = self._render_news_pages(news_items, page_characters)
        summary_html = self._render_summary_page(news_items)
        
        return f"{cover_html}{news_html}{summary_html}"

    def create_styles(self):
        base_css = self.load_template(PATH_CONFIG["style_file"])
        
        # .format() 대신 .replace()를 사용하여 안전하게 치환
        css = base_css
        css = css.replace("{{page_width}}", str(OUTPUT_CONFIG["page_width"]))
        css = css.replace("{{page_height}}", str(OUTPUT_CONFIG["page_height"]))
        css = css.replace("{{cover_title_size}}", FONT_CONFIG["cover_title"])
        css = css.replace("{{cover_subtitle_size}}", FONT_CONFIG["cover_subtitle"])
        css = css.replace("{{news_title_size}}", FONT_CONFIG["news_title"])
        css = css.replace("{{news_description_size}}", FONT_CONFIG["news_description"])
        css = css.replace("{{news_category_size}}", FONT_CONFIG["news_category"])
        css = css.replace("{{news_number_size}}", FONT_CONFIG["news_number"])
        css = css.replace("{{link_text_size}}", FONT_CONFIG["link_text"])
        css = css.replace("{{summary_title_size}}", FONT_CONFIG["summary_title"])
        css = css.replace("{{summary_subtitle_size}}", FONT_CONFIG["summary_subtitle"])
        css = css.replace("{{summary_item_title_size}}", FONT_CONFIG["summary_item_title"])
        
        return css

    async def generate_combined_image(self, news_items):
        html_content = await self.create_html(news_items)
        css_content = self.create_styles()
        
        final_html = f"<html><head><style>{css_content}</style></head><body>{html_content}</body></html>"
        
        logging.info("전체 페이지 스크린샷 생성 중...")
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(final_html)
            
            # 뷰포트 너비 설정, 높이는 기본값으로 두되 full_page=True로 전체를 캡처
            await page.set_viewport_size({
                "width": OUTPUT_CONFIG["page_width"], 
                "height": 1080  # 임시 높이, full_page가 실제 컨텐츠 높이를 사용함
            })
            
            await page.screenshot(path=self.generated_image_path, full_page=True)
            await browser.close()
            
        logging.info(f"'{self.generated_image_path}' 파일 생성 완료")

    def read_file(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            logging.error(f"오류: 파일 '{filepath}'을(를) 찾을 수 없습니다.")
            return ""
        except IOError as e:
            logging.error(f"파일 읽기 오류: {e}")
            return ""

    def create_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)

    async def generate(self):
        self.create_directory()
        logging.info("=== 긱뉴스 카드뉴스 생성 시작 ===")
        news_items = await self.fetch_news()
        
        if not news_items:
            logging.warning("뉴스를 가져오지 못했습니다.")
            return

        logging.info(f"{len(news_items)}개의 뉴스를 가져왔습니다.")
        
        await self.generate_combined_image(news_items)

        logging.info("=== 긱뉴스 카드뉴스 생성 완료 ===")

generator = GeekNewsCardGenerator()
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("FastAPI 애플리케이션 시작")
    if not os.path.exists(generator.generated_image_path):
        logging.info("초기 이미지 생성...")
        await generator.generate()
    
    scheduler.add_job(generator.generate, 'cron', hour=0)
    scheduler.start()
    logging.info("스케줄러 시작, 매일 자정 뉴스 업데이트가 예약되었습니다.")
    yield
    # Shutdown
    scheduler.shutdown()
    logging.info("스케줄러 종료")

app = FastAPI(lifespan=lifespan)

@app.get("/news", response_class=FileResponse)
async def get_news_image():
    if not os.path.exists(generator.generated_image_path):
        return Response(content="뉴스 이미지를 찾을 수 없습니다. 생성 중일 수 있으니 잠시 후 다시 시도해주세요.", status_code=404)
    return FileResponse(generator.generated_image_path, media_type="image/png")

@app.post("/news/refresh")
async def refresh_news():
    asyncio.create_task(generator.generate())
    return Response(content="뉴스 생성을 시작했습니다. 완료까지 몇 분 정도 소요될 수 있습니다.", status_code=202)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 