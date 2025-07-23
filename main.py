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
import logging
from openai import AsyncOpenAI
import json
import aioboto3
import time
from fastapi.staticfiles import StaticFiles

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

class KakaoManager:
    def __init__(self, config):
        self.kakao_config = config
        self.token_file = self.kakao_config['token_file']
        self.session = httpx.AsyncClient()

    async def _load_tokens(self):
        try:
            with open(self.token_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.warning(f"카카오 토큰 파일({self.token_file})을 찾을 수 없습니다. kakao_auth.py를 먼저 실행하세요.")
            return None
    
    async def _save_tokens(self, tokens):
        with open(self.token_file, 'w') as f:
            json.dump(tokens, f)

    async def _refresh_token_if_needed(self):
        tokens = await self._load_tokens()
        if not tokens:
            return None

        # 실제로는 만료 시간을 확인해야 하지만, 여기서는 리프레시 토큰 존재 여부로 간략화
        if 'refresh_token' in tokens:
            url = "https://kauth.kakao.com/oauth/token"
            data = {
                "grant_type": "refresh_token",
                "client_id": self.kakao_config['rest_api_key'],
                "refresh_token": tokens['refresh_token']
            }
            response = await self.session.post(url, data=data)
            
            if response.status_code == 200:
                new_tokens = response.json()
                # 카카오는 리프레시 토큰을 한 번 사용하면 만료될 수 있으므로, 새 리프레시 토큰이 오면 업데이트
                tokens['access_token'] = new_tokens['access_token']
                if 'refresh_token' in new_tokens:
                    tokens['refresh_token'] = new_tokens['refresh_token']
                
                await self._save_tokens(tokens)
                logging.info("카카오 액세스 토큰을 성공적으로 갱신했습니다.")
                return tokens['access_token']
            else:
                logging.error(f"카카오 토큰 갱신 실패: {response.text}")
                return None
        return tokens.get('access_token')

    async def send_card_news(self, image_path, news_items):
        logging.info("카카오톡 채널 메시지 전송 시작")
        access_token = await self._refresh_token_if_needed()
        if not access_token:
            logging.error("유효한 카카오 액세스 토큰이 없어 메시지를 보낼 수 없습니다.")
            return

        # 1. 친구 목록 가져오기
        friend_list_url = "https://kapi.kakao.com/v1/api/talk/friends"
        headers = {"Authorization": f"Bearer {access_token}"}
        response = await self.session.get(friend_list_url, headers=headers)
        
        if response.status_code != 200:
            logging.error(f"카카오 친구 목록 가져오기 실패: {response.text}")
            return
            
        friends = response.json().get("elements", [])
        if not friends:
            logging.warning("메시지를 보낼 채널 친구가 없습니다. 채널을 추가한 사용자가 있는지 확인하세요.")
            return

        friend_uuids = [friend["uuid"] for friend in friends]
        logging.info(f"총 {len(friend_uuids)}명의 친구에게 메시지를 보냅니다.")

        # 2. 이미지 URL 생성
        image_filename = os.path.basename(image_path)
        # FastAPI 정적 파일 경로에 맞춰 URL 구성
        image_url = f"{self.kakao_config['server_base_url']}/static/{image_filename}"
        
        # 3. 카카오톡 피드 템플릿 메시지 구성
        template_object = {
            "object_type": "feed",
            "content": {
                "title": f"{time.strftime('%Y년 %m월 %d일')} 오늘의 뉴스",
                "description": "새로운 소식이 도착했어요! 아래 버튼을 눌러 확인해보세요.",
                "image_url": image_url,
                "image_width": 1200,
                "image_height": 630,
                "link": {
                    "web_url": f"{self.kakao_config['server_base_url']}/news/html",
                    "mobile_web_url": f"{self.kakao_config['server_base_url']}/news/html",
                }
            },
            "buttons": [
                {
                    "title": "뉴스 보러가기",
                    "link": {
                        "web_url": f"{self.kakao_config['server_base_url']}/news/html",
                        "mobile_web_url": f"{self.kakao_config['server_base_url']}/news/html"
                    }
                }
            ]
        }
        
        # 4. 메시지 전송
        message_api_url = "https://kapi.kakao.com/v1/api/talk/friends/message/default/send"
        payload = {
            'receiver_uuids': json.dumps(friend_uuids),
            'template_object': json.dumps(template_object)
        }
        
        response = await self.session.post(message_api_url, headers=headers, data=payload)
        if response.status_code == 200 and response.json().get('successful_receiver_uuids'):
            logging.info("채널 친구에게 메시지를 성공적으로 보냈습니다.")
        else:
            logging.error(f"메시지 보내기 실패: {response.text}")

class GeekNewsCardGenerator:
    def __init__(self):
        self.output_dir = PATH_CONFIG["output_dir"]
        self.template_dir = PATH_CONFIG["template_dir"]
        self.generated_image_path = os.path.join(self.output_dir, "geek_news.png")
        self.generated_html_path = os.path.join(self.output_dir, "geek_news.html")
        self.summary_mode = AI_CONFIG["summary_mode"]
        if KAKAO_CONFIG.get("enabled"):
            self.kakao_manager = KakaoManager(KAKAO_CONFIG)
        else:
            self.kakao_manager = None
        self._initialize_model()

    def _initialize_model(self):
        if self.summary_mode == "huggingface":
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
            logging.info("허깅페이스 모델 로딩 완료")
        elif self.summary_mode == "openai":
            logging.info("OpenAI 클라이언트 초기화 중...")
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
            logging.info("OpenAI 클라이언트 초기화 완료")
        else:
            raise ValueError(f"지원하지 않는 요약 모드입니다: {self.summary_mode}")

    async def summarize_text(self, text):
        if not text or len(text.strip()) < 50:
            logging.info(f"  [요약 건너뜀] 텍스트가 너무 짧습니다: {text[:100]}")
            return ""
        logging.info(f"  [요약 원문] {text[:150]}...")
        try:
            if self.summary_mode == "huggingface":
                return await self._summarize_huggingface(text)
            elif self.summary_mode == "openai":
                return await self._summarize_openai(text)
            else:
                raise ValueError(f"지원하지 않는 요약 모드입니다: {self.summary_mode}")
        except Exception as e:
            logging.error(f"  [요약 실패] 오류: {e}")
            return "요약 생성에 실패했습니다."

    async def _summarize_huggingface(self, text):
        return await asyncio.to_thread(self._summarize_huggingface_sync, text)

    def _summarize_huggingface_sync(self, text):
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
        logging.info(f"  [허깅페이스 요약 성공] {summary.strip()}")
        return summary.strip()

    async def _summarize_openai(self, text):
        prompt_template = self.load_template(PATH_CONFIG["summary_prompt"])
        prompt = prompt_template.format(text=text)
        
        response = await self.openai_client.chat.completions.create(
            model=AI_CONFIG["openai_model"],
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes Korean text."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=AI_CONFIG["openai_max_tokens"],
            temperature=AI_CONFIG["openai_temperature"]
        )
        summary = response.choices[0].message.content.strip()
        logging.info(f"  [OpenAI 요약 성공] {summary}")
        return summary

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
                summarized_desc = await self.summarize_text(desc)
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

        character_iterator = iter(page_characters)

        for i in range(0, len(news_items), 2):
            page_news_items = news_items[i:i+2]
            news_html_parts = []

            for idx, news_item in enumerate(page_news_items):
                item_number = i + idx + 1
                domain_label = "원문"
                link = news_item.get('link')
                if link:
                    if "github.com" in link: domain_label = "GitHub"
                    elif "youtube.com" in link or "youtu.be" in link: domain_label = "YouTube"
                    elif "blog" in link or "medium.com" in link: domain_label = "블로그"
                    elif "news" in link: domain_label = "뉴스"
                
                geek_news_link = f"https://news.hada.io/topic?id={news_item['topic_id']}" if news_item.get('topic_id') else ""
                topic_category = "기술"
                category_class = "tech"
                link_lower = link.lower() if link else ''
                title_lower = news_item.get('title', '').lower()
                if "github" in link_lower or "git" in title_lower: 
                    topic_category = "개발"
                    category_class = "dev"
                elif "youtube" in link_lower: 
                    topic_category = "영상"
                    category_class = "video"
                elif "blog" in link_lower: 
                    topic_category = "블로그"
                    category_class = "blog"
                elif "ai" in title_lower or "gemini" in title_lower or "gpt" in title_lower: 
                    topic_category = "AI"
                    category_class = "ai"
                elif "데이터" in title_lower or "data" in title_lower: 
                    topic_category = "데이터"
                    category_class = "data"
                links_html = ""
                if geek_news_link:
                    links_html += f'<div class="link-item"><span class="link-label">토론:</span><a href="{geek_news_link}" target="_blank">{geek_news_link}</a></div>'
                if link:
                    links_html += f'<div class="link-item"><span class="link-label">{domain_label}:</span><a href="{link}" target="_blank">{link}</a></div>'
                try:
                    char_src = next(character_iterator)
                    character_html = f'<img src="{char_src}" class="page-character" alt="페이지 캐릭터" />'
                except StopIteration:
                    character_html = "" # 캐릭터가 더 없으면 빈 문자열
                news_html_parts.append(f"""
                    <div class="news-item">
                        <div class="news-header">
                            <span class="news-category {category_class}">{topic_category}</span>
                        </div>
                        <h2 class="news-title"><span>{news_item['title']}</span></h2>
                        <p class="news-description">{news_item['description']}</p>
                        <div class="links">{links_html}</div>
                        {character_html}
                    </div>
                """)
            news_content_for_page = f'<div class="news-container">{"".join(news_html_parts)}</div>'
            


            pages_html += news_template.format(
                page_number=(i // 2) + 1,
                news_content=news_content_for_page)
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
        css = base_css
        return css

    async def generate_and_save_html(self, news_items):
        html_content = await self.create_html(news_items)
        css_content = self.create_styles()
    
        combined_template = self.load_template(PATH_CONFIG["combined_template"])
        final_html = combined_template.replace("{{css_content}}", css_content).replace("{{html_content}}", html_content)

        with open(self.generated_html_path, 'w', encoding='utf-8') as f:
            f.write(final_html)
        logging.info(f"'{self.generated_html_path}' 파일 생성 완료")

        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            await page.set_content(final_html)
            
            # 카톡 미리보기용 썸네일 이미지 생성
            preview_image_path = os.path.join(self.output_dir, "geek_news_preview.png")
            
            # body의 첫번째 자식 요소(커버)만 스크린샷
            cover_element = await page.query_selector('body > div:first-child')
            if cover_element:
                 await cover_element.screenshot(path=preview_image_path)
                 logging.info(f"'{preview_image_path}' 썸네일 생성 완료")

            await browser.close()
        
        return self.generated_html_path, preview_image_path

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
        try:
            logging.info("=== 긱뉴스 카드뉴스 생성 시작 ===")
            self.create_directory()
            news_items = await self.fetch_news()

            if not news_items:
                logging.info("가져올 뉴스가 없습니다.")
                return

            logging.info(f"{len(news_items)}개의 뉴스를 가져왔습니다.")
            
            html_path, preview_image_path = await self.generate_and_save_html(news_items)
            
            # PNG 대신 HTML을 보내도록 send_card_news를 호출하지만, 
            # 카톡 템플릿에는 썸네일 이미지가 필요하므로 preview_image_path를 전달합니다.
            if html_path and self.kakao_manager:
                await self.kakao_manager.send_card_news(preview_image_path, news_items)
            
            logging.info("=== 긱뉴스 카드뉴스 생성 완료 ===")

        except Exception as e:
            logging.error(f"카드뉴스 생성 중 오류 발생: {e}", exc_info=True)

generator = GeekNewsCardGenerator()
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logging.info("FastAPI 애플리케이션 시작")
    if not os.path.exists(generator.generated_html_path):
        logging.info("초기 HTML 생성...")
        await generator.generate()
    
    scheduler.add_job(generator.generate, 'cron', hour=0)
    scheduler.start()
    logging.info("스케줄러 시작, 매일 자정 뉴스 업데이트가 예약되었습니다.")
    yield
    scheduler.shutdown()
    logging.info("스케줄러 종료")

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=PATH_CONFIG["output_dir"]), name="static")

@app.get("/", response_class=FileResponse)
async def get_news_image_legacy():
    if not os.path.exists(generator.generated_image_path):
        return Response(content="뉴스 이미지를 찾을 수 없습니다. 생성 중일 수 있으니 잠시 후 다시 시도해주세요.", status_code=404)
    return FileResponse(generator.generated_image_path, media_type="image/png")

@app.get("/news/html", response_class=FileResponse)
async def get_news_html():
    if not os.path.exists(generator.generated_html_path):
        return Response(content="뉴스 HTML을 찾을 수 없습니다. 생성 중일 수 있으니 잠시 후 다시 시도해주세요.", status_code=404)
    return FileResponse(generator.generated_html_path, media_type="text/html")

@app.post("/news/refresh")
async def refresh_news():
    asyncio.create_task(generator.generate())
    return Response(content="뉴스 생성을 시작했습니다. 완료까지 몇 분 정도 소요될 수 있습니다.", status_code=202)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 