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

load_dotenv()

class GeekNewsCardGenerator:
    def __init__(self):
        self.output_dir = PATH_CONFIG["output_dir"]
        self.template_dir = PATH_CONFIG["template_dir"]
        
        print("한국어 요약 모델 로딩 중...")
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        self.tokenizer = AutoTokenizer.from_pretrained(
            AI_CONFIG["model_name"], token=hf_token, trust_remote_code=True
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            AI_CONFIG["model_name"], token=hf_token
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
            print(f"긱뉴스 상세 정보 가져오기 실패: {e}")
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
        """S3 URL로부터 캐릭터 이미지 소스를 생성합니다."""
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
        """커버 페이지 HTML을 렌더링합니다."""
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
        """뉴스 페이지들의 HTML을 렌더링합니다."""
        news_template = self.load_template(PATH_CONFIG["news_template"])
        pages_html = ""

        for page_index, news in enumerate(news_items):
            character_src = page_characters[page_index % len(page_characters)] if page_characters else None
            
            domain_label = "원문"
            if news.get('link'):
                if "github.com" in news['link']: domain_label = "GitHub"
                elif "youtube.com" in news['link'] or "youtu.be" in news['link']: domain_label = "YouTube"
                elif "blog" in news['link'] or "medium.com" in news['link']: domain_label = "블로그"
                elif "news" in news['link']: domain_label = "뉴스"

            geek_news_link = f"https://news.hada.io/topic?id={news['topic_id']}" if news.get('topic_id') else ""
            
            topic_category = "기술"
            link_lower = news.get('link', '').lower()
            title_lower = news.get('title', '').lower()
            if "github" in link_lower or "git" in title_lower: topic_category = "개발"
            elif "youtube" in link_lower: topic_category = "영상"
            elif "blog" in link_lower: topic_category = "블로그"
            elif "ai" in title_lower or "gemini" in title_lower: topic_category = "AI"
            elif "데이터" in title_lower or "data" in title_lower: topic_category = "데이터"

            links_html = ""
            if geek_news_link:
                links_html += f'<div class="link-item"><span class="link-label">토론:</span>{geek_news_link}</div>'
            if news.get('link'):
                links_html += f'<div class="link-item"><span class="link-label">{domain_label}:</span>{news["link"]}</div>'
            character_html = f'<img src="{character_src}" class="page-character" alt="캐릭터" />' if character_src else ''
            
            pages_html += news_template.format(
                topic_category=topic_category,
                page_number=page_index + 1,
                news_title=news['title'],
                news_description=news['description'],
                links=links_html,
                character_image=character_html
            )
        return pages_html

    def _render_summary_page(self, news_items):
        """요약 페이지 HTML을 렌더링합니다."""
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
        """모든 페이지의 HTML을 생성하고 결합합니다."""
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
        """템플릿과 설정값을 결합하여 최종 CSS를 생성합니다."""
        base_css = self.load_template(PATH_CONFIG["style_file"])
        
        replacements = {
            "{{page_width}}": str(OUTPUT_CONFIG["page_width"]),
            "{{page_height}}": str(OUTPUT_CONFIG["page_height"]),
            "{{cover_background}}": COLOR_CONFIG["cover_background"],
            "{{news_background}}": COLOR_CONFIG["news_background"],
            "{{summary_background}}": COLOR_CONFIG["summary_background"],
            "{{end_background}}": COLOR_CONFIG["end_background"],
            "{{cover_title_size}}": FONT_CONFIG["cover_title"],
            "{{cover_subtitle_size}}": FONT_CONFIG["cover_subtitle"],
            "{{news_title_size}}": FONT_CONFIG["news_title"],
            "{{news_description_size}}": FONT_CONFIG["news_description"],
            "{{news_category_size}}": FONT_CONFIG["news_category"],
            "{{news_number_size}}": FONT_CONFIG["news_number"],
            "{{link_text_size}}": FONT_CONFIG["link_text"],
            "{{summary_title_size}}": FONT_CONFIG["summary_title"],
            "{{summary_subtitle_size}}": FONT_CONFIG["summary_subtitle"],
            "{{summary_item_title_size}}": FONT_CONFIG["summary_item_title"],
        }
        
        customized_css = base_css
        for placeholder, value in replacements.items():
            customized_css = customized_css.replace(placeholder, value)
            
        return customized_css

    async def generate_all(self, html_content, css_content):
        """개별 페이지 PDF와 이미지들을 생성합니다."""
        final_html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <title>오늘의 긱뉴스</title>
            <style>{css_content}</style>
        </head>
        <body>{html_content}</body>
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
            image_dir = PATH_CONFIG["image_dir"]
            for i, page_element in enumerate(pages, 1):
                if OUTPUT_CONFIG["generate_png"]:
                    png_filename = f"geek_page_{i:02d}.png"
                    png_output_path = os.path.join(image_dir, png_filename)
                    await page_element.screenshot(path=png_output_path, type='png', omit_background=False)
                    print(f"'{png_filename}' 생성 완료")
                
                if OUTPUT_CONFIG["generate_jpg"]:
                    jpg_filename = f"geek_page_{i:02d}.jpg"
                    jpg_output_path = os.path.join(image_dir, jpg_filename)
                    await page_element.screenshot(path=jpg_output_path, type='jpeg', quality=OUTPUT_CONFIG["image_quality"], omit_background=False)
                    print(f"'{jpg_filename}' 생성 완료")
            
            await browser.close()

    async def generate_combined_image(self, news_items):
        """모든 페이지를 합친 단일 이미지를 생성합니다."""
        print("연속 그라데이션을 적용하여 이미지를 결합하는 중...")

        template_str = self.load_template("combined_template.html")
        if not template_str:
            print("통합 이미지 템플릿을 찾을 수 없습니다.")
            return

        html_content = await self.create_html(news_items)
        base_css = self.create_styles()

        # .replace()를 사용하여 모든 플레이스홀더를 순차적으로 안전하게 치환합니다.
        final_html = template_str.replace("{{ base_css }}", base_css)
        final_html = final_html.replace("{{ html_content }}", html_content)
        final_html = final_html.replace("{{ page_width }}", str(OUTPUT_CONFIG["page_width"]))
        final_html = final_html.replace("{{ page_height }}", str(OUTPUT_CONFIG["page_height"]))
        final_html = final_html.replace("{{ cover_background }}", COLOR_CONFIG["cover_background"])

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        combined_html_path = os.path.join(self.output_dir, f"combined_view_{timestamp}.html")
        combined_image_path = os.path.join(self.output_dir, "geek_news.png")

        with open(combined_html_path, "w", encoding="utf-8") as f:
            f.write(final_html)

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                html_path_url = f'file://{os.path.abspath(combined_html_path)}'
                
                await page.goto(html_path_url, wait_until='networkidle')
                await page.locator('#wrapper').screenshot(path=combined_image_path)
                await browser.close()
            
            print(f"'{combined_image_path}' 파일 생성 완료")
        
        except Exception as e:
            print(f"통합 이미지 생성 실패: {e}")
        
        finally:
            if os.path.exists(combined_html_path):
                os.remove(combined_html_path)

    def read_file(self, filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"오류: 파일 '{filepath}'을(를) 찾을 수 없습니다.")
            return ""

    def create_directory(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(PATH_CONFIG["image_dir"], exist_ok=True)

    async def generate(self):
        print("=== 긱뉴스 카드뉴스 생성 시작 ===")
        self.create_directory()
        news_items = await self.fetch_news()
        if not news_items:
            print("뉴스를 가져올 수 없습니다.")
            return
        print(f"{len(news_items)}개의 뉴스를 가져왔습니다.")
        html_content = await self.create_html(news_items)
        css_content = self.create_styles()
        await self.generate_all(html_content, css_content)
        await self.generate_combined_image(news_items)
        print("=== 긱뉴스 카드뉴스 생성 완료 ===")

async def main():
    generator = GeekNewsCardGenerator()
    await generator.generate()

if __name__ == "__main__":
    asyncio.run(main()) 