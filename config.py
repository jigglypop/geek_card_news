# 파일 경로 설정
PATH_CONFIG = {
    "output_dir": "output",
    "template_dir": "templates",
    "style_file": "style.css",
    "cover_template": "cover_template.html",
    "news_template": "news_template.html",
    "summary_template": "summary_template.html",
    "summary_item_template": "summary_item_template.html",
    "summary_prompt": "summary_prompt.txt",
    "image_dir": "output/images",
    "character_dir": "image/character"
}

# 크롤링 설정
CRAWLING_CONFIG = {
    "news_count": 6,  # 가져올 뉴스 개수
    "base_url": "https://news.hada.io/",
    "timeout": 30  # HTTP 요청 타임아웃 (초)
}

# PDF/이미지 생성 설정
OUTPUT_CONFIG = {
    "page_width": 1920,
    "page_height": 1080,
    "pdf_margin": {'top': '0px', 'right': '0px', 'bottom': '0px', 'left': '0px'},
    "image_quality": 95,  # JPG 품질 (1-100)
    "generate_png": True,
    "generate_jpg": True,
    "generate_pdf": True
}

# 이미지 설정
IMAGE_CONFIG = {
    "character_extensions": [".png", ".jpg", ".jpeg"],
    "main_character": "1",
    "all_characters": "all",
    "character_names": ["1", "2", "3", "4", "5", "6", "7"]
}
# S3 설정
S3_CONFIG = {
    "use_s3": True,
    "bucket_name": "jiggloghttps",
    "region": "ap-northeast-2",
    "base_url": "https://jiggloghttps.s3.ap-northeast-2.amazonaws.com/",
    "character_prefix": "image/",
    "qr_code_key": "image/QR.png",
    "font_prefix": "fonts/"
}
# 색상 커스터마이징 설정
COLOR_CONFIG = {
    "cover_background": "linear-gradient(160deg, #FF5F6D 0%, #FFC371 100%)",    # 16:9 비율에 맞게 각도 조정
    "news_background": "linear-gradient(160deg, #FF5F6D 0%, #FFC371 100%)",     # 모든 페이지 통일
    "summary_background": "linear-gradient(160deg, #FF5F6D 0%, #FFC371 100%)",  # 모든 페이지 통일
    "end_background": "linear-gradient(160deg, #FF5F6D 0%, #FFC371 100%)"       # 모든 페이지 통일
}

# 폰트 크기 커스터마이징 설정
FONT_CONFIG = {
    "cover_title": "200px",
    "cover_subtitle": "50px",
    "news_title": "56px",        # 48px → 56px
    "news_description": "36px",   # 32px → 36px
    "news_category": "24px",      # 18px → 24px
    "news_number": "36px",
    "link_text": "22px",         # 18px → 22px
    "summary_title": "72px",
    "summary_subtitle": "36px",
    "summary_item_title": "26px"  # 22px → 26px
}
# 텍스트 커스터마이징 설정
TEXT_CONFIG = {
    "cover_subtitle": "모여봐요 개발자와 AI의 숲",
    "cover_title": "모드뉴스",
    "news_card_prefix": "GeekNews", 
    "speech_bubble_text": "뉴-스!",
    "summary_title": "GeekNews 요약",
    "summary_subtitle": "오늘의 주요 뉴스",
    "summary_footer_text": "총 {count}개의 뉴스를 확인했어요",
    "summary_source": "출처: GeekNews (news.hada.io)"
}
# 이모지 커스터마이징 설정
EMOJI_CONFIG = {
    "speech_bubble": "💬",
    "lightbulb": "💡",
    "star": "⭐"
}
# AI 모델 설정
AI_CONFIG = {
    "model_name": "lcw99/t5-large-korean-text-summary",
    "max_input_length": 768,
    "max_output_length": 150,
    "min_output_length": 50,
    "length_penalty": 2.0,
    "num_beams": 4
} 