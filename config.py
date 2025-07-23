import os
from dotenv import load_dotenv

load_dotenv()

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
    "combined_template": "combined_template.html",
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
    # 요약 모드 설정: "huggingface" 또는 "openai"
    "summary_mode": "openai",  # 기본값은 허깅페이스
    # 허깅페이스 모델 설정
    "model_name": "lcw99/t5-large-korean-text-summary",
    "max_input_length": 768,
    "max_output_length": 100,
    "min_output_length": 50,
    "length_penalty": 2.0,
    "num_beams": 4,
    
    # OpenAI 모델 설정
    "openai_model": "gpt-4o-mini",  # 또는 "gpt-3.5-turbo"
    "openai_max_tokens": 80,
    "openai_temperature": 0.3
}

# 카카오톡 채널 설정
KAKAO_CONFIG = {
    "enabled": True, # 카카오톡 전송 기능 활성화 여부
    "rest_api_key": os.getenv("KAKAO_REST_API"), # [필수] .env 파일에서 읽어옴
    "redirect_uri": "https://example.com/oauth", # 카카오 개발자 사이트에 등록한 Redirect URI
    "token_file": "kakao_token.json", # 발급된 토큰이 저장될 파일명
    "server_base_url": "http://15.165.73.31:8000" # [필수] 외부에서 접근 가능한 서버 주소.
} 