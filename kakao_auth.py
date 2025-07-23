import requests
import json
import webbrowser
import time
from config import KAKAO_CONFIG

# --- 카카오 인증 토큰 받기 (최초 1회 실행 필요) ---
def get_kakao_token():
    # 인증 코드 받기
    auth_url = f"https://kauth.kakao.com/oauth/authorize?client_id={KAKAO_CONFIG['rest_api_key']}&redirect_uri={KAKAO_CONFIG['redirect_uri']}&response_type=code&scope=talk_message,friends"
    print("--- 카카오톡 채널 연동을 위한 최초 인증 ---")
    print("1. 아래 주소로 접속하여 '동의하고 계속하기'를 눌러주세요.")
    print(f"   => {auth_url}")
    print("2. 로그인 후 리다이렉트된 페이지의 주소창에서 'code=' 뒷부분의 값을 복사하세요.")
    
    webbrowser.open(auth_url)
    time.sleep(1) # 브라우저가 열릴 시간을 잠시 기다립니다.

    AUTHORIZATION_CODE = input("3. 복사한 인증 코드(code=...)를 여기에 붙여넣으세요: ").strip()

    # 토큰 받기
    token_url = 'https://kauth.kakao.com/oauth/token'
    data = {
        'grant_type': 'authorization_code',
        'client_id': KAKAO_CONFIG['rest_api_key'],
        'redirect_uri': KAKAO_CONFIG['redirect_uri'],
        'code': AUTHORIZATION_CODE,
    }
    
    try:
        response = requests.post(token_url, data=data)
        response.raise_for_status()
        tokens = response.json()

        if 'access_token' in tokens:
            print("\n>> 인증 성공! 엑세스 토큰이 발급되었습니다.")
            # 받은 토큰을 파일에 저장
            with open(KAKAO_CONFIG['token_file'], "w") as fp:
                json.dump(tokens, fp)
            print(f">> '{KAKAO_CONFIG['token_file']}' 파일에 토큰을 저장했습니다.")
            print(">> 이제 메인 애플리케이션을 실행하여 메시지를 보낼 수 있습니다.")
            return tokens
        else:
            print("\n>> [오류] 엑세스 토큰 발급에 실패했습니다. 응답을 확인하세요.")
            print(tokens)
            return None
    except requests.exceptions.HTTPError as err:
        print(f"\n>> [오류] 토큰 발급 요청 중 HTTP 오류가 발생했습니다: {err}")
        print(f">> 응답 내용: {err.response.text}")
        return None
    except Exception as e:
        print(f"\n>> [오류] 알 수 없는 오류가 발생했습니다: {e}")
        return None

if __name__ == "__main__":
    get_kakao_token() 