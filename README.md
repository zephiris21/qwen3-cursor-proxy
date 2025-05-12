# Qwen3 Cursor 프록시

이 프로젝트는 Ollama에서 실행되는 Qwen3 모델을 Cursor IDE에서 사용할 수 있게 해주는 프록시 서버입니다. 특히 Qwen3의 XML 기반 도구 호출 형식을 Cursor에서 사용 가능한 OpenAI 호환 형식으로 변환해줍니다.

## 주요 기능

- Ollama의 Qwen3 모델을 Cursor IDE에서 사용할 수 있는 OpenAI 호환 API 제공
- Qwen3의 XML 기반 도구 호출(function calling) 형식 지원
- `/think` 명령어를 통한 사고 과정 표시 기능
- Cloudflare 터널 통합으로 로컬 네트워크 제한 우회 (옵션)

## 설치 방법

1. 저장소 클론:

```bash
git clone https://github.com/your-username/qwen3-cursor-proxy.git
cd qwen3-cursor-proxy
```

2. 필요한 패키지 설치:

```bash
pip install -r requirements.txt
```

3. Ollama 설치 및 Qwen3 모델 다운로드:
   - [Ollama 설치 방법](https://ollama.com/download)
   - 아래 명령어로 Qwen3 모델 다운로드:
   ```bash
   ollama pull qwen3:8b
   ```

## 사용 방법

### 기본 사용법

프록시 서버 실행:

```bash
python ollama_proxy.py
```

이 명령어는 기본적으로 포트 8000에서 서버를 시작합니다. 포트를 변경하려면 `PORT` 환경 변수를 설정하세요:

```bash
PORT=8080 python ollama_proxy.py
```

### Cloudflare 터널 사용하기

Cursor IDE가 "Access to private networks is forbidden" 오류로 로컬 네트워크에 접근하지 못하는 경우, Cloudflare 터널을 통해 이 제한을 우회할 수 있습니다.

터널 기능을 활성화하려면:

```bash
set USE_CLOUDFLARE_TUNNEL=true; python ollama_proxy.py
```

이 명령어는 임시 Cloudflare 터널을 생성하고 공개 URL을 제공합니다. 이 URL을 Cursor 설정에 사용하세요.

## Cursor 설정

1. Cursor를 실행하고 Settings(설정)으로 이동합니다.
2. AI 설정 섹션으로 이동합니다.
3. API URL을 설정합니다:
   - 기본 로컬 모드: `http://localhost:8000`
   - Cloudflare 터널 모드: 터널에서 제공하는 URL (예: `https://example-tunnel.trycloudflare.com`)
4. API 키는 아무 값이나 입력하세요 (예: "anything").
5. 모델은 "qwen3"로 설정하세요.

## 도구 호출 테스트

Cursor 채팅에서 다음과 같은 질문으로 도구 호출 기능을 테스트해보세요:

```
서울의 현재 날씨는 어떻게 되나요?
```

Qwen3 모델이 도구 호출(웹 검색) 기능을 사용하여 응답할 것입니다.

### 사고 과정 표시

상세한 사고 과정을 보려면 `/think` 명령어를 메시지에 추가하세요:

```
/think 한국의 GDP는 얼마인가요?
```

## 환경 변수

- `PORT`: 서버가 실행될 포트 (기본값: 8000)
- `USE_CLOUDFLARE_TUNNEL`: Cloudflare 터널 사용 여부 ("true", "1", "yes"로 설정하면 활성화)

## 주의사항

- 이 프로젝트는 로컬 개발 환경에서 사용하기 위한 것입니다.
- Cloudflare 터널 기능을 사용하려면 `pycloudflared` 패키지가 필요합니다.
- 임시 Cloudflare 터널은 제한된 시간 동안만 유효할 수 있습니다. 