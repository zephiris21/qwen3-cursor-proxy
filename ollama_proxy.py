from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
import litellm
import os
import requests
import json
import re
from typing import Optional, Dict, Any, List, Tuple

# Cloudflare 터널 기능 가져오기
try:
    from cloudflare_tunnel import CloudflareTunnel
    CLOUDFLARE_TUNNEL_AVAILABLE = True
except ImportError:
    CLOUDFLARE_TUNNEL_AVAILABLE = False

# 전역 변수 초기화
tunnel_url = None

# FastAPI 앱 초기화
app = FastAPI()

# LiteLLM 디버그 활성화
litellm.set_verbose = True

# Qwen3 모델이 도구 호출을 지원하도록 등록
litellm.register_model(model_cost={
    "ollama_chat/qwen3": {
        "supports_function_calling": True
    },
})

# Ollama 서버 상태 확인 함수
def check_ollama_status():
    try:
        response = requests.get("http://localhost:11434/api/version")
        if response.status_code == 200:
            return True, "Ollama 서버가 정상적으로 실행 중입니다."
        return False, "Ollama 서버에 연결할 수 있지만 응답이 올바르지 않습니다."
    except Exception as e:
        return False, f"Ollama 서버 연결 오류: {str(e)}"

# Qwen3 모델 로드 확인 함수
def check_qwen3_loaded():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            for model in models:
                if "qwen3" in model.get("name", "").lower():
                    return True, f"Qwen3 모델이 로드되었습니다: {model.get('name')}"
        return False, "Qwen3 모델이 로드되지 않았습니다. 'ollama pull qwen3:8b' 명령을 실행하세요."
    except Exception as e:
        return False, f"모델 목록 확인 오류: {str(e)}"

# Qwen3의 /think 명령어 처리 함수
def process_thinking_mode(messages):
    # 메시지 복사본 생성
    processed_messages = messages.copy()
    
    # 마지막 메시지 확인 (사용자 입력)
    if processed_messages and processed_messages[-1].get("role") == "user":
        content = processed_messages[-1].get("content", "")
        
        # /think 명령어가 포함되어 있는지 확인
        has_think_command = "/think" in content
        has_no_think_command = "/no_think" in content
        
        # 시스템 메시지가 없는 경우 추가
        has_system_message = any(msg.get("role") == "system" for msg in processed_messages)
        
        if not has_system_message:
            # 사고 모드에 따른 시스템 메시지 추가
            if has_think_command:
                processed_messages.insert(0, {
                    "role": "system",
                    "content": "사용자 질문에 답변할 때 상세한 사고 과정을 보여주세요. 단계별로 문제를 분석하고 결론을 도출하세요."
                })
            elif has_no_think_command:
                processed_messages.insert(0, {
                    "role": "system",
                    "content": "사용자 질문에 간결하게 답변하세요. 상세한 사고 과정 없이 직접적인 답변만 제공하세요."
                })
    
    return processed_messages

# OpenAI 도구 호출 형식을 Qwen3 XML 형식으로 변환하는 함수
def convert_openai_tools_to_qwen3_xml(tools):
    if not tools:
        return None
        
    # OpenAI 도구 정의를 Qwen3 XML 형식으로 변환
    tools_xml = "<tools>\n"
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            tools_xml += f"<tool name=\"{func.get('name')}\">\n"
            tools_xml += f"<description>{func.get('description', '')}</description>\n"
            
            # 매개변수 정의 추가
            params = func.get("parameters", {})
            if params and params.get("properties"):
                tools_xml += "<parameters>\n"
                for param_name, param_info in params.get("properties", {}).items():
                    required = "true" if param_name in params.get("required", []) else "false"
                    tools_xml += f"<parameter name=\"{param_name}\" required=\"{required}\">\n"
                    tools_xml += f"<description>{param_info.get('description', '')}</description>\n"
                    tools_xml += f"<type>{param_info.get('type', 'string')}</type>\n"
                    if param_info.get("enum"):
                        tools_xml += "<enum>\n"
                        for enum_val in param_info.get("enum", []):
                            tools_xml += f"<value>{enum_val}</value>\n"
                        tools_xml += "</enum>\n"
                    tools_xml += "</parameter>\n"
                tools_xml += "</parameters>\n"
            
            tools_xml += "</tool>\n"
    tools_xml += "</tools>"
    
    return tools_xml

# Qwen3 XML 응답을 OpenAI 도구 호출 형식으로 변환하는 함수
def convert_qwen3_xml_to_openai_tool_calls(content):
    # 응답에서 사고 과정 필터링
    thinking_content = ""
    
    # 사고 과정 패턴 찾기 및 추출
    thinking_pattern = r"<thinking>(.*?)<\/thinking>"
    thinking_matches = re.findall(thinking_pattern, content, re.DOTALL)
    if thinking_matches:
        for match in thinking_matches:
            thinking_content += match.strip() + "\n\n"
        
        # 사고 과정 제거
        content = re.sub(thinking_pattern, "", content, flags=re.DOTALL)
    
    # 도구 호출 처리
    tool_calls = []
    
    # <tool_call>...</tool_call> 패턴 검색
    tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
    tool_call_matches = re.findall(tool_call_pattern, content, re.DOTALL)
    
    for i, match in enumerate(tool_call_matches):
        # 함수 이름과 인수 추출
        name_pattern = r'"name"\s*:\s*"([^"]+)"'
        args_pattern = r'"arguments"\s*:\s*({[^}]+})'
        
        name_match = re.search(name_pattern, match)
        args_match = re.search(args_pattern, match)
        
        if name_match and args_match:
            function_name = name_match.group(1)
            function_args = args_match.group(1)
            
            tool_calls.append({
                "id": f"call_{i}",
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": function_args
                }
            })
    
    # 도구 호출이 있으면 원본 컨텐츠에서 XML 태그 제거
    if tool_calls:
        content = re.sub(tool_call_pattern, "", content).strip()
    
    # 사고 과정이 있었다면 메타데이터로 추가
    result = {
        "content": content.strip(),
        "tool_calls": tool_calls
    }
    
    if thinking_content:
        result["thinking_content"] = thinking_content
    
    return result

# 시작 시 확인
@app.on_event("startup")
async def startup_event():
    # LiteLLM 설정
    litellm.set_verbose = True
    
    # Ollama 서버 상태 확인
    ollama_status, ollama_msg = check_ollama_status()
    if not ollama_status:
        print(f"경고: {ollama_msg}")
        print("Ollama 서버가 실행 중인지 확인하세요.")
    else:
        print(f"Ollama 서버 정보: {ollama_msg}")
    
    # Cloudflare 터널 설정 및 시작 (환경 변수로 제어)
    global tunnel_url
    tunnel_url = None
    
    # 항상 Cloudflare 터널을 사용하도록 설정
    # 환경 변수로도 제어 가능 (USE_CLOUDFLARE_TUNNEL)
    use_cloudflare_tunnel = os.environ.get("USE_CLOUDFLARE_TUNNEL", "true").lower() == "true"
    
    if use_cloudflare_tunnel:
        if CLOUDFLARE_TUNNEL_AVAILABLE:
            try:
                port = int(os.environ.get("PORT", 8000))
                tunnel = CloudflareTunnel(port=port, verbose=True)
                tunnel_url = tunnel.start()
                
                if tunnel_url:
                    print("\n" + "="*50)
                    print(f"Cloudflare 터널이 시작되었습니다: {tunnel_url}")
                    print("로컬 네트워크 제한을 우회하기 위해 이 URL을 Cursor 설정에 사용하세요.")
                    print_cursor_setup_guide_with_tunnel(tunnel_url)
                    print("="*50 + "\n")
                else:
                    print("\n경고: Cloudflare 터널을 시작하지 못했습니다.")
                    print("로컬 서버에 직접 연결하세요: http://localhost:8000\n")
            except Exception as e:
                print(f"\n오류: Cloudflare 터널 시작 실패: {str(e)}")
                print("로컬 서버에 직접 연결하세요: http://localhost:8000\n")
        else:
            print("\n경고: pycloudflared 패키지가 설치되지 않았습니다.")
            print("터널링 기능을 사용하려면 'pip install pycloudflared'를 실행하세요.")
            print("로컬 서버에 직접 연결하세요: http://localhost:8000\n")
    else:
        print("\nCloudflare 터널 비활성화됨. 로컬 서버를 직접 사용합니다.")
        print("터널을 활성화하려면 'USE_CLOUDFLARE_TUNNEL=true'로 설정하세요.\n")

@app.post("/chat/completions")
async def legacy_chat_completions(request: Request):
    # v1/chat/completions로 리디렉션
    data = await request.json()
    return await chat_completions(request)
        
        
        
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    data = await request.json()
    
    # 요청 파라미터 추출
    model_name = data.get("model", "qwen3")  # 기본값: qwen3
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    temperature = data.get("temperature", 0.7)
    max_tokens = data.get("max_tokens", 2000)
    tools = data.get("tools", None)
    tool_choice = data.get("tool_choice", None)
    
    # Qwen3 /think 명령어 처리
    processed_messages = process_thinking_mode(messages)
    
    # 모델 이름 변환
    if not model_name.startswith("ollama_chat/"):
        model_name = f"ollama_chat/{model_name}"
    
    # Qwen3 XML 도구 형식으로 변환
    qwen3_tools_xml = None
    if tools:
        qwen3_tools_xml = convert_openai_tools_to_qwen3_xml(tools)
        
        # 마지막 메시지에 도구 정보 추가 (Qwen3 형식)
        if qwen3_tools_xml and processed_messages:
            last_msg = processed_messages[-1].copy()
            last_msg["content"] = last_msg["content"] + "\n\n" + qwen3_tools_xml
            processed_messages[-1] = last_msg
    
    try:
        # LiteLLM을 통해 응답 생성
        response = litellm.completion(
            model=model_name,
            messages=processed_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            api_base="http://localhost:11434",
        )
        
        # 스트리밍 응답 처리
        if stream:
            def generate_stream():
                for chunk in response:
                    # 청크에 도구 호출 내용이 있는지 확인하고 변환
                    if chunk.choices and chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        result = convert_qwen3_xml_to_openai_tool_calls(content)
                        
                        # 변환된 내용으로 청크 업데이트
                        chunk.choices[0].delta.content = result["content"]
                        
                        if result.get("tool_calls"):
                            # OpenAI 형식에 맞게 도구 호출 정보 설정
                            chunk.choices[0].delta.tool_calls = result["tool_calls"]
                            
                        # 사고 과정이 있는 경우 메타데이터로 추가
                        if result.get("thinking_content"):
                            if not hasattr(chunk.choices[0].delta, "metadata"):
                                chunk.choices[0].delta.metadata = {}
                            chunk.choices[0].delta.metadata["thinking_content"] = result["thinking_content"]
                        
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(generate_stream(), media_type="text/event-stream")
        
        # 일반 응답에서 도구 호출 처리
        if response.choices and response.choices[0].message.content:
            content = response.choices[0].message.content
            result = convert_qwen3_xml_to_openai_tool_calls(content)
            
            # 변환된 내용으로 응답 업데이트
            response.choices[0].message.content = result["content"]
            
            if result.get("tool_calls"):
                response.choices[0].message.tool_calls = result["tool_calls"]
                
            # 사고 과정이 있는 경우 메타데이터로 추가
            if result.get("thinking_content"):
                if not hasattr(response.choices[0].message, "metadata"):
                    response.choices[0].message.metadata = {}
                response.choices[0].message.metadata["thinking_content"] = result["thinking_content"]
        
        return response
        
    except Exception as e:
        return {"error": str(e)}
    
    
    

@app.get("/v1/models")
async def list_models():
    try:
        # Ollama에서 사용 가능한 모델 목록 가져오기
        response = requests.get("http://localhost:11434/api/tags")
        models = response.json().get("models", [])
        
        # OpenAI API 형식으로 변환
        openai_models = []
        for model in models:
            model_name = model.get("name")
            if "qwen3" in model_name.lower():
                openai_models.append({
                    "id": model_name,
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "ollama",
                })
        
        return {"data": openai_models, "object": "list"}
    except Exception as e:
        return {"error": str(e)}

@app.get("/health")
async def health_check():
    ollama_status, ollama_msg = check_ollama_status()
    qwen3_status, qwen3_msg = check_qwen3_loaded()
    
    if ollama_status and qwen3_status:
        return {
            "status": "ok",
            "ollama": ollama_msg,
            "qwen3": qwen3_msg
        }
    else:
        return {
            "status": "error",
            "ollama_status": ollama_status,
            "ollama_message": ollama_msg,
            "qwen3_status": qwen3_status,
            "qwen3_message": qwen3_msg
        }
        
        
        
        
        
def print_cursor_setup_guide_with_tunnel(tunnel_url: str):
    """Cloudflare 터널 URL로 Cursor 설정 가이드를 출력합니다."""
    print("\n" + "="*50)
    print("Cursor 설정 가이드 (Cloudflare 터널 사용):")
    print("="*50)
    print("1. Cursor를 실행하고 Settings(설정)으로 이동")
    print("2. AI 설정 섹션으로 이동")
    print(f"3. API URL을 '{tunnel_url}'로 설정")
    print("4. API 키를 'anything'으로 설정 (이 값은 무시됨)")
    print("5. 모델은 'qwen3'로 설정")
    print("\n도구 호출 테스트:")
    print("Cursor 채팅에서 '서울의 현재 날씨는 어떻게 되나요?'와 같은 질문으로 테스트")
    print("'/think'를 추가하면 상세 사고 과정을 볼 수 있습니다")
    print("="*50 + "\n")

def print_cursor_setup_guide(port):
    print("\n" + "="*50)
    print("Cursor 설정 가이드:")
    print("="*50)
    print("1. Cursor를 실행하고 Settings(설정)으로 이동")
    print("2. AI 설정 섹션으로 이동")
    print(f"3. API URL을 'http://localhost:{port}'로 설정")
    print("4. API 키를 'anything'으로 설정 (이 값은 무시됨)")
    print("5. 모델은 'qwen3'로 설정")
    print("\n도구 호출 테스트:")
    print("Cursor 채팅에서 '서울의 현재 날씨는 어떻게 되나요?'와 같은 질문으로 테스트")
    print("'/think'를 추가하면 상세 사고 과정을 볼 수 있습니다")
    print("="*50 + "\n")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    
    print(f"LiteLLM 프록시 서버 시작 (포트: {port})")
    print("Ollama 모델은 '/v1/chat/completions'를 통해 접근 가능합니다")
    
    # 터널 URL이 없는 경우에만 로컬 설정 가이드 출력
    if not tunnel_url:
        print_cursor_setup_guide(port)
    
    uvicorn.run(app, host="0.0.0.0", port=port)