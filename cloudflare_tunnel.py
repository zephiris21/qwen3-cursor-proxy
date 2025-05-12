"""
Cloudflare 터널 관리 모듈

pycloudflared를 사용하여 Cloudflare 터널을 생성하고 관리하는 기능을 제공합니다.
이 모듈을 통해 로컬 서버를 Cloudflare를 통해 공개 URL로 노출시킬 수 있습니다.
"""

import os
import time
import threading
import subprocess
import logging
import atexit
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# 항상 Cloudflare 터널을 사용하도록 설정
use_cloudflare_tunnel = True  # 환경 변수 체크 대신 항상 True로 설정

class CloudflareTunnel:
    """Cloudflare 터널 관리 클래스"""
    
    def __init__(self, port: int = 8000, metrics_port: Optional[int] = None, 
                 hostname: Optional[str] = None, verbose: bool = False):
        """
        Cloudflare 터널 인스턴스를 초기화합니다.
        
        Args:
            port: 터널이 로컬에서 연결할 포트 번호 (기본값: 8000)
            metrics_port: Cloudflared 메트릭스를 노출할 포트 번호 (기본값: None)
            hostname: 사용자 정의 hostname (기본값: None, TryCloudflare 모드에서는 무시됨)
            verbose: 자세한 로그 출력 여부 (기본값: False)
        """
        self.port = port
        self.metrics_port = metrics_port
        self.hostname = hostname
        self.verbose = verbose
        self.process = None
        self.tunnel_url = None
        self._stop_event = threading.Event()
        self._url_ready = threading.Event()
        
        # 프로그램 종료 시 터널 정리를 위한 등록
        atexit.register(self.stop)
        
        try:
            # pycloudflared 라이브러리 가져오기 (pycloudflared 0.2.0 구조에 맞게 수정)
            import pycloudflared
            from pycloudflared.util import get_info
            
            info = get_info()
            if not Path(info.executable).exists():
                from pycloudflared.util import download
                self.cloudflared_path = download(info)
            else:
                self.cloudflared_path = info.executable
                
            logger.info(f"Cloudflared 실행 파일 경로: {self.cloudflared_path}")
            
        except ImportError:
            logger.error("pycloudflared 라이브러리가 설치되지 않았습니다. 'pip install pycloudflared'로 설치하세요.")
            raise
        except Exception as e:
            logger.error(f"Cloudflared 초기화 중 오류 발생: {str(e)}")
            raise
    
    def start(self, wait_for_url: bool = True, timeout: int = 30) -> Optional[str]:
        """
        Cloudflare 터널을 시작합니다.
        
        Args:
            wait_for_url: 터널 URL이 준비될 때까지 대기할지 여부 (기본값: True)
            timeout: URL을 기다릴 최대 시간(초) (기본값: 30)
            
        Returns:
            성공 시 터널 URL, 실패 시 None
        """
        if self.process and self.process.poll() is None:
            logger.warning("터널이 이미 실행 중입니다.")
            return self.tunnel_url
        
        # 명령 준비
        cmd = [self.cloudflared_path, "tunnel"]
        
        # 등록된 터널이 아닌 임시 터널 사용
        cmd.append("--url")
        cmd.append(f"http://localhost:{self.port}")
        
        # 메트릭스 포트 설정
        if self.metrics_port:
            cmd.extend(["--metrics", f"localhost:{self.metrics_port}"])
        
        # 호스트명 설정 (있는 경우)
        if self.hostname:
            cmd.extend(["--hostname", self.hostname])
        
        # 프로세스 시작
        self._url_ready.clear()
        self._stop_event.clear()
        
        try:
            logger.info(f"Cloudflare 터널을 시작합니다. 명령: {' '.join(cmd)}")
            self.process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # 라인 버퍼링
            )
            
            # 출력을 읽기 위한 스레드 시작
            output_thread = threading.Thread(
                target=self._read_output, 
                args=(self.process.stdout,),
                daemon=True
            )
            output_thread.start()
            
            # URL이 준비될 때까지 대기
            if wait_for_url:
                if self._url_ready.wait(timeout=timeout):
                    logger.info(f"Cloudflare 터널이 시작되었습니다. URL: {self.tunnel_url}")
                    return self.tunnel_url
                else:
                    logger.error(f"터널 URL을 {timeout}초 내에 찾지 못했습니다.")
                    self.stop()
                    return None
            
            return None
            
        except Exception as e:
            logger.error(f"터널 시작 중 오류 발생: {str(e)}")
            self.stop()
            return None
    
    def _read_output(self, output):
        """프로세스 출력을 읽고 필요한 정보를 추출합니다."""
        for line in iter(output.readline, ""):
            if self.verbose:
                logger.info(f"cloudflared: {line.strip()}")
                print(f"cloudflared: {line.strip()}")  # 콘솔에도 출력
            
            # 터널 URL 추출 (다양한 패턴 포함)
            if "trycloudflare.com" in line or "cloudflare.com" in line:
                if any(pattern in line.lower() for pattern in ["your tunnel has started", "tunnel running", "https://", "http://"]):
                    # 정규식으로 URL 추출
                    import re
                    url_pattern = re.compile(r'https?://[a-zA-Z0-9.-]+\.(?:trycloudflare|cloudflare)\.com')
                    matches = url_pattern.findall(line)
                    
                    if matches:
                        self.tunnel_url = matches[0]
                        print(f"\n터널 URL이 생성되었습니다: {self.tunnel_url}\n")
                        self._url_ready.set()
            
            # 프로세스가 중지되어야 하는지 확인
            if self._stop_event.is_set():
                break
    
    def stop(self):
        """실행 중인 Cloudflare 터널을 중지합니다."""
        if self.process and self.process.poll() is None:
            self._stop_event.set()
            logger.info("Cloudflare 터널을 중지합니다...")
            
            try:
                self.process.terminate()
                # 정상적으로 종료되길 잠시 기다림
                for _ in range(5):
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.5)
                # 여전히 실행 중이면 강제 종료
                if self.process.poll() is None:
                    self.process.kill()
                    self.process.wait()
            except Exception as e:
                logger.error(f"터널 중지 중 오류 발생: {str(e)}")
            
            self.tunnel_url = None
            logger.info("Cloudflare 터널이 중지되었습니다.")

def start_cloudflare_tunnel(port: int = 8000, verbose: bool = False) -> Optional[str]:
    """
    간단한 Cloudflare 터널을 시작하는 헬퍼 함수입니다.
    
    Args:
        port: 터널이 로컬에서 연결할 포트 번호 (기본값: 8000)
        verbose: 자세한 로그 출력 여부 (기본값: False)
        
    Returns:
        성공 시 터널 URL, 실패 시 None
    """
    tunnel = CloudflareTunnel(port=port, verbose=verbose)
    return tunnel.start()

# 모듈 테스트를 위한 코드
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        tunnel = CloudflareTunnel(port=8000, verbose=True)
        url = tunnel.start()
        
        if url:
            print(f"터널이 다음 URL에서 실행 중입니다: {url}")
            try:
                print("Ctrl+C를 누르면 터널이 중지됩니다...")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                tunnel.stop()
        else:
            print("터널을 시작하지 못했습니다.")
    except Exception as e:
        print(f"오류 발생: {str(e)}") 