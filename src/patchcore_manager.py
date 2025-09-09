"""
PatchCore 라이브러리 관리 시스템
Google Colab 코드를 로컬 환경으로 변환한 PatchCore 관리 도구
"""

import os
import subprocess
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import git  # GitPython 라이브러리를 사용

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatchCoreManager:
    """
    PatchCore 라이브러리의 설치, 관리, 실행을 담당하는 클래스
    
    이 클래스는 Google Colab에서 수행되던 다음 작업들을 자동화합니다:
    1. PatchCore 저장소 클론
    2. 의존성 설치
    3. 라이브러리 설치 및 설정
    4. 환경 검증
    """
    
    def __init__(self, base_dir: str = None):
        """
        PatchCore 매니저를 초기화합니다.
        
        Args:
            base_dir: PatchCore를 설치할 기본 디렉토리 경로
        """
        if base_dir is None:
            # 현재 프로젝트의 루트 디렉토리를 기본값으로 사용
            self.base_dir = Path.cwd()
        else:
            self.base_dir = Path(base_dir)
        
        # PatchCore가 설치될 경로 설정
        self.patchcore_dir = self.base_dir / "patchcore-inspection"
        
        # GitHub 저장소 URL
        self.repo_url = "https://github.com/amazon-science/patchcore-inspection.git"
        
        logger.info(f"PatchCore 매니저 초기화됨: {self.base_dir}")
    
    def check_git_availability(self) -> bool:
        """
        시스템에 Git이 설치되어 있는지 확인합니다.
        
        Returns:
            bool: Git 사용 가능 여부
        """
        try:
            subprocess.run(['git', '--version'], 
                         capture_output=True, 
                         check=True, 
                         timeout=10)
            logger.info("✅ Git이 정상적으로 설치되어 있습니다.")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.error("❌ Git이 설치되어 있지 않습니다.")
            logger.error("Git을 설치한 후 다시 시도해주세요: https://git-scm.com/download")
            return False
    
    def is_patchcore_installed(self) -> bool:
        """
        PatchCore가 이미 설치되어 있는지 확인합니다.
        
        Returns:
            bool: PatchCore 설치 여부
        """
        # 디렉토리 존재 여부 확인
        if not self.patchcore_dir.exists():
            return False
        
        # 주요 파일들이 존재하는지 확인
        required_files = [
            self.patchcore_dir / "setup.py",
            self.patchcore_dir / "src" / "patchcore",
            self.patchcore_dir / "bin" / "run_patchcore.py"
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                logger.warning(f"필수 파일이 없습니다: {file_path}")
                return False
        
        logger.info("✅ PatchCore가 이미 설치되어 있습니다.")
        return True
    
    def clone_patchcore_repository(self, force_reinstall: bool = False) -> bool:
        """
        PatchCore 저장소를 GitHub에서 클론합니다.
        
        Args:
            force_reinstall: 이미 설치되어 있어도 강제로 재설치할지 여부
            
        Returns:
            bool: 클론 성공 여부
        """
        # Git 사용 가능성 확인
        if not self.check_git_availability():
            return False
        
        # 이미 설치되어 있는 경우 처리
        if self.is_patchcore_installed() and not force_reinstall:
            logger.info("PatchCore가 이미 설치되어 있습니다. 재설치하려면 force_reinstall=True를 사용하세요.")
            return True
        
        # 강제 재설치인 경우 기존 디렉토리 삭제
        if force_reinstall and self.patchcore_dir.exists():
            logger.info("기존 PatchCore 설치를 제거합니다...")
            shutil.rmtree(self.patchcore_dir)
        
        try:
            logger.info(f"🔄 PatchCore 저장소를 클론합니다: {self.repo_url}")
            
            # GitPython을 사용한 클론 (진행 상황 표시 가능)
            git.Repo.clone_from(
                self.repo_url, 
                self.patchcore_dir,
                progress=self._git_progress_callback
            )
            
            logger.info("✅ PatchCore 저장소 클론이 완료되었습니다.")
            return True
            
        except git.exc.GitCommandError as e:
            logger.error(f"❌ Git 클론 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류 발생: {e}")
            return False
    
    def _git_progress_callback(self, op_code, cur_count, max_count=None, message=''):
        """
        Git 클론 진행 상황을 표시하는 콜백 함수
        """
        if max_count:
            progress = (cur_count / max_count) * 100
            logger.info(f"진행률: {progress:.1f}% - {message}")
        else:
            logger.info(f"진행 중... - {message}")
    
    def install_patchcore_dependencies(self) -> bool:
        """
        PatchCore의 의존성 라이브러리들을 설치합니다.
        
        이 함수는 Colab 코드의 다음 부분을 대체합니다:
        !pip install -r requirements.txt
        !pip install -e .
        
        Returns:
            bool: 설치 성공 여부
        """
        if not self.is_patchcore_installed():
            logger.error("❌ PatchCore가 설치되지 않았습니다. 먼저 clone_patchcore_repository()를 실행하세요.")
            return False
        
        try:
            # 현재 디렉토리를 PatchCore 디렉토리로 변경
            original_dir = os.getcwd()
            os.chdir(self.patchcore_dir)
            
            logger.info("📦 PatchCore 의존성을 설치합니다...")
            
            # requirements.txt가 있는지 확인하고 설치
            requirements_file = self.patchcore_dir / "requirements.txt"
            if requirements_file.exists():
                logger.info("requirements.txt 설치 중...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install', 
                    '-r', str(requirements_file)
                ], check=True)
                logger.info("✅ requirements.txt 설치 완료")
            else:
                logger.warning("requirements.txt 파일을 찾을 수 없습니다.")
            
            # PatchCore를 editable 모드로 설치
            logger.info("PatchCore를 editable 모드로 설치 중...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', '-e', '.'
            ], check=True)
            logger.info("✅ PatchCore 설치 완료")
            
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 의존성 설치 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류: {e}")
            return False
        finally:
            # 원래 디렉토리로 복원
            os.chdir(original_dir)
    
    def verify_installation(self) -> Dict[str, Any]:
        """
        PatchCore 설치 상태를 종합적으로 검증합니다.
        
        Returns:
            Dict: 검증 결과 정보
        """
        verification_results = {
            "repository_cloned": False,
            "dependencies_installed": False,
            "patchcore_importable": False,
            "main_script_exists": False,
            "installation_complete": False,
            "error_messages": []
        }
        
        try:
            # 1. 저장소 클론 확인
            verification_results["repository_cloned"] = self.is_patchcore_installed()
            
            # 2. 메인 스크립트 존재 확인
            main_script = self.patchcore_dir / "bin" / "run_patchcore.py"
            verification_results["main_script_exists"] = main_script.exists()
            
            # 3. PatchCore 모듈 import 가능성 확인
            try:
                # sys.path에 PatchCore src 디렉토리 임시 추가
                patchcore_src = str(self.patchcore_dir / "src")
                if patchcore_src not in sys.path:
                    sys.path.insert(0, patchcore_src)
                
                import patchcore
                verification_results["patchcore_importable"] = True
                logger.info("✅ PatchCore 모듈 import 성공")
                
            except ImportError as e:
                verification_results["error_messages"].append(f"PatchCore import 실패: {e}")
                logger.error(f"❌ PatchCore import 실패: {e}")
            
            # 4. 전체 설치 완료 여부 판단
            verification_results["installation_complete"] = all([
                verification_results["repository_cloned"],
                verification_results["main_script_exists"],
                verification_results["patchcore_importable"]
            ])
            
            # 결과 요약 출력
            if verification_results["installation_complete"]:
                logger.info("🎉 PatchCore 설치 검증이 완료되었습니다!")
            else:
                logger.warning("⚠️ PatchCore 설치에 문제가 있습니다. 상세 내용을 확인하세요.")
            
        except Exception as e:
            verification_results["error_messages"].append(f"검증 중 오류 발생: {e}")
            logger.error(f"❌ 검증 중 오류 발생: {e}")
        
        return verification_results
    
    def setup_patchcore_complete(self, force_reinstall: bool = False) -> bool:
        """
        PatchCore의 완전한 설치 과정을 자동으로 수행합니다.
        
        이 함수는 Google Colab 코드의 전체 설치 과정을 하나의 함수로 통합합니다.
        
        Args:
            force_reinstall: 강제 재설치 여부
            
        Returns:
            bool: 전체 설치 과정의 성공 여부
        """
        logger.info("🚀 PatchCore 완전 설치를 시작합니다...")
        
        # 1단계: 저장소 클론
        if not self.clone_patchcore_repository(force_reinstall=force_reinstall):
            logger.error("❌ 저장소 클론에 실패했습니다.")
            return False
        
        # 2단계: 의존성 설치
        if not self.install_patchcore_dependencies():
            logger.error("❌ 의존성 설치에 실패했습니다.")
            return False
        
        # 3단계: 설치 검증
        verification_results = self.verify_installation()
        
        if verification_results["installation_complete"]:
            logger.info("🎉 PatchCore 완전 설치가 성공적으로 완료되었습니다!")
            return True
        else:
            logger.error("❌ 설치 과정에서 문제가 발생했습니다.")
            for error in verification_results["error_messages"]:
                logger.error(f"   - {error}")
            return False

# 사용 예시 및 테스트를 위한 메인 함수
if __name__ == "__main__":
    # PatchCore 매니저 인스턴스 생성
    manager = PatchCoreManager()
    
    # 완전 설치 수행
    success = manager.setup_patchcore_complete()
    
    if success:
        print("\n✅ PatchCore 설치가 완료되었습니다!")
        print("이제 PatchCore를 사용한 모델 학습을 시작할 수 있습니다.")
    else:
        print("\n❌ PatchCore 설치에 실패했습니다.")
        print("오류 메시지를 확인하고 문제를 해결한 후 다시 시도해주세요.")