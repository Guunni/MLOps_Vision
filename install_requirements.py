#!/usr/bin/env python3
"""
MLOps Vision 프로젝트 의존성 설치 스크립트
환경에 따라 적절한 라이브러리를 자동으로 설치합니다.
"""

import subprocess
import sys
import argparse
import torch

def check_gpu_availability():
    """
    시스템에서 CUDA를 사용할 수 있는지 확인합니다.
    
    Returns:
        bool: CUDA 사용 가능 여부
    """
    try:
        # PyTorch를 통해 CUDA 사용 가능성 확인
        return torch.cuda.is_available()
    except ImportError:
        # PyTorch가 설치되지 않은 경우, nvidia-smi 명령어로 확인
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

def install_requirements(requirements_file):
    """
    지정된 requirements 파일을 사용해서 라이브러리를 설치합니다.
    
    Args:
        requirements_file (str): 설치할 requirements 파일 경로
    """
    print(f"📦 {requirements_file} 설치 중...")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            '-r', requirements_file
        ])
        print(f"✅ {requirements_file} 설치 완료")
    except subprocess.CalledProcessError as e:
        print(f"❌ {requirements_file} 설치 실패: {e}")
        sys.exit(1)

def main():
    """메인 설치 프로세스를 실행합니다."""
    parser = argparse.ArgumentParser(
        description='MLOps Vision 프로젝트 의존성 설치'
    )
    parser.add_argument(
        '--force-cpu', 
        action='store_true',
        help='GPU가 있어도 강제로 CPU 버전을 설치'
    )
    parser.add_argument(
        '--dev', 
        action='store_true',
        help='개발용 도구들도 함께 설치'
    )
    
    args = parser.parse_args()
    
    print("🚀 MLOps Vision 프로젝트 의존성 설치를 시작합니다...")
    
    # 1단계: 기본 라이브러리 설치
    install_requirements('requirements/requirements_base.txt')
    
    # 2단계: GPU/CPU 환경에 따른 라이브러리 설치
    if args.force_cpu:
        print("🖥️  강제 CPU 모드로 설치합니다.")
        install_requirements('requirements/requirements_cpu.txt')
    else:
        gpu_available = check_gpu_availability()
        
        if gpu_available:
            print("🎮 GPU가 감지되었습니다. GPU 버전을 설치합니다.")
            install_requirements('requirements/requirements_gpu.txt')
        else:
            print("🖥️  GPU가 감지되지 않았습니다. CPU 버전을 설치합니다.")
            install_requirements('requirements/requirements_cpu.txt')
    
    # 3단계: 개발 도구 설치 (선택사항)
    if args.dev:
        print("🛠️  개발 도구들을 설치합니다.")
        install_requirements('requirements/requirements_dev.txt')
    
    print("\n🎉 모든 의존성 설치가 완료되었습니다!")
    print("이제 MLOps Vision 프로젝트를 시작할 수 있습니다.")

if __name__ == '__main__':
    main()