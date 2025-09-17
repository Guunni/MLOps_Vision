"""
MLOps Vision 프로젝트 의존성 설치 스크립트
환경에 따라 적절한 라이브러리를 자동으로 설치합니다.
"""

import subprocess
import sys
import argparse
import os
import platform
from pathlib import Path

def print_step(step_num, message):
    """설치 단계를 명확하게 표시"""
    print(f"\n{'='*50}")
    print(f"🔸 {step_num}단계: {message}")
    print(f"{'='*50}")

def get_os_info():
    """운영체제 정보를 반환합니다."""
    os_name = platform.system()
    os_version = platform.release()
    architecture = platform.machine()
    
    print(f"운영체제: {os_name} {os_version}")
    print(f"아키텍처: {architecture}")
    
    return os_name, os_version, architecture

def check_nvidia_gpu():
    """
    nvidia-smi를 통해 NVIDIA GPU 존재 여부를 확인합니다.
    (PyTorch 설치 전에 확인하는 용도)
    """
    try:
        result = subprocess.run(['nvidia-smi'],
                               capture_output=True,
                               text=True,
                               timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU 감지됨")
            return True
        else:
            print("❌ NVIDIA GPU 미감지")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("❌ nvidia-smi 명령어를 찾을 수 없음 (GPU 미설치)")
        return False

def check_torch_cuda_after_install():
    """
    PyTorch 설치 후 CUDA가 실제로 사용 가능한지 확인합니다.
    """
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA 사용 가능 (GPU 개수: {torch.cuda.device_count()})")
            for i in range(torch.cuda.device_count()):
                print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
            return True
        else:
            print("❌ PyTorch는 설치되었지만 CUDA 사용 불가")
            return False
    except ImportError:
        print("❌ PyTorch 설치 실패 또는 import 불가")
        return False

def install_package(package_name):
    """개별 패키지를 설치합니다."""
    print(f"📦 {package_name} 설치 중...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', package_name
        ])
        print(f"✅ {package_name} 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {package_name} 설치 실패: {e}")
        return False

def install_requirements(requirements_file):
    """requirements 파일을 설치합니다."""
    if not os.path.exists(requirements_file):
        print(f"⚠️ {requirements_file} 파일이 존재하지 않습니다. 건너뜀.")
        return False
    
    print(f"📦 {requirements_file} 설치 중...")
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', requirements_file
        ])
        print(f"✅ {requirements_file} 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {requirements_file} 설치 실패: {e}")
        print("🔄 개별 패키지 설치로 전환합니다...")
        return False

def install_base_packages():
    """기본 패키지들을 설치합니다."""
    base_packages = [
        "numpy==2.1.2",
        "pillow==11.0.0", 
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.3.0",
        "opencv-python>=4.8.0",
        "tqdm>=4.65.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0"
    ]
    
    print("📋 기본 패키지 목록:")
    for pkg in base_packages:
        print(f"   - {pkg}")
    
    success_count = 0
    for package in base_packages:
        if install_package(package):
            success_count += 1
    
    print(f"📊 기본 패키지: {success_count}/{len(base_packages)} 설치 완료")
    return success_count > 0

def install_pytorch_gpu():
    """PyTorch GPU 버전을 설치합니다."""
    pytorch_packages = [
        "torch==2.7.1+cu118", 
        "torchvision==0.22.1+cu118", 
        "torchaudio==2.7.1+cu118"
    ]
    
    print("🎮 PyTorch GPU 버전 설치 중...")
    
    # PyTorch index URL 사용
    cmd = [
        sys.executable, '-m', 'pip', 'install'
    ] + pytorch_packages + [
        '--index-url', 'https://download.pytorch.org/whl/cu118'
    ]
    
    try:
        subprocess.check_call(cmd)
        print("✅ PyTorch GPU 버전 설치 완료")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ PyTorch GPU 설치 실패: {e}")
        return False

def install_pytorch_cpu():
    """PyTorch CPU 버전을 설치합니다."""
    pytorch_packages = [
        "torch>=2.0.0", 
        "torchvision>=0.15.0", 
        "torchaudio>=2.0.0"
    ]
    
    print("🖥️ PyTorch CPU 버전 설치 중...")
    
    success_count = 0
    for package in pytorch_packages:
        if install_package(package):
            success_count += 1
    
    return success_count == len(pytorch_packages)

def install_faiss_for_windows():
    """
    Windows 환경에서 faiss 설치를 시도합니다.
    """
    os_name = platform.system()
    if os_name != "Windows":
        return install_package("faiss-gpu>=1.7.0")
    
    print("🪟 Windows 환경에서 FAISS 설치 중...")
    
    # 1. conda로 faiss-gpu 시도
    try:
        print("   1️⃣ conda를 통한 faiss-gpu 설치 시도...")
        result = subprocess.run(['conda', 'install', '-c', 'conda-forge', 'faiss-gpu', '-y'],
                               capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✅ conda를 통한 faiss-gpu 설치 성공")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("   ❌ conda를 찾을 수 없거나 설치 실패")
    
    # 2. pip로 faiss-cpu 시도
    print("   2️⃣ pip를 통한 faiss-cpu 설치 시도...")
    return install_package("faiss-cpu>=1.7.0")

def install_ml_packages():
    """머신러닝 관련 패키지들을 설치합니다."""
    ml_packages = [
        "scikit-image>=0.21.0",
        "scipy>=1.11.0",
        "seaborn>=0.12.0"
    ]
    
    success_count = 0
    for package in ml_packages:
        if install_package(package):
            success_count += 1
    
    return success_count > 0

def install_dev_packages():
    """개발 도구들을 설치합니다."""
    dev_packages = [
        "jupyter>=1.0.0",
        "jupyterlab>=4.0.0", 
        "black>=23.0.0",
        "pytest>=7.0.0",
        "plotly>=5.15.0",
        "GitPython>=3.0.0"
    ]
    
    success_count = 0
    for package in dev_packages:
        if install_package(package):
            success_count += 1
    
    print(f"📊 개발 도구: {success_count}/{len(dev_packages)} 설치 완료")
    return success_count > 0

def create_requirements_directory():
    """requirements 디렉토리를 생성합니다."""
    req_dir = Path('requirements')
    req_dir.mkdir(exist_ok=True)
    return req_dir

def main():
    parser = argparse.ArgumentParser(
        description='MLOps Vision 프로젝트 의존성 설치',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python install_requirements.py              # 자동 감지 설치
  python install_requirements.py --force-cpu  # 강제 CPU 버전
  python install_requirements.py --dev        # 개발 도구 포함
  python install_requirements.py --dev --force-cpu  # CPU + 개발 도구
        """
    )
    parser.add_argument('--force-cpu', action='store_true', 
                       help='GPU가 있어도 강제로 CPU 버전을 설치')
    parser.add_argument('--dev', action='store_true', 
                       help='개발용 도구들도 함께 설치')
    parser.add_argument('--skip-base', action='store_true',
                       help='기본 패키지 설치 건너뛰기 (이미 설치된 경우)')
    parser.add_argument('--use-files', action='store_true',
                       help='requirements 파일 사용 (있는 경우)')
    args = parser.parse_args()

    print("🚀 MLOps Vision 프로젝트 의존성 설치 시작")
    
    # 시스템 정보 출력
    os_name, os_version, architecture = get_os_info()
    print(f"Python 버전: {sys.version}")
    print(f"pip 위치: {subprocess.check_output([sys.executable, '-m', 'pip', '--version'], text=True).strip()}")

    # requirements 디렉토리 확인
    req_dir = create_requirements_directory()
    
    # 1단계: 기본 패키지 설치
    if not args.skip_base:
        print_step(1, "기본 패키지 설치")
        
        if args.use_files and (req_dir / 'requirements_base.txt').exists():
            install_requirements(req_dir / 'requirements_base.txt')
        else:
            install_base_packages()
    else:
        print_step(1, "기본 패키지 설치 건너뜀")

    # 2단계: PyTorch 설치 (CPU vs GPU)
    print_step(2, "PyTorch 설치")
    
    if args.force_cpu:
        print("🖥️ 강제 CPU 모드로 설치합니다.")
        pytorch_success = install_pytorch_cpu()
        use_gpu = False
    else:
        has_nvidia_gpu = check_nvidia_gpu()
        if has_nvidia_gpu:
            print("🎮 GPU가 감지되었습니다. GPU 버전을 설치합니다.")
            pytorch_success = install_pytorch_gpu()
            use_gpu = True
        else:
            print("🖥️ GPU 미감지 → CPU 버전 설치")
            pytorch_success = install_pytorch_cpu()
            use_gpu = False

    # 3단계: PyTorch 설치 확인
    print_step(3, "PyTorch 설치 확인")
    if pytorch_success:
        if use_gpu:
            cuda_available = check_torch_cuda_after_install()
            if not cuda_available:
                print("⚠️ GPU 버전을 설치했지만 CUDA를 사용할 수 없습니다.")
                print("   CUDA 드라이버나 런타임이 올바르게 설치되지 않았을 수 있습니다.")
        else:
            try:
                import torch
                print(f"✅ PyTorch CPU 버전 설치 완료 (버전: {torch.__version__})")
            except ImportError:
                print("❌ PyTorch 설치 확인 실패")
    else:
        print("❌ PyTorch 설치에 문제가 있었습니다.")

    # 4단계: 추가 ML 패키지 설치
    print_step(4, "머신러닝 패키지 설치")
    install_ml_packages()
    
    # FAISS 별도 설치
    print("🔍 FAISS 설치 중...")
    install_faiss_for_windows()

    # 5단계: 개발 도구 설치 (선택사항)
    if args.dev:
        print_step(5, "개발 도구 설치")
        install_dev_packages()

    # 6단계: 설치 완료 요약
    print_step("완료", "설치 요약")
    print("🎉 모든 의존성 설치 완료!")
    print("\n📋 설치된 구성:")
    print(f"   • 기본 패키지: {'✅' if not args.skip_base else '⏭️ 건너뜀'}")
    print(f"   • PyTorch: {'🎮 GPU 버전' if use_gpu else '🖥️ CPU 버전'}")
    print(f"   • 개발 도구: {'✅' if args.dev else '❌'}")
    
    print("\n🔧 다음 단계:")
    print("   1. 'python -c \"import torch; print(torch.__version__)\"' 로 설치 확인")
    if use_gpu:
        print("   2. 'python -c \"import torch; print(torch.cuda.is_available())\"' 로 CUDA 확인")
    print("   3. 'python -c \"import numpy, pandas, cv2; print('모든 라이브러리 OK')\"' 로 전체 확인")

if __name__ == '__main__':
    main()