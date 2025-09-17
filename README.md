# 기본 설치 (GPU 자동 감지)
python install_requirements.py

# 개발 도구 포함
python install_requirements.py --dev

# CPU 강제 설치
python install_requirements.py --force-cpu --dev

# 기본 패키지 건너뛰고 PyTorch만
python install_requirements.py --skip-base

# 가상 환경 사용
venv_mlops_vision\Scripts\activate