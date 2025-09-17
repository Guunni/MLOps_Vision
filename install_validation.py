"""
MLOps Vision 프로젝트 환경 검증 스크립트
설치된 라이브러리들이 정상적으로 작동하는지 확인합니다.
"""

print('=== MLOps Vision 환경 검증 ===')
print()

validation_results = []

# 1. 기본 라이브러리 확인
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print('✅ 기본 과학 계산 라이브러리: 정상')
    validation_results.append(True)
except ImportError as e:
    print(f'❌ 기본 라이브러리 오류: {e}')
    validation_results.append(False)

# 2. 이미지 처리 라이브러리 확인
try:
    import cv2
    from PIL import Image
    print('✅ 이미지 처리 라이브러리: 정상')
    validation_results.append(True)
except ImportError as e:
    print(f'❌ 이미지 처리 라이브러리 오류: {e}')
    validation_results.append(False)

# 3. PyTorch 환경 확인
try:
    import torch
    import torchvision
    print(f'✅ PyTorch {torch.__version__}: 정상')
    print(f'   CUDA 사용 가능: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU 장치 수: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
    else:
        print('   CPU 모드로 실행됩니다')
    validation_results.append(True)
except ImportError as e:
    print(f'❌ PyTorch 오류: {e}')
    validation_results.append(False)

# 4. FAISS 라이브러리 확인
try:
    import faiss
    print('✅ FAISS 라이브러리: 정상')
    
    # FAISS 기능 테스트
    import numpy as np
    test_vectors = np.random.random((100, 64)).astype('float32')
    index = faiss.IndexFlatL2(64)
    index.add(test_vectors)
    print(f'   FAISS 인덱스 테스트: {index.ntotal}개 벡터 저장 성공')
    
    # GPU 지원 확인 (있는 경우)
    if hasattr(faiss, 'StandardGpuResources'):
        print('   FAISS GPU 지원: 사용 가능')
    else:
        print('   FAISS GPU 지원: CPU 전용')
    
    validation_results.append(True)
except ImportError as e:
    print(f'❌ FAISS 오류: {e}')
    validation_results.append(False)
except Exception as e:
    print(f'❌ FAISS 기능 테스트 실패: {e}')
    validation_results.append(False)

# 5. 웹 API 라이브러리 확인
try:
    import fastapi
    import uvicorn
    from fastapi import FastAPI
    print('✅ 웹 API 라이브러리: 정상')
    
    # FastAPI 기본 앱 생성 테스트
    app = FastAPI()
    print('   FastAPI 앱 생성 테스트: 성공')
    validation_results.append(True)
except ImportError as e:
    print(f'❌ 웹 API 라이브러리 오류: {e}')
    validation_results.append(False)

# 6. 시각화 라이브러리 확인
try:
    import plotly
    import sklearn
    import seaborn as sns
    print('✅ 시각화 및 ML 라이브러리: 정상')
    validation_results.append(True)
except ImportError as e:
    print(f'❌ 시각화 라이브러리 오류: {e}')
    validation_results.append(False)

# 7. 개발 도구 확인 (선택사항)
dev_tools_available = []
try:
    import jupyter
    dev_tools_available.append('Jupyter')
except ImportError:
    pass

try:
    import black
    dev_tools_available.append('Black')
except ImportError:
    pass

try:
    import pytest
    dev_tools_available.append('Pytest')
except ImportError:
    pass

if dev_tools_available:
    print(f'✅ 개발 도구: {", ".join(dev_tools_available)} 사용 가능')
else:
    print('ℹ️ 개발 도구: 설치되지 않음 (--dev 옵션으로 설치 가능)')

# 8. 추가 기능 테스트
print('\n--- 추가 기능 테스트 ---')

# NumPy 배열 연산 테스트
try:
    arr = np.random.random((100, 100))
    result = np.mean(arr)
    print(f'✅ NumPy 연산 테스트: 평균값 {result:.4f}')
except Exception as e:
    print(f'❌ NumPy 연산 테스트 실패: {e}')

# OpenCV 이미지 처리 테스트
try:
    import cv2
    import numpy as np
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
    print(f'✅ OpenCV 이미지 처리 테스트: {gray_img.shape} 그레이스케일 변환 성공')
except Exception as e:
    print(f'❌ OpenCV 테스트 실패: {e}')

# PyTorch 텐서 연산 테스트
try:
    import torch
    x = torch.randn(10, 10)
    y = torch.mm(x, x.t())
    print(f'✅ PyTorch 텐서 연산 테스트: {y.shape} 행렬 곱셈 성공')
    
    if torch.cuda.is_available():
        x_gpu = x.cuda()
        y_gpu = torch.mm(x_gpu, x_gpu.t())
        print(f'✅ PyTorch GPU 연산 테스트: 성공')
except Exception as e:
    print(f'❌ PyTorch 테스트 실패: {e}')

print('\n' + '='*50)

# 최종 결과
success_count = sum(validation_results)
total_count = len(validation_results)

print(f'📊 전체 검증 결과: {success_count}/{total_count} 통과')

if success_count == total_count:
    print('🎉 환경 검증이 완료되었습니다!')
else:
    print('⚠️ 일부 라이브러리에 문제가 있습니다.')
    print('\n🔧 해결 방법:')
    print('1. python install_requirements.py --dev 재실행')
    print('2. 가상환경이 활성화되어 있는지 확인')
    print('3. 개별 패키지 수동 설치:')
    
    if not validation_results[0]:
        print('   pip install numpy pandas matplotlib')
    if not validation_results[1]:
        print('   pip install opencv-python pillow')
    if not validation_results[2]:
        print('   pip install torch torchvision torchaudio')
    if not validation_results[3]:
        print('   pip install faiss-cpu')
    if not validation_results[4]:
        print('   pip install fastapi uvicorn')
    if not validation_results[5]:
        print('   pip install plotly scikit-learn seaborn')