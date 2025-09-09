print('=== MLOps Vision 환경 검증 ===')
print()

# 1. 기본 라이브러리 확인
try:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    print('✅ 기본 과학 계산 라이브러리: 정상')
except ImportError as e:
    print(f'❌ 기본 라이브러리 오류: {e}')

# 2. 이미지 처리 라이브러리 확인
try:
    import cv2
    from PIL import Image
    print('✅ 이미지 처리 라이브러리: 정상')
except ImportError as e:
    print(f'❌ 이미지 처리 라이브러리 오류: {e}')

# 3. PyTorch 환경 확인
try:
    import torch
    import torchvision
    print(f'✅ PyTorch {torch.__version__}: 정상')
    print(f'   CUDA 사용 가능: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'   GPU 장치 수: {torch.cuda.device_count()}')
    else:
        print('   CPU 모드로 실행됩니다')
except ImportError as e:
    print(f'❌ PyTorch 오류: {e}')

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
except ImportError as e:
    print(f'❌ FAISS 오류: {e}')

# 5. 웹 API 라이브러리 확인
try:
    import fastapi
    import uvicorn
    import sqlalchemy
    print('✅ 웹 API 라이브러리: 정상')
except ImportError as e:
    print(f'❌ 웹 API 라이브러리 오류: {e}')

# 6. 시각화 라이브러리 확인
try:
    import plotly
    import sklearn
    print('✅ 시각화 및 ML 라이브러리: 정상')
except ImportError as e:
    print(f'❌ 시각화 라이브러리 오류: {e}')

print()
print('🎉 환경 검증이 완료되었습니다!')
print('이제 PatchCore 구현을 시작할 준비가 되었습니다.')