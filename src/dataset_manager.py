"""
데이터셋 관리 시스템
다양한 형태의 vision 검사 데이터셋을 통합 관리하는 시스템
"""

import os
import shutil
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import requests
from tqdm import tqdm
import hashlib

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetManager:
    """
    Vision 검사용 데이터셋을 관리하는 클래스
    
    이 클래스는 다음 기능들을 제공합니다:
    1. 다양한 형태의 압축 파일 해제 (tar, zip, etc.)
    2. 데이터셋 구조 검증 및 정리
    3. MVTec과 같은 표준 데이터셋 자동 다운로드
    4. 사용자 정의 데이터셋 업로드 및 처리
    5. 데이터셋 메타데이터 관리
    """
    
    def __init__(self, data_dir: str = None):
        """
        데이터셋 매니저를 초기화합니다.
        
        Args:
            data_dir: 데이터셋을 저장할 기본 디렉토리
        """
        if data_dir is None:
            # 프로젝트 루트의 data 폴더를 기본값으로 사용
            self.data_dir = Path.cwd().parent / "data"
        else:
            self.data_dir = Path(data_dir)
        
        # 데이터 디렉토리 생성
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 각종 하위 디렉토리 설정
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.temp_dir = self.data_dir / "temp"
        
        # 하위 디렉토리들 생성
        for dir_path in [self.raw_data_dir, self.processed_data_dir, self.temp_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"데이터셋 매니저 초기화됨: {self.data_dir}")
    
    def extract_archive(self, archive_path: Path, extract_to: Path = None) -> bool:
        """
        다양한 형태의 압축 파일을 해제합니다.
        
        Args:
            archive_path: 압축 파일 경로
            extract_to: 압축 해제할 디렉토리 (None이면 자동 결정)
            
        Returns:
            bool: 압축 해제 성공 여부
        """
        if extract_to is None:
            extract_to = self.temp_dir / archive_path.stem
        
        extract_to.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"📦 압축 파일 해제 시작: {archive_path}")
            
            # 파일 확장자에 따른 압축 해제 방식 결정
            if archive_path.suffix.lower() in ['.tar', '.tar.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tar:
                    tar.extractall(path=extract_to)
                    
            elif archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
                    
            else:
                logger.error(f"❌ 지원하지 않는 압축 형식: {archive_path.suffix}")
                return False
            
            logger.info(f"✅ 압축 해제 완료: {extract_to}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 압축 해제 실패: {e}")
            return False
    
    def normalize_dataset_structure(self, dataset_path: Path, target_name: str) -> Dict[str, Any]:
        """
        데이터셋 구조를 표준화합니다.
        
        MVTec 데이터셋의 경우 다음과 같은 구조를 가져야 합니다:
        dataset_name/
        ├── train/
        │   └── good/
        └── test/
            ├── good/
            └── defect_type1/
            └── defect_type2/
        └── ground_truth/
            └── defect_type1/
            └── defect_type2/
            
        Args:
            dataset_path: 정리할 데이터셋 경로
            target_name: 최종 데이터셋 이름
            
        Returns:
            Dict: 데이터셋 구조 정보
        """
        logger.info(f"🔧 데이터셋 구조 정리 시작: {dataset_path}")
        
        result = {
            "success": False,
            "dataset_path": None,
            "categories": [],
            "train_samples": 0,
            "test_samples": 0,
            "structure": {}
        }
        
        try:
            # 최종 데이터셋 경로 설정
            final_dataset_path = self.processed_data_dir / target_name
            
            # 기존 처리된 데이터가 있다면 백업
            if final_dataset_path.exists():
                backup_path = final_dataset_path.with_suffix('.backup')
                if backup_path.exists():
                    shutil.rmtree(backup_path)
                shutil.move(final_dataset_path, backup_path)
                logger.info(f"기존 데이터셋을 백업했습니다: {backup_path}")
            
            # 데이터셋 구조 탐색 및 정리
            self._explore_and_organize_dataset(dataset_path, final_dataset_path, result)
            
            # 구조 검증
            if self._validate_dataset_structure(final_dataset_path, result):
                result["success"] = True
                result["dataset_path"] = str(final_dataset_path)
                logger.info(f"✅ 데이터셋 구조 정리 완료: {final_dataset_path}")
            else:
                logger.error("❌ 데이터셋 구조 검증 실패")
            
        except Exception as e:
            logger.error(f"❌ 데이터셋 구조 정리 중 오류: {e}")
        
        return result
    
    def _explore_and_organize_dataset(self, source_path: Path, target_path: Path, result: Dict):
        """
        데이터셋을 탐색하고 표준 구조로 정리합니다.
        """
        logger.info("🔍 데이터셋 구조 탐색 중...")
        
        # 가능한 데이터셋 루트 찾기
        dataset_root = self._find_dataset_root(source_path)
        
        if dataset_root is None:
            logger.error("❌ 유효한 데이터셋 구조를 찾을 수 없습니다.")
            return
        
        logger.info(f"📁 데이터셋 루트 발견: {dataset_root}")
        
        # 표준 구조로 복사
        target_path.mkdir(parents=True, exist_ok=True)
        
        # train, test, ground_truth 디렉토리 처리
        for subset in ['train', 'test', 'ground_truth']:
            source_subset = dataset_root / subset
            target_subset = target_path / subset
            
            if source_subset.exists():
                shutil.copytree(source_subset, target_subset, dirs_exist_ok=True)
                
                # 샘플 수 계산
                if subset == 'train':
                    result["train_samples"] = self._count_images(target_subset)
                elif subset == 'test':
                    result["test_samples"] = self._count_images(target_subset)
                
                # 카테고리 정보 수집
                if subset in ['test', 'ground_truth']:
                    categories = [d.name for d in target_subset.iterdir() if d.is_dir()]
                    result["categories"].extend(categories)
        
        # 중복 제거
        result["categories"] = list(set(result["categories"]))
        
        # 구조 정보 기록
        result["structure"] = self._analyze_structure(target_path)
    
    def _find_dataset_root(self, search_path: Path) -> Optional[Path]:
        """
        실제 데이터셋이 시작되는 루트 디렉토리를 찾습니다.
        """
        # 현재 디렉토리에서 train, test 폴더를 찾기
        if (search_path / 'train').exists() and (search_path / 'test').exists():
            return search_path
        
        # 하위 디렉토리에서 찾기 (최대 3단계까지)
        for depth in range(1, 4):
            for item in search_path.rglob('*'):
                if item.is_dir() and item.name in ['train', 'test']:
                    potential_root = item.parent
                    if (potential_root / 'train').exists() and (potential_root / 'test').exists():
                        return potential_root
        
        return None
    
    def _count_images(self, directory: Path) -> int:
        """
        디렉토리 내의 이미지 파일 수를 계산합니다.
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        count = 0
        
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                count += 1
        
        return count
    
    def _analyze_structure(self, dataset_path: Path) -> Dict[str, Any]:
        """
        데이터셋 구조를 분석합니다.
        """
        structure = {}
        
        for subset_dir in dataset_path.iterdir():
            if subset_dir.is_dir():
                subset_name = subset_dir.name
                structure[subset_name] = {}
                
                for category_dir in subset_dir.iterdir():
                    if category_dir.is_dir():
                        category_name = category_dir.name
                        image_count = self._count_images(category_dir)
                        structure[subset_name][category_name] = image_count
        
        return structure
    
    def _validate_dataset_structure(self, dataset_path: Path, result: Dict) -> bool:
        """
        데이터셋 구조가 올바른지 검증합니다.
        """
        required_dirs = ['train', 'test']
        
        for required_dir in required_dirs:
            dir_path = dataset_path / required_dir
            if not dir_path.exists():
                logger.error(f"❌ 필수 디렉토리가 없습니다: {required_dir}")
                return False
        
        # train/good 디렉토리 확인
        train_good = dataset_path / 'train' / 'good'
        if not train_good.exists():
            logger.error("❌ train/good 디렉토리가 없습니다.")
            return False
        
        # 최소한의 이미지 파일 확인
        if result["train_samples"] == 0:
            logger.error("❌ 훈련용 이미지가 없습니다.")
            return False
        
        if result["test_samples"] == 0:
            logger.error("❌ 테스트용 이미지가 없습니다.")
            return False
        
        logger.info("✅ 데이터셋 구조 검증 완료")
        return True
    
    def process_uploaded_dataset(self, archive_path: str, dataset_name: str) -> Dict[str, Any]:
        """
        업로드된 데이터셋 파일을 처리합니다.
        
        Args:
            archive_path: 업로드된 압축 파일 경로
            dataset_name: 데이터셋 이름
            
        Returns:
            Dict: 처리 결과 정보
        """
        logger.info(f"🚀 데이터셋 처리 시작: {archive_path}")
        
        archive_path = Path(archive_path)
        
        # 1단계: 압축 해제
        temp_extract_dir = self.temp_dir / f"{dataset_name}_extract"
        if not self.extract_archive(archive_path, temp_extract_dir):
            return {"success": False, "error": "압축 해제 실패"}
        
        # 2단계: 구조 정리
        result = self.normalize_dataset_structure(temp_extract_dir, dataset_name)
        
        # 3단계: 임시 파일 정리
        try:
            shutil.rmtree(temp_extract_dir)
            logger.info("🧹 임시 파일 정리 완료")
        except Exception as e:
            logger.warning(f"⚠️ 임시 파일 정리 실패: {e}")
        
        return result
    
    def list_available_datasets(self) -> List[Dict[str, Any]]:
        """
        사용 가능한 데이터셋 목록을 반환합니다.
        
        Returns:
            List: 데이터셋 정보 리스트
        """
        datasets = []
        
        for dataset_dir in self.processed_data_dir.iterdir():
            if dataset_dir.is_dir():
                # 메타데이터 파일이 있는지 확인
                metadata_file = dataset_dir / "dataset_info.json"
                
                dataset_info = {
                    "name": dataset_dir.name,
                    "path": str(dataset_dir),
                    "train_samples": self._count_images(dataset_dir / "train") if (dataset_dir / "train").exists() else 0,
                    "test_samples": self._count_images(dataset_dir / "test") if (dataset_dir / "test").exists() else 0,
                    "structure": self._analyze_structure(dataset_dir)
                }
                
                datasets.append(dataset_info)
        
        return datasets
    
    def get_dataset_info(self, dataset_name: str) -> Optional[Dict[str, Any]]:
        """
        특정 데이터셋의 상세 정보를 반환합니다.
        
        Args:
            dataset_name: 데이터셋 이름
            
        Returns:
            Dict: 데이터셋 상세 정보 (없으면 None)
        """
        dataset_path = self.processed_data_dir / dataset_name
        
        if not dataset_path.exists():
            return None
        
        return {
            "name": dataset_name,
            "path": str(dataset_path),
            "train_samples": self._count_images(dataset_path / "train") if (dataset_path / "train").exists() else 0,
            "test_samples": self._count_images(dataset_path / "test") if (dataset_path / "test").exists() else 0,
            "structure": self._analyze_structure(dataset_path),
            "categories": [d.name for d in (dataset_path / "test").iterdir() if d.is_dir()] if (dataset_path / "test").exists() else []
        }

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 데이터셋 매니저 인스턴스 생성
    manager = DatasetManager()
    
    # 사용 가능한 데이터셋 목록 출력
    datasets = manager.list_available_datasets()
    
    if datasets:
        print("📋 사용 가능한 데이터셋:")
        for dataset in datasets:
            print(f"  - {dataset['name']}: 훈련 {dataset['train_samples']}개, 테스트 {dataset['test_samples']}개")
    else:
        print("📂 아직 처리된 데이터셋이 없습니다.")
        print("데이터셋을 업로드하여 처리를 시작하세요.")