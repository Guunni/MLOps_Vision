"""
PatchCore 모델 관리 및 결과 분석 시스템
학습된 모델의 저장, 로드, 분석, 배포를 관리하는 시스템
"""

import os
import shutil
import zipfile
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatchCoreModelManager:
    """
    PatchCore 모델의 전체 생명주기를 관리하는 클래스
    
    주요 기능:
    1. 학습 완료된 모델 자동 탐지
    2. 모델 파일 검증 및 메타데이터 추출
    3. 모델 압축 및 아카이브 생성
    4. 모델 성능 분석 및 비교
    5. 모델 배포 준비
    """
    
    def __init__(self, results_dir: str = None, models_archive_dir: str = None):
        """
        모델 관리자를 초기화합니다.
        
        Args:
            results_dir: 학습 결과가 저장된 디렉토리
            models_archive_dir: 모델 아카이브를 저장할 디렉토리
        """
        if results_dir is None:
            self.results_dir = Path.cwd().parent / "models"
        else:
            self.results_dir = Path(results_dir)
        
        if models_archive_dir is None:
            self.models_archive_dir = Path.cwd().parent / "models" / "archives"
        else:
            self.models_archive_dir = Path(models_archive_dir)
        
        # 디렉토리 생성
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.models_archive_dir.mkdir(parents=True, exist_ok=True)
        
        # 모델 탐지를 위한 필수 파일들
        self.required_model_files = [
            "patchcore_params.pkl",
            "nnscorer_search_index.faiss"
        ]
        
        logger.info(f"모델 관리자 초기화됨")
        logger.info(f"  - 결과 디렉토리: {self.results_dir}")
        logger.info(f"  - 아카이브 디렉토리: {self.models_archive_dir}")
    
    def discover_trained_models(self) -> List[Dict[str, Any]]:
        """
        결과 디렉토리에서 학습 완료된 모델들을 자동으로 탐지합니다.
        
        Returns:
            List[Dict]: 발견된 모델들의 정보 리스트
        """
        logger.info("🔍 학습된 모델 탐지 중...")
        
        discovered_models = []
        
        # patchcore_params.pkl 파일을 찾아서 모델 디렉토리 식별
        for pkl_file in self.results_dir.rglob("patchcore_params.pkl"):
            model_dir = pkl_file.parent
            
            # 필수 파일들이 모두 있는지 확인
            model_info = self._analyze_model_directory(model_dir)
            
            if model_info["is_complete"]:
                discovered_models.append(model_info)
                logger.info(f"✅ 완전한 모델 발견: {model_info['model_name']}")
            else:
                logger.warning(f"⚠️ 불완전한 모델: {model_dir}")
                logger.warning(f"   누락된 파일: {model_info['missing_files']}")
        
        logger.info(f"총 {len(discovered_models)}개의 완전한 모델을 발견했습니다.")
        return discovered_models
    
    def _analyze_model_directory(self, model_dir: Path) -> Dict[str, Any]:
        """
        모델 디렉토리를 분석하여 상세 정보를 추출합니다.
        
        Args:
            model_dir: 분석할 모델 디렉토리
            
        Returns:
            Dict: 모델 분석 결과
        """
        model_info = {
            "model_path": str(model_dir),
            "model_name": model_dir.name,
            "is_complete": True,
            "missing_files": [],
            "available_files": [],
            "metadata": {},
            "creation_time": None,
            "file_sizes": {}
        }
        
        # 파일 존재 여부 확인
        for required_file in self.required_model_files:
            file_path = model_dir / required_file
            if file_path.exists():
                model_info["available_files"].append(required_file)
                model_info["file_sizes"][required_file] = file_path.stat().st_size
                
                # 생성 시간 업데이트 (가장 최근 파일 기준)
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if model_info["creation_time"] is None or file_mtime > model_info["creation_time"]:
                    model_info["creation_time"] = file_mtime
            else:
                model_info["missing_files"].append(required_file)
                model_info["is_complete"] = False
        
        # 추가 파일들도 탐지
        for file_path in model_dir.iterdir():
            if file_path.is_file() and file_path.name not in self.required_model_files:
                model_info["available_files"].append(file_path.name)
                model_info["file_sizes"][file_path.name] = file_path.stat().st_size
        
        # 메타데이터 추출 시도
        try:
            model_info["metadata"] = self._extract_model_metadata(model_dir)
        except Exception as e:
            logger.warning(f"메타데이터 추출 실패: {e}")
            model_info["metadata"] = {}
        
        return model_info
    
    def _extract_model_metadata(self, model_dir: Path) -> Dict[str, Any]:
        """
        모델 디렉토리에서 메타데이터를 추출합니다.
        
        Args:
            model_dir: 모델 디렉토리
            
        Returns:
            Dict: 추출된 메타데이터
        """
        metadata = {}
        
        # patchcore_params.pkl에서 설정 정보 추출
        params_file = model_dir / "patchcore_params.pkl"
        if params_file.exists():
            try:
                with open(params_file, 'rb') as f:
                    params = pickle.load(f)
                    metadata["model_params"] = self._sanitize_params(params)
            except Exception as e:
                logger.warning(f"파라미터 파일 읽기 실패: {e}")
        
        # 성능 로그 파일 찾기 및 분석
        performance_data = self._find_performance_data(model_dir)
        if performance_data:
            metadata["performance"] = performance_data
        
        # 디렉토리 구조 분석
        run_dir = model_dir.parent.parent  # results/log_group/run_X/model_dir
        if run_dir.exists():
            metadata["experiment_info"] = {
                "log_group": run_dir.parent.name,
                "run_id": run_dir.name,
                "full_path": str(run_dir)
            }
        
        return metadata
    
    def _sanitize_params(self, params: Any) -> Dict[str, Any]:
        """
        파라미터 객체를 JSON 직렬화 가능한 형태로 변환합니다.
        """
        if hasattr(params, '__dict__'):
            return {k: str(v) for k, v in params.__dict__.items()}
        elif isinstance(params, dict):
            return {k: str(v) for k, v in params.items()}
        else:
            return {"raw_params": str(params)}
    
    def _find_performance_data(self, model_dir: Path) -> Optional[Dict[str, Any]]:
        """
        모델의 성능 데이터를 찾아서 분석합니다.
        """
        performance_data = {}
        
        # results.csv 파일 찾기
        run_dir = model_dir.parent.parent
        results_csv = run_dir / "results.csv"
        
        if results_csv.exists():
            try:
                df = pd.read_csv(results_csv)
                if not df.empty:
                    # 마지막 행의 성능 지표 추출
                    last_row = df.iloc[-1]
                    performance_data = {
                        "image_auroc": float(last_row.get("image_auroc", 0)),
                        "pixel_auroc": float(last_row.get("pixel_auroc", 0)),
                        "image_aupro": float(last_row.get("image_aupro", 0)),
                        "pixel_aupro": float(last_row.get("pixel_aupro", 0))
                    }
            except Exception as e:
                logger.warning(f"성능 데이터 분석 실패: {e}")
        
        return performance_data if performance_data else None
    
    def create_model_archive(self, model_info: Dict[str, Any], archive_name: str = None) -> Dict[str, Any]:
        """
        모델을 압축 아카이브로 생성합니다.
        
        Args:
            model_info: 압축할 모델 정보
            archive_name: 아카이브 파일 이름 (None이면 자동 생성)
            
        Returns:
            Dict: 아카이브 생성 결과
        """
        model_dir = Path(model_info["model_path"])
        
        if archive_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_name = f"patchcore_{model_info['model_name']}_{timestamp}.zip"
        
        archive_path = self.models_archive_dir / archive_name
        
        try:
            logger.info(f"📦 모델 아카이브 생성 중: {archive_name}")
            
            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # 모델 파일들 추가
                for file_name in model_info["available_files"]:
                    file_path = model_dir / file_name
                    if file_path.exists():
                        zipf.write(file_path, f"model/{file_name}")
                
                # 메타데이터 JSON 파일 추가
                metadata_json = json.dumps(model_info, indent=2, default=str)
                zipf.writestr("metadata.json", metadata_json)
                
                # 성능 결과 파일이 있다면 추가
                run_dir = model_dir.parent.parent
                results_csv = run_dir / "results.csv"
                if results_csv.exists():
                    zipf.write(results_csv, "performance/results.csv")
            
            archive_size = archive_path.stat().st_size
            
            logger.info(f"✅ 아카이브 생성 완료: {archive_path}")
            logger.info(f"   파일 크기: {archive_size / 1024 / 1024:.2f} MB")
            
            return {
                "success": True,
                "archive_path": str(archive_path),
                "archive_name": archive_name,
                "archive_size": archive_size,
                "included_files": model_info["available_files"]
            }
            
        except Exception as e:
            logger.error(f"❌ 아카이브 생성 실패: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def load_model_for_inference(self, model_path: str) -> Optional[Any]:
        """
        추론을 위해 모델을 로드합니다.
        
        Args:
            model_path: 모델 디렉토리 경로
            
        Returns:
            로드된 모델 객체 (실패시 None)
        """
        model_dir = Path(model_path)
        
        try:
            logger.info(f"🔄 모델 로딩 중: {model_dir}")
            
            # PatchCore 모델 로딩을 위한 시스템 경로 설정
            import sys
            patchcore_src = Path.cwd().parent / "patchcore-inspection" / "src"
            if str(patchcore_src) not in sys.path:
                sys.path.insert(0, str(patchcore_src))
            
            # PatchCore 모듈 import
            import patchcore.patchcore
            import patchcore.utils
            
            # 모델 파라미터 로드
            params_file = model_dir / "patchcore_params.pkl"
            with open(params_file, 'rb') as f:
                model_params = pickle.load(f)
            
            # FAISS 인덱스 로드 (실제 구현에서는 더 복잡한 로딩 과정이 필요)
            faiss_file = model_dir / "nnscorer_search_index.faiss"
            
            logger.info("✅ 모델 로딩 완료")
            
            # 실제 PatchCore 객체 반환 (구현에 따라 달라질 수 있음)
            return {
                "model_params": model_params,
                "faiss_index_path": str(faiss_file),
                "model_dir": str(model_dir)
            }
            
        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            return None
    
    def compare_models(self, model_list: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        여러 모델의 성능을 비교합니다.
        
        Args:
            model_list: 비교할 모델들의 정보 리스트
            
        Returns:
            DataFrame: 모델 비교 결과
        """
        comparison_data = []
        
        for model_info in model_list:
            row = {
                "model_name": model_info["model_name"],
                "creation_time": model_info["creation_time"],
                "total_size_mb": sum(model_info["file_sizes"].values()) / 1024 / 1024
            }
            
            # 성능 지표 추가
            if "performance" in model_info["metadata"]:
                perf = model_info["metadata"]["performance"]
                row.update({
                    "image_auroc": perf.get("image_auroc", None),
                    "pixel_auroc": perf.get("pixel_auroc", None),
                    "image_aupro": perf.get("image_aupro", None),
                    "pixel_aupro": perf.get("pixel_aupro", None)
                })
            
            # 모델 파라미터 추가
            if "model_params" in model_info["metadata"]:
                params = model_info["metadata"]["model_params"]
                # 주요 파라미터들만 추출
                for key in ["backbone", "layers", "anomaly_scorer_num_nn", "sampling_ratio"]:
                    if key in params:
                        row[key] = params[key]
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        return df
    
    def get_best_model(self, models: List[Dict[str, Any]], metric: str = "image_auroc") -> Optional[Dict[str, Any]]:
        """
        지정된 메트릭 기준으로 최고 성능 모델을 찾습니다.
        
        Args:
            models: 모델 리스트
            metric: 비교 기준 메트릭
            
        Returns:
            Dict: 최고 성능 모델 정보 (없으면 None)
        """
        best_model = None
        best_score = -1
        
        for model in models:
            if "performance" in model["metadata"]:
                score = model["metadata"]["performance"].get(metric, -1)
                if score > best_score:
                    best_score = score
                    best_model = model
        
        return best_model
    
    def cleanup_old_models(self, keep_count: int = 5) -> int:
        """
        오래된 모델들을 정리합니다.
        
        Args:
            keep_count: 보관할 모델 수
            
        Returns:
            int: 삭제된 모델 수
        """
        models = self.discover_trained_models()
        
        if len(models) <= keep_count:
            logger.info(f"정리할 모델이 없습니다. (현재 {len(models)}개, 보관 기준 {keep_count}개)")
            return 0
        
        # 생성 시간 기준으로 정렬 (최신순)
        models.sort(key=lambda x: x["creation_time"], reverse=True)
        
        # 보관할 모델들을 제외한 나머지 삭제
        models_to_delete = models[keep_count:]
        deleted_count = 0
        
        for model in models_to_delete:
            try:
                model_path = Path(model["model_path"])
                if model_path.exists():
                    shutil.rmtree(model_path)
                    logger.info(f"🗑️ 모델 삭제됨: {model['model_name']}")
                    deleted_count += 1
            except Exception as e:
                logger.error(f"❌ 모델 삭제 실패: {model['model_name']} - {e}")
        
        logger.info(f"총 {deleted_count}개의 오래된 모델을 정리했습니다.")
        return deleted_count

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 모델 관리자 인스턴스 생성
    manager = PatchCoreModelManager()
    
    # 학습된 모델들 탐지
    models = manager.discover_trained_models()
    
    if models:
        print(f"📋 발견된 모델: {len(models)}개")
        
        for i, model in enumerate(models, 1):
            print(f"\n{i}. {model['model_name']}")
            print(f"   경로: {model['model_path']}")
            print(f"   생성시간: {model['creation_time']}")
            print(f"   파일 수: {len(model['available_files'])}개")
            
            if "performance" in model["metadata"]:
                perf = model["metadata"]["performance"]
                print(f"   성능: Image AUROC {perf.get('image_auroc', 'N/A')}")
        
        # 첫 번째 모델로 아카이브 생성 예시
        print(f"\n📦 첫 번째 모델의 아카이브 생성 중...")
        archive_result = manager.create_model_archive(models[0])
        
        if archive_result["success"]:
            print(f"✅ 아카이브 생성 완료: {archive_result['archive_name']}")
        else:
            print(f"❌ 아카이브 생성 실패: {archive_result['error']}")
            
    else:
        print("📂 아직 학습된 모델이 없습니다.")
        print("먼저 PatchCore 학습을 완료해주세요.")