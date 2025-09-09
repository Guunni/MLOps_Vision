"""
MLOps Vision 통합 웹 API
모든 PatchCore 관련 기능을 하나의 웹 인터페이스로 통합하는 메인 API
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from fastapi import Request
from fastapi.responses import RedirectResponse
import json
import logging
from pathlib import Path
import tempfile
import shutil
from datetime import datetime
import threading

# 우리가 만든 모듈들 import
from patchcore_manager import PatchCoreManager
from dataset_manager import DatasetManager
from training_manager import PatchCoreTrainer, PatchCoreConfig
from model_manager import PatchCoreModelManager

#static 파일 지원 추가
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MLOps Vision Platform",
    description="PatchCore 기반 Vision 검사 MLOps 통합 플랫폼",
    version="1.0.0"
)

# Static 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../templates")

# 전역 매니저 인스턴스들
patchcore_manager = PatchCoreManager()
dataset_manager = DatasetManager()
trainer = PatchCoreTrainer()
model_manager = PatchCoreModelManager()

# Pydantic 모델 정의 (API 요청/응답 스키마)
class TrainingConfigModel(BaseModel):
    """학습 설정을 위한 Pydantic 모델"""
    backbone: str = "wideresnet50"
    layers: List[str] = ["layer2", "layer3"]
    anomaly_scorer_num_nn: int = 7
    sampling_ratio: float = 0.3
    dataset_name: str  # 필수 필드로 변경
    resize_size: int = 320
    image_size: int = 256
    gpu_id: Optional[int] = None
    
    # 새로 추가된 필드들
    training_type: str = "new"  # "new", "transfer", "retrain"
    base_model: Optional[str] = None  # 전이학습/재학습 시 기존 모델명
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    train_ratio: float = 0.7
    val_ratio: float = 0.2

class SystemStatusResponse(BaseModel):
    """시스템 상태 응답 모델"""
    patchcore_installed: bool
    datasets_available: int
    models_trained: int
    training_active: bool
    system_ready: bool

# =============================================================================
# 시스템 상태 및 초기화 API
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def redirect_to_dashboard():
    """메인 페이지에서 대시보드로 리다이렉트"""
    return RedirectResponse(url="/dashboard")

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """메인 대시보드 페이지"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/ui/datasets", response_class=HTMLResponse)
async def get_datasets_page(request: Request):
    """데이터셋 관리 페이지"""
    return templates.TemplateResponse("datasets.html", {"request": request})

@app.get("/ui/training", response_class=HTMLResponse)
async def get_training_page(request: Request):
    """학습 관리 페이지"""
    return templates.TemplateResponse("training.html", {"request": request})

@app.get("/ui/models", response_class=HTMLResponse)
async def get_models_page(request: Request):
    """모델 관리 페이지"""
    return templates.TemplateResponse("models.html", {"request": request})

@app.get("/ui/monitoring", response_class=HTMLResponse)
async def get_monitoring_page(request: Request):
    """모니터링 페이지"""
    return templates.TemplateResponse("monitoring.html", {"request": request})

@app.get("/system/status", response_model=SystemStatusResponse)
async def get_system_status():
    """시스템 전체 상태를 확인합니다."""
    try:
        # PatchCore 설치 상태 확인
        patchcore_installed = patchcore_manager.is_patchcore_installed()
        
        # 사용 가능한 데이터셋 수 확인
        datasets = dataset_manager.list_available_datasets()
        datasets_count = len(datasets)
        
        # 학습된 모델 수 확인
        models = model_manager.discover_trained_models()
        models_count = len(models)
        
        # 현재 학습 진행 상태 확인
        training_active = trainer.is_training_active()
        
        # 전체 시스템 준비 상태
        system_ready = patchcore_installed and datasets_count > 0
        
        return SystemStatusResponse(
            patchcore_installed=patchcore_installed,
            datasets_available=datasets_count,
            models_trained=models_count,
            training_active=training_active,
            system_ready=system_ready
        )
        
    except Exception as e:
        logger.error(f"시스템 상태 확인 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/system/setup")
async def setup_system():
    """전체 시스템을 초기화합니다. (이미 설치된 구성요소는 건너뜁니다)"""
    try:
        logger.info("🔍 시스템 상태 확인 중...")
        
        setup_results = {
            "patchcore_status": "unknown",
            "actions_taken": [],
            "skipped_actions": [],
            "warnings": []
        }
        
        # 1. PatchCore 설치 상태 확인
        if patchcore_manager.is_patchcore_installed():
            setup_results["patchcore_status"] = "already_installed"
            setup_results["skipped_actions"].append("PatchCore 라이브러리 (이미 설치됨)")
            logger.info("✅ PatchCore가 이미 설치되어 있습니다.")
            
            # 추가 검증만 수행
            verification = patchcore_manager.verify_installation()
            if not verification["installation_complete"]:
                setup_results["warnings"].append("PatchCore 설치가 불완전할 수 있습니다.")
                setup_results["warnings"].extend(verification["error_messages"])
        else:
            logger.info("📦 PatchCore 설치가 필요합니다...")
            setup_result = patchcore_manager.setup_patchcore_complete()
            
            if setup_result:
                setup_results["patchcore_status"] = "newly_installed"
                setup_results["actions_taken"].append("PatchCore 라이브러리 설치 완료")
            else:
                setup_results["patchcore_status"] = "installation_failed"
                setup_results["warnings"].append("PatchCore 설치에 실패했습니다.")
        
        # 2. 기본 디렉토리 구조 확인 및 생성
        required_dirs = [
            dataset_manager.data_dir,
            dataset_manager.processed_data_dir,
            trainer.results_dir,
            model_manager.models_archive_dir
        ]
        
        dirs_created = []
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                dirs_created.append(str(dir_path.name))
        
        if dirs_created:
            setup_results["actions_taken"].append(f"디렉토리 생성: {', '.join(dirs_created)}")
        else:
            setup_results["skipped_actions"].append("모든 필수 디렉토리 (이미 존재)")
        
        # 3. 시스템 전체 상태 확인
        system_ready = setup_results["patchcore_status"] in ["already_installed", "newly_installed"]
        
        # 4. 결과 정리
        total_actions = len(setup_results["actions_taken"])
        total_skipped = len(setup_results["skipped_actions"])
        
        if system_ready:
            logger.info("✅ 시스템 준비 완료")
            return {
                "success": True,
                "system_ready": True,
                "message": f"시스템이 사용 준비되었습니다. ({total_actions}개 작업 수행, {total_skipped}개 작업 건너뜀)",
                "details": setup_results,
                "next_steps": [
                    "데이터셋을 업로드하세요 (/datasets/upload)",
                    "학습 설정을 구성하세요 (/training/start)", 
                    "모델 학습을 시작하세요"
                ]
            }
        else:
            logger.error("❌ 시스템 설정에 문제가 있습니다")
            return {
                "success": False,
                "system_ready": False,
                "message": "시스템 설정 중 문제가 발생했습니다.",
                "details": setup_results,
                "troubleshooting": [
                    "로그를 확인하여 구체적인 오류를 파악하세요",
                    "필요한 경우 수동으로 PatchCore를 설치해보세요",
                    "시스템 권한을 확인하세요"
                ]
            }
            
    except Exception as e:
        logger.error(f"시스템 설정 중 예외 발생: {e}")
        return {
            "success": False,
            "system_ready": False,
            "error": str(e),
            "message": "시스템 설정 중 예상치 못한 오류가 발생했습니다."
        }
    
@app.post("/system/setup/force-reinstall")
async def force_reinstall_system():
    """강제로 전체 시스템을 재설치합니다. (주의: 기존 설치를 삭제합니다)"""
    try:
        logger.warning("⚠️ 강제 재설치를 시작합니다...")
        
        # 강제 재설치 수행
        setup_result = patchcore_manager.setup_patchcore_complete(force_reinstall=True)
        
        if setup_result:
            logger.info("✅ 강제 재설치 완료")
            return {
                "success": True,
                "message": "시스템이 강제로 재설치되었습니다.",
                "warning": "모든 기존 설정이 삭제되었습니다."
            }
        else:
            raise HTTPException(status_code=500, detail="강제 재설치 실패")
            
    except Exception as e:
        logger.error(f"강제 재설치 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 데이터셋 관리 API
# =============================================================================

@app.get("/datasets")
async def list_datasets():
    """사용 가능한 데이터셋 목록을 반환합니다."""
    try:
        datasets = dataset_manager.list_available_datasets()
        return {
            "datasets": datasets,
            "total_count": len(datasets)
        }
    except Exception as e:
        logger.error(f"데이터셋 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Form(...)
):
    """데이터셋 파일을 업로드하고 처리합니다."""
    try:
        logger.info(f"📤 데이터셋 업로드 시작: {file.filename}")
        
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # 데이터셋 처리
            result = dataset_manager.process_uploaded_dataset(tmp_file_path, dataset_name)
            
            return {
                "success": result["success"],
                "dataset_name": dataset_name,
                "processing_result": result
            }
            
        finally:
            # 임시 파일 정리
            Path(tmp_file_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"데이터셋 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/datasets/{dataset_name}")
async def get_dataset_info(dataset_name: str):
    """특정 데이터셋의 상세 정보를 반환합니다."""
    try:
        dataset_info = dataset_manager.get_dataset_info(dataset_name)
        
        if dataset_info is None:
            raise HTTPException(status_code=404, detail=f"데이터셋을 찾을 수 없습니다: {dataset_name}")
        
        return dataset_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터셋 정보 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 학습 관리 API
# =============================================================================

@app.get("/training/status")
async def get_training_status():
    """현재 학습 상태를 반환합니다."""
    try:
        status = trainer.get_training_status()
        
        return {
            "is_training_active": trainer.is_training_active(),
            "current_training": status
        }
        
    except Exception as e:
        logger.error(f"학습 상태 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/start")
async def start_training(config: TrainingConfigModel):
    """모델 학습을 시작합니다 (새 학습, 전이학습, 재학습 지원)"""
    try:
        # 전이학습/재학습 시 기존 모델 확인
        if config.training_type in ["transfer", "retrain"] and not config.base_model:
            raise HTTPException(status_code=400, detail="전이학습/재학습 시 기존 모델을 선택해야 합니다")
        
        # 데이터셋 존재 확인
        dataset_info = dataset_manager.get_dataset_info(config.dataset_name)
        if dataset_info is None:
            raise HTTPException(status_code=404, detail=f"데이터셋을 찾을 수 없습니다: {config.dataset_name}")
        
        # 이미 학습이 진행 중인지 확인
        if trainer.is_training_active():
            raise HTTPException(status_code=400, detail="이미 학습이 진행 중입니다")
        
        # PatchCore 학습 설정 생성
        training_config = PatchCoreConfig(
            backbone=config.backbone,
            layers=config.layers,
            anomaly_scorer_num_nn=config.anomaly_scorer_num_nn,
            sampling_ratio=config.sampling_ratio,
            dataset_name=config.dataset_name,
            resize_size=config.resize_size,
            image_size=config.image_size,
            gpu_id=config.gpu_id
        )
        
        # 학습 타입에 따른 처리
        if config.training_type == "transfer":
            logger.info(f"전이학습 시작: {config.base_model} -> {config.dataset_name}")
            result = trainer.start_transfer_learning(
                config=training_config,
                dataset_path=dataset_info["path"],
                base_model_path=config.base_model,
                blocking=False  # 백그라운드 실행
            )
        elif config.training_type == "retrain":
            logger.info(f"재학습 시작: {config.base_model} 기반 전체 재학습")
            result = trainer.start_retraining(
                config=training_config,
                dataset_path=dataset_info["path"],
                base_model_path=config.base_model,
                blocking=False  # 백그라운드 실행
            )
        else:  # new training
            logger.info(f"새 학습 시작: {config.dataset_name}")
            result = trainer.start_training(
                config=training_config,
                dataset_path=dataset_info["path"],
                blocking=False  # 백그라운드 실행
            )
        
        if result["success"]:
            training_type_kr = {
                "new": "새 학습",
                "transfer": "전이학습", 
                "retrain": "재학습"
            }[config.training_type]
            
            return {
                "success": True,
                "message": f"{training_type_kr}이 시작되었습니다",
                "training_info": result["training_info"],
                "training_type": config.training_type
            }
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "학습 시작 실패"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"학습 시작 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/training/stop")
async def stop_training():
    """진행 중인 학습을 중단합니다."""
    try:
        if not trainer.is_training_active():
            raise HTTPException(status_code=400, detail="진행 중인 학습이 없습니다")
        
        success = trainer.stop_training()
        
        if success:
            return {"success": True, "message": "학습이 중단되었습니다"}
        else:
            raise HTTPException(status_code=500, detail="학습 중단 실패")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"학습 중단 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 모델 관리 API
# =============================================================================

@app.get("/models")
async def list_models():
    """학습된 모델 목록을 반환합니다."""
    try:
        models = model_manager.discover_trained_models()
        
        # 모델 비교 데이터 생성
        comparison_df = None
        if models:
            comparison_df = model_manager.compare_models(models)
            comparison_data = comparison_df.to_dict('records')
        else:
            comparison_data = []
        
        return {
            "models": models,
            "total_count": len(models),
            "comparison": comparison_data
        }
        
    except Exception as e:
        logger.error(f"모델 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_name}/archive")
async def create_model_archive(model_name: str):
    """모델을 압축 아카이브로 생성합니다."""
    try:
        # 모델 찾기
        models = model_manager.discover_trained_models()
        target_model = None
        
        for model in models:
            if model["model_name"] == model_name:
                target_model = model
                break
        
        if target_model is None:
            raise HTTPException(status_code=404, detail=f"모델을 찾을 수 없습니다: {model_name}")
        
        # 아카이브 생성
        result = model_manager.create_model_archive(target_model)
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result.get("error", "아카이브 생성 실패"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"모델 아카이브 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/best")
async def get_best_model(metric: str = "image_auroc"):
    """최고 성능 모델을 반환합니다."""
    try:
        models = model_manager.discover_trained_models()
        
        if not models:
            raise HTTPException(status_code=404, detail="학습된 모델이 없습니다")
        
        best_model = model_manager.get_best_model(models, metric)
        
        if best_model is None:
            raise HTTPException(status_code=404, detail=f"{metric} 기준으로 최고 모델을 찾을 수 없습니다")
        
        return {
            "best_model": best_model,
            "metric": metric,
            "score": best_model["metadata"]["performance"].get(metric, None)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"최고 모델 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# 애플리케이션 시작
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    import subprocess
    import sys
    
    logger.info("🚀 MLOps Vision Platform 시작 중...")
    logger.info("웹 브라우저에서 http://localhost:8080 으로 접속하세요")
    
    try:
        # uvicorn을 subprocess로 실행 (가장 안정적인 방법)
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "mlops_api:app", 
            "--host", "127.0.0.1", 
            "--port", "8080", 
            "--reload"
        ]
        
        logger.info("서버를 시작합니다...")
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        logger.info("서버가 중단되었습니다.")
    except Exception as e:
        logger.error(f"서버 시작 실패: {e}")
        logger.info("수동으로 다음 명령어를 실행해주세요:")
        logger.info("uvicorn mlops_api:app --host 127.0.0.1 --port 8080 --reload")