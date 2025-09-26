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
import threading
import sqlite3
import uuid
import pickle
import shutil
from training_manager import PatchCoreTrainer, PatchCoreConfig
from fastapi import File, UploadFile, Form
from fastapi.responses import FileResponse
from typing import List

# 우리가 만든 모듈들 import
from patchcore_manager import PatchCoreManager
from dataset_manager import DatasetManager
from training_manager import PatchCoreTrainer, PatchCoreConfig
from model_manager import PatchCoreModelManager

#static 파일 지원 추가
from fastapi.templating import Jinja2Templates

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="MLOps Vision Platform",
    description="Vision 검사 MLOps 통합 플랫폼",
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
trainer = PatchCoreTrainer()
BASE_DATA_PATH = Path(__file__).resolve().parent / "data" / "processed"

def init_database():
    """데이터베이스 초기화"""
    conn = sqlite3.connect('mlops_vision.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL UNIQUE,
            description TEXT,
            status TEXT DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 데이터셋 테이블 (프로젝트 연결)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_datasets (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            name TEXT NOT NULL,
            path TEXT NOT NULL,
            image_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id)
        )
    ''')
    
    # 모델 테이블 (프로젝트 연결)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS project_models (
            id TEXT PRIMARY KEY,
            project_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            name TEXT NOT NULL,
            backbone TEXT,
            performance_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (project_id) REFERENCES projects (id),
            FOREIGN KEY (dataset_id) REFERENCES project_datasets (id)
        )
    ''')
    
    conn.commit()
    conn.close()

# 애플리케이션 시작 시 DB 초기화
init_database()

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

@app.get("/ui/projects", response_class=HTMLResponse)
async def get_projects_page(request: Request):
    """프로젝트 관리 페이지"""
    return templates.TemplateResponse("projects.html", {"request": request})

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
# 프로젝트 관리 API
# =============================================================================

@app.get("/api/projects")
async def get_projects():
    """모든 프로젝트 목록 조회"""
    try:
        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.id, p.name, p.description, p.status,
                p.created_at, p.updated_at,
                COUNT(DISTINCT d.id) as dataset_count,
                COUNT(DISTINCT m.id) as model_count
            FROM projects p
            LEFT JOIN project_datasets d ON p.id = d.project_id
            LEFT JOIN project_models m ON p.id = m.project_id
            GROUP BY p.id
        ''')
        
        rows = cursor.fetchall()
        projects = []

        for row in rows:
            projects.append({
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "status": row[3],
            "created_at": row[4],
            "updated_at": row[5],
            "dataset_count": row[6],
            "model_count": row[7]
            })

        conn.close()
        return projects # 배열만 return
        
    except Exception as e:
        logger.error(f"프로젝트 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/projects")
async def create_project(project_data: dict):
    """새 프로젝트 생성"""
    try:
        project_name = project_data.get("name")
        project_description = project_data.get("description", "")
        project_status = project_data.get('status', 'active')
        
        if not project_name:
            raise HTTPException(status_code=400, detail="프로젝트 이름이 필요합니다")
        
        project_id = str(uuid.uuid4())
        
        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO projects (id, name, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (project_id, project_name, project_description, project_status))
        
        conn.commit()
        
        # 생성된 프로젝트 정보 반환
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()
        
        project = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "created_at": row[3],
            "updated_at": row[4],
            "dataset_count": 0,
            "model_count": 0
        }
        
        conn.close()
        return {"success": True, "project": project}
        
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="이미 존재하는 프로젝트 이름입니다")
    except Exception as e:
        logger.error(f"프로젝트 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, project_data: dict):
    """프로젝트 정보 수정"""
    try:
        project_name = project_data.get("name")
        project_description = project_data.get("description", "")
        project_status = project_data.get('status', 'active')
        
        if not project_name:
            raise HTTPException(status_code=400, detail="프로젝트 이름이 필요합니다")
        
        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE projects
            SET name = ?, description = ?, status = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (project_name, project_description, project_status, project_id))
        
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
        
        conn.commit()
        
        # 수정된 프로젝트 정보 반환
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        row = cursor.fetchone()
        
        project = {
            "id": row[0],
            "name": row[1],
            "description": row[2],
            "status": row[3],
            "created_at": row[4],
            "updated_at": row[5]
        }
        
        conn.close()
        return {"success": True, "project": project}
        
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="이미 존재하는 프로젝트 이름입니다")
    except Exception as e:
        logger.error(f"프로젝트 수정 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    """프로젝트 삭제"""
    try:
        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()
        
        # 프로젝트 존재 확인
        cursor.execute('SELECT name FROM projects WHERE id = ?', (project_id,))
        project = cursor.fetchone()
        
        if not project:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
        
        # 관련 데이터 삭제 (모델 -> 데이터셋 -> 프로젝트 순)
        cursor.execute('DELETE FROM project_models WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM project_datasets WHERE project_id = ?', (project_id,))
        cursor.execute('DELETE FROM projects WHERE id = ?', (project_id,))
        
        conn.commit()
        conn.close()
        
        return {"success": True, "message": f"프로젝트 '{project[0]}'가 삭제되었습니다"}
        
    except Exception as e:
        logger.error(f"프로젝트 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/api/projects/{project_id}/export")
async def export_project(project_id: str):
    """
    프로젝트 + 데이터셋 + 모델 내보내기
    (JSON 형식으로 반환)
    """
    try:
        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()

        # 프로젝트 가져오기
        cursor.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        project_row = cursor.fetchone()
        if not project_row:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        project = {
            "id": project_row[0],
            "name": project_row[1],
            "description": project_row[2],
            "status": project_row[3],
            "created_at": project_row[4],
            "updated_at": project_row[5],
        }

        # 데이터셋 가져오기
        cursor.execute("SELECT * FROM project_datasets WHERE project_id=?", (project_id,))
        dataset_rows = cursor.fetchall()
        datasets = [dict(zip([col[0] for col in cursor.description], row)) for row in dataset_rows]

        # 모델 가져오기
        cursor.execute("SELECT * FROM project_models WHERE project_id=?", (project_id,))
        model_rows = cursor.fetchall()
        models = [dict(zip([col[0] for col in cursor.description], row)) for row in model_rows]

        conn.close()

        return {
            "project": project,
            "datasets": datasets,
            "models": models
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/projects/import")
async def import_project(file: UploadFile = File(...)):
    """
    프로젝트 + 데이터셋 + 모델 가져오기 (ZIP 파일 업로드)
    """
    import uuid, os, zipfile, shutil, json
    try:
        # 1. 업로드된 zip 저장
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        zip_path = os.path.join(upload_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 압축 해제
        extract_dir = os.path.join(upload_dir, str(uuid.uuid4()))
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 3. JSON 로드
        with open(os.path.join(extract_dir, "project.json"), "r", encoding="utf-8") as f:
            import_data = json.load(f)

        project = import_data.get("project")
        datasets = import_data.get("datasets", [])
        models = import_data.get("models", [])

        if not project:
            raise HTTPException(status_code=400, detail="잘못된 프로젝트 데이터")

        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()

        # 4. 새 프로젝트 ID 생성
        new_project_id = str(uuid.uuid4())
        cursor.execute("""
            INSERT INTO projects (id, name, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (
            new_project_id,
            project.get("name") + " (가져오기)",
            project.get("description"),
            project.get("status", "active")
        ))

        dataset_id_map = {}
        # 5. 데이터셋 복사
        for ds in datasets:
            new_dataset_id = str(uuid.uuid4())
            dataset_id_map[ds["id"]] = new_dataset_id

            src_path = os.path.join(extract_dir, "datasets", ds["id"])
            dst_path = os.path.join("processed", new_project_id, new_dataset_id)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            if os.path.exists(src_path):
                shutil.copytree(src_path, dst_path)

            cursor.execute("""
                INSERT INTO project_datasets (id, project_id, name, path, image_count, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                new_dataset_id,
                new_project_id,
                ds.get("name"),
                dst_path,
                ds.get("image_count", 0)
            ))

        # 6. 모델 복사
        for m in models:
            new_model_id = str(uuid.uuid4())
            new_dataset_id = dataset_id_map.get(m["dataset_id"], None)

            src_model_path = os.path.join(extract_dir, "models", m["id"])
            dst_model_path = os.path.join("saved_models", new_model_id)
            if os.path.exists(src_model_path):
                shutil.copytree(src_model_path, dst_model_path)

            cursor.execute("""
                INSERT INTO project_models (id, project_id, dataset_id, name, backbone, performance_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                new_model_id,
                new_project_id,
                new_dataset_id,
                m.get("name"),
                m.get("backbone"),
                m.get("performance_data")
            ))

        conn.commit()
        conn.close()

        return {"success": True, "new_project_id": new_project_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/projects/{project_id}/duplicate")
async def duplicate_project(project_id: str):
    """
    프로젝트 + 데이터셋 + 모델 완전 복제
    """
    try:
        import uuid, shutil, os

        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()

        # 1. 원본 프로젝트 조회
        cursor.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        project_row = cursor.fetchone()
        if not project_row:
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        new_project_id = str(uuid.uuid4())
        new_project_name = project_row[1] + " (복사본)"

        # 2. 프로젝트 복사
        cursor.execute("""
            INSERT INTO projects (id, name, description, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (new_project_id, new_project_name, project_row[2], project_row[3]))

        # 3. 데이터셋 복사
        cursor.execute("SELECT * FROM project_datasets WHERE project_id=?", (project_id,))
        dataset_rows = cursor.fetchall()
        dataset_id_map = {}

        for ds in dataset_rows:
            new_dataset_id = str(uuid.uuid4())
            dataset_id_map[ds[0]] = new_dataset_id

            new_path = ds[3].replace(project_id, new_project_id)
            cursor.execute("""
                INSERT INTO project_datasets (id, project_id, name, path, image_count, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (new_dataset_id, new_project_id, ds[2], new_path, ds[4]))

            if os.path.exists(ds[3]):
                shutil.copytree(ds[3], new_path)

        # 4. 모델 복사
        cursor.execute("SELECT * FROM project_models WHERE project_id=?", (project_id,))
        model_rows = cursor.fetchall()

        for m in model_rows:
            new_model_id = str(uuid.uuid4())
            new_dataset_id = dataset_id_map.get(m[2], m[2])

            cursor.execute("""
                INSERT INTO project_models (id, project_id, dataset_id, name, backbone, performance_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (new_model_id, new_project_id, new_dataset_id, m[3], m[4], m[5]))

            src_model_path = os.path.join("saved_models", m[0])
            dst_model_path = os.path.join("saved_models", new_model_id)
            if os.path.exists(src_model_path):
                shutil.copytree(src_model_path, dst_model_path)

        conn.commit()
        conn.close()

        return {"success": True, "new_project_id": new_project_id, "new_name": new_project_name}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects/{project_id}")
async def get_project_detail(project_id: str):
    """프로젝트 상세 정보 조회"""
    try:
        conn = sqlite3.connect('mlops_vision.db')
        cursor = conn.cursor()
        
        # 프로젝트 기본 정보
        cursor.execute('SELECT * FROM projects WHERE id = ?', (project_id,))
        project_row = cursor.fetchone()
        
        if not project_row:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")
        
        # 데이터셋 목록
        cursor.execute('SELECT * FROM project_datasets WHERE project_id = ?', (project_id,))
        dataset_rows = cursor.fetchall()
        
        datasets = []
        for row in dataset_rows:
            datasets.append({
                "id": row[0],
                "name": row[1],
                "description": row[2],
                "status": row[3],
                "created_at": row[4],
                "updated_at": row[5]
            })
        
        # 모델 목록
        cursor.execute('SELECT * FROM project_models WHERE project_id = ?', (project_id,))
        model_rows = cursor.fetchall()
        
        models = []
        for row in model_rows:
            models.append({
                "id": row[0],
                "dataset_id": row[2],
                "name": row[3],
                "backbone": row[4],
                "performance_data": json.loads(row[5]) if row[5] else {},
                "created_at": row[6]
            })
        
        project = {
            "id": project_row[0],
            "name": project_row[1],
            "description": project_row[2],
            "created_at": project_row[3],
            "updated_at": project_row[4],
            "datasets": datasets,
            "models": models
        }
        
        conn.close()
        return {"success": True, "project": project}
        
    except Exception as e:
        logger.error(f"프로젝트 상세 조회 실패: {e}")
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
    
@app.post("/datasets/{project_id}/{dataset_id}/upload")
async def upload_dataset_images(
    project_id: str,
    dataset_id: str,
    files: List[UploadFile] = File(...)
):
    try:
        # 경로를 BASE_DATA_PATH 기준으로 통일
        dataset_path = BASE_DATA_PATH / project_id / dataset_id
        dataset_path.mkdir(parents=True, exist_ok=True)

        saved_filenames = []
        for file in files:
            file_path = dataset_path / file.filename
            with open(file_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            saved_filenames.append(file.filename)

        return {"success": True, "saved_files": saved_filenames}

    except Exception as e:
        logger.error(f"이미지 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")

# 이미지 조회 API는 동일하게 추가
@app.get("/api/image/{project_id}/{dataset_id}/{filename}")
async def get_image(project_id: str, dataset_id: str, filename: str):
    try:
        file_path = BASE_DATA_PATH / project_id / dataset_id / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

        # 이미지 파일을 직접 StreamingResponse로 전송
        return FileResponse(str(file_path))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"이미지 조회 실패: {str(e)}")
    
@app.get("/datasets/{project_id}/{dataset_id}/images")
async def list_uploaded_images(project_id: str, dataset_id: str):
    try:
        dataset_path = BASE_DATA_PATH / project_id / dataset_id
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다")
        
        images = [f.name for f in dataset_path.iterdir() if f.is_file()]
        return {"images": images}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 목록 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")

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
    """데이터셋 타입별 모델 학습을 시작합니다"""
    try:
        # 데이터셋 존재 확인
        dataset_info = dataset_manager.get_dataset_info(config.dataset_name)
        if dataset_info is None:
            raise HTTPException(status_code=404, detail=f"데이터셋을 찾을 수 없습니다: {config.dataset_name}")
        
        # 이미 학습이 진행 중인지 확인
        if trainer.is_training_active():
            raise HTTPException(status_code=400, detail="이미 학습이 진행 중입니다")
        
        # 데이터셋 타입에 따른 학습 분기
        dataset_type = dataset_info.get("type", "anomaly")  # 기본값 anomaly
        
        if dataset_type == "anomal":  # Anomaly Detection
            # PatchCore 설정으로 학습
            patchcore_config = PatchCoreConfig(
                backbone=config.backbone,
                layers=config.layers,
                anomaly_scorer_num_nn=config.anomaly_scorer_num_nn,
                sampling_ratio=config.sampling_ratio,
                dataset_name=config.dataset_name,
                resize_size=config.resize_size,
                image_size=config.image_size,
                gpu_id=config.gpu_id
            )
            
            logger.info(f"Anomaly Detection 학습 시작: {config.dataset_name}")
            result = trainer.start_training(
                config=patchcore_config,
                dataset_path=dataset_info["path"],
                blocking=False
            )
            
        elif dataset_type == "classify":  # Classification
            logger.info(f"Classification 학습 시작: {config.dataset_name}")
            # Classification 학습 로직 (추후 구현)
            return {"success": False, "error": "Classification 모델은 아직 지원하지 않습니다"}
            
        elif dataset_type == "obd":  # Object Detection
            logger.info(f"Object Detection 학습 시작: {config.dataset_name}")
            # Object Detection 학습 로직 (추후 구현)  
            return {"success": False, "error": "Object Detection 모델은 아직 지원하지 않습니다"}
            
        elif dataset_type == "seg":  # Segmentation
            logger.info(f"Segmentation 학습 시작: {config.dataset_name}")
            # Segmentation 학습 로직 (추후 구현)
            return {"success": False, "error": "Segmentation 모델은 아직 지원하지 않습니다"}
            
        else:
            raise HTTPException(status_code=400, detail=f"지원하지 않는 데이터셋 타입: {dataset_type}")
        
        if result["success"]:
            return {
                "success": True,
                "message": f"{dataset_type} 모델 학습이 시작되었습니다",
                "training_info": result["training_info"],
                "dataset_type": dataset_type
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
    
# 학습 진행률 API 추가
@app.get("/training/progress")
async def get_training_progress():
    """실시간 학습 진행률 조회"""
    try:
        status = trainer.get_training_status()
        progress = 0
        
        if status and status.get("log_group"):
            progress = parse_training_log(status["log_group"])
            
        return {
            "success": True,
            "status": status,
            "progress": progress,
            "is_active": trainer.is_training_active()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 학습 로그 API 추가
@app.get("/training/logs/{log_group}")
async def get_training_logs(log_group: str, lines: int = 50):
    """학습 로그 조회"""
    try:
        log_file = trainer.results_dir / f"{log_group}_training.log"
        if not log_file.exists():
            raise HTTPException(status_code=404, detail="로그 파일을 찾을 수 없습니다")
        
        with open(log_file, 'r') as f:
            all_lines = f.readlines()
            recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
        
        return {
            "success": True,
            "logs": recent_lines,
            "total_lines": len(all_lines)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def parse_training_log(log_group: str) -> int:
    """로그 파일에서 진행률 추출"""
    try:
        log_file = trainer.results_dir / f"{log_group}_training.log"
        if not log_file.exists():
            return 0
            
        with open(log_file, 'r') as f:
            content = f.read()
            
        # PatchCore 로그에서 진행률 파싱
        import re
        
        # "Epoch 3/10" 패턴 찾기
        epoch_pattern = r'Epoch (\d+)/(\d+)'
        matches = re.findall(epoch_pattern, content)
        
        if matches:
            current, total = map(int, matches[-1])
            return min(int((current / total) * 100), 100)
            
        # "Training progress: 45%" 같은 패턴도 찾기
        progress_pattern = r'progress:\s*(\d+)%'
        progress_matches = re.findall(progress_pattern, content, re.IGNORECASE)
        
        if progress_matches:
            return min(int(progress_matches[-1]), 100)
        
        return 0
    except Exception:
        return 0

# =============================================================================
# 모델 관리 API
# =============================================================================

@app.get("/models")
async def list_models():
    """학습된 모델 목록을 반환합니다 (기존 학습 + 업로드된 모델 포함)"""
    try:
        # 기존 학습된 모델
        trained_models = model_manager.discover_trained_models()
        
        # 업로드된 모델 추가
        uploaded_models = []
        saved_models_dir = Path("saved_models")
        
        if saved_models_dir.exists():
            for model_dir in saved_models_dir.iterdir():
                if model_dir.is_dir():
                    model_info_file = model_dir / "model_info.json"
                    if model_info_file.exists():
                        try:
                            with open(model_info_file, 'r', encoding='utf-8') as f:
                                model_info = json.load(f)
                            uploaded_models.append(model_info)
                        except Exception as e:
                            logger.error(f"업로드 모델 정보 로드 실패: {e}")
        
        # 모든 모델 합치기
        all_models = trained_models + uploaded_models
        
        # 모델 비교 데이터 생성
        comparison_data = []
        if all_models:
            # 기존 comparison 로직 + 업로드된 모델도 포함
            for model in uploaded_models:
                comparison_data.append({
                    "model_name": model["model_name"],
                    "backbone": model["config"].get("backbone", "unknown"),
                    "accuracy": model["performance"].get("accuracy", 0),
                    "f1_score": model["performance"].get("f1_score", 0),
                    "precision": model["performance"].get("precision", 0),
                    "recall": model["performance"].get("recall", 0),
                    "source": "uploaded"
                })
        
        return {
            "models": all_models,
            "total_count": len(all_models),
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
    
@app.post("/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    project_id: Optional[str] = Form(None)
):
    """모델 파일 업로드 및 분석"""
    try:
        # 파일 확장자 확인
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['pkl', 'pth']:
            raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다")
        
        # 모델 ID 생성
        model_id = f"model_{int(datetime.now().timestamp())}"
        model_name = file.filename.rsplit('.', 1)[0]
        
        # 저장 디렉토리 생성
        saved_models_dir = Path("saved_models")
        saved_models_dir.mkdir(exist_ok=True)
        
        model_dir = saved_models_dir / f"{model_name}_{model_id}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 파일 저장
        model_path = model_dir / file.filename
        with open(model_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # pkl 파일 분석
        model_info = analyze_uploaded_model(str(model_path), model_id, model_name, project_id)
        
        # model_info.json 저장
        with open(model_dir / "model_info.json", 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"모델 업로드 완료: {model_path}")
        return {"success": True, "model": model_info}
        
    except Exception as e:
        logger.error(f"모델 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")

@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """업로드된 모델 삭제"""
    try:
        saved_models_dir = Path("saved_models")
        
        if not saved_models_dir.exists():
            raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")
        
        for model_dir in saved_models_dir.iterdir():
            if model_dir.is_dir():
                model_info_file = model_dir / "model_info.json"
                
                if model_info_file.exists():
                    try:
                        with open(model_info_file, 'r', encoding='utf-8') as f:
                            model_info = json.load(f)
                        
                        if model_info.get('id') == model_id:
                            shutil.rmtree(model_dir)
                            logger.info(f"모델 삭제 완료: {model_dir}")
                            return {"success": True, "message": "모델이 삭제되었습니다"}
                    except Exception:
                        continue
        
        raise HTTPException(status_code=404, detail="모델을 찾을 수 없습니다")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"모델 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"삭제 실패: {str(e)}")

def analyze_uploaded_model(file_path: str, model_id: str, model_name: str, project_id: str = None):
    """업로드된 pkl 파일에서 모델 정보 추출"""
    analysis_status = "success"
    error_message = None
    
    try:
        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)

        logger.info(f"=== 모델 파일 분석 시작: {model_name} ===")
        logger.info(f"데이터 타입: {type(model_data)}")
        
        if isinstance(model_data, dict):
            logger.info(f"딕셔너리 키들: {list(model_data.keys())}")
            for key, value in model_data.items():
                logger.info(f"  {key}: {type(value)}")
        else:
            logger.info(f"객체 속성들: {dir(model_data)}")
        
        logger.info("=== 분석 완료 ===")
        
        
        # 기본값 (분석 실패 시 사용)
        performance = {
            "accuracy": None,
            "f1_score": None,
            "precision": None,
            "recall": None
        }
        
        config = {
            "backbone": None,
            "dataset_name": None,
            "epochs": None,
            "learning_method": None
        }
        
        # 실제 pkl 데이터 구조 분석
        logger.info(f"분석 중인 모델 데이터 타입: {type(model_data)}")
        
        if isinstance(model_data, dict):
            logger.info(f"모델 데이터 키들: {list(model_data.keys())}")
            
            # 성능 데이터 추출 시도
            if 'performance' in model_data:
                perf_data = model_data['performance']
                if isinstance(perf_data, dict):
                    performance.update(perf_data)
                    logger.info("성능 데이터 추출 완료")
            
            elif 'metrics' in model_data:
                metrics_data = model_data['metrics']
                if isinstance(metrics_data, dict):
                    performance.update(metrics_data)
                    logger.info("메트릭 데이터 추출 완료")
            
            elif 'test_results' in model_data:
                test_data = model_data['test_results']
                if isinstance(test_data, dict):
                    # test_results에서 성능 지표 매핑
                    if 'image_auroc' in test_data:
                        performance['accuracy'] = test_data['image_auroc']
                    if 'pixel_auroc' in test_data:
                        performance['f1_score'] = test_data['pixel_auroc']
                    logger.info("테스트 결과에서 성능 데이터 추출 완료")
            
            # 설정 데이터 추출 시도
            if 'config' in model_data:
                config_data = model_data['config']
                if isinstance(config_data, dict):
                    config.update(config_data)
                    logger.info("설정 데이터 추출 완료")
            
            # 백본 정보 추출
            if 'backbone' in model_data:
                config['backbone'] = model_data['backbone']
            elif 'model' in model_data and hasattr(model_data['model'], '__class__'):
                config['backbone'] = model_data['model'].__class__.__name__
            
            # PatchCore 특화 분석
            if 'anomaly_maps' in model_data or 'patch_lib' in model_data:
                config['learning_method'] = 'PatchCore'
                logger.info("PatchCore 모델로 식별됨")
        
        else:
            # dict가 아닌 경우 (모델 객체 등)
            logger.info(f"모델이 dict가 아님. 객체 분석 시도: {model_data.__class__.__name__ if hasattr(model_data, '__class__') else 'unknown'}")
            config['backbone'] = getattr(model_data, '__class__', {}).get('__name__', 'unknown')
            analysis_status = "partial"
            error_message = "모델 구조가 예상과 다름"
        
        # 분석된 데이터가 없으면 실패로 처리
        all_performance_none = all(v is None for v in performance.values())
        all_config_none = all(v is None for v in config.values())
        
        if all_performance_none and all_config_none:
            analysis_status = "failed"
            error_message = "모델에서 성능 데이터나 설정 정보를 찾을 수 없음"
        
        return {
            "id": model_id,
            "model_name": model_name,
            "project_id": project_id,
            "config": config,
            "performance": performance,
            "training_info": {
                "training_time": "업로드됨",
                "image_count": 0,
                "created_date": datetime.now().isoformat(),
                "file_size": Path(file_path).stat().st_size
            },
            "model_path": file_path,
            "status": "uploaded",
            "analysis_status": analysis_status,
            "analysis_error": error_message
        }
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"모델 분석 실패: {e}")
        return {
            "id": model_id,
            "model_name": model_name,
            "project_id": project_id,
            "config": {
                "backbone": "분석 실패",
                "dataset_name": "업로드됨",
                "epochs": 0,
                "learning_method": "분석 실패"
            },
            "performance": {
                "accuracy": 0.0,
                "f1_score": 0.0,
                "precision": 0.0,
                "recall": 0.0
            },
            "training_info": {
                "training_time": "업로드됨",
                "image_count": 0,
                "created_date": datetime.now().isoformat()
            },
            "status": "analysis_failed",
            "analysis_status": "failed",
            "analysis_error": error_message
        }

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