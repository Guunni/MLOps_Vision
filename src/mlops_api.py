"""
MLOps Vision 통합 웹 API
모든 PatchCore 관련 기능을 하나의 웹 인터페이스로 통합하는 메인 API
"""

# FastAPI 관련
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 타입 관련
from typing import Dict, Any, List, Optional

# 표준 라이브러리
import os
import json
import logging
import sqlite3
import uuid
import shutil
import pickle
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path
import threading

from datetime import datetime, timezone, timedelta
KST = timezone(timedelta(hours=9))

# 우리가 만든 모듈
from patchcore_manager import PatchCoreManager
from dataset_manager import DatasetManager
from training_manager import PatchCoreTrainer, PatchCoreConfig
from model_manager import PatchCoreModelManager

# 템플릿 (정적 HTML 렌더링)
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

DB_PATH = "mlops_vision.db"

def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # ← 컬럼명으로 접근
    return conn

def now_kst_str():
    from datetime import datetime, timezone, timedelta
    KST = timezone(timedelta(hours=9))
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")  # ← DB와 동일 포맷

def to_camel_case(row: dict) -> dict:
    mapping = {
        "project_id": "projectId",
        "dataset_id": "datasetId",
        "created_at": "createdAt",
        "updated_at": "updatedAt",
        "image_count": "imageCount",
        "total_size": "totalSize",
    }
    return {mapping.get(k, k): v for k, v in row.items()}

def recalc_dataset_stats(conn, project_id: str, dataset_id: str):
    """폴더 기준으로 image_count/total_size 재계산 후 DB 반영"""
    ds_dir = BASE_DATA_PATH / project_id / dataset_id
    ds_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    if ds_dir.exists():
        for entry in ds_dir.iterdir():
            if entry.is_file():
                st = entry.stat()
                count += 1
                total += st.st_size
    cur = conn.cursor()
    cur.execute("""
        UPDATE project_datasets
        SET image_count=?, total_size=?, updated_at=CURRENT_TIMESTAMP
        WHERE id=? AND project_id=?
    """, (count, total, dataset_id, project_id))
    conn.commit()
    return count, total

# 이미지 목록 조회 (통일된 v2)
from typing import List  # 파일 상단에 이미 있다면 중복 import 무시

@app.get("/api/projects/{project_id}/datasets/{dataset_id}/images")
async def list_images_v2(project_id: str, dataset_id: str):
    try:
        conn = get_conn()
        cur = conn.cursor()

        # 데이터셋 존재 확인
        cur.execute(
            "SELECT 1 FROM project_datasets WHERE id=? AND project_id=?",
            (dataset_id, project_id),
        )
        row = cur.fetchone()

        ds_dir = BASE_DATA_PATH / project_id / dataset_id

        # 🔥 디버깅 로그 추가
        print("=== [DEBUG] list_images_v2 ===")
        print(" project_id:", project_id)
        print(" dataset_id:", dataset_id)
        print(" DB row exists?:", bool(row))
        print(" ds_dir path:", ds_dir)
        print(" ds_dir.exists():", ds_dir.exists())
        print("================================")

        if not row:
            if ds_dir.exists():
                cur.execute(
                    "INSERT INTO project_datasets (id, project_id, name, path, type) VALUES (?, ?, ?, ?, ?)",
                    (dataset_id, project_id, dataset_id, str(ds_dir), "classify"),
                )
                conn.commit()
            else:
                conn.close()
                raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다")

        if not ds_dir.exists():
            conn.close()
            raise HTTPException(status_code=404, detail="데이터셋 디렉토리를 찾을 수 없습니다")

        images = []
        for file_path in ds_dir.glob("*.*"):
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                stat = file_path.stat()
                images.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

        conn.close()
        return {"images": images}

    except HTTPException:
        raise
    except Exception as e:
        print("list_images_v2 error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# 이미지 업로드
@app.post("/api/projects/{project_id}/datasets/{dataset_id}/images")
async def upload_images_v2(project_id: str, dataset_id: str, files: List[UploadFile] = File(...)):
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()

        ds_dir = BASE_DATA_PATH / project_id / dataset_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        # 데이터셋 존재 확인 (없으면 폴더 기준으로 자동 등록) - path 반드시 포함
        cur.execute("SELECT 1 FROM project_datasets WHERE id=? AND project_id=?", (dataset_id, project_id))
        if not cur.fetchone():
            cur.execute("""
                INSERT INTO project_datasets
                    (id, project_id, name, path, image_count, created_at, type, description, total_size, updated_at)
                VALUES
                    (?,  ?,          ?,    ?,    0,           CURRENT_TIMESTAMP, ?,    ?,           0,          CURRENT_TIMESTAMP)
            """, (dataset_id, project_id, dataset_id, str(ds_dir), "classify", ""))
            conn.commit()

        # 파일 저장
        for file in files:
            dst = ds_dir / file.filename
            with open(dst, "wb") as f:
                shutil.copyfileobj(file.file, f)

        # 통계 갱신
        count, total = recalc_dataset_stats(conn, project_id, dataset_id)

        # 업로드 후 최신 이미지 목록 생성 (프론트가 기대하는 구조)
        images = []
        for file_path in ds_dir.glob("*.*"):
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                st = file_path.stat()
                images.append({
                    "name": file_path.name,
                    "size": st.st_size,
                    "created_at": datetime.fromtimestamp(st.st_ctime).isoformat(),
                    "updated_at": datetime.fromtimestamp(st.st_mtime).isoformat(),
                })

        conn.commit()
        return {"success": True, "images": images, "imageCount": count, "totalSize": total}

    except HTTPException:
        if conn: conn.close()
        raise
    except Exception as e:
        if conn: conn.close()
        logger.error(f"이미지 업로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"업로드 실패: {str(e)}")
    finally:
        if conn:
            conn.close()

# Static 파일과 템플릿 설정
app.mount("/static", StaticFiles(directory="../static"), name="static")
templates = Jinja2Templates(directory="../templates")

# 전역 매니저 인스턴스들
patchcore_manager = PatchCoreManager()
dataset_manager = DatasetManager()
trainer = PatchCoreTrainer()
model_manager = PatchCoreModelManager()
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

     # 누락 컬럼 추가
    def add_column_if_missing(table, column_def):
        col_name = column_def.split()[0]
        cursor.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cursor.fetchall()]
        if col_name not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")

    # datasets 확장 컬럼
    add_column_if_missing("project_datasets", "type TEXT")
    add_column_if_missing("project_datasets", "description TEXT")
    add_column_if_missing("project_datasets", "total_size INTEGER DEFAULT 0")

    # ❗DEFAULT CURRENT_TIMESTAMP는 허용 안 됨 → 기본값 없이 추가
    add_column_if_missing("project_datasets", "updated_at TIMESTAMP")

    # ✅ 기존 행 보정(널이면 현재 시각으로)
    cursor.execute("""
        UPDATE project_datasets
        SET updated_at = COALESCE(updated_at, CURRENT_TIMESTAMP)
    """)

    # ✅ 트리거로 updated_at 자동 갱신
    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_project_datasets_set_updated_at_on_update
        AFTER UPDATE ON project_datasets
        FOR EACH ROW
        BEGIN
            UPDATE project_datasets
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = NEW.id;
        END;
    """)

    cursor.execute("""
        CREATE TRIGGER IF NOT EXISTS trg_project_datasets_set_updated_at_on_insert
        AFTER INSERT ON project_datasets
        FOR EACH ROW
        WHEN NEW.updated_at IS NULL
        BEGIN
            UPDATE project_datasets
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = NEW.id;
        END;
    """)
    
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
    test_ratio: float = 0.1

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

# 프로젝트 목록 조회
@app.get("/api/projects")
async def list_projects():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
        cur = conn.cursor()

        # 프로젝트 기본 목록
        cur.execute("SELECT * FROM projects ORDER BY created_at ASC")
        p_rows = cur.fetchall()

        result = []
        for p in p_rows:
            project = to_camel_case(p)

            # 날짜 보정 (– 방지)
            created_at_raw = p.get("created_at")
            updated_at_raw = p.get("updated_at")
            now = now_kst_str()
            project["createdAt"] = project.get("createdAt") or created_at_raw or now
            project["updatedAt"] = project.get("updatedAt") or updated_at_raw or project["createdAt"]

            # 데이터셋 목록 (* 로 안전 조회)
            cur.execute("SELECT * FROM project_datasets WHERE project_id=?", (p["id"],))
            ds_rows = cur.fetchall()

            dataset_list = []
            for d in ds_rows:
                item = {
                    "id": d.get("id"),
                    "name": d.get("name"),
                    "type": (d.get("type") or "classify"),
                    "createdAt": d.get("created_at"),
                    "imageCount": d.get("image_count", 0),
                    "totalSize": d.get("total_size", 0),
                    "description": d.get("description") or ""
                }
                # 과거 템플릿 호환용 별칭
                item["date"] = item["createdAt"]
                dataset_list.append(item)

            project["datasetList"] = dataset_list
            project["datasetCount"] = len(dataset_list)

            # 모델 개수 (테이블 없거나 오류여도 0)
            try:
                cur.execute("SELECT COUNT(*) AS cnt FROM project_models WHERE project_id=?", (p["id"],))
                mc = cur.fetchone()
                project["modelCount"] = mc.get("cnt", 0) if mc else 0
            except Exception:
                project["modelCount"] = 0

            result.append(project)

        return result
    except Exception as e:
        logger.error(f"/api/projects 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
    
# 프로젝트 생성
@app.post("/api/projects")
async def create_project(project_data: dict):
    try:
        project_id = str(uuid.uuid4())
        name = project_data.get("name")
        description = project_data.get("description", "")
        status = project_data.get("status", "active")
        if not name:
            raise HTTPException(status_code=400, detail="프로젝트 이름이 필요합니다")

        now = now_kst_str()

        conn = get_conn()
        cur = conn.cursor()
        # 컬럼을 명시하고, 우리가 넣을 값과 순서를 1:1로 맞춥니다.
        cur.execute("""
            INSERT INTO projects (id, name, description, created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (project_id, name, description, now, now, status))

        conn.commit()
        cur.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        row = cur.fetchone()
        conn.close()

        project = {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        return {"success": True, "project": project}

    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="이미 존재하는 프로젝트 이름입니다")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 프로젝트 수정
@app.put("/api/projects/{project_id}")
async def update_project(project_id: str, project_data: dict):
    try:
        name = project_data.get("name")
        description = project_data.get("description", "")
        status = project_data.get("status", "active")
        if not name:
            raise HTTPException(status_code=400, detail="프로젝트 이름이 필요합니다")

        now = now_kst_str()
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("""
            UPDATE projects
               SET name=?, description=?, status=?, updated_at=?
             WHERE id=?
        """, (name, description, status, now, project_id))

        if cur.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        conn.commit()
        cur.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        row = cur.fetchone()
        conn.close()

        project = {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        return {"success": True, "project": project}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 프로젝트 삭제
@app.delete("/api/projects/{project_id}")
async def delete_project(project_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute("DELETE FROM projects WHERE id=?", (project_id,))
        if cursor.rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        conn.commit()
        conn.close()
        return {"success": True, "message": "프로젝트가 삭제되었습니다"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# 프로젝트 내보내기
@app.get("/api/projects/{project_id}/export")
async def export_project(project_id: str, mode: str = "meta"):
    """
    프로젝트 내보내기
    - mode=meta : project.json + datasets.json + models/zip
    - mode=full : project.json + datasets/zip + models/zip
    """
    try:
        conn = get_conn()
        cur = conn.cursor()

        # 1. 프로젝트 조회
        cur.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        p_row = cur.fetchone()
        if not p_row:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        project = {
            "id": p_row["id"],
            "name": p_row["name"],
            "description": p_row["description"],
            "status": p_row["status"],
            "created_at": p_row["created_at"],
            "updated_at": p_row["updated_at"],
            "models": []
        }

        # 2. 데이터셋 조회
        cur.execute("SELECT * FROM project_datasets WHERE project_id=?", (project_id,))
        datasets = [{
            "id": ds["id"],
            "project_id": ds["project_id"],
            "name": ds["name"],
            "path": ds["path"],
            "image_count": ds["image_count"],
            "created_at": ds["created_at"],
        } for ds in cur.fetchall()]

        # 3. 모델 조회
        cur.execute("SELECT * FROM project_models WHERE project_id=?", (project_id,))
        for m in cur.fetchall():
            project["models"].append({
                "id": m["id"],
                "project_id": m["project_id"],
                "dataset_id": m["dataset_id"],
                "name": m["name"],
                "backbone": m["backbone"],
                "performance_data": m["performance_data"],
                "created_at": m["created_at"]
            })

        conn.close()

        # 4. 임시 zip 생성
        tmpdir = tempfile.mkdtemp()
        zip_path = os.path.join(tmpdir, f"{project['name']}_export_{mode}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            # project.json
            zipf.writestr("project.json", json.dumps(project, indent=2, ensure_ascii=False))

            if mode == "meta":
                # datasets.json
                zipf.writestr("datasets.json", json.dumps(datasets, indent=2, ensure_ascii=False))

            elif mode == "full":
                # dataset별 zip 생성
                for ds in datasets:
                    src_path = ds["path"]
                    if os.path.exists(src_path):
                        ds_zip_name = f"datasets/{ds['id']}.zip"
                        with tempfile.TemporaryDirectory() as ds_tmp:
                            ds_zip = os.path.join(ds_tmp, f"{ds['id']}.zip")
                            with zipfile.ZipFile(ds_zip, "w") as dsf:
                                for root, _, files in os.walk(src_path):
                                    for f in files:
                                        abs_path = os.path.join(root, f)
                                        rel_path = os.path.relpath(abs_path, src_path)
                                        dsf.write(abs_path, rel_path)
                            zipf.write(ds_zip, ds_zip_name)

            # models
            for m in project["models"]:
                src_model_path = os.path.join("saved_models", m["id"])
                if os.path.exists(src_model_path):
                    model_zip_name = f"models/{m['id']}.zip"
                    with tempfile.TemporaryDirectory() as m_tmp:
                        m_zip = os.path.join(m_tmp, f"{m['id']}.zip")
                        with zipfile.ZipFile(m_zip, "w") as mf:
                            for root, _, files in os.walk(src_model_path):
                                for f in files:
                                    abs_path = os.path.join(root, f)
                                    rel_path = os.path.relpath(abs_path, src_model_path)
                                    mf.write(abs_path, rel_path)
                        zipf.write(m_zip, model_zip_name)

        return FileResponse(zip_path, filename=os.path.basename(zip_path))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/projects/import")
async def import_project(file: UploadFile = File(...)):
    """
    프로젝트 가져오기 (Meta Export 또는 Full Export zip 업로드)
    """
    import uuid, os, zipfile, shutil, json, tempfile

    try:
        # 1. 업로드 zip 저장
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        zip_path = os.path.join(upload_dir, file.filename)
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 2. 압축 해제
        extract_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        # 3. project.json 로드
        project_json_path = os.path.join(extract_dir, "project.json")
        if not os.path.exists(project_json_path):
            raise HTTPException(status_code=400, detail="project.json이 없습니다")

        with open(project_json_path, "r", encoding="utf-8") as f:
            project_data = json.load(f)

        # 새 프로젝트 ID
        new_project_id = str(uuid.uuid4())
        now = now_kst_str()

        conn = get_conn()
        cur = conn.cursor()

        # 4. 프로젝트 삽입 (created_at은 원본 유지, updated_at은 현재 시각)
        cur.execute("""
            INSERT INTO projects (id, name, description, created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            new_project_id,
            project_data.get("name") + " (가져오기)",
            project_data.get("description", ""),
            project_data.get("created_at"),  # 원본
            now,                             # 현재 시각
            project_data.get("status", "active")
        ))

        dataset_id_map = {}

        # 5. datasets.json 처리 (meta export)
        datasets_json_path = os.path.join(extract_dir, "datasets.json")
        if os.path.exists(datasets_json_path):
            with open(datasets_json_path, "r", encoding="utf-8") as f:
                datasets_data = json.load(f)

            for ds in datasets_data:
                new_dataset_id = str(uuid.uuid4())
                dataset_id_map[ds["id"]] = new_dataset_id

                ds_zip_path = os.path.join(extract_dir, "datasets", f"{ds['id']}.zip")
                dst_path = os.path.join("processed", new_project_id, new_dataset_id)
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                if os.path.exists(ds_zip_path):
                    with zipfile.ZipFile(ds_zip_path, "r") as ds_zip:
                        ds_zip.extractall(dst_path)

                cur.execute("""
                    INSERT INTO project_datasets (id, project_id, name, path, image_count, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    new_dataset_id,
                    new_project_id,
                    ds.get("name"),
                    dst_path if os.path.exists(ds_zip_path) else ds.get("path"),
                    ds.get("image_count", 0),
                    ds.get("created_at")  # 원본 생성일 유지
                ))

        # 6. 모델 복원
        if "models" in project_data:
            for m in project_data["models"]:
                new_model_id = str(uuid.uuid4())
                new_dataset_id = dataset_id_map.get(m["dataset_id"], None)

                cur.execute("""
                    INSERT INTO project_models (id, project_id, dataset_id, name, backbone, performance_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    new_model_id,
                    new_project_id,
                    new_dataset_id,
                    m.get("name"),
                    m.get("backbone"),
                    m.get("performance_data"),
                    m.get("created_at")  # 원본 생성일 유지
                ))

                # 모델 파일 복원
                src_model_zip = os.path.join(extract_dir, "models", f"{m['id']}.zip")
                dst_model_path = os.path.join("saved_models", new_model_id)
                if os.path.exists(src_model_zip):
                    os.makedirs(dst_model_path, exist_ok=True)
                    with zipfile.ZipFile(src_model_zip, "r") as mzip:
                        mzip.extractall(dst_model_path)

        conn.commit()
        conn.close()

        return {"success": True, "new_project_id": new_project_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/projects/{project_id}/duplicate")
async def duplicate_project(project_id: str):
    """
    프로젝트 Deep Copy (프로젝트 + 데이터셋 + 이미지 + 모델 + 모델 파일)
    - 컬럼명 접근으로 인덱스 꼬임 방지
    - created_at/updated_at: KST "YYYY-MM-DD HH:MM:SS"
    """
    import os, shutil, uuid
    try:
        conn = get_conn()
        cur = conn.cursor()

        # 1) 원본 프로젝트
        cur.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        proj = cur.fetchone()
        if not proj:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        new_project_id = str(uuid.uuid4())
        new_project_name = f'{proj["name"]} (복사본)'
        status_value = proj["status"] or "active"
        now = now_kst_str()

        # 2) 새 프로젝트 insert (컬럼 순서 명시)
        cur.execute("""
            INSERT INTO projects (id, name, description, created_at, updated_at, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (new_project_id, new_project_name, proj["description"], now, now, status_value))

        # 3) 데이터셋 복제
        cur.execute("SELECT * FROM project_datasets WHERE project_id=?", (project_id,))
        ds_rows = cur.fetchall()
        ds_id_map = {}
        for ds in ds_rows:
            new_ds_id = str(uuid.uuid4())
            ds_id_map[ds["id"]] = new_ds_id

            src_path = ds["path"]
            dst_path = src_path.replace(project_id, new_project_id)

            cur.execute("""
                INSERT INTO project_datasets (id, project_id, name, path, image_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (new_ds_id, new_project_id, ds["name"], dst_path, ds["image_count"], now))

            if os.path.exists(src_path):
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                shutil.copytree(src_path, dst_path)

        # 4) 모델 복제
        cur.execute("SELECT * FROM project_models WHERE project_id=?", (project_id,))
        mdl_rows = cur.fetchall()
        for m in mdl_rows:
            new_model_id = str(uuid.uuid4())
            new_ds_id = ds_id_map.get(m["dataset_id"], m["dataset_id"])

            cur.execute("""
                INSERT INTO project_models (id, project_id, dataset_id, name, backbone, performance_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (new_model_id, new_project_id, new_ds_id, m["name"], m["backbone"], m["performance_data"], now))

            src_model_path = os.path.join("saved_models", m["id"])
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
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute("SELECT * FROM projects WHERE id=?", (project_id,))
        p = cur.fetchone()
        if not p:
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        cur.execute("SELECT * FROM project_datasets WHERE project_id=?", (project_id,))
        datasets = [to_camel_case(dict(ds)) for ds in cur.fetchall()]

        cur.execute("SELECT * FROM project_models WHERE project_id=?", (project_id,))
        models = [to_camel_case(dict(m)) for m in cur.fetchall()]

        project = to_camel_case(dict(p))
        project["datasets"] = datasets
        project["models"] = models

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
    
@app.get("/api/projects/{project_id}/datasets")
async def list_project_datasets(project_id: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
        cur = conn.cursor()

        # 프로젝트 존재 확인
        cur.execute("SELECT id FROM projects WHERE id=?", (project_id,))
        if not cur.fetchone():
            conn.close()
            raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다")

        # 데이터셋 조회(*로 안전 조회) → camelCase
        cur.execute("""
            SELECT * FROM project_datasets
            WHERE project_id=?
            ORDER BY created_at ASC
        """, (project_id,))
        rows = cur.fetchall()
        conn.close()

        # camelCase + 기본값 보정
        datasets = []
        for r in rows:
            r_c = to_camel_case(r)
            r_c.setdefault("type", "classify")
            r_c.setdefault("description", "")
            r_c.setdefault("totalSize", 0)
            return_map = {
                **r_c,
                "imageCount": r_c.get("imageCount", 0)
            }
            datasets.append(return_map)

        return datasets
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터셋 조회 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/projects/{project_id}/datasets")
async def create_dataset(project_id: str, dataset_data: dict):
    try:
        name = dataset_data.get("name", "").strip()
        dtype = dataset_data.get("type", "classify")
        description = dataset_data.get("description", "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="데이터셋 이름이 필요합니다")

        new_id = str(uuid.uuid4())
        now = now_kst_str()
        ds_path = BASE_DATA_PATH / project_id / new_id
        ds_path.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO project_datasets
            (id, project_id, name, path, image_count, created_at, type, description, total_size, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (new_id, project_id, name, str(ds_path), 0, now, dtype, description, 0, now))
        conn.commit()
        conn.close()

        return {
            "success": True,
            "dataset": {
                "id": new_id,
                "projectId": project_id,
                "name": name,
                "type": dtype,
                "description": description,
                "imageCount": 0,
                "totalSize": 0,
                "createdAt": now,
                "updatedAt": now
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터셋 생성 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/projects/{project_id}/datasets/{dataset_id}")
async def delete_dataset(project_id: str, dataset_id: str):
    """
    데이터셋 삭제:
    - 파일시스템: {BASE_DATA_PATH}/{project_id}/{dataset_id} 디렉토리 삭제
    - DB: project_datasets 행 삭제
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
        cur = conn.cursor()

        # 존재 확인 + 경로 조회
        cur.execute(
            "SELECT path FROM project_datasets WHERE id=? AND project_id=?",
            (dataset_id, project_id)
        )
        row = cur.fetchone()
        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다")

        # 폴더 삭제 (경로가 DB에 없으면 기본 규칙으로 추정)
        ds_path = Path(row.get("path") or (BASE_DATA_PATH / project_id / dataset_id))
        try:
            if ds_path.exists():
                shutil.rmtree(ds_path, ignore_errors=True)
        except Exception as fe:
            # 폴더 삭제 실패해도 DB 정리 가능하도록 로그만 남김
            logger.error(f"데이터셋 폴더 삭제 실패: {fe}")

        # DB 삭제
        cur.execute(
            "DELETE FROM project_datasets WHERE id=? AND project_id=?",
            (dataset_id, project_id)
        )
        conn.commit()
        conn.close()

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"데이터셋 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# 통일된 이미지 삭제 API (권장 엔드포인트)
@app.delete("/api/projects/{project_id}/datasets/{dataset_id}/images/{filename}")
async def delete_image_v2(project_id: str, dataset_id: str, filename: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = dict_factory
        cur = conn.cursor()

        ds_dir = BASE_DATA_PATH / project_id / dataset_id
        ds_dir.mkdir(parents=True, exist_ok=True)

        # ✅ 데이터셋 존재 확인 → 없으면 폴더가 있을 때 자동 등록
        cur.execute("SELECT 1 FROM project_datasets WHERE id=? AND project_id=?", (dataset_id, project_id))
        if not cur.fetchone():
            if ds_dir.exists():
                cur.execute("""
                    INSERT INTO project_datasets (id, project_id, name, type, created_at, updated_at, image_count, total_size, description)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP, 0, 0, ?)
                """, (dataset_id, project_id, dataset_id, "classify", ""))
                conn.commit()
            else:
                conn.close()
                raise HTTPException(status_code=404, detail="데이터셋을 찾을 수 없습니다")

        file_path = ds_dir / filename
        if not file_path.exists():
            conn.close()
            raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다")

        # 파일 삭제
        try:
            file_path.unlink()
        except Exception as fe:
            conn.close()
            raise HTTPException(status_code=500, detail=f"파일 삭제 실패: {fe}")

        # 통계 갱신
        count, total = recalc_dataset_stats(conn, project_id, dataset_id)
        conn.commit()
        conn.close()

        return {"success": True, "imageCount": count, "totalSize": total}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"이미지 삭제 실패: {e}")
        raise HTTPException(status_code=500, detail=f"삭제 실패: {str(e)}")

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

# 이미지 조회 API
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
        raw_type = dataset_info.get("type", "anomaly")
        dataset_type = "anomaly" if raw_type == "anomal" else raw_type
                
        logger.info(
            f"Start training: ds={config.dataset_name} raw_type={raw_type} -> use={dataset_type} "
            f"lr={config.learning_rate} bs={config.batch_size} epochs={config.epochs} "
            f"split=({config.train_ratio}/{config.val_ratio}/{config.test_ratio}) "
            f"training_type={config.training_type}"
        )
        
        if dataset_type == "anomaly":  # Anomaly Detection
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
    try:
        trained_models = model_manager.discover_trained_models()
        uploaded_models = []
        saved_models_dir = Path("saved_models")

        if saved_models_dir.exists():
            for model_dir in saved_models_dir.iterdir():
                if model_dir.is_dir():
                    model_info_file = model_dir / "model_info.json"
                    if model_info_file.exists():
                        with open(model_info_file, "r", encoding="utf-8") as f:
                            model_info = json.load(f)
                        uploaded_models.append(model_info)

        all_models = trained_models + uploaded_models

        # ✅ camelCase 변환
        all_models_camel = []
        for m in all_models:
            all_models_camel.append(to_camel_case(m))

        return {
            "models": all_models_camel,
            "totalCount": len(all_models_camel)
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