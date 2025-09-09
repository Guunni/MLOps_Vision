"""
PatchCore 학습 관리 시스템
팀원의 Colab 학습 코드를 Python 클래스로 변환한 학습 관리자
"""

import os
import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import threading
from dataclasses import dataclass, asdict

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatchCoreConfig:
    """
    PatchCore 학습 설정을 담는 데이터 클래스
    팀원의 Colab 코드에서 사용된 모든 매개변수들을 체계화
    """
    # 모델 관련 설정
    backbone: str = "wideresnet50"  # 백본 네트워크
    layers: List[str] = None  # 특징 추출 레이어
    pretrain_embed_dimension: int = 1024
    target_embed_dimension: int = 1024
    
    # 이상 탐지 관련 설정
    anomaly_scorer_num_nn: int = 7  # k-NN에서 사용할 이웃 수
    patchsize: int = 3
    
    # 샘플링 관련 설정
    sampling_ratio: float = 0.3  # 샘플링 비율 (p 값)
    sampler_name: str = "approx_greedy_coreset"
    
    # 데이터 관련 설정
    dataset_name: str = "cable"  # 데이터셋 카테고리
    dataset_type: str = "mvtec"  # 데이터셋 타입
    resize_size: int = 320
    image_size: int = 256
    
    # 학습 관련 설정
    seed: int = 0
    gpu_id: int = 0 if None else None  # GPU 사용 설정
    save_model: bool = True
    
    # 실험 관리
    log_group: str = None  # 자동 생성됨
    
    def __post_init__(self):
        """초기화 후 처리"""
        if self.layers is None:
            self.layers = ["layer2", "layer3"]
        
        if self.log_group is None:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            self.log_group = f"MLOps_Vision_{self.backbone}_k{self.anomaly_scorer_num_nn}_p{int(self.sampling_ratio*100)}_{timestamp}"

class PatchCoreTrainer:
    """
    PatchCore 모델 학습을 관리하는 클래스
    Colab의 bash 명령어를 Python으로 변환하여 더 유연한 제어를 제공
    """
    
    def __init__(self, patchcore_dir: str = None, results_dir: str = None):
        """
        학습 관리자를 초기화합니다.
        
        Args:
            patchcore_dir: PatchCore 설치 디렉토리
            results_dir: 학습 결과를 저장할 디렉토리
        """
        # PatchCore 디렉토리 설정
        if patchcore_dir is None:
            self.patchcore_dir = Path.cwd().parent / "patchcore-inspection"
        else:
            self.patchcore_dir = Path(patchcore_dir)
        
        # 결과 디렉토리 설정
        if results_dir is None:
            self.results_dir = Path.cwd().parent / "models"
        else:
            self.results_dir = Path(results_dir)
        
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 실행 스크립트 경로
        self.run_script = self.patchcore_dir / "bin" / "run_patchcore.py"
        
        # 현재 실행 중인 학습 정보
        self.current_training = None
        self.training_thread = None
        
        logger.info(f"PatchCore 학습 관리자 초기화됨")
        logger.info(f"  - PatchCore 디렉토리: {self.patchcore_dir}")
        logger.info(f"  - 결과 저장 디렉토리: {self.results_dir}")
    
    def validate_setup(self) -> Dict[str, bool]:
        """
        학습을 위한 환경이 올바르게 설정되었는지 검증합니다.
        
        Returns:
            Dict: 검증 결과
        """
        validation = {
            "patchcore_directory_exists": self.patchcore_dir.exists(),
            "run_script_exists": self.run_script.exists(),
            "results_directory_writable": True,
            "python_environment_ready": True
        }
        
        # 결과 디렉토리 쓰기 권한 테스트
        try:
            test_file = self.results_dir / "test_write.tmp"
            test_file.touch()
            test_file.unlink()
        except Exception:
            validation["results_directory_writable"] = False
        
        # Python 환경 테스트
        try:
            import sys
            sys.path.append(str(self.patchcore_dir / "src"))
            import patchcore
            logger.info("✅ PatchCore 모듈 import 성공")
        except ImportError:
            validation["python_environment_ready"] = False
            logger.error("❌ PatchCore 모듈 import 실패")
        
        # 검증 결과 로깅
        all_valid = all(validation.values())
        if all_valid:
            logger.info("✅ 학습 환경 검증 완료")
        else:
            logger.error("❌ 학습 환경에 문제가 있습니다:")
            for key, value in validation.items():
                if not value:
                    logger.error(f"  - {key}: 실패")
        
        return validation
    
    def start_transfer_learning(self, config: PatchCoreConfig, dataset_path: Path, base_model_path: str, blocking: bool = True) -> Dict[str, Any]:
        """
        전이학습을 시작합니다.
        기존 학습된 모델을 기반으로 새 데이터에 대해 추가 학습을 수행합니다.
        """
        try:
            if self.is_training_active():
                return {"success": False, "error": "이미 학습이 진행 중입니다"}
            
            logger.info(f"🔄 전이학습 시작: {base_model_path} -> {config.dataset_name}")
            
            # 기존 모델 경로 확인
            base_model_full_path = self.results_dir / base_model_path / "weights" / "lightning" / "model.ckpt"
            if not base_model_full_path.exists():
                return {"success": False, "error": f"기존 모델을 찾을 수 없습니다: {base_model_path}"}
            
            # 학습 정보 설정
            training_info = {
                "type": "transfer_learning",
                "base_model": base_model_path,
                "dataset": config.dataset_name,
                "start_time": datetime.now().isoformat(),
                "status": "starting",
                "progress": 0
            }
            
            self.current_training = training_info
            
            if blocking:
                result = self._run_transfer_learning(config, dataset_path, base_model_full_path)
                self.current_training = None
                return result
            else:
                # 백그라운드에서 실행
                thread = threading.Thread(
                    target=self._run_transfer_learning_background,
                    args=(config, dataset_path, base_model_full_path)
                )
                thread.start()
                return {"success": True, "training_info": training_info}
                
        except Exception as e:
            logger.error(f"전이학습 시작 실패: {e}")
            self.current_training = None
            return {"success": False, "error": str(e)}

    def start_retraining(self, config: PatchCoreConfig, dataset_path: Path, base_model_path: str, blocking: bool = True) -> Dict[str, Any]:
        """
        재학습을 시작합니다.
        기존 데이터와 새 데이터를 모두 합쳐서 처음부터 다시 학습합니다.
        """
        try:
            if self.is_training_active():
                return {"success": False, "error": "이미 학습이 진행 중입니다"}
            
            logger.info(f"🔁 재학습 시작: 기존 모델 {base_model_path} 기반으로 전체 재학습")
            
            # 기존 모델의 데이터셋 경로 확인
            base_model_dir = self.results_dir / base_model_path
            if not base_model_dir.exists():
                return {"success": False, "error": f"기존 모델을 찾을 수 없습니다: {base_model_path}"}
            
            # 학습 정보 설정
            training_info = {
                "type": "retraining",
                "base_model": base_model_path,
                "dataset": config.dataset_name,
                "start_time": datetime.now().isoformat(),
                "status": "starting",
                "progress": 0
            }
            
            self.current_training = training_info
            
            if blocking:
                result = self._run_retraining(config, dataset_path, base_model_dir)
                self.current_training = None
                return result
            else:
                # 백그라운드에서 실행
                thread = threading.Thread(
                    target=self._run_retraining_background,
                    args=(config, dataset_path, base_model_dir)
                )
                thread.start()
                return {"success": True, "training_info": training_info}
                
        except Exception as e:
            logger.error(f"재학습 시작 실패: {e}")
            self.current_training = None
            return {"success": False, "error": str(e)}

    def _run_transfer_learning(self, config: PatchCoreConfig, dataset_path: Path, base_model_path: Path) -> Dict[str, Any]:
        """전이학습 실행 (내부 메서드)"""
        try:
            logger.info("🔄 전이학습 실행 중...")
            
            # anomalib 모델 로드
            from anomalib.models import Patchcore
            from anomalib.engine import Engine
            from anomalib.data import Folder
            
            # 기존 모델 로드
            model = Patchcore.load_from_checkpoint(str(base_model_path))
            
            # 새 데이터셋 설정
            datamodule = Folder(
                root=str(dataset_path),
                normal_dir="good",
                abnormal_dir="defect",
                train_batch_size=32,
                eval_batch_size=32,
                image_size=(config.image_size, config.image_size),
            )
            
            # 전이학습용 엔진 설정
            engine = Engine(
                task="segmentation",
                model=model,
                datamodule=datamodule,
                max_epochs=10,  # 전이학습은 적은 에폭으로
                resume_from_checkpoint=str(base_model_path)  # 체크포인트에서 재개
            )
            
            # 결과 저장 디렉토리 생성
            result_dir = self.results_dir / f"{config.log_group}_transfer"
            result_dir.mkdir(parents=True, exist_ok=True)
            
            # 학습 실행
            logger.info("전이학습 진행 중...")
            if self.current_training:
                self.current_training["status"] = "training"
                self.current_training["progress"] = 50
            
            engine.fit()
            
            # 테스트 실행
            logger.info("전이학습 테스트 중...")
            if self.current_training:
                self.current_training["status"] = "testing"
                self.current_training["progress"] = 90
            
            test_results = engine.test()
            
            # 결과 저장
            model_save_path = result_dir / "weights" / "lightning"
            model_save_path.mkdir(parents=True, exist_ok=True)
            engine.trainer.save_checkpoint(model_save_path / "model.ckpt")
            
            if self.current_training:
                self.current_training["status"] = "completed"
                self.current_training["progress"] = 100
            
            logger.info("✅ 전이학습 완료")
            return {
                "success": True,
                "model_path": str(result_dir),
                "test_results": test_results
            }
            
        except Exception as e:
            logger.error(f"전이학습 실행 실패: {e}")
            if self.current_training:
                self.current_training["status"] = "failed"
                self.current_training["error"] = str(e)
            return {"success": False, "error": str(e)}

    def _run_retraining(self, config: PatchCoreConfig, new_dataset_path: Path, base_model_dir: Path) -> Dict[str, Any]:
        """재학습 실행 (내부 메서드)"""
        try:
            logger.info("🔁 재학습 실행 중...")
            
            # 기존 데이터셋과 새 데이터셋 병합
            merged_dataset_path = self._merge_datasets(base_model_dir, new_dataset_path)
            
            # 일반 학습과 동일하지만 병합된 데이터셋 사용
            return self._run_training_with_dataset(config, merged_dataset_path, is_retraining=True)
            
        except Exception as e:
            logger.error(f"재학습 실행 실패: {e}")
            if self.current_training:
                self.current_training["status"] = "failed"
                self.current_training["error"] = str(e)
            return {"success": False, "error": str(e)}

    def _merge_datasets(self, base_model_dir: Path, new_dataset_path: Path) -> Path:
        """기존 데이터셋과 새 데이터셋을 병합"""
        import shutil
        
        # 병합된 데이터셋 저장할 임시 디렉토리
        merged_path = self.results_dir / "temp_merged_dataset"
        if merged_path.exists():
            shutil.rmtree(merged_path)
        merged_path.mkdir(parents=True)
        
        # good, defect 폴더 생성
        (merged_path / "good").mkdir()
        (merged_path / "defect").mkdir()
        
        # 기존 데이터셋 복사 (기존 모델 디렉토리에서 원본 데이터 경로 찾기)
        # 실제 구현에서는 모델 메타데이터에서 원본 데이터 경로를 찾아야 함
        
        # 새 데이터셋 복사
        if (new_dataset_path / "good").exists():
            for img_file in (new_dataset_path / "good").glob("*"):
                shutil.copy2(img_file, merged_path / "good")
        
        if (new_dataset_path / "defect").exists():
            for img_file in (new_dataset_path / "defect").glob("*"):
                shutil.copy2(img_file, merged_path / "defect")
        
        logger.info(f"데이터셋 병합 완료: {merged_path}")
        return merged_path

    def _run_transfer_learning_background(self, config: PatchCoreConfig, dataset_path: Path, base_model_path: Path):
        """백그라운드 전이학습 실행"""
        try:
            result = self._run_transfer_learning(config, dataset_path, base_model_path)
            logger.info(f"백그라운드 전이학습 완료: {result}")
        except Exception as e:
            logger.error(f"백그라운드 전이학습 실패: {e}")
        finally:
            self.current_training = None

    def _run_retraining_background(self, config: PatchCoreConfig, dataset_path: Path, base_model_dir: Path):
        """백그라운드 재학습 실행"""
        try:
            result = self._run_retraining(config, dataset_path, base_model_dir)
            logger.info(f"백그라운드 재학습 완료: {result}")
        except Exception as e:
            logger.error(f"백그라운드 재학습 실패: {e}")
        finally:
            self.current_training = None
    
    def build_command(self, config: PatchCoreConfig, dataset_path: str) -> List[str]:
        """
        PatchCore 학습 명령어를 구성합니다.
        
        이 함수는 팀원의 Colab 코드에서 사용된 복잡한 bash 명령어를
        Python 리스트로 변환합니다.
        
        Args:
            config: 학습 설정
            dataset_path: 데이터셋 경로
            
        Returns:
            List[str]: 실행할 명령어 리스트
        """
        cmd = [
            "python", str(self.run_script),
            "--seed", str(config.seed),
            "--log_group", config.log_group,
            str(self.results_dir),
            "patch_core"
        ]
        
        # GPU 설정
        if config.gpu_id is not None:
            cmd.extend(["--gpu", str(config.gpu_id)])
        
        # 모델 저장 설정
        if config.save_model:
            cmd.append("--save_patchcore_model")
        
        # 백본 네트워크 설정
        cmd.extend(["-b", config.backbone])
        
        # 레이어 설정
        for layer in config.layers:
            cmd.extend(["-le", layer])
        
        # 임베딩 차원 설정
        cmd.extend([
            "--pretrain_embed_dimension", str(config.pretrain_embed_dimension),
            "--target_embed_dimension", str(config.target_embed_dimension)
        ])
        
        # 이상 탐지 설정
        cmd.extend([
            "--anomaly_scorer_num_nn", str(config.anomaly_scorer_num_nn),
            "--patchsize", str(config.patchsize)
        ])
        
        # 샘플러 설정
        cmd.extend([
            "sampler", 
            "-p", str(config.sampling_ratio),
            config.sampler_name
        ])
        
        # 데이터셋 설정
        cmd.extend([
            "dataset",
            "--resize", str(config.resize_size),
            "--imagesize", str(config.image_size),
            "-d", config.dataset_name,
            config.dataset_type,
            str(dataset_path)
        ])
        
        logger.info(f"학습 명령어 구성 완료: {len(cmd)}개 인자")
        return cmd
    
    def start_training(self, config: PatchCoreConfig, dataset_path: str, blocking: bool = False) -> Dict[str, Any]:
        """
        PatchCore 모델 학습을 시작합니다.
        
        Args:
            config: 학습 설정
            dataset_path: 데이터셋 경로
            blocking: True면 학습 완료까지 대기, False면 백그라운드 실행
            
        Returns:
            Dict: 학습 시작 결과
        """
        # 환경 검증
        validation = self.validate_setup()
        if not all(validation.values()):
            return {
                "success": False,
                "error": "환경 검증 실패",
                "validation": validation
            }
        
        # 이미 학습이 진행 중인지 확인
        if self.is_training_active():
            return {
                "success": False,
                "error": "이미 학습이 진행 중입니다"
            }
        
        try:
            # 명령어 구성
            cmd = self.build_command(config, dataset_path)
            
            # 학습 정보 기록
            training_info = {
                "config": asdict(config),
                "dataset_path": dataset_path,
                "start_time": datetime.now().isoformat(),
                "status": "running",
                "log_group": config.log_group,
                "results_path": str(self.results_dir / config.log_group)
            }
            
            self.current_training = training_info
            
            # 로그 파일 설정
            log_file_path = self.results_dir / f"{config.log_group}_training.log"
            
            logger.info(f"🚀 PatchCore 학습 시작: {config.log_group}")
            logger.info(f"📊 설정: {config.backbone}, k={config.anomaly_scorer_num_nn}, p={config.sampling_ratio}")
            logger.info(f"📁 데이터셋: {dataset_path}")
            logger.info(f"📝 로그 파일: {log_file_path}")
            
            if blocking:
                # 동기식 실행 (학습 완료까지 대기)
                result = self._run_training_sync(cmd, log_file_path)
            else:
                # 비동기식 실행 (백그라운드에서 실행)
                result = self._run_training_async(cmd, log_file_path)
            
            return {
                "success": True,
                "training_info": training_info,
                "execution_result": result
            }
            
        except Exception as e:
            logger.error(f"❌ 학습 시작 실패: {e}")
            self.current_training = None
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_training_sync(self, cmd: List[str], log_file_path: Path) -> Dict[str, Any]:
        """
        동기식으로 학습을 실행합니다.
        """
        start_time = time.time()
        
        try:
            with open(log_file_path, 'w') as log_file:
                # 작업 디렉토리를 PatchCore 디렉토리로 변경
                process = subprocess.Popen(
                    cmd,
                    cwd=self.patchcore_dir,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=self._get_training_environment()
                )
                
                # 프로세스 완료 대기
                return_code = process.wait()
                
                end_time = time.time()
                duration = end_time - start_time
                
                if return_code == 0:
                    logger.info(f"✅ 학습 완료 (소요시간: {duration:.1f}초)")
                    self.current_training["status"] = "completed"
                    self.current_training["end_time"] = datetime.now().isoformat()
                    self.current_training["duration"] = duration
                    
                    return {
                        "success": True,
                        "return_code": return_code,
                        "duration": duration
                    }
                else:
                    logger.error(f"❌ 학습 실패 (종료 코드: {return_code})")
                    self.current_training["status"] = "failed"
                    self.current_training["error_code"] = return_code
                    
                    return {
                        "success": False,
                        "return_code": return_code,
                        "duration": duration
                    }
                    
        except Exception as e:
            logger.error(f"❌ 학습 실행 중 오류: {e}")
            if self.current_training:
                self.current_training["status"] = "error"
                self.current_training["error"] = str(e)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def _run_training_async(self, cmd: List[str], log_file_path: Path) -> Dict[str, Any]:
        """
        비동기식으로 학습을 실행합니다.
        """
        def training_worker():
            """백그라운드에서 실행될 학습 작업"""
            try:
                result = self._run_training_sync(cmd, log_file_path)
                logger.info(f"백그라운드 학습 완료: {result}")
            except Exception as e:
                logger.error(f"백그라운드 학습 오류: {e}")
                if self.current_training:
                    self.current_training["status"] = "error"
                    self.current_training["error"] = str(e)
        
        # 백그라운드 스레드 시작
        self.training_thread = threading.Thread(target=training_worker, daemon=True)
        self.training_thread.start()
        
        return {
            "success": True,
            "mode": "async",
            "thread_id": self.training_thread.ident
        }
    
    def _get_training_environment(self) -> Dict[str, str]:
        """
        학습을 위한 환경 변수를 설정합니다.
        """
        env = os.environ.copy()
        
        # PYTHONPATH에 PatchCore src 디렉토리 추가
        patchcore_src = str(self.patchcore_dir / "src")
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = f"{patchcore_src}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env["PYTHONPATH"] = patchcore_src
        
        return env
    
    def is_training_active(self) -> bool:
        """
        현재 학습이 진행 중인지 확인합니다.
        
        Returns:
            bool: 학습 진행 상태
        """
        if self.current_training is None:
            return False
        
        if self.current_training.get("status") in ["completed", "failed", "error"]:
            return False
        
        # 백그라운드 스레드가 살아있는지 확인
        if self.training_thread and self.training_thread.is_alive():
            return True
        
        return False
    
    def get_training_status(self) -> Optional[Dict[str, Any]]:
        """
        현재 학습 상태 정보를 반환합니다.
        
        Returns:
            Dict: 학습 상태 정보 (학습 중이 아니면 None)
        """
        if self.current_training is None:
            return None
        
        status = self.current_training.copy()
        status["is_active"] = self.is_training_active()
        
        return status
    
    def stop_training(self) -> bool:
        """
        진행 중인 학습을 중단합니다.
        
        Returns:
            bool: 중단 성공 여부
        """
        if not self.is_training_active():
            logger.warning("중단할 학습이 없습니다.")
            return False
        
        try:
            # 여기에 프로세스 종료 로직을 추가할 수 있습니다
            # 현재는 기본적인 상태 변경만 수행
            if self.current_training:
                self.current_training["status"] = "stopped"
                self.current_training["stop_time"] = datetime.now().isoformat()
            
            logger.info("🛑 학습이 중단되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"❌ 학습 중단 실패: {e}")
            return False

# 사용 예시 및 테스트
if __name__ == "__main__":
    # 학습 매니저 인스턴스 생성
    trainer = PatchCoreTrainer()
    
    # 환경 검증
    validation = trainer.validate_setup()
    print(f"환경 검증 결과: {validation}")
    
    if all(validation.values()):
        print("✅ 학습 환경이 준비되었습니다!")
        
        # 기본 설정으로 학습 설정 생성
        config = PatchCoreConfig(
            dataset_name="cable",
            anomaly_scorer_num_nn=7,
            sampling_ratio=0.3
        )
        
        print(f"학습 설정: {config.log_group}")
        print("실제 학습을 시작하려면 데이터셋 경로를 제공하세요.")
    else:
        print("❌ 학습 환경에 문제가 있습니다.")