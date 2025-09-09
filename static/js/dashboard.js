// 대시보드 JavaScript 기능들

class DashboardManager {
    constructor() {
        this.init();
    }

    async init() {
        await this.loadSystemStatus();
        this.setupEventListeners();
        this.startPeriodicUpdates();
    }

    // 시스템 상태 로드
    async loadSystemStatus() {
        this.showLoading();
        
        try {
            const response = await fetch('/system/status');
            const data = await response.json();
            
            this.updateStatusDisplay(data);
            this.addLogEntry('시스템 상태가 업데이트되었습니다.');
            
        } catch (error) {
            console.error('시스템 상태 로드 실패:', error);
            this.addLogEntry('시스템 상태 로드에 실패했습니다.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // 상태 표시 업데이트
    updateStatusDisplay(data) {
        // 시스템 상태
        const systemStatus = document.getElementById('system-status');
        const systemDetail = document.getElementById('system-detail');
        
        if (data.system_ready) {
            systemStatus.textContent = '정상';
            systemStatus.className = 'status-value status-success';
            systemDetail.textContent = 'PatchCore 준비 완료';
        } else {
            systemStatus.textContent = '설정 필요';
            systemStatus.className = 'status-value status-warning';
            systemDetail.textContent = '시스템 설정이 필요합니다';
        }

        // 데이터셋 수
        document.getElementById('datasets-count').textContent = data.datasets_available;

        // 모델 수
        document.getElementById('models-count').textContent = data.models_trained;

        // 학습 상태
        const trainingStatus = document.getElementById('training-status');
        const trainingDetail = document.getElementById('training-detail');
        
        if (data.training_active) {
            trainingStatus.textContent = '학습 중';
            trainingStatus.className = 'status-value status-info';
            trainingDetail.textContent = '모델 학습이 진행 중입니다';
        } else {
            trainingStatus.textContent = '대기 중';
            trainingStatus.className = 'status-value';
            trainingDetail.textContent = '현재 진행 중인 학습 없음';
        }
    }

    // 이벤트 리스너 설정
    setupEventListeners() {
        // 새로고침 버튼
        window.refreshStatus = () => this.loadSystemStatus();
        
        // 빠른 액션 버튼들
        window.setupSystem = () => this.setupSystem();
        window.uploadDataset = () => this.navigateToDatasets();
        window.startTraining = () => this.navigateToTraining();
        window.viewModels = () => this.navigateToModels();
    }

    // 시스템 설정
    async setupSystem() {
        this.showLoading('시스템을 설정하고 있습니다...');
        
        try {
            const response = await fetch('/system/setup', {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addLogEntry('시스템 설정이 완료되었습니다.', 'success');
                await this.loadSystemStatus(); // 상태 새로고침
            } else {
                this.addLogEntry('시스템 설정에 실패했습니다.', 'error');
            }
            
        } catch (error) {
            console.error('시스템 설정 실패:', error);
            this.addLogEntry('시스템 설정 중 오류가 발생했습니다.', 'error');
        } finally {
            this.hideLoading();
        }
    }

    // 페이지 네비게이션
    navigateToDatasets() {
        window.location.href = '/ui/datasets';
    }

    navigateToTraining() {
        window.location.href = '/ui/training';
    }

    navigateToModels() {
        window.location.href = '/ui/models';
    }

    // 로그 항목 추가
    addLogEntry(message, type = 'info') {
        const logContainer = document.getElementById('activity-log');
        const logEntry = document.createElement('div');
        logEntry.className = 'log-entry';
        
        const iconClass = {
            'info': 'fas fa-info-circle',
            'success': 'fas fa-check-circle',
            'warning': 'fas fa-exclamation-triangle',
            'error': 'fas fa-times-circle'
        }[type] || 'fas fa-info-circle';
        
        const now = new Date();
        const timeString = now.toLocaleTimeString('ko-KR', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
        
        logEntry.innerHTML = `
            <i class="${iconClass}"></i>
            <span class="log-time">${timeString}</span>
            <span class="log-message">${message}</span>
        `;
        
        // 최신 로그를 맨 위에 추가
        logContainer.insertBefore(logEntry, logContainer.firstChild);
        
        // 로그 항목이 너무 많으면 오래된 것 제거
        const maxLogs = 20;
        while (logContainer.children.length > maxLogs) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }

    // 주기적 업데이트 시작
    startPeriodicUpdates() {
        // 30초마다 상태 업데이트
        setInterval(() => {
            this.loadSystemStatus();
        }, 30000);
    }

    // 로딩 표시
    showLoading(message = '처리 중...') {
        const overlay = document.getElementById('loading-overlay');
        const messageElement = overlay.querySelector('p');
        messageElement.textContent = message;
        overlay.classList.add('show');
    }

    // 로딩 숨김
    hideLoading() {
        const overlay = document.getElementById('loading-overlay');
        overlay.classList.remove('show');
    }
}

// 페이지 로드 시 대시보드 초기화
document.addEventListener('DOMContentLoaded', () => {
    new DashboardManager();
});

// 전역 함수들 (HTML에서 호출)
function refreshStatus() {
    if (window.dashboardManager) {
        window.dashboardManager.loadSystemStatus();
    }
}