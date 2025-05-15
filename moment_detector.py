import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from datetime import datetime
import json
import os
import asyncio
import torch
import warnings
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MOMENT-Detector")

# 경고 무시
warnings.filterwarnings("ignore", category=UserWarning)

# MOMENT 기반 이상치 탐지 클래스
class MomentAnomalyDetector:
    """
    MOMENT 파운데이션 모델을 사용한 이상치 탐지를 위한 클래스
    
    MOMENT는 시계열 데이터를 위한 파운데이션 모델로, 이상치 탐지에 효과적입니다.
    이 클래스는 MOMENT 모델의 reconstruction 기능을 활용하여 이상치를 탐지합니다.
    """
    
    def __init__(self, use_moment=True, contamination=0.05, model_name="AutonLab/MOMENT-1-large"):
        """
        모델 초기화
        
        매개변수:
        use_moment (bool): MOMENT 모델 사용 여부 (False이면 IsolationForest 사용)
        contamination (float): 이상치 비율 예상치
        model_name (str): 사용할 MOMENT 모델 이름
        """
        self.use_moment = use_moment
        self.contamination = contamination
        self.model_name = model_name
        self.scaler = None
        self.isolation_forest = None
        
        # MOMENT 모델 관련 속성
        self.model = None
        self.is_moment_available = False
        self.seq_len = 512  # MOMENT 모델 입력 요구 길이
        
        # MOMENT 모델 로드
        if use_moment:
            try:
                # MOMENT 라이브러리 임포트
                from momentfm import MOMENTPipeline
                
                logger.info(f"MOMENT 이상치 탐지 모델 {model_name} 로드 중...")
                
                # MOMENT 모델 로드 (reconstruction 모드로)
                self.model = MOMENTPipeline.from_pretrained(
                    model_name,
                    model_kwargs={"task_name": "reconstruction"}
                )
                self.model.init()
                
                # CUDA 사용 가능 여부 확인
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                logger.info(f"MOMENT 모델이 {self.device} 장치를 사용합니다.")
                
                # 모델을 장치로 이동
                self.model = self.model.to(self.device)
                
                logger.info(f"MOMENT 이상치 탐지 모델 초기화 성공: {model_name}")
                self.is_moment_available = True
                
            except Exception as e:
                logger.error(f"MOMENT 모델 초기화 실패: {e}. IsolationForest를 대신 사용합니다.")
                self.is_moment_available = False
        else:
            logger.info("MOMENT 모델 사용이 비활성화되었습니다. IsolationForest를 사용합니다.")
    
    def fit(self, X, contamination=None):
        """
        데이터에 모델 학습
        
        매개변수:
        X (numpy.ndarray): 학습 데이터
        contamination (float): 이상치 비율 (None이면 초기화 시 설정된 값 사용)
        """
        if contamination is not None:
            self.contamination = contamination
            
        # 데이터 전처리
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # MOMENT 모델이 사용 불가능하거나 사용하지 않는 경우 IsolationForest 학습
        if not self.is_moment_available or not self.use_moment:
            logger.info(f"IsolationForest 모델 학습 중 (contamination={self.contamination})...")
            self.isolation_forest = IsolationForest(
                contamination=self.contamination, 
                random_state=42,
                n_estimators=200
            )
            self.isolation_forest.fit(X_scaled)
            logger.info("IsolationForest 모델 학습 완료")
        
        # MOMENT는 사전 학습된 모델이므로 별도 학습 필요 없음
        return self
    
    def _prepare_data_for_moment(self, X_scaled):
        """
        MOMENT 모델 입력을 위한 데이터 준비
        
        매개변수:
        X_scaled (numpy.ndarray): 스케일링된 입력 데이터 [n_samples, n_features]
        
        반환:
        torch.Tensor: MOMENT 모델 입력 형식으로 변환된 데이터 [batch_size, n_channels, seq_len]
        torch.Tensor: 입력 마스크 [batch_size, seq_len]
        """
        n_samples, n_features = X_scaled.shape
        
        # MOMENT는 [batch_size, n_channels, seq_len] 형태의 입력을 기대함
        # 우리는 전체 X를 하나의 배치로 처리
        batch_size = 1
        
        # 모든 특성(channels)을 포함한 텐서 생성
        X_tensor = np.zeros((batch_size, n_features, self.seq_len))
        
        # 실제 데이터 길이가 seq_len보다 짧은 경우
        actual_len = min(n_samples, self.seq_len)
        
        # 시계열 데이터를 channel 차원으로 변환 (transpose)
        # X_scaled의 각 열(특성)이 X_tensor의 각 채널이 됨
        for i in range(n_features):
            X_tensor[0, i, :actual_len] = X_scaled[:actual_len, i]
            
        # 모든 시점이 관찰됨을 나타내는 입력 마스크 (1: 관찰됨, 0: 관찰되지 않음)
        input_mask = np.ones((batch_size, self.seq_len))
        
        # 부족한 길이만큼 마스크를 0으로 설정 (padding 부분)
        if actual_len < self.seq_len:
            input_mask[0, actual_len:] = 0
            
        # PyTorch 텐서로 변환
        X_tensor = torch.tensor(X_tensor, dtype=torch.float32)
        input_mask = torch.tensor(input_mask, dtype=torch.float32)
        
        return X_tensor, input_mask
    
    def predict(self, X):
        """
        이상치 예측
        
        매개변수:
        X (numpy.ndarray): 예측할 데이터
        
        반환:
        numpy.ndarray: 이상치 레이블 (0: 정상, 1: 이상)
        """
        # 데이터 전처리
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        
        # MOMENT 모델 사용 가능하고 사용하기로 한 경우
        if self.is_moment_available and self.use_moment:
            try:
                logger.info("MOMENT 모델을 사용하여 이상치 탐지 중...")
                
                # 데이터 준비
                X_tensor, input_mask = self._prepare_data_for_moment(X_scaled)
                
                # 장치로 이동
                X_tensor = X_tensor.to(self.device)
                input_mask = input_mask.to(self.device)
                
                # 예측
                with torch.no_grad():
                    outputs = self.model(x_enc=X_tensor, input_mask=input_mask)
                    reconstruction = outputs.reconstruction
                    
                    # 재구성된 데이터와 원본 데이터 간의 오차 계산
                    n_samples = min(X_scaled.shape[0], self.seq_len)
                    
                    # 재구성 오류를 이상치 점수로 사용
                    # 튜토리얼에서와 같이 MSE 사용
                    error_scores = []
                    
                    for i in range(X_scaled.shape[1]):  # 각 특성에 대해
                        original = X_tensor[0, i, :n_samples].cpu().numpy()
                        recon = reconstruction[0, i, :n_samples].cpu().numpy()
                        mse = np.square(original - recon)
                        error_scores.append(mse)
                    
                    # 모든 특성의 평균 MSE를 이상치 점수로 사용
                    anomaly_scores = np.mean(error_scores, axis=0)
                    
                    # 남은 샘플에 대해서는 평균값으로 채우기
                    if n_samples < X_scaled.shape[0]:
                        mean_score = np.mean(anomaly_scores)
                        padding_scores = np.ones(X_scaled.shape[0] - n_samples) * mean_score
                        anomaly_scores = np.concatenate([anomaly_scores, padding_scores])
                    
                    # 임계값 기반으로 이상치 판별
                    threshold = np.percentile(anomaly_scores, 100 * (1 - self.contamination))
                    predictions = (anomaly_scores > threshold).astype(int)
                
                logger.info(f"MOMENT 모델 이상치 탐지 완료: {np.sum(predictions)}/{len(predictions)} 이상치 발견")
                return predictions
                
            except Exception as e:
                logger.error(f"MOMENT 모델 예측 실패, IsolationForest로 전환합니다: {e}")
                logger.exception("상세 오류:")
                # MOMENT 실패 시 IsolationForest로 대체
                if self.isolation_forest is None:
                    self.fit(X, self.contamination)
                
        # MOMENT 모델 사용 불가능하거나 실패한 경우 IsolationForest 사용
        if self.isolation_forest is None:
            logger.info(f"IsolationForest 모델 학습 중 (contamination={self.contamination})...")
            self.isolation_forest = IsolationForest(
                contamination=self.contamination, 
                random_state=42,
                n_estimators=200
            )
            self.isolation_forest.fit(X_scaled)
        
        # IsolationForest로 예측 (-1: 이상치, 1: 정상)
        predictions = (self.isolation_forest.predict(X_scaled) == -1).astype(int)
        logger.info(f"IsolationForest 이상치 탐지 완료: {np.sum(predictions)}/{len(predictions)} 이상치 발견")
        return predictions
    
    def get_anomaly_scores(self, X):
        """
        이상치 점수 계산
        
        매개변수:
        X (numpy.ndarray): 입력 데이터
        
        반환:
        numpy.ndarray: 이상치 점수 (높을수록 이상 가능성 높음)
        """
        # 데이터 전처리
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        
        # MOMENT 모델 사용 가능하고 사용하기로 한 경우
        if self.is_moment_available and self.use_moment:
            try:
                # 데이터 준비
                X_tensor, input_mask = self._prepare_data_for_moment(X_scaled)
                
                # 장치로 이동
                X_tensor = X_tensor.to(self.device)
                input_mask = input_mask.to(self.device)
                
                # 예측
                with torch.no_grad():
                    outputs = self.model(x_enc=X_tensor, input_mask=input_mask)
                    reconstruction = outputs.reconstruction
                    
                    # 재구성된 데이터와 원본 데이터 간의 오차 계산
                    n_samples = min(X_scaled.shape[0], self.seq_len)
                    
                    # 재구성 오류를 이상치 점수로 사용
                    error_scores = []
                    
                    for i in range(X_scaled.shape[1]):  # 각 특성에 대해
                        original = X_tensor[0, i, :n_samples].cpu().numpy()
                        recon = reconstruction[0, i, :n_samples].cpu().numpy()
                        mse = np.square(original - recon)
                        error_scores.append(mse)
                    
                    # 모든 특성의 평균 MSE를 이상치 점수로 사용
                    anomaly_scores = np.mean(error_scores, axis=0)
                    
                    # 남은 샘플에 대해서는 평균값으로 채우기
                    if n_samples < X_scaled.shape[0]:
                        mean_score = np.mean(anomaly_scores)
                        padding_scores = np.ones(X_scaled.shape[0] - n_samples) * mean_score
                        anomaly_scores = np.concatenate([anomaly_scores, padding_scores])
                
                return anomaly_scores
                
            except Exception as e:
                logger.error(f"MOMENT 모델 점수 계산 실패: {e}")
                # 실패 시 IsolationForest 사용
        
        # IsolationForest 사용하여 이상 점수 계산
        if self.isolation_forest is None:
            self.fit(X, self.contamination)
        
        # 이상 점수 계산 (-값이 클수록 이상일 가능성이 높음)
        # 부호를 바꿔서 양수값이 클수록 이상일 가능성이 높게 변환
        return -self.isolation_forest.score_samples(X_scaled)

# 비동기 이상치 탐지 함수
async def detect_anomalies_with_moment_async(data, features, contamination=0.05, use_moment=True):
    """
    MOMENT 파운데이션 모델을 활용한 이상치 탐지 (비동기 버전)
    
    매개변수:
    data (pandas.DataFrame): 시계열 데이터
    features (list): 분석할 특성 목록
    contamination (float): 이상치 비율 예상치
    use_moment (bool): MOMENT 모델 사용 여부
    
    반환:
    pandas.DataFrame: 이상치 탐지 결과가 추가된 데이터프레임
    dict: 성능 지표
    """
    # 이 함수 내부에서는 CPU 집약적인 연산이 있으므로 별도의 스레드 풀에서 실행
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        lambda: detect_anomalies_with_moment(data, features, contamination, use_moment)
    )

# 동기 이상치 탐지 함수 (실제 계산 수행)
def detect_anomalies_with_moment(data, features, contamination=0.05, use_moment=True):
    """
    MOMENT 파운데이션 모델을 활용한 이상치 탐지
    
    매개변수:
    data (pandas.DataFrame): 시계열 데이터
    features (list): 분석할 특성 목록
    contamination (float): 이상치 비율 예상치
    use_moment (bool): MOMENT 모델 사용 여부
    
    반환:
    pandas.DataFrame: 이상치 탐지 결과가 추가된 데이터프레임
    dict: 성능 지표
    """
    try:
        # 특성 데이터 추출
        X = data[features].values
        
        # 모델 초기화 및 학습
        start_time = datetime.now()
        model = MomentAnomalyDetector(use_moment=use_moment, contamination=contamination)
        model.fit(X, contamination)
        
        # 이상치 점수 계산
        anomaly_scores = model.get_anomaly_scores(X)
        
        # 임계값 계산 및 이상치 예측
        threshold = np.percentile(anomaly_scores, 100 * (1 - contamination))
        predictions = (anomaly_scores > threshold).astype(int)
        
        # 처리 시간 계산
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # 결과 데이터프레임 생성
        result_df = data.copy()
        result_df['이상치_점수'] = anomaly_scores
        result_df['이상치'] = predictions
        
        # 실제 이상치와 비교 (실제 이상치 레이블이 있는 경우)
        metrics = {
            'processing_time': processing_time,
            'model_type': 'MOMENT' if use_moment and model.is_moment_available else 'IsolationForest'
        }
        
        if '이상여부' in result_df.columns:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
            
            true_anomalies = result_df['이상여부'].values
            predicted_anomalies = result_df['이상치'].values
            
            precision = precision_score(true_anomalies, predicted_anomalies)
            recall = recall_score(true_anomalies, predicted_anomalies)
            f1 = f1_score(true_anomalies, predicted_anomalies)
            cm = confusion_matrix(true_anomalies, predicted_anomalies)
            
            metrics.update({
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm.tolist()
            })
        
        return result_df, metrics
        
    except Exception as e:
        logger.error(f"이상치 탐지 중 오류 발생: {e}")
        logger.exception("상세 오류:")
        # 오류 발생 시 빈 결과 반환
        result_df = data.copy()
        result_df['이상치'] = np.zeros(len(data))
        return result_df, {'error': str(e)}

# 테스트 코드
if __name__ == "__main__":
    # 테스트 데이터 생성
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    X = np.random.randn(n_samples, n_features)
    
    # 이상치 추가
    X[0:10, :] = X[0:10, :] * 5
    
    # 테스트 데이터프레임 생성
    test_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    test_df['timestamp'] = pd.date_range(start='2023-01-01', periods=n_samples, freq='H')
    test_df.set_index('timestamp', inplace=True)
    
    # 테스트 실행 - IsolationForest 사용
    print("=== IsolationForest 테스트 ===")
    result_df, metrics = detect_anomalies_with_moment(
        test_df, 
        [f'feature_{i}' for i in range(n_features)], 
        use_moment=False
    )
    print(f"IsolationForest 이상치 수: {result_df['이상치'].sum()}")
    print(f"처리 시간: {metrics['processing_time']:.4f}초\n")
    
    # 테스트 실행 - MOMENT 사용
    print("=== MOMENT 모델 테스트 ===")
    result_df, metrics = detect_anomalies_with_moment(
        test_df, 
        [f'feature_{i}' for i in range(n_features)], 
        use_moment=True
    )
    print(f"사용된 모델: {metrics['model_type']}")
    print(f"MOMENT 이상치 수: {result_df['이상치'].sum()}")
    print(f"처리 시간: {metrics['processing_time']:.4f}초\n")
