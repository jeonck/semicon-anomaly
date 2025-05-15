import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_semiconductor_data(n_samples=1000, n_features=5, anomaly_percentage=0.05):
    """
    반도체 공정 데이터와 이상치를 생성합니다.
    
    매개변수:
    n_samples (int): 생성할 샘플 수
    n_features (int): 특성 수 (온도, 압력, 유량 등)
    anomaly_percentage (float): 이상치 비율 (0~1)
    
    반환:
    pandas.DataFrame: 생성된 반도체 공정 데이터
    """
    # 특성 이름 정의
    feature_names = [
        "온도", "압력", "유량", "전압", "전류", 
        "가스농도", "습도", "진동", "소음", "두께"
    ]
    
    # 사용할 특성 선택
    selected_features = feature_names[:n_features]
    
    # 각 특성의 정상 범위 설정
    normal_ranges = {
        "온도": (150, 170),  # 섭씨
        "압력": (95, 105),   # kPa
        "유량": (45, 55),    # L/min
        "전압": (220, 240),  # V
        "전류": (9.5, 10.5), # A
        "가스농도": (28, 32), # %
        "습도": (40, 50),    # %
        "진동": (0.05, 0.15), # mm
        "소음": (60, 70),    # dB
        "두께": (0.9, 1.1)   # mm
    }
    
    # 시간 인덱스 생성
    start_time = datetime.now() - timedelta(hours=n_samples)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_samples)]
    
    # 데이터프레임 생성
    data = pd.DataFrame(index=timestamps)
    
    # 정상 데이터 생성
    for feature in selected_features:
        low, high = normal_ranges[feature]
        mean = (low + high) / 2
        std = (high - low) / 6  # 표준편차 (±3시그마가 범위 내에 있도록)
        
        # 약간의 주기적인 변동 추가
        base_values = np.random.normal(mean, std, n_samples)
        periodic = np.sin(np.linspace(0, 10*np.pi, n_samples)) * std * 0.5
        
        # 추세 추가
        trend = np.linspace(-std*0.3, std*0.3, n_samples)
        
        data[feature] = base_values + periodic + trend
    
    # 이상치 주입
    n_anomalies = int(n_samples * anomaly_percentage)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # 이상치 레이블 열 추가 (0: 정상, 1: 이상)
    data['이상여부'] = 0
    
    for idx in anomaly_indices:
        # 랜덤하게 선택된 특성에 이상치 주입
        feature = np.random.choice(selected_features)
        low, high = normal_ranges[feature]
        
        # 이상치 유형 선택 (급격한 스파이크, 드리프트, 단계적 변화)
        anomaly_type = np.random.choice(['spike', 'drift', 'step'])
        
        if anomaly_type == 'spike':
            # 급격한 스파이크 (정상 범위의 1.5-3배)
            multiplier = np.random.uniform(1.5, 3.0)
            direction = np.random.choice([1, -1])
            deviation = (high - low) * multiplier * direction
            data.iloc[idx, data.columns.get_loc(feature)] = data.iloc[idx, data.columns.get_loc(feature)] + deviation
        
        elif anomaly_type == 'drift':
            # 점진적 드리프트 (연속된 값들이 점차 증가 또는 감소)
            if idx < n_samples - 10:  # 최소 10개의 샘플이 남아있는지 확인
                drift_length = min(10, n_samples - idx)
                direction = np.random.choice([1, -1])
                drift_values = np.linspace(0, (high - low) * 1.2 * direction, drift_length)
                
                for i in range(drift_length):
                    data.iloc[idx + i, data.columns.get_loc(feature)] += drift_values[i]
                    data.iloc[idx + i, data.columns.get_loc('이상여부')] = 1
        
        elif anomaly_type == 'step':
            # 단계적 변화 (갑자기 레벨이 변하고 유지됨)
            if idx < n_samples - 5:  # 최소 5개의 샘플이 남아있는지 확인
                step_length = min(5, n_samples - idx)
                direction = np.random.choice([1, -1])
                step_value = (high - low) * 0.8 * direction
                
                for i in range(step_length):
                    data.iloc[idx + i, data.columns.get_loc(feature)] += step_value
                    data.iloc[idx + i, data.columns.get_loc('이상여부')] = 1
        
        # 단일 이상치에 대한 레이블 설정
        data.iloc[idx, data.columns.get_loc('이상여부')] = 1
    
    # 생산 배치 번호 추가
    batch_size = 100  # 각 배치당 샘플 수
    n_batches = int(np.ceil(n_samples / batch_size))
    batch_numbers = np.repeat(np.arange(1, n_batches + 1), batch_size)[:n_samples]
    data['배치번호'] = batch_numbers
    
    # 공정단계 추가 (5개 단계로 가정)
    process_stages = np.random.randint(1, 6, n_samples)
    data['공정단계'] = process_stages
    
    # 장비ID 추가 (3개 장비로 가정)
    equipment_ids = np.random.choice(['EQ-A', 'EQ-B', 'EQ-C'], n_samples)
    data['장비ID'] = equipment_ids
    
    return data

# 데이터 생성 및 CSV 파일로 저장
np.random.seed(42)  # 재현성을 위한 시드 설정
data = generate_semiconductor_data(n_samples=1000, n_features=5, anomaly_percentage=0.05)
data.to_csv('data/semiconductor_process_data.csv')

print(f"생성된 데이터 형태: {data.shape}")
print(f"이상치 수: {data['이상여부'].sum()}")
print("데이터 저장 완료: data/semiconductor_process_data.csv")
