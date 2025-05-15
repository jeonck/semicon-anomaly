import torch
import numpy as np
from momentfm import MOMENTPipeline

def test_moment_model():
    """MOMENT 모델의 이상치 탐지 기능을 테스트하는 간단한 스크립트"""
    
    print("MOMENT 모델 테스트 시작...")
    
    try:
        # 모델 로드
        model = MOMENTPipeline.from_pretrained(
            "AutonLab/MOMENT-1-large",
            model_kwargs={"task_name": "reconstruction"}
        )
        model.init()
        
        # 모델 메서드 확인
        print(f"MOMENT 모델 메서드 목록: {[m for m in dir(model) if not m.startswith('_')]}")
        print(f"MOMENT 모델 task_name: {getattr(model, 'task_name', 'Unknown')}")
        
        # 테스트 데이터 생성
        batch_size = 1
        n_channels = 5  # 특성 개수
        seq_len = 512  # MOMENT 모델의 기본 입력 길이
        
        # 랜덤 데이터 생성
        test_data = torch.randn(batch_size, n_channels, seq_len)
        print(f"테스트 데이터 형태: {test_data.shape}")
        
        # 각 메서드 시도
        print("\n1. reconstruction 메서드 시도...")
        try:
            outputs = model.reconstruction(x_enc=test_data)
            print(f"reconstruction 성공! 결과 타입: {type(outputs)}")
            print(f"결과 속성: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            if hasattr(outputs, 'reconstruction'):
                recon = outputs.reconstruction
                print(f"reconstruction 데이터 형태: {recon.shape}")
                
                # 이상치 점수 계산
                anomaly_scores = torch.mean(torch.abs(test_data - recon), dim=2).squeeze()
                print(f"계산된 이상치 점수 형태: {anomaly_scores.shape}")
        except Exception as e:
            print(f"reconstruction 실패: {e}")
        
        print("\n2. detect_anomalies 메서드 시도...")
        try:
            outputs = model.detect_anomalies(x_enc=test_data)
            print(f"detect_anomalies 성공! 결과 타입: {type(outputs)}")
            print(f"결과 속성: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            if hasattr(outputs, 'anomaly_scores'):
                scores = outputs.anomaly_scores
                print(f"anomaly_scores 형태: {scores.shape}")
        except Exception as e:
            print(f"detect_anomalies 실패: {e}")
        
        print("\n3. reconstruct 메서드 시도...")
        try:
            outputs = model.reconstruct(x_enc=test_data)
            print(f"reconstruct 성공! 결과 타입: {type(outputs)}")
            print(f"결과 속성: {[attr for attr in dir(outputs) if not attr.startswith('_')]}")
            if hasattr(outputs, 'reconstruction'):
                recon = outputs.reconstruction
                print(f"reconstruct 데이터 형태: {recon.shape}")
                
                # 이상치 점수 계산
                anomaly_scores = torch.mean(torch.abs(test_data - recon), dim=2).squeeze()
                print(f"계산된 이상치 점수 형태: {anomaly_scores.shape}")
        except Exception as e:
            print(f"reconstruct 실패: {e}")
            
    except Exception as e:
        print(f"모델 초기화 실패: {e}")
    
    print("\nMOMENT 모델 테스트 완료.")

if __name__ == "__main__":
    test_moment_model()
