from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly
import subprocess
import sys
from pathlib import Path
import asyncio
import jinja2
import logging

# 사용자 정의 모듈 임포트
from moment_detector import detect_anomalies_with_moment_async

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AnomalyApp")

# 데이터 경로
DATA_PATH = 'data/semiconductor_process_data.csv'

# Jinja2 환경 설정 및 사용자 정의 필터 추가
def now_format(value='', format='%Y-%m-%d'):
    return datetime.now().strftime(format)

# 템플릿 설정 (사용자 정의 필터 추가)
templates = Jinja2Templates(directory='templates')
templates.env.filters['now'] = now_format

# 비동기 데이터 로드 함수
async def load_data():
    """데이터를 로드하고 전처리합니다."""
    # 비동기로 처리하기 위해 스레드 풀에서 실행
    loop = asyncio.get_event_loop()
    data = await loop.run_in_executor(None, pd.read_csv, DATA_PATH)
    
    # 타임스탬프 처리
    data['timestamp'] = pd.to_datetime(data.iloc[:, 0])
    data.set_index('timestamp', inplace=True)
    return data

# 시각화 함수들 (비동기 처리를 위한 래퍼)
async def create_time_series_plot_async(data, feature, group_by=None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_time_series_plot, data, feature, group_by)

async def create_distribution_plot_async(data, feature):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_distribution_plot, data, feature)

async def create_heatmap_async(data, features):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_heatmap, data, features)

async def create_anomaly_by_feature_plot_async(data, features):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, create_anomaly_by_feature_plot, data, features)

# 시각화 함수들 (동기 처리)
def create_time_series_plot(data, feature, group_by=None):
    """시계열 데이터의 시각화를 생성합니다."""
    fig = go.Figure()
    
    # 이상치 컬럼이 있는 경우 사용, 없으면 '예측_이상여부' 사용
    anomaly_col = '이상치' if '이상치' in data.columns else '예측_이상여부'
    
    if group_by is not None and group_by in data.columns:
        # 그룹별로 시각화
        groups = data[group_by].unique()
        for group in groups:
            group_data = data[data[group_by] == group]
            
            # 정상 데이터와 이상 데이터 분리
            normal = group_data[group_data[anomaly_col] == 0]
            anomaly = group_data[group_data[anomaly_col] == 1]
            
            # 정상 데이터 플롯
            fig.add_trace(go.Scatter(
                x=normal.index, 
                y=normal[feature],
                mode='lines',
                name=f'{group} - 정상',
                line=dict(color='#4361ee', width=2)
            ))
            
            # 이상 데이터 플롯
            fig.add_trace(go.Scatter(
                x=anomaly.index, 
                y=anomaly[feature],
                mode='markers',
                name=f'{group} - 이상',
                marker=dict(color='#f72585', size=10, symbol='circle')
            ))
    else:
        # 전체 데이터에 대한 시각화
        normal = data[data[anomaly_col] == 0]
        anomaly = data[data[anomaly_col] == 1]
        
        # 정상 데이터 플롯
        fig.add_trace(go.Scatter(
            x=normal.index, 
            y=normal[feature],
            mode='lines',
            name='정상',
            line=dict(color='#4361ee', width=2)
        ))
        
        # 이상 데이터 플롯
        fig.add_trace(go.Scatter(
            x=anomaly.index, 
            y=anomaly[feature],
            mode='markers',
            name='이상',
            marker=dict(color='#f72585', size=10, symbol='circle')
        ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'{feature} 시계열 데이터',
        xaxis_title='시간',
        yaxis_title=feature,
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(247, 249, 252, 0.8)',
        paper_bgcolor='rgba(247, 249, 252, 0)',
    )
    
    # Grid 설정
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    
    return plotly.io.to_json(fig)

def create_distribution_plot(data, feature):
    """특성의 분포를 시각화합니다."""
    fig = go.Figure()
    
    # 이상치 컬럼이 있는 경우 사용, 없으면 '예측_이상여부' 사용
    anomaly_col = '이상치' if '이상치' in data.columns else '예측_이상여부'
    
    # 정상 데이터와 이상 데이터 분리
    normal = data[data[anomaly_col] == 0]
    anomaly = data[data[anomaly_col] == 1]
    
    # 정상 데이터 히스토그램
    fig.add_trace(go.Histogram(
        x=normal[feature],
        name='정상',
        marker_color='#4361ee',
        opacity=0.7,
        nbinsx=20
    ))
    
    # 이상 데이터 히스토그램
    fig.add_trace(go.Histogram(
        x=anomaly[feature],
        name='이상',
        marker_color='#f72585',
        opacity=0.7,
        nbinsx=20
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title=f'{feature} 분포',
        xaxis_title=feature,
        yaxis_title='빈도',
        barmode='overlay',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(247, 249, 252, 0.8)',
        paper_bgcolor='rgba(247, 249, 252, 0)',
    )
    
    # Grid 설정
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    
    return plotly.io.to_json(fig)

def create_heatmap(data, features):
    """특성 간의 상관관계 히트맵을 생성합니다."""
    # 수치형 특성만 선택
    numeric_data = data[features]
    
    # 상관관계 계산
    corr_matrix = numeric_data.corr()
    
    # 히트맵 생성
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.index,
        colorscale=[
            [0.0, "#4cc9f0"],
            [0.5, "#f8f9fa"],
            [1.0, "#f72585"]
        ],
        zmin=-1, zmax=1,
        showscale=True,
    ))
    
    fig.update_layout(
        title='특성 간 상관관계',
        template='plotly_white',
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(247, 249, 252, 0.8)',
        paper_bgcolor='rgba(247, 249, 252, 0)',
    )
    
    return plotly.io.to_json(fig)

def create_anomaly_by_feature_plot(data, features):
    """각 특성별 이상치 비율을 시각화합니다."""
    # 이상치 컬럼이 있는 경우 사용, 없으면 '예측_이상여부' 사용
    anomaly_col = '이상치' if '이상치' in data.columns else '예측_이상여부'
    
    anomaly_counts = []
    
    for feature in features:
        # 각 특성의 값을 4분위로 나누기
        data['quartile'] = pd.qcut(data[feature], 4, labels=False)
        
        # 각 분위별 이상치 비율 계산
        quartile_anomaly_ratio = data.groupby('quartile')[anomaly_col].mean()
        
        # 결과 저장
        for quartile, ratio in quartile_anomaly_ratio.items():
            anomaly_counts.append({
                'feature': feature,
                'quartile': f'Q{quartile+1}',
                'anomaly_ratio': ratio
            })
    
    # 결과를 데이터프레임으로 변환
    ratio_df = pd.DataFrame(anomaly_counts)
    
    # 색상 팔레트 생성
    colors = ['#4361ee', '#4895ef', '#4cc9f0', '#3f37c9', '#f72585']
    
    # 그래프 생성
    fig = go.Figure()
    
    for i, feature in enumerate(features):
        feature_data = ratio_df[ratio_df['feature'] == feature]
        fig.add_trace(go.Bar(
            x=feature_data['quartile'],
            y=feature_data['anomaly_ratio'],
            name=feature,
            marker_color=colors[i % len(colors)]
        ))
    
    fig.update_layout(
        title='특성별 이상치 비율',
        xaxis_title='분위수',
        yaxis_title='이상치 비율',
        barmode='group',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(247, 249, 252, 0.8)',
        paper_bgcolor='rgba(247, 249, 252, 0)',
    )
    
    # Grid 설정
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    
    return plotly.io.to_json(fig)

def create_anomaly_score_plot(data, features, anomaly_col='이상치_점수'):
    """이상치 점수를 시각화합니다."""
    fig = go.Figure()
    
    # 시계열 인덱스
    x = data.index
    
    # 이상치 여부
    anomaly_label = '이상치' if '이상치' in data.columns else '예측_이상여부'
    is_anomaly = data[anomaly_label].astype(bool)
    
    # 정상 데이터 점수
    normal_scores = data[~is_anomaly][anomaly_col]
    
    # 이상 데이터 점수
    anomaly_scores = data[is_anomaly][anomaly_col]
    
    # 이상치 점수 시각화
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[anomaly_col],
        mode='lines',
        name='이상치 점수',
        line=dict(color='#4361ee', width=1.5)
    ))
    
    # 이상 데이터 표시
    fig.add_trace(go.Scatter(
        x=data.index[is_anomaly],
        y=data[anomaly_col][is_anomaly],
        mode='markers',
        name='이상치',
        marker=dict(color='#f72585', size=10, symbol='circle')
    ))
    
    # 임계값 표시
    threshold = np.percentile(data[anomaly_col], 100 * (1 - 0.05))  # 5% 이상치 가정
    fig.add_trace(go.Scatter(
        x=[data.index.min(), data.index.max()],
        y=[threshold, threshold],
        mode='lines',
        name='임계값',
        line=dict(color='#ff9e00', width=2, dash='dash')
    ))
    
    # 레이아웃 설정
    fig.update_layout(
        title='이상치 점수 시계열',
        xaxis_title='시간',
        yaxis_title='이상치 점수',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='rgba(247, 249, 252, 0.8)',
        paper_bgcolor='rgba(247, 249, 252, 0)',
    )
    
    # Grid 설정
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(220, 220, 220, 0.5)'
    )
    
    return plotly.io.to_json(fig)

# 라우트 핸들러 함수들
async def index(request):
    """메인 페이지"""
    # 데이터 파일이 없으면 생성
    if not os.path.exists(DATA_PATH):
        logger.info("데이터 파일을 생성합니다...")
        data_dir = os.path.dirname(DATA_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        subprocess.run([sys.executable, 'data_generator.py'])
    
    data = await load_data()
    
    # 특성 목록 가져오기
    numeric_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categoric_features = data.select_dtypes(include=['object']).columns.tolist()
    
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "numeric_features": numeric_features,
            "categoric_features": categoric_features
        }
    )

async def detect(request):
    """이상치 탐지 처리"""
    # POST 데이터 파싱
    form_data = await request.form()
    features = form_data.getlist('features[]')
    contamination = float(form_data.get('contamination', '0.05'))
    use_moment = form_data.get('use_moment', '') == 'true'  # MOMENT 모델 사용 여부
    
    logger.info(f"이상치 탐지 요청: 특성={features}, 이상치비율={contamination}, MOMENT사용={use_moment}")
    
    # 데이터 로드
    data = await load_data()
    
    # MOMENT 모델을 사용한 이상치 탐지 수행
    start_time = datetime.now()
    result_df, metrics = await detect_anomalies_with_moment_async(
        data, features, contamination, use_moment=use_moment
    )
    processing_time = (datetime.now() - start_time).total_seconds()
    
    logger.info(f"이상치 탐지 완료: 모델={metrics.get('model_type', 'Unknown')}, 처리시간={processing_time:.2f}초")
    
    # 선택된 특성에 대한 시각화 생성
    primary_feature = features[0] if features else data.select_dtypes(include=['float64']).columns[0]
    time_series_plot = await create_time_series_plot_async(result_df, primary_feature)
    distribution_plot = await create_distribution_plot_async(result_df, primary_feature)
    heatmap_plot = await create_heatmap_async(result_df, features)
    anomaly_by_feature_plot = await create_anomaly_by_feature_plot_async(result_df, features)
    
    # 이상치 점수 시각화 추가
    anomaly_score_plot = create_anomaly_score_plot(result_df, features)
    
    # 이상치 탐지 결과 요약
    anomaly_col = '이상치' if '이상치' in result_df.columns else '예측_이상여부'
    anomaly_count = result_df[anomaly_col].sum()
    total_count = len(result_df)
    anomaly_ratio = anomaly_count / total_count
    
    # 이상치가 있는 배치 또는 장비 분석
    batch_anomalies = {}
    if '배치번호' in result_df.columns:
        batch_anomalies = result_df.groupby('배치번호')[anomaly_col].mean().to_dict()
    
    equipment_anomalies = {}
    if '장비ID' in result_df.columns:
        equipment_anomalies = result_df.groupby('장비ID')[anomaly_col].mean().to_dict()
    
    # MOMENT 통계 추가
    moment_stats = {
        'model_used': metrics.get('model_type', 'Unknown'),
        'processing_time': processing_time
    }
    
    # 성능 지표가 있으면 추가
    if 'precision' in metrics:
        moment_stats.update({
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score']
        })
    
    # 결과 반환
    response = {
        'success': True,
        'anomaly_count': int(anomaly_count),
        'total_count': total_count,
        'anomaly_ratio': float(anomaly_ratio),
        'batch_anomalies': batch_anomalies,
        'equipment_anomalies': equipment_anomalies,
        'moment_stats': moment_stats,
        'metrics': metrics,
        'time_series_plot': time_series_plot,
        'distribution_plot': distribution_plot,
        'heatmap_plot': heatmap_plot,
        'anomaly_by_feature_plot': anomaly_by_feature_plot,
        'anomaly_score_plot': anomaly_score_plot
    }
    
    return JSONResponse(response)

async def plot_feature(request):
    """특정 특성에 대한 시각화"""
    feature = request.path_params['feature']
    
    # 데이터 로드
    data = await load_data()
    
    # 이상치 탐지 (간단한 방법으로)
    result_df, _ = await detect_anomalies_with_moment_async(data, [feature])
    
    # 시각화 생성
    time_series_plot = await create_time_series_plot_async(result_df, feature)
    distribution_plot = await create_distribution_plot_async(result_df, feature)
    
    return JSONResponse({
        'time_series_plot': time_series_plot,
        'distribution_plot': distribution_plot
    })

async def group_analysis(request):
    """그룹별 이상치 분석"""
    # POST 데이터 파싱
    form_data = await request.form()
    feature = form_data.get('feature')
    group_by = form_data.get('group_by')
    
    # 데이터 로드
    data = await load_data()
    
    # 이상치 탐지
    result_df, _ = await detect_anomalies_with_moment_async(data, [feature])
    
    # 그룹별 시각화 생성
    time_series_plot = await create_time_series_plot_async(result_df, feature, group_by)
    
    # 그룹별 이상치 비율 계산
    anomaly_col = '이상치' if '이상치' in result_df.columns else '예측_이상여부'
    group_anomaly_ratios = result_df.groupby(group_by)[anomaly_col].mean().to_dict()
    
    return JSONResponse({
        'time_series_plot': time_series_plot,
        'group_anomaly_ratios': group_anomaly_ratios
    })

async def model_compare(request):
    """MOMENT와 IsolationForest 모델 비교"""
    # POST 데이터 파싱
    form_data = await request.form()
    features = form_data.getlist('features[]')
    contamination = float(form_data.get('contamination', '0.05'))
    
    # 데이터 로드
    data = await load_data()
    
    # IsolationForest 이상치 탐지
    isolation_start = datetime.now()
    isolation_df, isolation_metrics = await detect_anomalies_with_moment_async(
        data, features, contamination, use_moment=False
    )
    isolation_time = (datetime.now() - isolation_start).total_seconds()
    
    # MOMENT 이상치 탐지
    moment_start = datetime.now()
    moment_df, moment_metrics = await detect_anomalies_with_moment_async(
        data, features, contamination, use_moment=True
    )
    moment_time = (datetime.now() - moment_start).total_seconds()
    
    # 결과 비교
    if '이상여부' in data.columns:
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        # 실제 이상치와 비교
        true_anomalies = data['이상여부'].values
        
        # IsolationForest 성능
        isolation_predictions = isolation_df['이상치'].values
        isolation_precision = precision_score(true_anomalies, isolation_predictions)
        isolation_recall = recall_score(true_anomalies, isolation_predictions)
        isolation_f1 = f1_score(true_anomalies, isolation_predictions)
        
        # MOMENT 성능
        moment_predictions = moment_df['이상치'].values
        moment_precision = precision_score(true_anomalies, moment_predictions)
        moment_recall = recall_score(true_anomalies, moment_predictions)
        moment_f1 = f1_score(true_anomalies, moment_predictions)
        
        # 결과 반환
        return JSONResponse({
            'success': True,
            'isolation_forest': {
                'anomaly_count': int(isolation_df['이상치'].sum()),
                'processing_time': isolation_time,
                'precision': float(isolation_precision),
                'recall': float(isolation_recall),
                'f1_score': float(isolation_f1)
            },
            'moment': {
                'anomaly_count': int(moment_df['이상치'].sum()),
                'processing_time': moment_time,
                'precision': float(moment_precision),
                'recall': float(moment_recall),
                'f1_score': float(moment_f1),
                'model_type': moment_metrics.get('model_type', 'Unknown')
            }
        })
    else:
        # 레이블이 없는 경우 일치도 계산
        agreement_ratio = np.mean(isolation_df['이상치'] == moment_df['이상치'])
        
        # 결과 반환
        return JSONResponse({
            'success': True,
            'isolation_forest': {
                'anomaly_count': int(isolation_df['이상치'].sum()),
                'processing_time': isolation_time
            },
            'moment': {
                'anomaly_count': int(moment_df['이상치'].sum()),
                'processing_time': moment_time,
                'model_type': moment_metrics.get('model_type', 'Unknown')
            },
            'agreement_ratio': float(agreement_ratio)
        })

# 라우트 설정
routes = [
    Route('/', index),
    Route('/detect', detect, methods=['POST']),
    Route('/plot/{feature:str}', plot_feature),
    Route('/group_analysis', group_analysis, methods=['POST']),
    Route('/model_compare', model_compare, methods=['POST']),
    Mount('/static', StaticFiles(directory='static'), name='static')
]

# CORS 미들웨어 설정
middleware = [
    Middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'])
]

# Starlette 애플리케이션 생성
app = Starlette(
    debug=True,
    routes=routes,
    middleware=middleware
)

if __name__ == '__main__':
    # 데이터 폴더 확인 및 생성
    if not os.path.exists('data'):
        os.makedirs('data')
    
    # Uvicorn으로 서버 실행
    uvicorn.run(app, host='0.0.0.0', port=5000)
