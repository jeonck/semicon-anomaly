$(document).ready(function() {
    // 이상치 비율 슬라이더 값 업데이트
    $('#contamination').on('input', function() {
        const value = $(this).val();
        $('#contaminationValue').text(Math.round(value * 100) + '%');
    });
    
    // MOMENT 모델 사용 여부에 따른 UI 업데이트
    $('#use_moment').on('change', function() {
        const useMoment = $(this).is(':checked');
        
        if (useMoment) {
            $('#modelInfo').html(`
                <div class="alert alert-info">
                    <i class="bx bx-info-circle"></i> MOMENT 모델이 활성화되었습니다. 딥러닝 기반 시계열 파운데이션 모델로, 복잡한 패턴을 더 잘 검출할 수 있습니다.
                </div>
            `);
        } else {
            $('#modelInfo').html(`
                <div class="alert alert-warning">
                    <i class="bx bx-info-circle"></i> MOMENT 모델이 비활성화되었습니다. 기본 IsolationForest 알고리즘이 사용됩니다.
                </div>
            `);
        }
    });
    
    // 이상치 탐지 폼 제출
    $('#detectForm').on('submit', function(e) {
        e.preventDefault();
        
        // 선택된 특성이 있는지 확인
        if ($('#features').val().length === 0) {
            alert('분석할 특성을 하나 이상 선택해주세요.');
            return;
        }
        
        // 로딩 표시 및 결과 패널 초기화
        $('#loadingIndicator').removeClass('d-none');
        $('#resultsPanel').addClass('d-none');
        $('#groupResultsPanel').addClass('d-none');
        $('#compareResultsPanel').addClass('d-none');
        
        // FormData 객체 생성을 통한 다중 선택 값 처리
        const formData = new FormData(this);
        
        // MOMENT 모델 사용 여부 확인
        const useMoment = document.getElementById('use_moment').checked;
        if (useMoment) {
            formData.append('use_moment', 'true');
        } else {
            formData.append('use_moment', 'false');
        }
        
        // AJAX 요청
        $.ajax({
            url: '/detect',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // 로딩 표시 숨기기
                $('#loadingIndicator').addClass('d-none');
                
                if (response.success) {
                    // 결과 패널 표시
                    $('#resultsPanel').removeClass('d-none');
                    
                    // 이상치 통계 업데이트
                    $('#anomalyCount').text(response.anomaly_count);
                    $('#totalCount').text(response.total_count);
                    $('#anomalyRatio').text((response.anomaly_ratio * 100).toFixed(2) + '%');
                    
                    // MOMENT 모델 정보 표시
                    if (response.moment_stats) {
                        let modelHtml = `
                            <div class="alert alert-${useMoment ? 'info' : 'secondary'} mt-3">
                                <p class="mb-1"><i class='bx bx-chip me-1'></i> <strong>사용된 모델:</strong> ${response.moment_stats.model_used}</p>
                                <p class="mb-1"><i class='bx bx-time me-1'></i> <strong>처리 시간:</strong> ${response.moment_stats.processing_time.toFixed(2)}초</p>
                            </div>
                        `;
                        $('#modelInfoList').html(modelHtml);
                    }
                    
                    // 성능 지표 업데이트 (있는 경우)
                    if (response.metrics && 'precision' in response.metrics) {
                        $('#precision').text((response.metrics.precision * 100).toFixed(2) + '%');
                        $('#recall').text((response.metrics.recall * 100).toFixed(2) + '%');
                        $('#f1Score').text((response.metrics.f1_score * 100).toFixed(2) + '%');
                        $('#metricsContainer').parent().parent().show();
                    } else {
                        $('#metricsContainer').parent().parent().hide();
                    }
                    
                    // 시각화 그래프 업데이트
                    updatePlot('timeSeriesPlot', response.time_series_plot);
                    updatePlot('distributionPlot', response.distribution_plot);
                    updatePlot('heatmapPlot', response.heatmap_plot);
                    updatePlot('anomalyByFeaturePlot', response.anomaly_by_feature_plot);
                    updatePlot('anomalyScorePlot', response.anomaly_score_plot);
                    
                    // 탭 활성화
                    $('#time-series-tab').tab('show');
                } else {
                    alert('이상치 탐지 중 오류가 발생했습니다.');
                }
            },
            error: function() {
                $('#loadingIndicator').addClass('d-none');
                alert('서버 요청 중 오류가 발생했습니다.');
            }
        });
    });
    
    // 그룹별 분석 폼 제출
    $('#groupAnalysisForm').on('submit', function(e) {
        e.preventDefault();
        
        // 로딩 표시 및 결과 패널 초기화
        $('#loadingIndicator').removeClass('d-none');
        $('#groupResultsPanel').addClass('d-none');
        
        // FormData 객체 생성
        const formData = new FormData(this);
        
        // AJAX 요청
        $.ajax({
            url: '/group_analysis',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // 로딩 표시 숨기기
                $('#loadingIndicator').addClass('d-none');
                
                // 결과 패널 표시
                $('#groupResultsPanel').removeClass('d-none');
                
                // 시각화 그래프 업데이트
                updatePlot('groupTimeSeriesPlot', response.time_series_plot);
                
                // 그룹별 이상치 비율 테이블 생성
                let tableHtml = '<table class="table table-striped">';
                tableHtml += '<thead><tr><th>그룹</th><th>이상치 비율</th></tr></thead>';
                tableHtml += '<tbody>';
                
                for (const [group, ratio] of Object.entries(response.group_anomaly_ratios)) {
                    tableHtml += `<tr><td>${group}</td><td>${(ratio * 100).toFixed(2)}%</td></tr>`;
                }
                
                tableHtml += '</tbody></table>';
                $('#groupAnomalyRatios').html(tableHtml);
                
                // 스크롤 이동
                $('html, body').animate({
                    scrollTop: $('#groupResultsPanel').offset().top - 20
                }, 500);
            },
            error: function() {
                $('#loadingIndicator').addClass('d-none');
                alert('서버 요청 중 오류가 발생했습니다.');
            }
        });
    });
    
    // 모델 비교 버튼 클릭 이벤트
    $('#compareModelsBtn').on('click', function() {
        // 선택된 특성이 있는지 확인
        if ($('#features').val().length === 0) {
            alert('분석할 특성을 하나 이상 선택해주세요.');
            return;
        }
        
        // 로딩 표시 및 결과 패널 초기화
        $('#loadingIndicator').removeClass('d-none');
        $('#compareResultsPanel').addClass('d-none');
        
        // FormData 객체 생성
        const formData = new FormData($('#detectForm')[0]);
        
        // AJAX 요청
        $.ajax({
            url: '/model_compare',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                // 로딩 표시 숨기기
                $('#loadingIndicator').addClass('d-none');
                
                if (response.success) {
                    // 결과 패널 표시
                    $('#compareResultsPanel').removeClass('d-none');
                    
                    // 비교 결과 표시
                    let resultHtml = `
                        <div class="row">
                            <div class="col-md-6">
                                <div class="card bg-light mb-3">
                                    <div class="card-header bg-secondary text-white">
                                        <i class='bx bx-tree me-1'></i> IsolationForest 모델
                                    </div>
                                    <div class="card-body">
                                        <p><i class='bx bx-error-circle me-1 text-danger'></i> <strong>이상치 수:</strong> ${response.isolation_forest.anomaly_count}</p>
                                        <p><i class='bx bx-time me-1 text-primary'></i> <strong>처리 시간:</strong> ${response.isolation_forest.processing_time.toFixed(2)}초</p>
                    `;
                    
                    // 성능 지표 추가 (있는 경우)
                    if (response.isolation_forest.precision !== undefined) {
                        resultHtml += `
                            <p><i class='bx bx-bullseye me-1 text-danger'></i> <strong>정밀도:</strong> ${(response.isolation_forest.precision * 100).toFixed(2)}%</p>
                            <p><i class='bx bx-receipt me-1 text-success'></i> <strong>재현율:</strong> ${(response.isolation_forest.recall * 100).toFixed(2)}%</p>
                            <p><i class='bx bx-bar-chart-alt-2 me-1 text-primary'></i> <strong>F1 점수:</strong> ${(response.isolation_forest.f1_score * 100).toFixed(2)}%</p>
                        `;
                    }
                    
                    resultHtml += `
                                    </div>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="card bg-light mb-3">
                                    <div class="card-header bg-info text-white">
                                        <i class='bx bx-analyse me-1'></i> MOMENT 모델
                                    </div>
                                    <div class="card-body">
                                        <p><i class='bx bx-error-circle me-1 text-danger'></i> <strong>이상치 수:</strong> ${response.moment.anomaly_count}</p>
                                        <p><i class='bx bx-time me-1 text-primary'></i> <strong>처리 시간:</strong> ${response.moment.processing_time.toFixed(2)}초</p>
                                        <p><i class='bx bx-chip me-1 text-info'></i> <strong>모델 타입:</strong> ${response.moment.model_type}</p>
                    `;
                    
                    // 성능 지표 추가 (있는 경우)
                    if (response.moment.precision !== undefined) {
                        resultHtml += `
                            <p><i class='bx bx-bullseye me-1 text-danger'></i> <strong>정밀도:</strong> ${(response.moment.precision * 100).toFixed(2)}%</p>
                            <p><i class='bx bx-receipt me-1 text-success'></i> <strong>재현율:</strong> ${(response.moment.recall * 100).toFixed(2)}%</p>
                            <p><i class='bx bx-bar-chart-alt-2 me-1 text-primary'></i> <strong>F1 점수:</strong> ${(response.moment.f1_score * 100).toFixed(2)}%</p>
                        `;
                    }
                    
                    resultHtml += `
                                    </div>
                                </div>
                            </div>
                        </div>
                    `;
                    
                    // 일치율 추가 (레이블이 없는 경우)
                    if (response.agreement_ratio !== undefined) {
                        resultHtml += `
                            <div class="alert alert-warning mt-3">
                                <i class='bx bx-info-circle me-2'></i>
                                <strong>모델 간 일치율:</strong> ${(response.agreement_ratio * 100).toFixed(2)}%
                                <p class="mb-0 mt-2 small">참고: 실제 레이블이 없는 경우 두 모델이 얼마나 유사한 결과를 내는지 나타냅니다.</p>
                            </div>
                        `;
                    }
                    
                    // 성능 비교 요약
                    if (response.moment.f1_score !== undefined && response.isolation_forest.f1_score !== undefined) {
                        const f1Diff = response.moment.f1_score - response.isolation_forest.f1_score;
                        const speedDiff = response.isolation_forest.processing_time - response.moment.processing_time;
                        
                        let betterModel = '';
                        let performanceSummary = '';
                        
                        if (f1Diff > 0.05) {
                            betterModel = 'MOMENT';
                            performanceSummary = `MOMENT 모델이 IsolationForest보다 F1 점수가 ${(f1Diff * 100).toFixed(2)}% 더 높습니다.`;
                        } else if (f1Diff < -0.05) {
                            betterModel = 'IsolationForest';
                            performanceSummary = `IsolationForest 모델이 MOMENT보다 F1 점수가 ${(Math.abs(f1Diff) * 100).toFixed(2)}% 더 높습니다.`;
                        } else {
                            betterModel = '두 모델 모두';
                            performanceSummary = '두 모델의 F1 점수가 유사합니다 (5% 이내 차이).';
                        }
                        
                        let speedSummary = '';
                        if (speedDiff > 1) {
                            speedSummary = `IsolationForest가 MOMENT보다 ${speedDiff.toFixed(2)}초 더 느립니다.`;
                        } else if (speedDiff < -1) {
                            speedSummary = `MOMENT가 IsolationForest보다 ${Math.abs(speedDiff).toFixed(2)}초 더 느립니다.`;
                        } else {
                            speedSummary = '두 모델의 처리 시간이 유사합니다 (1초 이내 차이).';
                        }
                        
                        resultHtml += `
                            <div class="alert alert-success mt-3">
                                <i class='bx bx-check-double me-2'></i>
                                <strong>성능 분석 요약:</strong><br>
                                ${performanceSummary}<br>
                                ${speedSummary}<br>
                                <span class="fw-bold mt-2 d-block">추천 모델: ${betterModel}</span>
                            </div>
                        `;
                    }
                    
                    $('#compareResultsContent').html(resultHtml);
                    
                    // 스크롤 이동
                    $('html, body').animate({
                        scrollTop: $('#compareResultsPanel').offset().top - 20
                    }, 500);
                } else {
                    alert('모델 비교 중 오류가 발생했습니다.');
                }
            },
            error: function() {
                $('#loadingIndicator').addClass('d-none');
                alert('서버 요청 중 오류가 발생했습니다.');
            }
        });
    });
    
    // 특성 선택에 따른 시각화 업데이트
    $('#features').on('change', function() {
        const selectedValue = $(this).val();
        
        // 단일 특성 선택 시 그룹별 분석의 특성 선택도 업데이트
        if (selectedValue && selectedValue.length === 1) {
            $('#groupFeature').val(selectedValue[0]);
        }
    });
    
    // 탭 전환 이벤트
    $('#visualizationTabs button').on('click', function(e) {
        e.preventDefault();
        $(this).tab('show');
    });
});

// Plotly 그래프 업데이트 함수
function updatePlot(elementId, plotData) {
    if (plotData) {
        try {
            const figure = JSON.parse(plotData);
            Plotly.newPlot(elementId, figure.data, figure.layout);
        } catch (error) {
            console.error('그래프 업데이트 중 오류 발생:', error);
            $(`#${elementId}`).html('<div class="alert alert-danger">그래프를 로드할 수 없습니다.</div>');
        }
    }
}
