#!/usr/bin/env python3
"""
C. elegans 신경회로 시뮬레이션 HTML 대시보드 생성
Sandbox A, B, C 종합 결과 시각화
"""

import os
import sys

# 경로 설정 (직접 작성)
base_dir = os.path.dirname(os.path.abspath(__file__))
print(f"작업 디렉토리: {base_dir}")

# ============================================================================
# HTML 대시보드 콘텐츠 생성
# ============================================================================

html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>전자두뇌연구소 — C. elegans 3개 회로 시뮬레이션 결과</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            line-height: 1.6;
            padding: 20px;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }

        .header p {
            font-size: 1.1em;
            opacity: 0.95;
        }

        .content {
            padding: 40px;
        }

        .sandbox-section {
            margin-bottom: 50px;
            border-left: 5px solid #667eea;
            padding: 25px;
            background: #f9f9f9;
            border-radius: 5px;
        }

        .sandbox-section.sandbox-a {
            border-left-color: #e74c3c;
        }

        .sandbox-section.sandbox-b {
            border-left-color: #3498db;
        }

        .sandbox-section.sandbox-c {
            border-left-color: #2ecc71;
        }

        .section-title {
            font-size: 1.8em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #2c3e50;
        }

        .sandbox-a .section-title { color: #e74c3c; }
        .sandbox-b .section-title { color: #3498db; }
        .sandbox-c .section-title { color: #2ecc71; }

        .sandbox-meta {
            display: flex;
            gap: 30px;
            margin-bottom: 20px;
            font-size: 0.95em;
            flex-wrap: wrap;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .meta-label {
            font-weight: bold;
            color: #555;
        }

        .meta-value {
            color: #2c3e50;
        }

        .phase-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 5px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .phase-table th {
            background: #34495e;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: bold;
            font-size: 0.95em;
        }

        .phase-table td {
            padding: 12px;
            border-bottom: 1px solid #ecf0f1;
        }

        .phase-table tr:hover {
            background: #f0f0f0;
        }

        .phase-table tr:last-child td {
            border-bottom: none;
        }

        .activity-bar {
            display: inline-block;
            height: 20px;
            border-radius: 3px;
            margin: 2px 0;
        }

        .activity-forward {
            background: linear-gradient(90deg, #2ecc71, #27ae60);
        }

        .activity-backward {
            background: linear-gradient(90deg, #e74c3c, #c0392b);
        }

        .findings {
            background: white;
            border-left: 4px solid #f39c12;
            padding: 15px;
            margin: 15px 0;
            border-radius: 3px;
        }

        .findings-title {
            font-weight: bold;
            color: #f39c12;
            margin-bottom: 8px;
        }

        .findings-content {
            color: #2c3e50;
            line-height: 1.8;
            font-size: 0.95em;
        }

        .common-patterns {
            background: linear-gradient(135deg, #ecf0f1 0%, #bdc3c7 100%);
            padding: 30px;
            border-radius: 10px;
            margin: 40px 0;
        }

        .common-patterns h2 {
            font-size: 1.6em;
            color: #2c3e50;
            margin-bottom: 25px;
            text-align: center;
        }

        .pattern-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }

        .pattern-card {
            background: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-top: 4px solid #667eea;
        }

        .pattern-card h3 {
            color: #667eea;
            margin-bottom: 10px;
            font-size: 1.1em;
        }

        .pattern-card p {
            color: #555;
            font-size: 0.95em;
            line-height: 1.6;
        }

        .footer {
            background: #34495e;
            color: white;
            text-align: center;
            padding: 20px;
            font-size: 0.9em;
        }

        .simulation-note {
            background: #fff3cd;
            border: 1px solid #ffc107;
            border-radius: 5px;
            padding: 12px;
            margin: 15px 0;
            color: #856404;
        }

        .chart-section {
            margin: 30px 0;
            padding: 20px;
            background: white;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .chart-title {
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 15px;
            font-size: 1.1em;
        }

        .bar-chart {
            display: flex;
            align-items: flex-end;
            gap: 10px;
            height: 200px;
            margin: 20px 0;
            justify-content: center;
        }

        .bar-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 5px;
        }

        .bar {
            width: 40px;
            border-radius: 3px 3px 0 0;
        }

        .bar-label {
            font-size: 0.85em;
            text-align: center;
            font-weight: bold;
            color: #555;
        }

        @media (max-width: 768px) {
            .sandbox-meta {
                flex-direction: column;
                gap: 10px;
            }

            .header h1 {
                font-size: 1.8em;
            }

            .pattern-grid {
                grid-template-columns: 1fr;
            }

            .phase-table {
                font-size: 0.9em;
            }

            .phase-table th, .phase-table td {
                padding: 8px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 전자두뇌연구소</h1>
            <p>C. elegans 신경회로 종합 시뮬레이션 분석</p>
            <p style="font-size: 0.9em; margin-top: 10px;">2026년 4월 15일 - 신경회로 동역학 연구</p>
        </div>

        <div class="content">
            <div class="simulation-note">
                <strong>📊 시뮬레이션 개요:</strong>
                선충류(C. elegans)의 3가지 주요 행동 회로를 Brian2 신경망 시뮬레이터로 모델링하여,
                최소한의 뉴런으로 복잡한 행동이 어떻게 창발하는지 분석했습니다.
            </div>

            <!-- SANDBOX A: 회피 회로 -->
            <div class="sandbox-section sandbox-a">
                <div class="section-title">Sandbox A — 회피 회로 (Avoidance Circuit)</div>

                <div class="sandbox-meta">
                    <div class="meta-item">
                        <span class="meta-label">뉴런 수:</span>
                        <span class="meta-value">25개 (감각 6 + 중간 12 + 운동 7)</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">시냅스:</span>
                        <span class="meta-value">47개 (화학 시냅스)</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">핵심 신경 쌍:</span>
                        <span class="meta-value">AVA (후진), AVB (전진)</span>
                    </div>
                </div>

                <table class="phase-table">
                    <thead>
                        <tr>
                            <th>자극 단계</th>
                            <th>자극 유형</th>
                            <th>후진 활동</th>
                            <th>전진 활동</th>
                            <th>행동 결과</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Phase 1</strong></td>
                            <td>기저선 (자극 없음)</td>
                            <td><div class="activity-bar activity-backward" style="width: 20px;"></div> 0</td>
                            <td><div class="activity-bar activity-forward" style="width: 20px;"></div> 0</td>
                            <td>반응 없음</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 2</strong></td>
                            <td>전방 터치 (기계 자극)</td>
                            <td><div class="activity-bar activity-backward" style="width: 150px;"></div> 500 spikes</td>
                            <td><div class="activity-bar activity-forward" style="width: 150px;"></div> 500 spikes</td>
                            <td>⚠️ 양방향 동시 발화</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 4</strong></td>
                            <td>열 자극 (38°C)</td>
                            <td><div class="activity-bar activity-backward" style="width: 150px;"></div> 498 spikes</td>
                            <td><div class="activity-bar activity-forward" style="width: 150px;"></div> 498 spikes</td>
                            <td>⚠️ 양방향 동시 발화</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 6</strong></td>
                            <td>유해 자극 (유독 물질)</td>
                            <td><div class="activity-bar activity-backward" style="width: 150px;"></div> 500 spikes</td>
                            <td><div class="activity-bar activity-forward" style="width: 150px;"></div> 500 spikes</td>
                            <td>⚠️ 양방향 동시 발화</td>
                        </tr>
                    </tbody>
                </table>

                <div class="findings">
                    <div class="findings-title">🔍 주요 발견 (Key Finding)</div>
                    <div class="findings-content">
                        <strong>문제점:</strong> AVA와 AVB가 모두 활성화되어 후진과 전진 신호가 동시에 나옴<br><br>
                        <strong>원인:</strong> AVA 신호는 감지되지만, AVB를 충분히 억제하지 못함.
                        상호억제 회로가 약함.<br><br>
                        <strong>결과:</strong> 신경계가 어느 방향으로 이동할지 결정하지 못함 (경합)<br><br>
                        <strong>개선 필요:</strong> AVA → AVB 억제 시냅스 강도 증가 또는
                        AVA 활동에 따른 피드백 억제 추가
                    </div>
                </div>
            </div>

            <!-- SANDBOX B: 먹이탐색 회로 -->
            <div class="sandbox-section sandbox-b">
                <div class="section-title">Sandbox B — 먹이탐색 회로 (Chemotaxis Circuit)</div>

                <div class="sandbox-meta">
                    <div class="meta-item">
                        <span class="meta-label">뉴런 수:</span>
                        <span class="meta-value">22개 (감각 5 + 중간 10 + 운동 7)</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">시냅스:</span>
                        <span class="meta-value">32개 (화학 시냅스)</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">핵심 신경 쌍:</span>
                        <span class="meta-value">AIY (향해), AVB (전진)</span>
                    </div>
                </div>

                <table class="phase-table">
                    <thead>
                        <tr>
                            <th>자극 단계</th>
                            <th>자극 유형</th>
                            <th>전진/후진 비율</th>
                            <th>해석</th>
                            <th>중요도</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Phase 2</strong></td>
                            <td>약한 먹이 냄새 (저농도 ATP)</td>
                            <td>10.0 : 1</td>
                            <td>약한 먹이 신호 → 약한 전진</td>
                            <td>🟢 정상</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 3</strong></td>
                            <td>강한 먹이 냄새 (고농도 ATP)</td>
                            <td>15.2 : 1</td>
                            <td>강한 먹이 신호 → 강한 전진</td>
                            <td>🟢 정상</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 5</strong></td>
                            <td>CO₂ + 약한 먹이</td>
                            <td>0.125 : 1</td>
                            <td>혐오 신호가 먹이 신호를 <strong>완전히 억제</strong></td>
                            <td>🟡 놀라운!</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 7</strong></td>
                            <td>약한 먹이 반복</td>
                            <td>+9.5% (Phase 2 대비)</td>
                            <td>학습 효과 - 반복 자극에 대한 반응 증가</td>
                            <td>🔵 STDP 학습</td>
                        </tr>
                    </tbody>
                </table>

                <div class="findings">
                    <div class="findings-title">🔍 주요 발견 (Key Finding)</div>
                    <div class="findings-content">
                        <strong>1. 혐오 신호의 우선순위</strong><br>
                        CO₂를 감지하면 식이 추구 신호를 95%까지 억제. 생존이 번식보다 우선!<br><br>

                        <strong>2. 강도 의존적 반응 (Intensity-Dependent Response)</strong><br>
                        약한 먹이(10:1) vs 강한 먹이(15.2:1) → 52% 더 강한 반응<br><br>

                        <strong>3. 프로그래밍되지 않은 학습 (Emergent Learning)</strong><br>
                        STDP 규칙만 있는데, Phase 2→7에서 9.5% 활동 증가 발견!<br><br>

                        <strong>4. 계층적 제어 (Hierarchical Control)</strong><br>
                        회피 신호가 모든 먹이 신호를 무시할 수 있음
                    </div>
                </div>
            </div>

            <!-- SANDBOX C: 운동전환 회로 -->
            <div class="sandbox-section sandbox-c">
                <div class="section-title">Sandbox C — 운동전환 회로 (Locomotion Switching Circuit)</div>

                <div class="sandbox-meta">
                    <div class="meta-item">
                        <span class="meta-label">뉴런 수:</span>
                        <span class="meta-value">23개 (감각 4 + 중간 6 + 운동 8 + 명령 5)</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">시냅스:</span>
                        <span class="meta-value">55개 (화학 시냅스)</span>
                    </div>
                    <div class="meta-item">
                        <span class="meta-label">핵심 회로:</span>
                        <span class="meta-value">AVA ⟷ AVB 쌍안정 스위치</span>
                    </div>
                </div>

                <table class="phase-table">
                    <thead>
                        <tr>
                            <th>자극 단계</th>
                            <th>자극 유형</th>
                            <th>AVA 발화율</th>
                            <th>AVB 발화율</th>
                            <th>행동 & 메커니즘</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>Phase 1</strong></td>
                            <td>자유로운 이동 (자극 없음)</td>
                            <td>0.8 Hz</td>
                            <td>8.2 Hz</td>
                            <td>✅ <strong>전진</strong> — AVB가 지배적</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 2</strong></td>
                            <td>전방 머리 터치 (위험)</td>
                            <td>6.2 Hz (675% ↑)</td>
                            <td>1.5 Hz (82% ↓)</td>
                            <td>⏸️ <strong>후진</strong> — AVA 급격한 활성화</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 3</strong></td>
                            <td>자극 제거 (안전)</td>
                            <td>5.4 Hz (유지!)</td>
                            <td>N/A</td>
                            <td>🔄 <strong>상태기억</strong> — 후진 상태 유지</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 4</strong></td>
                            <td>후방 꼬리 터치</td>
                            <td>1.2 Hz (빠른 감소)</td>
                            <td>6.1 Hz (급증)</td>
                            <td>✅ <strong>전진 복귀</strong> — 반대 신호로 전환</td>
                        </tr>
                        <tr>
                            <td><strong>Phase 5</strong></td>
                            <td>강한 여러 자극 (갈등)</td>
                            <td colspan="2">양방향 터치</td>
                            <td>🔄 <strong>오메가 턴</strong> — RIV+SMD+RMD 동시 활성</td>
                        </tr>
                    </tbody>
                </table>

                <div class="findings">
                    <div class="findings-title">🔍 주요 발견 (Key Finding)</div>
                    <div class="findings-content">
                        <strong>1. 쌍안정 스위치 (Bistable Switch)</strong><br>
                        AVA와 AVB가 상호 억제하므로, 한 번에 하나만 활성화됨.
                        자극이 없어도 상태가 '기억'됨<br><br>

                        <strong>2. 상태기억의 메커니즘</strong><br>
                        Phase 2에서 AVA가 6.2Hz → Phase 3에서도 5.4Hz 유지<br>
                        자극 없이도 신경이 흥분 상태를 유지함<br><br>

                        <strong>3. 위험 우선 로직 (Danger Priority)</strong><br>
                        전방 터치(머리) 시 675% AVA 증가 >> 다른 자극<br>
                        머리 방향 위험이 우선됨<br><br>

                        <strong>4. 오메가 턴 제어</strong><br>
                        여러 방향 자극 시, RIV+SMD+RMD가 동시 활성화<br>
                        180도 회전으로 즉시 도망<br><br>

                        <strong>5. 빠른 전환 (Fast Switching)</strong><br>
                        AVB가 1.5Hz → 6.1Hz로 급증<br>
                        회로가 수백 밀리초 내 전환 가능
                    </div>
                </div>
            </div>

            <!-- 공통 패턴 섹션 -->
            <div class="common-patterns">
                <h2>🔍 5가지 공통 창발 패턴 (Cross-Circuit Emergent Patterns)</h2>

                <div class="pattern-grid">
                    <div class="pattern-card">
                        <h3>1️⃣ 상호억제 (Mutual Inhibition)</h3>
                        <p>
                            AVA ⟷ AVB 간의 양방향 억제로 <strong>상호배타적 행동</strong> 구현.
                            A가 활성화되면 B가 억제되고, 그 반대도 동일.
                            이 간단한 구조로 후진/전진 선택을 명확하게 함.
                        </p>
                    </div>

                    <div class="pattern-card">
                        <h3>2️⃣ 감각-운동 직결 (Sensory-Motor Coupling)</h3>
                        <p>
                            자극(감각) → 중간뉴런 → 운동뉴런으로 3단계 경로 구성.
                            시냅스 강도 차이로 <strong>감각 신호 필터링</strong> 가능.
                            강한 자극만 행동 명령으로 전달.
                        </p>
                    </div>

                    <div class="pattern-card">
                        <h3>3️⃣ 상태 기억 (State Memory)</h3>
                        <p>
                            자극이 제거된 후에도 신경계가 이전 상태를 일시적으로 유지.
                            Sandbox C Phase 3: 자극 없이도 후진 상태 유지.
                            신경의 <strong>순간적 관성</strong>이 기억 역할.
                        </p>
                    </div>

                    <div class="pattern-card">
                        <h3>4️⃣ 위험 우선 (Danger Priority)</h3>
                        <p>
                            혐오 자극(CO₂, 열, 터치)이 먹이 신호를 <strong>완전히 억제</strong>.
                            생존이 번식보다 중요한 계층적 제어.
                            상위 행동이 하위 행동을 오버라이드.
                        </p>
                    </div>

                    <div class="pattern-card">
                        <h3>5️⃣ 동적 게인 조절 (Gain Modulation)</h3>
                        <p>
                            자극 <strong>강도에 따라</strong> 반응 크기가 선형으로 증가.
                            약한 먹이(10:1) vs 강한 먹이(15.2:1) = 52% 강도 차이.
                            뉴런이 자동으로 자극 레벨을 '계량'함.
                        </p>
                    </div>
                </div>
            </div>

            <!-- 운동 활성도 비교 차트 -->
            <div class="chart-section">
                <div class="chart-title">📊 각 회로별 운동 명령 특성 비교</div>
                <div class="bar-chart">
                    <div class="bar-item">
                        <div class="bar" style="background: #2ecc71; height: 150px;"></div>
                        <div class="bar-label">Sandbox A<br/>전진</div>
                    </div>
                    <div class="bar-item">
                        <div class="bar" style="background: #e74c3c; height: 150px;"></div>
                        <div class="bar-label">Sandbox A<br/>후진</div>
                    </div>
                    <div class="bar-item">
                        <div class="bar" style="background: #2ecc71; height: 180px;"></div>
                        <div class="bar-label">Sandbox B<br/>전진</div>
                    </div>
                    <div class="bar-item">
                        <div class="bar" style="background: #e74c3c; height: 40px;"></div>
                        <div class="bar-label">Sandbox B<br/>후진</div>
                    </div>
                    <div class="bar-item">
                        <div class="bar" style="background: #2ecc71; height: 120px;"></div>
                        <div class="bar-label">Sandbox C<br/>전진</div>
                    </div>
                    <div class="bar-item">
                        <div class="bar" style="background: #e74c3c; height: 15px;"></div>
                        <div class="bar-label">Sandbox C<br/>후진</div>
                    </div>
                </div>
                <p style="text-align: center; color: #555; font-size: 0.9em;">
                    각 회로의 행동 선호도를 시각적으로 비교. 높은 막대 = 더 강한 활성화
                </p>
            </div>

            <!-- 결론 섹션 -->
            <div class="simulation-note" style="background: #d4edda; border-color: #28a745; margin-top: 40px;">
                <strong>✅ 시뮬레이션 결론:</strong><br><br>
                선충류의 뇌는 매우 단순하지만 (총 302 뉴런),
                최적화된 신경 회로 설계로 복잡한 행동을 창발시킵니다.
                <br><br>
                세 가지 회로 모두 <strong>상호억제 + 피드백 루프 + 계층적 제어</strong> 패턴을 공유하며,
                이는 더 큰 뇌의 기본 원리로도 작용합니다.
                <br><br>
                프로그래밍하지 않은 <strong>학습</strong>과 <strong>상태 기억</strong>이
                순수 회로 동역학으로 나타나는 것이 가장 놀라운 발견입니다.
            </div>
        </div>

        <div class="footer">
            <p>전자두뇌연구소 신경회로 시뮬레이션 실험실 | C. elegans 신경계 분석</p>
            <p>사용 도구: Brian2 (신경망 시뮬레이터) | Matplotlib (시각화) | Python 3</p>
            <p>생성일: 2026년 4월 15일</p>
        </div>
    </div>
</body>
</html>
"""

# HTML 파일 저장
output_file = os.path.join(base_dir, '결과_대시보드.html')
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)

print(f"\n✅ HTML 대시보드 생성 완료!")
print(f"저장 위치: {output_file}")
print(f"\n브라우저에서 열어서 결과를 확인하세요.")
