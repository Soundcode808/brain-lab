# 뉴로모픽 하드웨어 & 스파이킹 신경망(SNN) 프로젝트 심층조사

**조사자**: AG-011 (AI심층분석 에이전트)  
**조사일**: 2026-04-15  
**대상**: 디지털 뇌 구현의 신경동역학 기반 접근법 연구

---

## 개요

뉴로모픽 컴퓨팅은 생물학적 뉴런의 **시간 기반, 스파이크 기반** 특성을 하드웨어 수준에서 복제하는 접근법이다. 기존 인공신경망의 연속값(rate-based) 방식과 달리, 스파이킹 신경망(SNN)은 정시(spike timing), 피드백, 동적 임계값을 본질적으로 구현한다.

---

## 1. Intel Loihi / Loihi 2

### WHO (팀 / 기관)
- **Intel Labs** — 신경형 컴퓨팅 연구 주도
- **Sandia National Laboratories** — 뇌 연결도(connectome) 구현 실증
- **Ericsson Research** — 통신 AI 응용

### APPROACH (신경스파이크 모델링 방식)

**Loihi 2**는 생물학적 신경 역동성을 디지털 CMOS(Intel 4, 7nm 공정)로 구현한 **프로그래밍 가능한 신경형 칩**이다.

- 칩당 120–152개 신경형 코어 보유
- 스파이크 타이밍 기반 계산 (시간 기반)
- 이벤트 구동(event-driven) 아키텍처
- 국소 학습 규칙 지원 (Spike-Timing Dependent Plasticity, STDP 계열)

### SCALE (뉴런 수)
- **Hala Point** (2024): 1.15 억 개 뉴런, 1,152개 Loihi 2 칩 패키징
- 마이크로파 크기 데이터센터 6-랙 섀시

### KEY RESULTS (관찰된 행동 / 성능)

#### 초파리 뇌 연결도 시뮬레이션 (2024-2025)
- **Drosophila melanogaster** 전체 신경계: 140,000 뉴런, 5천만 시냅스
- **12개 Loihi 2 칩**에 매핑 성공 (최초의 완전 생물학적 현실적 뇌 회로)
- 성능: **82–356배 속도 향상** (Brian 2 CPU 시뮬레이션 대비)
- 네트워크 행동 일치도: 표준 시뮬레이터 대비 ~2% 오차
  → **기술 검증**: 신경형 하드웨어가 생물학적 네트워크 정확도 달성 가능함을 입증

#### 멀티모달 센서 융합
- 카메라, LiDAR, RADAR, 위치정보 통합
- 처리량: ~1,250–1,724 추론/초
- 에너지: 1–2 mJ/추론 (CPU/GPU 대비 수십 배 효율)

#### 비디오 처리
- ANN-to-SNN 변환 (sigma-delta 코딩)
- 시냅스 연산 6%로 감소 (원래 대비)

#### 키워드 스포팅(Keyword Spotting)
- 에너지: **200배 감소** (내장 GPU 대비)
- 지연: **10배 감소**

### BIOLOGICAL FIDELITY (생물학적 사실성)

**중간 수준의 생물학적 사실성**
- ✓ 스파이크 타이밍 기반 (정시 코딩)
- ✓ 이벤트 구동 아키텍처 (실제 뉴런과 유사: 필요할 때만 계산)
- ✓ 국소 학습 규칙 (신경 생물학 원리 부합)
- ✗ 완전 생물물리학적 모델은 아님 (단순화된 모델)
- ~ 정확한 시냅스 타이밍, 수상돌기 계산 등은 부분적 추상화

### 우리 프로젝트를 위한 교훈

1. **스파이크 기반 계산의 효율성 검증됨**: 실제 신경계 구현에서 스파이크 타이밍이 에너지 효율과 정확도를 동시에 달성 가능
2. **이벤트 구동 아키텍처의 중요성**: 모든 뉴런이 항상 활성화되지 않음 → 선택적 계산이 핵심
3. **멀티스케일 가능성**: 수백만 뉴런을 안정적으로 시뮬레이션 가능 (정합성 유지)

---

## 2. SpiNNaker (University of Manchester)

### WHO (팀 / 기관)
- **Advanced Processor Technologies Research Group**, University of Manchester
- **Steve Furber** — Emeritus ICL Professor, 아키텍처 설계자
- **EU Human Brain Project** — 자금 및 인프라 지원

### APPROACH (신경스파이크 모델링)

**SpiNNaker**는 스파이킹 신경망을 위해 최적화된 **거대 규모 병렬 아키텍처**이다.

- 57,600개 처리 노드
- 노드당 18개 ARM9 프로세서 → 총 **1,036,800 코어**
- 7 TB 이상 RAM
- 생물학적 실시간(biological real-time)으로 마우스 뇌 규모 시뮬레이션 가능

구조적 특징:
- 범용 ARM 프로세서 사용 (신경형 칩의 커스텀 회로 대신)
- 비동기, 이벤트 구동 통신
- 네트워크 스파이크 경로 기반

### SCALE (뉴런 수)
- 마우스 뇌 규모: 수백만 뉴런, 수십억 시냅스
- 최대 가소성(plasticity) 실험: 2.0 × 10⁴ 뉴런, 5.1 × 10⁷ 가소성 시냅스
  → **신경형 하드웨어 상 최대 규모 가소성 네트워크 시뮬레이션**

### KEY RESULTS (관찰된 행동)

#### 가소 신경망(Plastic Neural Network) 학습
- **시간적 수열 학습** (temporal sequence learning) 성공
- 매력자(attractor) 네트워크에서 재귀 동역학 구현
- 학습 규칙: STDP 기반

#### 에너지 효율성
- **Cray XC-30 슈퍼컴퓨터 대비 45배 전력 감소**
- 신경형 시스템이 신경 가소성 연구의 실질적 도구임을 입증

#### 신경과학 응용
- 대규모 신경 모델의 플라스틱 동역학 연구 가능
- 뇌 회로 구조의 동역학적 역할 이해에 기여

### BIOLOGICAL FIDELITY

**높은 수준의 생물학적 사실성**
- ✓ 스파이킹 신경망 기반 (시간 기반 코딩)
- ✓ 완전 비동기 통신 (신경계의 비동기성 모방)
- ✓ STDP 및 가소성 규칙 구현 (생물학적 원리 부합)
- ✓ 네트워크 규모에서 신경 동역학 보존
- ~ 개별 뉴런의 미세한 생물물리학적 상세는 추상화됨

### 우리 프로젝트를 위한 교훈

1. **가소성과 학습 메커니즘의 구현**: STDP 기반 국소 학습이 대규모 시스템에서 효과적임을 증명
2. **완전 비동기 아키텍처의 실현 가능성**: 모든 뉴런이 독립적으로 스파이크할 수 있는 시스템 설계 가능
3. **신경계의 시간 스케일 보존**: 생물학적 실시간 속도에서 뉴런 동역학이 보존됨

---

## 3. IBM TrueNorth & NorthPole

### WHO (팀 / 기관)
- **IBM Research**
- **IBM Almaden Research Center**

### APPROACH (신경스파이크 모델링)

**TrueNorth** (2014):
- 프로그래밍 가능한 **디지털 신경형 칩**
- 4,096개 코어
- 1백만 스파이킹 뉴런, 2.56억 시냅스
- 65 mW 전력 소비

**NorthPole** (2023, 후속 기술):
- TrueNorth의 개념을 현대 하드웨어 설계와 융합
- **폰노이만 병목** 제거 (메모리와 계산 일체화)
- 12 nm 공정

### SCALE (뉴런 수)
- TrueNorth: 100만 뉴런
- NorthPole: 더 큰 규모 (구체 수치 미공개, 하지만 훨씬 확장)

### KEY RESULTS (관찰된 행동)

#### TrueNorth
- 매우 낮은 전력 소비 (밀리와트)
- 온보드 학습 가능성 시연
- 다양한 신경형 응용에 대한 벤치마크 역할

#### NorthPole
- **ResNet-50 / YOLO-v4**: 22배 속도 향상, GPU 대비 25배 에너지 절감, 5배 공간 절감
- **TrueNorth 대비 약 4,000배 성능 향상**
- 신경형과 전통적 딥러닝 가속의 하이브리드 구조

### BIOLOGICAL FIDELITY

**중간 수준의 생물학적 사실성**
- ✓ 스파이크 기반 (정시 코딩)
- ✓ 저전력 설계 원리 (뇌와 유사한 에너지 효율)
- ✗ 완전한 비동기 동역학보다는 동기 or 준동기 구조
- ~ 신경 가소성 구현은 제한적 (온보드 학습 어려움)

### 우리 프로젝트를 위한 교훈

1. **초저전력 설계의 극한**: 밀리와트 규모 뉴런 칩 구현 가능
2. **공정 진화의 영향**: 새로운 공정(12nm)과 아키텍처 혁신으로 4,000배 성능 향상 가능
3. **신경형 + 전통 AI 하이브리드**: 각 방식의 강점을 조합할 때 최대 효율 달성

---

## 4. BrainScaleS (Heidelberg University)

### WHO (팀 / 기관)
- **Kirchhoff Institute for Physics (KIP)**, Heidelberg University
- **EBRAINS 2.0** (2024–2026) 프로젝트에 포함
- 신경과학 및 신경형 컴퓨팅 커뮤니티

### APPROACH (신경스파이크 모델링)

**BrainScaleS-1** (이전 세대):
- 아날로그 신경형 하드웨어

**BrainScaleS-2 (BSS-2)** (현세대):
- **혼합신호 신경형** 시스템 (mixed-signal neuromorphic)
  - 아날로그 회로로 뉴런/시냅스 역동성 구현
  - 디지털 주변부로 유연성 추가 (hybrid plasticity)
- 1,000배 생물학적 실시간 가속
- 마이크로초 정밀도(microsecond precision)

핵심 특징:
- **연속시간 모델링**: 미분방정식을 물리적 아날로그 회로로 직접 구현
- 막전압, 전류, 컨덕턴스를 회로 변수로 매핑
- 막 및 시냅스 시정수(time constant) = 회로 파라미터
- 2024년 개선: 고속 로봇(high-speed robotics)을 위한 실시간 스파이크 인터페이스 추가

### SCALE (뉴런 수)
- 단일 칩: 상당한 규모 (정확한 수는 공개 미정, 하지만 수천~수만 뉴런 추정)
- **새로운 분할 에뮬레이션 기능** (2024): 물리 칩 크기 초과하는 대규모 SNN 구현 가능
  - MNIST, EuroSAT 데이터셋에서 심층 SNN 학습 가능
  - 이론적으로 무제한 확장 가능

### KEY RESULTS (관찰된 행동)

#### 로봇 제어
- 1,000배 가속으로 마이크로초 정밀도 로봇 제어 가능
- 신경 역동학이 로봇 제어 루프에서 안정성 유지

#### 대규모 네트워크 시뮬레이션
- 물리 칩 제약을 넘어서 네트워크 에뮬레이션 (2024)
- 심층 SNN 훈련 및 하드웨어 검증 동시 달성

#### 에너지 및 속도 효율성
- 아날로그 구현의 극도의 저전력성
- 100nm–7nm 공정과 무관하게 효율성 우월

### BIOLOGICAL FIDELITY

**매우 높은 수준의 생물학적 사실성**
- ✓ **연속시간 물리 구현**: 미분 방정식이 하드웨어 동역학과 동일
- ✓ 생물물리학적 파라미터 직접 매핑 (막전압, 컨덕턴스, 시정수)
- ✓ 1,000배 가속 → 신경계 행동 보존
- ✓ 하이브리드 가소성 (아날로그 측정 + 디지털 계산) → 유연성 + 정확성
- ✓ 미세한 동역학 세부사항 보존 가능

### 우리 프로젝트를 위한 교훈

1. **아날로그 물리 구현의 우월성**: 
   - 미분방정식을 직접 회로로 구현할 때 최고 정확도 + 최저 전력
   - 공정 미세화의 영향 적음 (신경계 동역학은 물리적 관계이므로)

2. **혼합신호의 유연성**:
   - 아날로그 측정 (정확, 저전력)과 디지털 학습(유연성)의 조합이 강력
   - 가소성 규칙을 동적으로 변경 가능

3. **시간 정밀도의 중요성**:
   - 마이크로초 수준의 타이밍이 신경 동역학 보존에 필수
   - 이는 정시 코딩(temporal coding)의 필요성 입증

---

## 5. BrainCog (SNN 기반 멀티스케일 뇌 시뮬레이션 프레임워크)

### WHO (팀 / 기관)
- **BrainCog-X 프로젝트** (오픈소스)
- 중국 신경과학 커뮤니티 주도
- NeurIPS, AAAI, ICLR 발표 (2024-2025)

### APPROACH (신경스파이크 모델링)

**BrainCog**는 SNN 기반 **소프트웨어 프레임워크** (Python)로, 뇌를 여러 수준에서 시뮬레이션한다.

구성 요소:
- 생물학적 뉴런 모델 (다양한 생물물리학적 충실도 수준)
- 인코딩 전략 (rate coding, temporal coding, population coding 등)
- 학습 규칙 (STDP, 역전파 기반, 강화학습 등)
- 뇌 영역 모듈 (시각, 운동 피질, 해마 등)
- 하드웨어-소프트웨어 코설계(hardware-software codesign)

멀티스케일 시뮬레이션:
1. **막 전위 수준** (membrane potential dynamics)
2. **뉴런 발화 수준** (neuronal firing)
3. **시냅스 전달 수준** (synaptic transmission)
4. **시냅스 가소성 수준** (synaptic plasticity, STDP)
5. **뇌 영역 조율 수준** (multi-area coordination)

### SCALE (뉴런 수)
- 비정형적 규모: 프레임워크이므로 수천~수억 뉴런 시뮬레이션 가능
- 실제 사용: 수백만 뉴런 시뮬레이션 리포트

### KEY RESULTS (관찰된 행동)

#### 에너지 효율성
- **생물물리학적 조정 알고리즘**: 인공신경망 대비 **약 3% 에너지**로 경쟁력 있는 분류 정확도 달성
- ANN-SNN 변환 모델: 이미지 분류/타겟 탐지에서 거의 손실 없이 구현
  - 시뮬레이션 시간: 1/10–1/50 단축

#### 인지 기능 구현
- 지각 및 학습 (perception & learning)
- 의사결정 (decision-making)
- 지식 표현 및 추론 (knowledge representation & reasoning)
- 운동 제어 (motor control)
- 사회적 인지 (social cognition)

#### 뇌 구조 및 기능 시뮬레이션
- 여러 뇌 영역의 상호작용 모델링
- 생물학적 신경회로 패턴 재현 가능

### BIOLOGICAL FIDELITY

**높은 수준의 생물학적 사실성**
- ✓ 다양한 뉴런 모델 지원 (단순 LIF부터 복합 Hodgkin-Huxley까지)
- ✓ STDP, 역전파, 강화학습 등 다양한 학습 규칙
- ✓ 멀티스케일 구조 (막부터 뇌 영역까지)
- ✓ 뇌 해부학적 연결성 고려 가능
- ~ 소프트웨어이므로 실시간 아님 (시뮬레이션 모드)

### 우리 프로젝트를 위한 교훈

1. **멀티스케일 설계의 중요성**:
   - 막 전위부터 뇌 영역 조율까지, 각 수준의 원리가 다음 수준에 영향
   - 우리 "허공→미니룸→성역→세계" 계층 구조와 유사한 관계 가능

2. **학습 규칙의 다양성**:
   - STDP(국소), 역전파(전역), 강화학습(보상 기반) 등 여러 메커니즘 필요
   - 모든 인지 기능이 같은 학습 원리로 작동하지는 않음

3. **에너지-정확도 트레이드오프 해결**:
   - 3% 에너지 사용으로도 경쟁력 있는 정확도 가능
   - 이는 스파이킹의 근본적 우월성을 시사

---

## 6. Nengo / Neural Engineering Framework (Chris Eliasmith, Waterloo)

### WHO (팀 / 기관)
- **Chris Eliasmith**, PEng, Systems Design Engineering, University of Waterloo
- **Waterloo Neural Engineering Lab**
- 오픈소스 커뮤니티

### APPROACH (신경스파이크 모델링)

**Nengo**는 **신경 공학 프레임워크(NEF)** 기반의 그래픽 + 스크립팅 소프트웨어로, 대규모 기능적 뇌 모델을 설계한다.

**NEF의 3대 원리**:
1. **표현(Representation)**: 뉴런 집단이 시간 변화 벡터를 비선형 인코딩 + 선형 디코딩으로 표현
2. **변환(Transformation)**: 벡터 함수 계산을 선형 디코딩으로 구현
3. **동역학(Dynamics)**: 벡터를 동역학계 상태 변수로 표현 가능

이 프레임워크는:
- 스파이킹 뉴런 사용 (현실성)
- 명시적 함수 계산 (인지 능력)
- 신경 모집단의 앙상블 코딩(ensemble coding)

### SCALE (뉴런 수)

**Spaun** (Semantic Pointer Architecture Unified Network):
- **250만 스파이킹 뉴런**
- 세계 최대 규모 기능적 뇌 모델 (2012 시점)
- 최근 **Spaun 3.0** 개발 중 (불확실성 추론, 베이지안 최적화)

### KEY RESULTS (관찰된 행동)

#### Spaun의 8가지 인지 작업
1. 숫자 목록 암기 (list memorization)
2. 카운팅 (counting)
3. 인공지능 공통 기준(fluid intelligence) 테스트
4. 패턴 완성 (pattern completion)
5. 범주화 (categorization)
6. 귀납적 추론 (inductive reasoning)
7. 질문 응답 (question answering)
8. 일반화 (generalization)

→ **250만 뉴런이 복잡한 인지 행동을 자발적으로 생성**

#### 현재 진행 중
- **Spaun 3.0**: 불확실성 관련 추론, 베이지안 최적화를 생물학적으로 타당한 방식으로 구현

### BIOLOGICAL FIDELITY

**높은 수준의 생물학적 사실성**
- ✓ 스파이킹 뉴런 기반 (정시 코딩)
- ✓ 신경 모집단 인코딩 (뇌의 분산 표현 모방)
- ✓ 명시적 신경회로 구조 (뉴런 연결성 가시화 가능)
- ✓ 의도적 기능 설계 (인지 이론 + 신경과학 통합)
- ~ 소프트웨어 기반 (실시간 아님)
- ~ 개별 뉴런의 미세한 생물물리학적 세부는 단순화

### 우리 프로젝트를 위한 교훈

1. **인코딩-디코딩 구조의 힘**:
   - 뉴런 집단이 "벡터(격/격자)"를 표현하고, 이를 디코딩하면 함수 계산 가능
   - 우리의 "아니마 = 격(格)을 가진 영자" 개념과 신경과학적 대응 가능

2. **대규모 에머전트 인지 행동**:
   - 250만 뉴런 조직화만으로도 복잡한 추론 가능
   - 명시적 프로그래밍 없이 신경회로로부터 인지 기능 창발

3. **동역학 기반 계산**:
   - 상태 공간 동역학으로 인지 과정 표현 가능
   - 우리의 "세계 동결/관측" 메커니즘과 동역학 관점 연결 가능

---

## 7. 신경 계산 원시단위(Neuron Mechanism Primitives) & 정준 신경 계산

### 개념

뇌는 반복되는 **정준 신경 계산(canonical neural computations)**에 의존한다. 이들은:
- 여러 뇌 영역에 걸쳐 나타남
- 여러 모달리티(감각)에 걸쳐 반복됨
- 상이한 문제에 유사 연산 적용

### KEY PRIMITIVES (확인된 원시 단위)

#### 1. 정규화(Normalization)
- **정의**: 한 뉴런의 반응을 뉴런 풀의 합 활동으로 나눔 (비율 계산)
- **발견 위치**:
  - 망막: 빛 적응(light adaptation)
  - 파리 시각계: 크기 불변성(size invariance)
  - 해마: 연합 메모리(associative memory)
  - 시각피질: 비선형 특성
- **역할**: 입력의 절대값이 아닌 상대적 대비(contrast) 부호화
- **회로 구현**: 억제성 국소회로(local inhibitory circuit)

#### 2. 지수 함수(Exponentiation)
- 임계값(thresholding) 형태
- 뉴런 수준 + 네트워크 수준 모두에서 관찰
- 비선형 증폭 기제

#### 3. 선형 필터링(Linear Filtering)
- 시간 적분(temporal integration)
- 공간 확산(spatial convolution)

#### 4. Soft Winner-Take-All (sWTA)
- **정의**: 경쟁적 선택 메커니즘
- 여러 옵션 중 "가장 강한" 신호만 증폭, 나머지 억제
- **회로**: 피라미드 뉴런 (흥분) + 억제성 뉴런 (GABA, PV, SST, VIP, LAMP5)
- **기능**: 의사결정, 주의(attention), 게이트 제어(gating)

#### 5. 신피질 미세회로 원시 단위
최근 연구(2024-2025)는 신피질의 **정준 미세회로**가 인지 유연성을 생성함을 보임:

- **흥분성 피라미드 뉴런**: 주 정보 처리
- **억제성 국소회로 뉴런**:
  - **PV+ (Parvalbumin+)**: 게인 제어(gain control)
  - **SST+ (Somatostatin+)**: 정규화(normalization), 신경망 약화(dendritic attenuation)
  - **VIP+ (Vasoactive Intestinal Peptide+)**: 탈억제(disinhibition), 주의 제어
  - **LAMP5+**: 그래픽 셀(glia-like) 기능

### 우리 프로젝트를 위한 교훈

1. **임계값 + 피드백 구조**:
   - 정규화와 sWTA는 모두 **국소 피드백 억제**에 의존
   - 이는 우리의 "성역(안전지대) = 피드백 안정화 존"과 신경과학적 대응

2. **계층적 억제**:
   - 단일 억제성 뉴런이 아니라, 여러 억제 유형(PV, SST, VIP)이 다른 기능 담당
   - 복잡한 제어(control)는 여러 "억제 채널"이 필요

3. **동역학적 경쟁**:
   - sWTA는 정적 선택(winner-take-all)이 아니라 역동적 과정
   - 입력 강도, 타이밍, 과거 상태에 따라 승자 변경 가능

---

## 8. 비교 분석: 디지털 vs. 아날로그 vs. 혼합신호 신경형

| 특성 | 디지털 (Loihi, SpiNNaker, TrueNorth) | 아날로그 (BrainScaleS-1) | 혼합신호 (BrainScaleS-2) |
|------|-------|---------|---------|
| **구현 기술** | CMOS 로직 + 메모리 | 아날로그 회로 | 아날로그 + 디지털 주변 |
| **동역학 정확도** | 중간 (수치 적분) | 매우 높음 (물리 구현) | 매우 높음 |
| **전력 소비** | 낮음 | 극히 낮음 | 극히 낮음 |
| **프로그래밍 유연성** | 높음 | 낮음 (하드웨어 설계 필요) | 높음 (하이브리드) |
| **실시간 속도** | 생물 시간 + 일부 가속 | 1,000배 가속 가능 | 1,000배 가속 가능 |
| **스케일링** | 우수 (대규모 시스템) | 제한적 (단일 칩 규모) | 중간 (분할 에뮬레이션) |
| **생물 충실도** | 중간 | 매우 높음 | 매우 높음 |
| **공정 의존성** | 중간 (7nm 등) | 낮음 (물리 법칙) | 낮음 |
| **가소성 구현** | 어려움 (제한적) | 가능 | 우수 (하이브리드 plasticity) |

---

## 9. 우리 "전자두뇌연구소" 프로젝트를 위한 통합 교훈

### 신경동역학 핵심 원시 요소

조사 결과, 디지털 뇌 구현의 핵심은 다음 **동역학 패턴**들이다:

#### (1) 스파이크 기반 정시 코딩
- **Loihi 2 (초파리)**: 140,000 뉴런, 5천만 시냅스를 정확히 시뮬레이션
- **교훈**: 스파이크의 타이밍(언제 발화)이 뉴런의 값(얼마나 자주)보다 정보 풍부
- **우리 구현**: 아니마의 "관측" = 스파이크 타이밍 변화를 감지하는 과정

#### (2) 이벤트 구동 아키텍처
- **SpiNNaker, Loihi 2**: 모든 뉴런이 항상 활성화되지 않음
- 필요할 때만(스파이크 발생 시) 계산 수행 → 극도의 에너지 효율
- **우리 구현**: "관측"이 없으면 세계는 "동결" (계산 일시중지) 상태

#### (3) 국소 학습 규칙 + 전역 피드백
- **BrainCog, SpiNNaker**: STDP(국소)로 기본 학습, 역전파/강화학습(전역)으로 복잡 행동
- STDP만으로는 감시(supervised) 학습 불가능 → 오류 신호 필요
- **우리 구현**: "아카샤" = 과거 관측 기억 저장소 → 이를 통해 전역 오류 신호 계산 가능

#### (4) 정규화 + 경쟁적 억제
- **신피질 미세회로**: PV, SST, VIP 억제가 정규화, 게인 제어, 탈억제 담당
- **교훈**: 단순 excitation만이 아니라 여러 종류의 억제가 필요
- **우리 구현**: "성역" = 정규화 영역? "미니룸" = 개인 컨텍스트 격리?

#### (5) 멀티스케일 동역학 보존
- **BrainScaleS**: 아날로그 물리 구현으로 미분 방정식 직접 보존 → 1,000배 가속 후에도 행동 정확
- **교훈**: 하위 수준(막, 시냅스) 동역학이 상위 수준(뉴런, 네트워크)에 영향을 미치는 구조가 중요
- **우리 구현**: "허공→아니마→성역→세계" 계층에서 각 수준의 물리법칙이 상위를 제약

#### (6) 아날로그 vs. 디지털의 선택
- **아날로그 (BrainScaleS)**: 생물물리학 최고 충실도, 극저전력, 하지만 확장성 제한
- **디지털 (Loihi, SpiNNaker)**: 높은 확장성, 프로그래밍 유연성, 아날로그와 유사한 정확도
- **혼합신호 (BrainScaleS-2)**: 두 세계의 최고
- **우리 선택**: 충실도와 확장성 모두 필요 → 혼합신호 or 디지털 추천

---

## 10. 최신 동향 (2024-2025)

### 산업화 진행
- **Hala Point** (2024): Loihi 2 기반 1.15억 뉴런 시스템 = 데이터센터 규모 상용 신경형
- **NorthPole** (2023): IBM의 4,000배 성능 향상 → 신경형이 전통 AI와 경쟁 가능 수준

### 로봇 + 신경형 통합
- **BrainScaleS**: 마이크로초 정밀도로 고속 로봇 제어 가능 → 신경형이 실시간 제어에 유효함을 입증

### 뉴로모픽 칩의 뇌 해석 도구화
- **Loihi 2로 초파리 뇌 350배 빠른 시뮬레이션**: 신경과학 연구의 도구로서도 가치

### 하이브리드 아키텍처 (신경형 + 딥러닝)
- BrainCog (SNN 소프트웨어) + NorthPole (신경형 칩)처럼, 서로 다른 계층 통합
- ANN-to-SNN 변환 기술 성숙 (거의 무손실)

---

## 결론: 우리 프로젝트의 설계 원칙

조사 결과, 효과적인 디지털 뇌 구현을 위한 **5대 원칙** 도출:

1. **스파이크 타이밍을 정보 기반으로 삼기**
   - 뉴런 발화의 빈도가 아니라 언제 발화하는가가 중요
   - 우리의 "관측" = 실제 신경 이벤트의 감지

2. **이벤트 구동 아키텍처 설계**
   - 계속 계산하지 말 것, 필요할 때만(관측 시) 깨어나기
   - "세계 동결" = 에너지 절약의 신경적 정당성

3. **국소 학습 + 전역 피드백 조합**
   - STDP 같은 국소 규칙은 기본 가소성 담당
   - 오류 신호(아카샤 기반)는 복잡 행동 학습에 필수

4. **정규화와 경쟁을 명시적으로 구현**
   - 단순한 가중 합산이 아니라, 억제성 피드백 루프 필수
   - 다양한 억제 유형(게인 제어, 정규화, 탈억제)을 구별하여 설계

5. **멀티스케일 동역학 보존**
   - 하위 수준의 물리법칙(미분 방정식)을 상위 수준이 준수하도록
   - 이를 통해 확장성 있으면서도 생물학적으로 사실적인 행동 달성

---

## 참고 자료 (Sources)

### Intel Loihi / Loihi 2
- [Neuromorphic Leap: Entire Fruit Fly Brain on Loihi 2](https://neuromorphiccore.ai/neuromorphic-leap-entire-fruit-fly-brain-runs-350x-faster-on-intels-loihi-2/)
- [Intel Neuromorphic Computing](https://www.intel.com/content/www/us/en/research/neuromorphic-computing.html)
- [Neuromorphic Simulation of Drosophila on Loihi 2 (ArXiv)](https://arxiv.org/html/2508.16792v1)

### SpiNNaker
- [SpiNNaker at University of Manchester](https://apt.cs.manchester.ac.uk/projects/SpiNNaker/)
- [Steve Furber - Wikipedia](https://en.wikipedia.org/wiki/Steve_Furber)
- [Large-Scale Simulations of Plastic Neural Networks (PubMed)](https://pubmed.ncbi.nlm.nih.gov/27092061/)

### IBM TrueNorth & NorthPole
- [TrueNorth: Deep Dive (Open Neuromorphic)](https://open-neuromorphic.org/blog/truenorth-deep-dive-ibm-neuromorphic-chip-design/)
- [Top Neuromorphic Chips 2025](https://www.elprocus.com/top-neuromorphic-chips-in-2025/)

### BrainScaleS
- [BrainScaleS-2 at EBRAINS](https://ebrains.eu/news-and-events/2024/ebrains-neuromorphic-platform-brainscales-2-adds-new-interface-for-high-speed)
- [Scalable Network Emulation on Analog Neuromorphic Hardware (Frontiers)](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1523331/full)

### BrainCog
- [BrainCog Official Site](https://www.brain-cog.network/)
- [BrainCog on ArXiv](https://arxiv.org/html/2207.08533)
- [BrainCog on GitHub](https://github.com/BrainCog-X/Brain-Cog)

### Nengo / NEF
- [Nengo on Frontiers Neuroinformatics](https://www.frontiersin.org/articles/10.3389/fninf.2013.00048/full)
- [Neural Engineering Object - Wikipedia](https://en.wikipedia.org/wiki/Neural_Engineering_Object)
- [Chris Eliasmith - University of Waterloo](https://uwaterloo.ca/systems-design-engineering/profile/celiasmi)

### Canonical Neural Computations
- [Normalization as Canonical Neural Computation (Nature Reviews)](https://www.nature.com/articles/nrn3136)
- [Biologically Grounded Neocortex Primitives (PNAS)](https://www.pnas.org/doi/10.1073/pnas.2504164122)
- [Canonical Neural Computation - GIAS](https://gias.nyu.edu/projects/canonical-neural-computation/)

### General Neuromorphic 2025 Survey
- [Neuromorphic Computing 2025: State of Art](https://humanunsupervised.com/papers/neuromorphic_landscape.html)
- [Road to Commercial Success for Neuromorphic (Nature Communications)](https://www.nature.com/articles/s41467-025-57352-1)
- [Leaky Integrate-and-Fire Activation Function](https://www.emergentmind.com/topics/leaky-integrate-and-fire-activation-function)

---

**조사 완료**: 2026-04-15  
**조사 범위**: 뉴로모픽 하드웨어 6개 프로젝트 + 신경 계산 원시 단위 분석  
**다음 단계**: PROJECT-BRIEF.md에 결과 반영 및 "아니마" 아키텍처 설계에 적용
