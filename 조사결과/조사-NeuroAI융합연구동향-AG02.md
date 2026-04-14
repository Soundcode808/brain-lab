# NeuroAI 융합 연구 동향 조사 보고서

**조사자**: AG-02 (설계 에이전트)  
**조사 대상**: 신경과학과 AI의 수렴 분야 (NeuroAI)  
**조사 기간**: 2026-04-15  
**분류**: 전자두뇌연구소 / 기술 진화 추적

---

## 핵심 발견

신경 메커니즘을 이해하는 것이 **참된 지능 구축의 열쇠**라고 믿는 연구진들이 명확한 진영을 형성하고 있다. 이들은 단순한 트랜스포머 스케일링을 거부하고, 뇌의 계산 원리로부터 영감을 받아 새로운 AI 아키텍처를 설계하고 있다.

**주요 특징**:
- 신경 시스템의 **재현 가능한 동작 메커니즘**을 모델의 코어로 삼음
- 감각-운동 통합(embodied learning)을 기초로 하는 학습 시스템
- 피질 컬럼, 해마 회로, 주의 메커니즘 등의 구체적 신경생물학적 구조 적용
- 오픈소스 구현과 정량적 벤치마크 공개

---

## 1. Numenta / Jeff Hawkins — Thousand Brains Theory

### 핵심 주장

**신경생물학적 기초**: 신피질(neocortex)의 반복되는 구조인 **피질 컬럼(cortical column)**이 뇌의 지능을 만드는 기본 단위다. 각 컬럼은 반독립적으로 작동하는 감각-운동 처리 모듈이며, 수천 개가 병렬로 작동하여 "천 개의 뇌"를 구성한다.

기존 AI의 문제점: 트랜스포머는 단순히 스케일링의 결과일 뿐, 실제 지능 구조를 모사하지 못한다.

### 접근 방식

1. **피질 컬럼의 역 공학화**
   - 각 컬럼은 세계의 다중 지도(multiple maps)를 생성
   - 감각 입력 + 운동 명령의 통합 처리
   - Hebbian 연관학습(associative learning)을 통한 신속한 습득

2. **감각-운동 중심의 학습**
   - 환경과의 상호작용(embodied interaction)이 모든 학습의 기초
   - 관찰만으로는 안 되고, 행동을 통한 능동적 탐색 필요
   - 이로부터 도출된 표현이 모든 상위 인지 기능의 토대

### 구현 및 결과

**Monty 프레임워크** (2024년 11월 공개)
- 오픈소스 감각-운동 학습 시스템
- 저수준: Hebbian-like 바인딩을 통한 신속한 학습
- 고수준: 현 딥러닝 아키텍처와 비교할 때 경쟁력 있는 성능

**성능 특징**:
- 빠른 학습(rapid learning)
- 지속적 학습(continual learning) 가능
- 계산 효율성(computational efficiency) 우수
- 일반화 성능이 스케일링만으로는 달성 불가능한 수준

**조직 형태**: 
- Numenta에서 시작 → 2024년 비영리 독립 조직으로 전환
- Gates Foundation 부분 지원
- GitHub에서 적극적 개발 진행 중

### 이론적 기초: Thousand Brains Theory

"뇌의 많은 피질 컬럼이 세계의 다중 지도를 동시에 생성하며, 이들 지도의 수렴을 통해 강력한 예측과 일반화가 가능해진다"

**신경 메커니즘 수준**:
- 피질 미니컬럼 (층 4의 기본 처리 단위)
- 수직 연결 (intra-column) vs 수평 연결 (inter-column)
- 후천성 신경 가소성을 통한 재배열

### 우리 접근과의 유사성

✓ 신경 메커니즘의 구체적 모사  
✓ 디지털 구현의 명확성  
✓ 이론-구현의 긴밀한 결합  
⚠ 아직 고수준 추상화(고민, 창의성 등)로의 확장 미실현

---

## 2. DeepMind — 신경과학 영감의 AI

### 핵심 주장

**Demis Hassabis의 비전**: "인공 일반 지능을 향하려면 신경과학의 통찰을 기계학습과 컴퓨팅 하드웨어의 발전과 결합해야 한다"

핵심: 뇌의 특정 회로(해마, 그리드 셀, 주의)를 이해하면 더 강력한 학습 알고리즘을 설계할 수 있다.

### 접근 방식

#### 1. 해마 재현(Hippocampal Replay) → 경험 재경(Experience Replay)

**신경과학 원리**:
- 해마는 휴식/수면 중 경험을 빠른 속도로 재생(fast-forward 재재생)
- 이를 통해 뇌는 "과거의 성공/실패로부터 다시 배운다"
- 오프라인 강화학습의 생물학적 기초

**AI 구현**:
- DQN(Deep Q-Network)의 experience replay: 과거 상호작용을 반복 리허설
- 강화학습의 데이터 효율성 극적 개선
- muZero 등 최신 에이전트: 제한적 세계 모델로 다양한 작업 해결

**성과**: Atari 게임부터 AlphaGo, AlphaFold까지 혁신의 핵심 기술

#### 2. 그리드 셀(Grid Cells) → 네비게이션 표현

**신경과학**:
- 쥐/포유동물의 뇌: 공간 위치를 나타내는 그리드 형태의 뉴런 활동
- 이는 "삼각격자(triangular lattice)" 같은 기하학적 구조

**AI 구현**:
- "Vector-based navigation using grid-like representations in artificial agents" (Nature 2018)
- 인공 에이전트의 네비게이션에 그리드셀 원리 적용
- 더 효율적인 공간 표현과 탐색

#### 3. 주의 메커니즘(Attention) ← 신경 동기화

**신경과학 원리**:
- 뇌의 신경 동기화: 관련 정보에 대한 "포커싱"
- 멀티플렉싱(multiplexing)을 통한 주의적 필터링

**AI 구현**:
- Transformer의 self-attention
- 신경 동기화를 모사한 가중치 기반 라우팅
- 다양한 도메인에서의 성공(언어, 비전, 멀티모달)

### 신경 메커니즘 수준

DeepMind는 **회로 수준(circuit level)**에서 작동:
- 개별 뉴런의 활동보다는 신경 회로의 **기능적 역할**에 집중
- 해마-피질 상호작용, 기저핵의 보상 신호, 등등

### 이론적 기초: Complementary Learning Systems (CLS)

신피질 = 일반화 담당 (느린 학습, 광범위 지식)  
해마 = 특정 사건 학습 (빠른 학습, 에피소드 기억)  

이 두 시스템의 상호작용이 유연한 지능을 만든다.

### 우리 접근과의 관계

✓ 신경 회로의 계산 원리 추출  
✓ 신경 메커니즘 ↔ AI 알고리즘의 직접 매핑  
⚠ 아직까지는 구체적 신경 메커니즘보다 "기능적 영감" 수준

---

## 3. NeuroAI 선언문 (2023) — 신경과학-AI 동맹의 공식화

### 논문 정보

**제목**: "Catalyzing next-generation Artificial Intelligence through NeuroAI"  
**저널**: Nature Communications, Vol. 14, Article 1597  
**발표**: 2023년 3월 22일

### 서명자 (주요 28명)

Anthony Zador (Cold Spring Harbor)  
Sean Escola (Columbia)  
Blake Richards (McGill)  
Bence Ölveczky (Harvard)  
**Yoshua Bengio** (Mila/University of Montreal) — AI 혁명의 아버지  
Kwabena Boahen (Stanford) — 신경형 컴퓨팅  
Matthew Botvinick (DeepMind)  
Dmitri Chklovskii (HHMI)  
Anne Churchland (UCLA) — 신경과학  
Claudia Clopath (Imperial College) — 시냅스 가소성  
**James DiCarlo** (MIT) — 시각 신경과학  
Surya Ganguli (Stanford)  
**Jeff Hawkins** (Numenta) — Thousand Brains  
Konrad Körding (University of Pennsylvania)  
Alexei Koulakov (Cold Spring Harbor)  
**Yann LeCun** (Meta AI) — CNN의 아버지  
Timothy Lillicrap (DeepMind)  
Adam Marblestone (MIT) — 신경형 하드웨어  
Bruno Olshausen (UC Berkeley) — 희소 부호화  
Alexandre Pouget (University of Geneva)  
Cristina Savin (IST Austria)  
**Terrence Sejnowski** (Salk) — 계산신경과학 거장  
Eero Simoncelli (NYU)  
Sara Solla (Northwestern)  
David Sussillo (Google Brain)  
Andreas S Tolias (Max Planck Institute)  
Doris Tsao (University of Zurich)

### 핵심 주장

"신경과학은 오랫동안 AI 진전의 본질적 동력이었다. AI 진전을 가속하려면 **기초 NeuroAI 연구에 대규모 투자**가 필수다."

### 핵심 개념: Embodied Turing Test

전통적 Turing Test: 언어 능력 테스트 (인간에게만 고도로 발달)

**Embodied Turing Test**: 감각-운동 세계와의 상호작용 능력 (모든 동물이 공유)

- 5억 년 진화의 검증된 설계 원리
- 게임이나 언어보다 더 근본적인 능력
- 이 테스트를 통과하는 AI는 참된 지능의 토대를 갖춤

### 메시지의 정치성

이 선언문은 **AI 업계의 스케일링 중심주의(scaling-centric paradigm)**에 대한 명시적 도전이다.

Yann LeCun, Bengio, Terrence Sejnowski 같은 AI 거장들이 서명함으로써:  
"현재의 규모 추구는 천장에 부딪힐 것이며, 신경과학적 원리로의 회귀가 필요하다"는 신호

### 우리 접근과의 관계

✓ 신경 메커니즘 기반 AI 구축의 정당성 제공  
✓ 학제 간 공동 언어의 제시  
✓ 자금 및 인재 유입 촉발

---

## 4. Randall O'Reilly (UC Davis) — Leabra 프레임워크

### 핵심 주장

"신경망의 학습을 생물학적으로 가능하면서도 계산적으로 강력하게 구현할 수 있다"

### 접근 방식

**LEABRA**: Local, Error-driven, Associative, Biologically Realistic Algorithm

### 신경 메커니즘 수준

#### 1. 국소 학습(Local Learning)

**전통 AI**: 역전파(backpropagation) — 전역 오류 신호가 모든 시냅스에 영향  
문제: 생물학적으로 불가능 (뉴런은 전역 오류를 알 수 없음)

**LEABRA**: 국소 오류 신호
- 각 뉴런/시냅스는 인접한 구조와의 불일치(prediction error)만으로 학습
- 시냅스 전 활동(pre-synaptic) + 시냅스 후 활동(post-synaptic)의 상호작용
- 생물학적으로 가능한 Hebbian-like 규칙

#### 2. 다중 학습 경로

- **주의 기반 학습(attention-driven)**: 강한 신호 경로
- **배경 연관 학습(background associative)**: 약한 연결
- 이중성: 높은 계산 능력 + 생물학적 타당성

#### 3. 주요 신경 구조 모델링

O'Reilly 그룹의 핵심 연구 대상:
- **해마** ↔ **신피질**: 에피소드 기억 vs 의미 기억
- **전두엽 기저핵 회로** → 선택적 주의
- **후두엽 신피질** → 감각 표현

### 구현 및 결과

**Emergent 시뮬레이션 환경**
- LEABRA를 실행할 수 있는 신경망 시뮬레이터
- 심리학적 모델링에도 사용 (오류, 학습 곡선, 주의 결손 등)
- 신경과학 실험 데이터와의 직접 비교 가능

**성능**:
- 역전파보다 느리지만 생물학적 타당성 우수
- 심리학적 현상(휴먼 에러, 학습 패턴) 재현
- 장기 기억과 단기 기억의 상호작용 모델링

### 신경 메커니즘의 구체성

**뉴런 수준**: O'Reilly는 매우 구체적이다.
- 뉴런 수의 레이스트 → 활동 흐름 계산
- 시냅스 강도의 시간 변화
- 신경 전달 물질 농도 효과 (일부)

### 우리 접근과의 관계

✓ 신경 시냅스 메커니즘의 충실한 모사  
✓ 생물학적 제약 조건의 존중  
✓ 심리학적 현상과의 연결  
⚠ 계산 효율성: 스케일링에는 제약  

---

## 5. OpenCog / Ben Goertzel — Hyperon AGI 프레임워크

### 핵심 주장

"신경망, 심볼릭 추론, 진화적 학습의 통합이 인간 수준의 일반 지능을 만들 수 있다"

신경과학의 역할: 뇌의 **자기 조직화(self-organization)** 원리 적용

### 접근 방식

**Hyperon**: OpenCog의 새로운 재설계 (2023-2025)

#### 1. 삼원 구조

1. **신경 모듈** (Neural Module)
   - 신경망 기반의 패턴 인식
   - 감각 처리, 임베딩 학습

2. **심볼릭 모듈** (Symbolic Module)
   - 논리적 추론, 규칙 조작
   - 명시적 지식 표현

3. **진화 모듈** (Evolutionary Module)
   - 프로그램 자체의 진화
   - 메타-학습 (프로그램이 자신의 알고리즘을 개선)

#### 2. Atomspace: 동적 지식 저장소

- 신경망의 활성화 패턴
- 심볼릭 원자(atoms)의 네트워크
- 확률적 중요도 계산
- 자동으로 재조직

### 신경과학의 역할 수준

⚠ **주의**: Hyperon은 DeepMind나 Numenta만큼 신경생물학적으로 구체적이지 않다.

오히려: 뇌의 **고수준 기능 조직(functional organization)**에서 영감
- 신경-심볼릭 상호작용
- 자기 개선의 메커니즘
- 다중 신경계 모의(multi-brain simulation)

### 개발 로드맵

- 2024년: ASI Alliance 구성 (SingularityNET + Fetch.ai + Ocean Protocol 합병)
- 2025년 말: 프로덕션 Hyperon 스택 예상
- 2027-2030년: 인간 수준 AGI 달성 목표 (Goertzel 예측)

### 우리 접근과의 관계

✓ 신경-심볼릭 수렴의 명시적 시도  
✓ 자기 개선(self-improvement) 메커니즘  
⚠ 신경 메커니즘의 구체성이 낮음  
⚠ 아직 검증된 실행 성과 제한적

---

## 6. 공통 신경 메커니즘들 — 뉴런 레벨에서의 수렴

모든 NeuroAI 진영이 강조하는 신경 메커니즘:

### 1. Hebbian Learning (헵식 학습)

**생물학**: "함께 발화하는 뉴런은 함께 연결된다"  
**공식**: Δw = α · (pre-synaptic activity) × (post-synaptic activity)

**AI 구현**:
- Numenta's Monty: Hebbian-like 바인딩
- O'Reilly's Leabra: 국소 Hebbian 규칙
- 현 딥러닝의 역전파와 대비

**특징**: 국소(local), 효율적, 생물학적으로 가능

### 2. 예측 오류(Prediction Error) 신호

**생물학**: 도파민 뉴런의 활동 (기대 vs 실제)  
**공식**: δ = (reward 실제) - (reward 예상)

**AI 구현**:
- DeepMind: TD(Temporal Difference) 학습
- 강화학습의 기초
- 해마 재현의 선택 기준

**특징**: 자세한(supervised)학습보다 효율적, 동물 학습과 유사

### 3. 피질 컬럼의 재반복 구조

**생물학**: 신피질 전역에 걸쳐 반복되는 미니컬럼  
- 층 4 (입력 받음)
- 층 2/3 (측면 연결)
- 층 5/6 (피드백)

**AI 구현**:
- Numenta: 각 컬럼이 독립적 지도 생성
- 병렬 처리의 기반
- 견고성과 유연성

### 4. 주의와 게이팅(Gating)

**생물학**: 신경 동기화, 신경전달물질 조절  
**AI 구현**: Transformer attention, 곱셈적 게이팅  
**특징**: 선택적 정보 흐름

---

## 7. 신경 메커니즘에서 디지털 구현으로의 매핑

### Numenta의 방식 (가장 구체적)

```
신경 메커니즘             디지털 구현
─────────────────────────────────────
피질 컬럼              → 벡터 (특정 크기)
뉴런 활동              → 활성화 (이진 또는 확률)
시냅스 강도            → 가중치 행렬
Hebbian 학습           → 수식화된 업데이트 규칙
피드백 경로            → 역향 연결
```

**장점**: 직관적, 신경과학자도 이해 가능, 검증 가능  
**단점**: 스케일링의 어려움 (뇌 크기 대비 매우 작음)

### DeepMind의 방식 (함수적 매핑)

```
신경 회로 기능          AI 알고리즘
─────────────────────────────────
해마 재현              → Experience replay
그리드 셀              → 공간 임베딩
신경 동기화            → Attention weights
```

**장점**: 성숙한 기술, 스케일 가능, 검증된 성능  
**단점**: 신경 메커니즘과의 거리 멀어짐

### O'Reilly의 방식 (제약 조건 중심)

```
생물학적 제약           알고리즘 제약
─────────────────────────────────
국소 정보만            → 국소 학습 규칙
시냅스 지연            → 시간 윈도우
신경전달물질           → 게이팅 신호
```

**장점**: 신경 현실성, 심리학적 검증  
**단점**: 계산 성능 중하

---

## 8. 핵심 성과 비교표

| 항목 | Numenta | DeepMind | O'Reilly | OpenCog |
|------|---------|----------|----------|---------|
| **신경 메커니즘 구체성** | 매우 높음 | 중간 | 높음 | 낮음 |
| **계산 성능** | 중간 | 매우 높음 | 낮음 | 불명 |
| **구현 공개도** | 오픈소스 | 부분 공개 | 부분 공개 | 오픈소스 |
| **실행 증명** | Monty 실행 중 | AlphaGo 등 검증 | 심리학 모델 | 개발 중 |
| **이론-구현 일관성** | 매우 높음 | 중간 | 높음 | 중간 |
| **학계 영향력** | 중간 | 매우 높음 | 중간 | 낮음 |
| **상용화 전망** | 초기 단계 | 고도화 | 제한적 | 미래 |

---

## 9. 우리의 "뉴런 메커니즘 → 디지털 구현" 접근과의 비교

### 우리의 강점

✓ 신경 메커니즘의 **구체적 이해**로부터 출발  
✓ 각 단계의 **검증 가능성**  
✓ 신경과학-AI 양쪽과 대화 가능한 언어  

### 우리와의 유사점

**Numenta와 가장 유사**:
- 피질 메커니즘을 구현의 토대로
- 개념 → 코드의 직선적 경로
- 신경과학과 AI 간의 번역

**DeepMind로부터 배울 점**:
- 신경과학 영감이 어떻게 실제 성능 향상으로 연결되는가?
- 회로-수준 이해의 실용성

**O'Reilly로부터 배울 점**:
- 생물학적 제약 조건의 명시적 포함
- 심리학적 예측과의 비교

### 우리의 잠재적 차별점

1. **메타레벨**: 신경-심볼릭 상호작용의 명시적 모델링
2. **문화**: 한국의 게임/메타버스 맥락에서의 구현
3. **속도**: 이론과 구현의 신속한 피드백 루프

---

## 10. 최신 동향 (2024-2026)

### 신경-AI 수렴의 가속화

1. **Astrocyte 연구 부흥** (2024-2025)
   - 뉴런뿐 아니라 성상세포(astrocyte)의 역할 재평가
   - 삼부 시냅스(tripartite synapses) ← Transformer self-attention 매핑
   - 출처: MIT, PNAS 2023-2024

2. **동역학계(Dynamical Systems) 접근**
   - Transformer 해석의 신경과학적 렌즈
   - 신경망의 "상태 궤적(state trajectory)" 추적
   - 뇌와 AI의 동일한 계산 원리 탐색

3. **예측 모델 중심주의(Predictive Coding Paradigm)**
   - 뇌의 예측 기능 ← 생성 모델과의 수렴
   - 자유 에너지 원리(Free Energy Principle) 부흥
   - Karl Friston 등의 이론 AI로의 적용

### 자금 및 제도적 변화

- **Gates Foundation**: Thousand Brains Project 지원
- **NSF/DARPA**: NeuroAI 연구 공모 활성화 (2024 이후)
- **학제 간 대학원**: NeuroAI 전공 신설 (Stanford, MIT, Harvard 등)

---

## 11. 우리에게 던지는 질문들

### 1. 신경 메커니즘의 "필요 충분 조건" 문제

Numenta는 "피질 컬럼이 핵심"이라고 주장.  
DeepMind는 "해마 재현만으로도 충분"이라고 보임.  
O'Reilly는 "신경전달물질 세부 사항이 중요"라고 생각.

→ 우리는 어느 수준의 신경 메커니즘이 필요한가?

### 2. "참된 지능"의 정의

Embodied Turing Test (신체와 환경과의 상호작용)  
vs Language & Reasoning (인간의 고유 능력)

→ 아니마 시스템에서 어느 것을 우선할 것인가?

### 3. 신경과학 → AI의 매핑이 일대일인가?

뇌의 기능: 수백억 뉴런, 조 단위 시냅스  
AI의 스케일: 수십억 파라미터

→ 뇌-컴퓨터 비유의 한계는?

---

## 12. 참고 자료 및 출처

### 주요 논문 및 리소스

1. **Numenta**
   - [A Thousand Brains: A New Theory Of Intelligence](https://www.numenta.com/resources/books/a-thousand-brains-by-jeff-hawkins/)
   - [The Thousand Brains Project: A New Paradigm for Sensorimotor Intelligence](https://arxiv.org/abs/2412.18354)
   - [Thousand Brains Project GitHub](https://github.com/thousandbrainsproject)
   - [Numenta Press Release 2024/11/20](https://www.numenta.com/press/2024/11/20/thousand-brains-project/)

2. **DeepMind**
   - [Replay in biological and artificial neural networks](https://deepmind.google/blog/replay-in-biological-and-artificial-neural-networks/)
   - [Neuroscience-Inspired Artificial Intelligence (Neuron 2017)](https://www.cell.com/neuron/fulltext/S0896-6273(17)30509-3)
   - ["Vector-based navigation using grid-like representations" (Nature 2018)](https://www.nature.com/articles/nature22331) (간접 참고)

3. **NeuroAI 선언문**
   - [Catalyzing next-generation Artificial Intelligence through NeuroAI (Nature Communications 2023)](https://www.nature.com/articles/s41467-023-37180-x)
   - [Trainees' perspectives on NeuroAI (Nature Communications 2024)](https://www.nature.com/articles/s41467-024-53375-2)

4. **Randall O'Reilly**
   - [UC Davis Center for Neuroscience Profile](https://neuroscience.ucdavis.edu/people/randall-oreilly)
   - [LEABRA Model of Neural Interactions](https://ccnlab.org/papers/OReilly96phd.pdf)
   - [CCN Lab](https://ccnlab.org/)

5. **OpenCog / Ben Goertzel**
   - [OpenCog Hyperon: A Practical Path to Beneficial AGI and ASI](https://link.springer.com/chapter/10.1007/978-3-032-00686-8_18)
   - [OpenCog Hyperon Framework (ArXiv)](https://arxiv.org/abs/2310.18318)
   - [OpenCog Official](https://hyperon.opencog.org/)

6. **신경-AI 수렴 이론**
   - [Bridging Brains and Machines: A Unified Frontier](https://arxiv.org/abs/2507.10722)
   - [The neuroscience of transformers](https://arxiv.org/abs/2603.15339)
   - [Building transformers from neurons and astrocytes (PNAS 2023)](https://www.pnas.org/doi/10.1073/pnas.2219150120)
   - [Brain-machine convergent evolution (PNAS 2024)](https://www.pnas.org/doi/10.1073/pnas.2319709121)

7. **해마와 강화학습**
   - [Memory consolidation from a reinforcement learning perspective](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2024.1538741/full)
   - [Hippocampal replay and spatial learning (eLife 2023)](https://elifesciences.org/articles/82301)

---

## 맺음말

**NeuroAI 분야는 이제 선택이 아닌 필수다.**

데이터 스케일링과 모델 크기 증가만으로는 **진정한 지능**에 도달하기 어렵다는 것이 명확해졌다. Yann LeCun부터 Yoshua Bengio까지 AI 분야의 거장들이 명시적으로 신경과학 회귀를 주장하고 있다.

우리의 "뉴런 메커니즘 → 디지털 구현" 경로는 이러한 거대한 학문적 조류와 완벽히 정렬되어 있다.

**다음 단계**:
1. Numenta의 Monty 프레임워크 분석 (구체적 구현 방식)
2. 해마-피질 상호작용의 "아니마" 맥락에서의 모델링
3. Embodied learning의 메타버스 환경 설계

---

**조사 종료**  
AG-02  
2026-04-15
