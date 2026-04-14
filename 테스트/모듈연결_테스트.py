"""
모듈연결_테스트.py — 두 피질 컬럼 회로의 정보 전달 검증
안정화 파라미터: NMDA 포화 모델 (s 게이팅) + I_bg=10mV

실험1: C1에 자극 → C2 활동이 올라가는가?
실험2: C1에 다른 패턴(A vs B) 입력 → C2에 다른 패턴이 생기는가?
"""

import numpy as np
import json, os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '실험결과')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 파라미터 (B4 안정화 버전) ─────────────────────────────────
N_E, N_I = 400, 100
N        = N_E + N_I
E_idx    = np.arange(N_E)
I_idx    = np.arange(N_E, N)

DT       = 0.5e-3       # 타임스텝
TAU_E    = 20e-3        # 흥분성 막 시정수
TAU_I    = 10e-3        # 억제성 막 시정수
TAU_SLOW = 150e-3       # NMDA (E→E)
TAU_FAST = 5e-3         # GABA/AMPA

V_REST   = -70e-3
V_THRESH = -55e-3
V_RESET  = -70e-3
REF_MS   = 2e-3
REF_STEPS = int(REF_MS / DT)

G_EE    = 0.07e-3       # NMDA 포화 가중치 (안정화 버전)
W_EI    = 0.25e-3       # E→I (AMPA)
W_IE    = -1.20e-3      # I→E (GABA)
W_II    = -0.40e-3      # I→I (GABA)

P_EE, P_EI, P_IE, P_II = 0.35, 0.30, 0.40, 0.20

I_BG    = 10e-3         # 배경 전류 E 뉴런 (시상 입력 시뮬)
NOISE   = 1.5e-3
STIM    = 20e-3         # 자극 강도

# 인터모듈 파라미터
P_12    = 0.10          # C1→C2 연결 확률 (0.05보다 크게)
W_12    = 0.50e-3       # C1→C2 가중치

# 감쇠 상수 (루프 밖에서 한 번만 계산)
DECAY_S = np.exp(-DT / TAU_SLOW)
DECAY_F = np.exp(-DT / TAU_FAST)

# I_bg 벡터 (E 뉴런에만 적용)
IBG_VEC  = np.zeros(N);  IBG_VEC[:N_E]  = I_BG
# 시정수 벡터
TAU_VEC  = np.where(np.arange(N) < N_E, TAU_E, TAU_I)


# ── 연결 행렬 생성 ────────────────────────────────────────────

def build_circuit(seed):
    """
    단일 피질 컬럼 회로 연결 생성
    ee_mask : (N_E, N_E) bool — E→E 연결 (NMDA)
    cf      : (N, N) float  — 나머지 시냅스 가중치 (GABA/AMPA)
    """
    rng = np.random.RandomState(seed)
    ee_mask = np.zeros((N_E, N_E), dtype=bool)
    cf      = np.zeros((N, N))

    for j in range(N_E):               # pre = E neuron j
        m = rng.rand(N_E) < P_EE
        m[j] = False
        ee_mask[m, j] = True           # E→E 이진 마스크
        mi = rng.rand(N_I) < P_EI
        cf[I_idx[mi], j] = W_EI        # E→I

    for ji, j in enumerate(I_idx):     # pre = I neuron j
        m = rng.rand(N_E) < P_IE
        cf[E_idx[m], j] = W_IE         # I→E
        mi = rng.rand(N_I) < P_II
        mi[ji] = False
        cf[I_idx[mi], j] = W_II        # I→I

    return ee_mask, cf


def build_inter(seed):
    """
    C1 E 뉴런 → C2 E 뉴런 인터모듈 연결
    반환: (N_E, N_E) float — c12[post_in_C2, pre_in_C1]
    """
    rng = np.random.RandomState(seed)
    c12 = np.zeros((N_E, N_E))
    for j in range(N_E):               # pre: C1 E neuron j
        m = rng.rand(N_E) < P_12
        c12[m, j] = W_12
    return c12


print("=" * 60)
print("모듈 연결 테스트 — 두 피질 컬럼 간 정보 전달")
print(f"G_EE={G_EE*1000:.2f}mV  I_bg={I_BG*1000:.0f}mV  "
      f"P_12={P_12}  W_12={W_12*1000:.2f}mV")
print("=" * 60)

print("\n연결 행렬 생성 중...")
ee1, cf1 = build_circuit(seed=0)
ee2, cf2 = build_circuit(seed=1)
c12      = build_inter(seed=999)

nz12 = int(np.sum(c12 > 0))
print(f"  C1 내부 E→E: {int(ee1.sum())} 연결")
print(f"  C2 내부 E→E: {int(ee2.sum())} 연결")
print(f"  C1→C2 인터:  {nz12} 연결  (기대 ≈ {int(N_E * N_E * P_12)})")

if nz12 == 0:
    raise RuntimeError("c12 연결이 0개입니다 — 파라미터를 확인하세요.")


# ── 코어 시뮬레이터 ──────────────────────────────────────────

def sim_dual(duration_ms, stim_mask_c1, stim_start_ms, stim_end_ms, seed=42):
    """
    두 회로 동시 시뮬레이션.
    stim_mask_c1 : (N,) bool — C1에 가할 외부 자극 마스크
    반환: (st1, sn1), (st2, sn2), pat2
      st/sn : 스파이크 시간(ms), 뉴런 인덱스
      pat2  : (N_E,) — 자극 기간 중 C2 E 뉴런 발화 횟수
    """
    steps = int(duration_ms * 1e-3 / DT)
    rng   = np.random.RandomState(seed)

    # 상태 초기화
    v1  = np.full(N, V_REST) + rng.randn(N) * 1e-3
    v2  = np.full(N, V_REST) + rng.randn(N) * 1e-3
    s1  = np.zeros(N_E)       # C1 NMDA gating (E 뉴런)
    s2  = np.zeros(N_E)       # C2 NMDA gating
    If1 = np.zeros(N)         # C1 빠른 시냅스 전류
    If2 = np.zeros(N)         # C2 빠른 시냅스 전류
    Ii  = np.zeros(N_E)       # C1→C2 인터모듈 전류 (C2 E 뉴런)
    r1  = np.zeros(N, dtype=int)
    r2  = np.zeros(N, dtype=int)

    st1, sn1, st2, sn2 = [], [], [], []
    pat2 = np.zeros(N_E)      # 자극 기간 중 C2 E 발화 횟수

    for step in range(steps):
        t = step * DT * 1000   # ms

        # ── CIRCUIT 1 ────────────────────────────────────
        Iext1 = np.zeros(N)
        if stim_start_ms <= t < stim_end_ms:
            Iext1[stim_mask_c1] = STIM

        # NMDA 전류: I[i] = G_EE * sum_j(ee1[i,j] * s1[j])
        Inmda1         = np.zeros(N)
        Inmda1[:N_E]   = ee1.dot(s1) * G_EE    # (N_E,N_E)·(N_E,) → (N_E,)

        n1 = rng.randn(N) * NOISE * np.sqrt(DT / TAU_VEC)
        in_r1 = r1 > 0
        dv1 = (V_REST - v1 + Iext1 + Inmda1 + If1 + IBG_VEC) * (DT / TAU_VEC) + n1
        v1  = np.where(in_r1, V_RESET, v1 + dv1)
        r1  = np.maximum(r1 - 1, 0)

        f1 = (v1 >= V_THRESH) & ~in_r1
        fi1 = np.where(f1)[0]
        ef1 = fi1[fi1 < N_E]

        if len(fi1) > 0:
            st1.extend([t] * len(fi1)); sn1.extend(fi1.tolist())
            v1[f1] = V_RESET;  r1[f1] = REF_STEPS
            if len(ef1) > 0:
                s1[ef1] += 0.5 * (1.0 - s1[ef1])   # NMDA 포화
            If1 += cf1[:, fi1].sum(axis=1)

        s1  *= DECAY_S
        If1 *= DECAY_F

        # ── INTER-MODULE: C1 E 스파이크 → C2 E 전류 ─────
        if len(ef1) > 0:
            Ii += c12[:, ef1].sum(axis=1)   # (N_E,N_E)[:,ef1].sum → (N_E,)
        Ii *= DECAY_F

        # ── CIRCUIT 2 ────────────────────────────────────
        Inmda2        = np.zeros(N)
        Inmda2[:N_E]  = ee2.dot(s2) * G_EE

        Imod2         = np.zeros(N)
        Imod2[:N_E]   = Ii              # 인터모듈 전류 C2 E 뉴런에 주입

        n2 = rng.randn(N) * NOISE * np.sqrt(DT / TAU_VEC)
        in_r2 = r2 > 0
        dv2 = (V_REST - v2 + Imod2 + Inmda2 + If2 + IBG_VEC) * (DT / TAU_VEC) + n2
        v2  = np.where(in_r2, V_RESET, v2 + dv2)
        r2  = np.maximum(r2 - 1, 0)

        f2 = (v2 >= V_THRESH) & ~in_r2
        fi2 = np.where(f2)[0]
        ef2 = fi2[fi2 < N_E]

        if len(fi2) > 0:
            st2.extend([t] * len(fi2)); sn2.extend(fi2.tolist())
            v2[f2] = V_RESET;  r2[f2] = REF_STEPS
            if len(ef2) > 0:
                s2[ef2] += 0.5 * (1.0 - s2[ef2])
                if stim_start_ms <= t < stim_end_ms:
                    pat2[ef2] += 1          # 자극 기간 발화 기록
            If2 += cf2[:, fi2].sum(axis=1)

        s2  *= DECAY_S
        If2 *= DECAY_F

    return (np.array(st1), np.array(sn1)), (np.array(st2), np.array(sn2)), pat2


def rate_hz(st, sn, t0, t1, neurons):
    if len(st) == 0:
        return 0.0
    m = (st >= t0) & (st < t1) & np.isin(sn, neurons)
    return float(np.sum(m)) / (len(neurons) * (t1 - t0) * 1e-3)


def to_python(obj):
    if isinstance(obj, (np.bool_,)):  return bool(obj)
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, dict):         return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [to_python(v) for v in obj]
    return obj


results = {}

# ════════════════════════════════════════════════════════════
# 실험1 — 신호 전달: C1 자극 → C2 활동 상승 확인
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("실험1: C1 자극(300~700ms) → C2 반응 여부")

# C1 자극 마스크: E 뉴런 50%
rng0 = np.random.RandomState(42)
mask_c1 = np.zeros(N, dtype=bool)
mask_c1[np.where(rng0.rand(N_E) < 0.5)[0]] = True

(st1, sn1), (st2, sn2), _ = sim_dual(
    duration_ms    = 1000,
    stim_mask_c1   = mask_c1,
    stim_start_ms  = 300,
    stim_end_ms    = 700,
    seed           = 42
)

r1_pre  = rate_hz(st1, sn1,   0, 300, E_idx)
r1_dur  = rate_hz(st1, sn1, 300, 700, E_idx)
r1_post = rate_hz(st1, sn1, 700,1000, E_idx)

r2_pre  = rate_hz(st2, sn2,   0, 300, E_idx)
r2_dur  = rate_hz(st2, sn2, 300, 700, E_idx)
r2_post = rate_hz(st2, sn2, 700,1000, E_idx)

print(f"\n  회로1 (자극 인가):")
print(f"    자극 전:  {r1_pre:.1f} Hz")
print(f"    자극 중:  {r1_dur:.1f} Hz  ← 외부 자극")
print(f"    자극 후:  {r1_post:.1f} Hz")

print(f"\n  회로2 (자극 없음, C1→C2 신호만):")
print(f"    C1 발화 전:  {r2_pre:.1f} Hz")
print(f"    C1 발화 중:  {r2_dur:.1f} Hz  ← C1 신호 수신")
print(f"    C1 발화 후:  {r2_post:.1f} Hz")

# 판정: C2가 C1 발화 중에 유의미하게 올라갔는가
baseline = r2_pre + 0.5   # 베이스라인 (0이면 0.5Hz 기준)
passed1  = bool(r2_dur > baseline * 2.0 and r1_dur > 5.0)

print(f"\n  C2 상승폭: {r2_pre:.1f} → {r2_dur:.1f} Hz")
print(f"  C1 정상 발화: {'✅' if r1_dur > 5 else '❌'} ({r1_dur:.1f} Hz)")
print(f"  결과: {'✅ 신호 전달 확인' if passed1 else '❌ 신호 전달 미확인'}")

results['exp1_signal_transfer'] = {
    'c1_pre_hz':  round(r1_pre,  1),
    'c1_dur_hz':  round(r1_dur,  1),
    'c1_post_hz': round(r1_post, 1),
    'c2_pre_hz':  round(r2_pre,  1),
    'c2_dur_hz':  round(r2_dur,  1),
    'c2_post_hz': round(r2_post, 1),
    'passed':     passed1,
}


# ════════════════════════════════════════════════════════════
# 실험2 — 패턴 구별: C1이 다른 패턴(A/B)을 입력 → C2에 다른 패턴
# ════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("실험2: C1 패턴A(앞쪽 뉴런) vs 패턴B(뒤쪽 뉴런) → C2 패턴 차이")

# 패턴 A: E 뉴런 앞쪽 절반 자극
mask_A = np.zeros(N, dtype=bool)
mask_A[:N_E // 2] = True

# 패턴 B: E 뉴런 뒤쪽 절반 자극
mask_B = np.zeros(N, dtype=bool)
mask_B[N_E // 2: N_E] = True

(_, _), (st2_A, sn2_A), pat2_A = sim_dual(
    duration_ms=700, stim_mask_c1=mask_A,
    stim_start_ms=200, stim_end_ms=600, seed=42
)
(_, _), (st2_B, sn2_B), pat2_B = sim_dual(
    duration_ms=700, stim_mask_c1=mask_B,
    stim_start_ms=200, stim_end_ms=600, seed=42
)

r2A = rate_hz(st2_A, sn2_A, 200, 600, E_idx)
r2B = rate_hz(st2_B, sn2_B, 200, 600, E_idx)

# 활성 패턴 비교 — 상위 20% 발화 뉴런 집합의 Jaccard 유사도
thr   = max(np.percentile(pat2_A, 80), np.percentile(pat2_B, 80), 0.5)
setA  = set(np.where(pat2_A >= thr)[0].tolist())
setB  = set(np.where(pat2_B >= thr)[0].tolist())

union = len(setA | setB)
inter = len(setA & setB)
jaccard = inter / (union + 1e-9) if union > 0 else 1.0

passed2 = bool(jaccard < 0.70 and (r2A > 0.5 or r2B > 0.5))

print(f"\n  C1 패턴A 자극 → C2 반응:  {r2A:.1f} Hz  (활성 뉴런 {len(setA)}개)")
print(f"  C1 패턴B 자극 → C2 반응:  {r2B:.1f} Hz  (활성 뉴런 {len(setB)}개)")
print(f"  두 패턴 Jaccard 유사도:   {jaccard:.3f}  (기준 < 0.70 이면 구별됨)")
print(f"  결과: {'✅ 패턴 구별 가능' if passed2 else '❌ 패턴 구별 불가'}")

results['exp2_pattern_distinction'] = {
    'c2_rate_A_hz':       round(r2A, 1),
    'c2_rate_B_hz':       round(r2B, 1),
    'active_neurons_A':   len(setA),
    'active_neurons_B':   len(setB),
    'intersection':       inter,
    'jaccard_similarity': round(float(jaccard), 4),
    'passed':             passed2,
}


# ════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
all_passed = bool(passed1 and passed2)
verdict = ("PASS — 모듈 간 정보 전달 확인" if all_passed
           else f"PARTIAL — {'신호전달 ❌' if not passed1 else '신호전달 ✅'} "
                f"{'패턴구별 ❌' if not passed2 else '패턴구별 ✅'}")
print(f"최종 판정: {verdict}")
print(f"  실험1 신호 전달:  {'✅' if passed1 else '❌'}")
print(f"  실험2 패턴 구별:  {'✅' if passed2 else '❌'}")

output = to_python({
    'experiment': '모듈 연결 검증 (C1→C2)',
    'timestamp':  datetime.now().isoformat(),
    'params': {
        'N_E': N_E, 'N_I': N_I,
        'G_EE_mV':   G_EE * 1000,
        'I_bg_mV':   I_BG * 1000,
        'P_12':      P_12,
        'W_12_mV':   W_12 * 1000,
        'tau_NMDA_ms': int(TAU_SLOW * 1000),
        'tau_GABA_ms': int(TAU_FAST * 1000),
    },
    'results':    results,
    'all_passed': all_passed,
    'verdict':    verdict,
})

out_path = os.path.join(RESULTS_DIR, 'V3-모듈연결-결과.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n결과 저장: {out_path}")
