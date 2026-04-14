"""
검증_경량.py — Brian2 없이 순수 numpy로 돌리는 피질 컬럼 검증
핵심 수정: E→E는 NMDA 느린 시냅스(150ms), I→E/E→I는 빠른 GABA(5ms) — 생물학적으로 정확
"""

import numpy as np
import json, os
from datetime import datetime

np.random.seed(42)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '실험결과')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 회로 파라미터 ─────────────────────────────────────────
N_E, N_I = 400, 100
N = N_E + N_I

DT       = 0.5e-3    # 0.5ms 스텝
TAU_E    = 20e-3     # 흥분성 막 시정수
TAU_I    = 10e-3     # 억제성 막 시정수
TAU_SLOW = 150e-3    # NMDA: E→E 재귀 연결만 (사고를 "붙들어두는" 핵심)
TAU_FAST = 5e-3      # GABA: I 관련 모든 연결 (빠르게 소멸)

V_REST   = -70e-3
V_THRESH = -55e-3    # 임계 전위 (V_REST + 15mV)
V_RESET  = -70e-3

# 가중치: 1회 스파이크가 post-synaptic 전위에 주는 영향 (volt)
W_EE =  0.12e-3   # E→E, 느린 시냅스 (NMDA) — 개별은 작지만 재귀로 누적
W_EI =  0.25e-3   # E→I, 빠른 시냅스 (AMPA)
W_IE = -1.20e-3   # I→E, 빠른 시냅스 (GABA) — 억제
W_II = -0.40e-3   # I→I, 빠른 시냅스 (GABA)

P_EE, P_EI, P_IE, P_II = 0.35, 0.30, 0.40, 0.20

# 자극: -70mV + 20mV = -50mV 정상상태 → 임계값(-55mV) 초과 보장
STIM_STR = 20e-3
NOISE    = 1.5e-3   # 막 잡음

E_idx = np.arange(N_E)
I_idx = np.arange(N_E, N)

# ── 연결 행렬 생성 ────────────────────────────────────────
print("=" * 60)
print("피질 컬럼 경량 검증 (NMDA/GABA 분리 버전)")
print(f"N_E={N_E}, N_I={N_I} | tau_NMDA={int(TAU_SLOW*1000)}ms tau_GABA={int(TAU_FAST*1000)}ms")
print("=" * 60)
print("연결 행렬 생성 중...")

# [post, pre] 행렬 — 타입별 분리
conn_ee = np.zeros((N, N))   # E→E (느린 NMDA)
conn_fast = np.zeros((N, N)) # 나머지 (빠른 GABA/AMPA)

rng = np.random.RandomState(0)
for j in E_idx:   # pre=E
    # E→E
    mask = (rng.rand(N_E) < P_EE)
    mask[j] = False  # 자기 연결 제외
    conn_ee[E_idx[mask], j] = W_EE
    # E→I
    mask_i = (rng.rand(N_I) < P_EI)
    conn_fast[I_idx[mask_i], j] = W_EI

for j in I_idx:   # pre=I
    # I→E
    mask = (rng.rand(N_E) < P_IE)
    conn_fast[E_idx[mask], j] = W_IE
    # I→I
    mask_i = (rng.rand(N_I) < P_II)
    mask_i[j - N_E] = False
    conn_fast[I_idx[mask_i], j] = W_II

print(f"  E→E(NMDA): {int(np.sum(conn_ee > 0))} 연결")
print(f"  I→E(GABA): {int(np.sum(conn_fast[np.ix_(E_idx, I_idx)] < 0))} 연결")


# ── 시뮬레이션 함수 ───────────────────────────────────────
def simulate(duration_ms, stim_start_ms=None, stim_end_ms=None,
             stim_strength=STIM_STR, v_init_range=None, stim_frac=0.5):

    steps = int(duration_ms * 1e-3 / DT)

    # 자극 뉴런 고정 (E 뉴런 중 stim_frac 비율)
    stim_mask = np.zeros(N, dtype=bool)
    chosen = np.where(np.random.rand(N_E) < stim_frac)[0]
    stim_mask[chosen] = True

    # 초기 전위
    if v_init_range is None:
        v = np.full(N, V_REST) + np.random.randn(N) * 1e-3  # 1mV 잡음으로 동기화 방지
    else:
        v = np.random.uniform(v_init_range[0], v_init_range[1], N)

    I_slow = np.zeros(N)   # NMDA (E→E)
    I_fast = np.zeros(N)   # GABA/AMPA
    ref    = np.zeros(N, dtype=int)
    REF_STEPS = int(2e-3 / DT)

    spike_times, spike_neurons = [], []
    decay_slow = np.exp(-DT / TAU_SLOW)
    decay_fast = np.exp(-DT / TAU_FAST)

    for step in range(steps):
        t_ms = step * DT * 1000

        # 외부 자극
        I_ext = np.zeros(N)
        if stim_start_ms is not None and stim_end_ms is not None:
            if stim_start_ms <= t_ms < stim_end_ms:
                I_ext[stim_mask] = stim_strength

        tau = np.where(np.arange(N) < N_E, TAU_E, TAU_I)
        noise_v = np.random.randn(N) * NOISE * np.sqrt(DT / tau)

        in_ref = ref > 0
        I_total = I_ext + I_slow + I_fast
        dv = (V_REST - v + I_total) * (DT / tau) + noise_v

        v_new = np.where(in_ref, V_RESET, v + dv)
        v = v_new
        ref = np.maximum(ref - 1, 0)

        fired = (v >= V_THRESH) & (~in_ref)
        fired_idx = np.where(fired)[0]

        if len(fired_idx) > 0:
            spike_times.extend([t_ms] * len(fired_idx))
            spike_neurons.extend(fired_idx.tolist())
            v[fired] = V_RESET
            ref[fired] = REF_STEPS

            # 느린 NMDA (E→E만)
            I_slow += conn_ee[:, fired_idx].sum(axis=1)
            # 빠른 AMPA/GABA (나머지)
            I_fast += conn_fast[:, fired_idx].sum(axis=1)

        I_slow *= decay_slow
        I_fast *= decay_fast

    return np.array(spike_times), np.array(spike_neurons)


def rate_hz(st, sn, t0, t1, neuron_set):
    mask = (st >= t0) & (st < t1) & np.isin(sn, neuron_set)
    count = np.sum(mask)
    dur = (t1 - t0) * 1e-3
    return count / (len(neuron_set) * dur)


def to_python(obj):
    if isinstance(obj, (np.bool_,)):  return bool(obj)
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, dict):         return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [to_python(v) for v in obj]
    return obj


results = {}
all_passed = True

# ════════════════════════════════════════════════════════════
# V1 — 작업기억 (Working Memory)
# 자극 400ms → 제거 → 500ms 후에도 활동 ≥30% 유지
# ════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("V1: 작업기억 검증")

st, sn = simulate(1200, stim_start_ms=50, stim_end_ms=450)

r_during = rate_hz(st, sn,  50, 450, E_idx)
r_after  = rate_hz(st, sn, 450, 950, E_idx)
ratio_v1 = (r_after / r_during) if r_during > 0 else 0
passed_v1 = bool(ratio_v1 >= 0.30)

print(f"  자극 중:   {r_during:.2f} Hz")
print(f"  제거 후:   {r_after:.2f} Hz")
print(f"  유지 비율: {ratio_v1:.2%}  (기준 ≥30%)")
print(f"  결과: {'✅ 통과' if passed_v1 else '❌ 실패'}")

results['V1_working_memory'] = {
    'rate_during_hz': round(r_during, 3),
    'rate_after_hz':  round(r_after, 3),
    'retention_ratio': round(ratio_v1, 4),
    'criterion': 0.30,
    'passed': passed_v1
}
if not passed_v1:
    all_passed = False


# ════════════════════════════════════════════════════════════
# V2 — 맥락의존성 (Context Dependency)
# 사전 활성화 없음 vs 있음 → 동일 탐침에 대한 반응 차이 ≥30%
# 이것이 맥락의존성의 진짜 정의:
#   "이전에 무슨 일이 있었는지(맥락)에 따라 같은 자극에 다르게 반응"
# ════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("V2: 맥락의존성 검증 (사전활성화 없음 vs 있음)")

def simulate_two_phase(prime_start, prime_end, probe_start, probe_end,
                       total_ms, prime_str=STIM_STR, probe_str=STIM_STR,
                       v_init_range=None):
    """프라이밍 자극 후 탐침 자극 — 두 단계 시뮬레이션"""
    steps = int(total_ms * 1e-3 / DT)

    # 프라이밍 뉴런과 탐침 뉴런을 각각 50%씩 고정
    prime_mask = np.zeros(N, dtype=bool)
    probe_mask = np.zeros(N, dtype=bool)
    prime_mask[np.where(np.random.rand(N_E) < 0.5)[0]] = True
    probe_mask[np.where(np.random.rand(N_E) < 0.5)[0]] = True

    if v_init_range is None:
        v = np.full(N, V_REST) + np.random.randn(N) * 1e-3
    else:
        v = np.random.uniform(v_init_range[0], v_init_range[1], N)

    I_slow = np.zeros(N)
    I_fast = np.zeros(N)
    ref    = np.zeros(N, dtype=int)
    REF_STEPS = int(2e-3 / DT)
    decay_slow = np.exp(-DT / TAU_SLOW)
    decay_fast = np.exp(-DT / TAU_FAST)

    spike_times, spike_neurons = [], []

    for step in range(steps):
        t_ms = step * DT * 1000
        I_ext = np.zeros(N)
        if prime_start <= t_ms < prime_end:
            I_ext[prime_mask] = prime_str
        if probe_start <= t_ms < probe_end:
            I_ext[probe_mask] = probe_str

        tau = np.where(np.arange(N) < N_E, TAU_E, TAU_I)
        noise_v = np.random.randn(N) * NOISE * np.sqrt(DT / tau)
        in_ref = ref > 0
        dv = (V_REST - v + I_ext + I_slow + I_fast) * (DT / tau) + noise_v
        v_new = np.where(in_ref, V_RESET, v + dv)
        v = v_new
        ref = np.maximum(ref - 1, 0)

        fired = (v >= V_THRESH) & (~in_ref)
        fired_idx = np.where(fired)[0]
        if len(fired_idx) > 0:
            spike_times.extend([t_ms] * len(fired_idx))
            spike_neurons.extend(fired_idx.tolist())
            v[fired] = V_RESET
            ref[fired] = REF_STEPS
            I_slow += conn_ee[:, fired_idx].sum(axis=1)
            I_fast += conn_fast[:, fired_idx].sum(axis=1)
        I_slow *= decay_slow
        I_fast *= decay_fast

    return np.array(spike_times), np.array(spike_neurons)

# 맥락 없음: 탐침만 (300~500ms)
st_no, sn_no = simulate_two_phase(
    prime_start=9999, prime_end=9999,   # 프라이밍 없음
    probe_start=300, probe_end=500,
    total_ms=600)

# 맥락 있음: 프라이밍(50~200ms) → 휴지(200~300ms) → 탐침(300~500ms)
st_ctx, sn_ctx = simulate_two_phase(
    prime_start=50, prime_end=200,
    probe_start=300, probe_end=500,
    total_ms=600)

r_no  = rate_hz(st_no,  sn_no,  300, 500, E_idx)
r_ctx = rate_hz(st_ctx, sn_ctx, 300, 500, E_idx)
diff_v2 = abs(r_ctx - r_no) / (max(r_no, r_ctx) + 1e-9)
passed_v2 = bool(diff_v2 >= 0.30)

print(f"  맥락 없음 탐침 반응: {r_no:.2f} Hz")
print(f"  맥락 있음 탐침 반응: {r_ctx:.2f} Hz")
print(f"  상대 차이: {diff_v2:.2%}  (기준 ≥30%)")
print(f"  결과: {'✅ 통과' if passed_v2 else '❌ 실패'}")

results['V2_context_dependency'] = {
    'rate_no_context_hz':   round(r_no, 3),
    'rate_with_context_hz': round(r_ctx, 3),
    'relative_diff': round(diff_v2, 4),
    'criterion': 0.30,
    'passed': passed_v2
}
if not passed_v2:
    all_passed = False


# ════════════════════════════════════════════════════════════
# V3 — 반복 학습 (STDP 근사)
# 동일 자극 5회 반복 → 반응 크기 ≥5% 변화
# ════════════════════════════════════════════════════════════
print("\n" + "─"*60)
print("V3: 반복 적응 검증 (STDP 근사 — 가중치 누적)")

# STDP 완전 구현 대신: 재귀 가중치가 반복 발화로 강화되는 효과를 근사
# 매 trial 후 W_EE를 STDP 규칙으로 미세 업데이트
w_ee_current = conn_ee.copy()
A_PLUS, A_MINUS = 0.005, 0.006
trial_rates = []

for trial in range(5):
    # 이번 trial 시뮬레이션 (200ms, 자극 50-150ms)
    st_t, sn_t = simulate(250, stim_start_ms=30, stim_end_ms=180)
    r = rate_hz(st_t, sn_t, 30, 180, E_idx)
    trial_rates.append(r)
    print(f"  Trial {trial+1}: {r:.2f} Hz")

    # STDP 근사: 이번 trial에서 발화한 E 뉴런 쌍에 가중치 조정
    fired_this = np.unique(np.array(sn_t)[np.isin(sn_t, E_idx)]) if len(sn_t) > 0 else []
    if len(fired_this) > 0:
        for pre in fired_this[:20]:   # 계산량 제한
            for post in fired_this[:20]:
                if pre != post and conn_ee[post, pre] > 0:
                    conn_ee[post, pre] = min(
                        conn_ee[post, pre] * (1 + A_PLUS),
                        W_EE * 3  # 최대 3배까지 강화
                    )

first_r, last_r = trial_rates[0], trial_rates[-1]
chg_v3 = abs(last_r - first_r) / (first_r + 1e-9)
passed_v3 = bool(chg_v3 >= 0.05)

print(f"  변화량: {chg_v3:.2%}  (기준 ≥5%)")
print(f"  결과: {'✅ 통과' if passed_v3 else '❌ 실패'}")

results['V3_stdp_adaptation'] = {
    'trial_rates': [round(r, 3) for r in trial_rates],
    'first_trial_hz': round(first_r, 3),
    'last_trial_hz':  round(last_r, 3),
    'change_ratio':   round(chg_v3, 4),
    'criterion': 0.05,
    'passed': passed_v3
}
if not passed_v3:
    all_passed = False

conn_ee = w_ee_current  # 복원


# ════════════════════════════════════════════════════════════
print("\n" + "="*60)
verdict = "PASS — 최소 사고 단위 조건 충족" if all_passed else "PARTIAL — 일부 조건 미충족"
print(f"최종 판정: {verdict}")
print(f"  V1 작업기억:   {'✅' if results['V1_working_memory']['passed'] else '❌'}")
print(f"  V2 맥락의존성: {'✅' if results['V2_context_dependency']['passed'] else '❌'}")
print(f"  V3 STDP적응:   {'✅' if results['V3_stdp_adaptation']['passed'] else '❌'}")

output = to_python({
    'experiment': '피질컬럼 최소 사고단위 검증 (NMDA/GABA 분리)',
    'timestamp': datetime.now().isoformat(),
    'params': {
        'N_E': N_E, 'N_I': N_I,
        'tau_NMDA_ms': int(TAU_SLOW * 1000),
        'tau_GABA_ms': int(TAU_FAST * 1000),
        'p_EE': P_EE, 'W_EE_mV': W_EE * 1000,
        'stim_mV': STIM_STR * 1000
    },
    'results': results,
    'all_passed': all_passed,
    'verdict': verdict
})

out_path = os.path.join(RESULTS_DIR, 'V1-피질컬럼검증-결과.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n결과 저장: {out_path}")
