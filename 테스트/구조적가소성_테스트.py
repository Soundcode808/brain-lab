"""
구조적가소성_테스트.py — 스스로 새 연결을 만드는 회로

설계 원칙: 죽음 없음. 경쟁 없음.
  - 연결 못 한 뉴런은 사라지지 않고 조용히 기다린다.
  - 주변에 활동이 충분해지면 그때 뻗어나간다.
  - 기회는 모든 뉴런에게 열려 있다.

검증 목표:
  1. 훈련 후 새 연결이 실제로 생기는가
  2. 새 연결은 자주 같이 발화한 뉴런들 사이에 집중되는가
  3. 거의 안 쓰인 연결은 약해지는가 (가지치기)
"""

import numpy as np
import json, os
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '실험결과')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── 기본 파라미터 (안정화 버전 동일) ─────────────────────────
N_E, N_I = 400, 100
N        = N_E + N_I
E_idx    = np.arange(N_E)
I_idx    = np.arange(N_E, N)

DT       = 0.5e-3
TAU_E    = 20e-3
TAU_I    = 10e-3
TAU_SLOW = 150e-3
TAU_FAST = 5e-3

V_REST   = -70e-3
V_THRESH = -55e-3
V_RESET  = -70e-3
REF_STEPS = int(2e-3 / DT)

G_EE    = 0.07e-3
W_EI    = 0.25e-3
W_IE    = -1.20e-3
W_II    = -0.40e-3
P_EE, P_EI, P_IE, P_II = 0.35, 0.30, 0.40, 0.20

I_BG    = 10e-3
NOISE   = 1.5e-3
STIM    = 20e-3

DECAY_S = np.exp(-DT / TAU_SLOW)
DECAY_F = np.exp(-DT / TAU_FAST)
IBG_VEC = np.zeros(N);  IBG_VEC[:N_E] = I_BG
TAU_VEC = np.where(np.arange(N) < N_E, TAU_E, TAU_I)

# ── 구조적 가소성 파라미터 ───────────────────────────────────
# "횟수"가 아니라 "일관성" 기준
# 자극 뉴런은 매 트라이얼마다 활성 → 높은 일관성
# 간접 활성 뉴런은 들쑥날쑥 → 낮은 일관성
CONSISTENCY_FORM_THRESH = 6   # N번 트라이얼에서 같이 활성 → 연결 생성
CONSISTENCY_PRUNE_MAX   = 2   # N번 이하 트라이얼에서 같이 활성 → 가지치기
P_FORM                  = 0.50
P_PRUNE                 = 0.20
MAX_NEW_CONN_RATIO      = 0.20


# ── 연결 행렬 생성 ────────────────────────────────────────────
def build_circuit(seed):
    rng = np.random.RandomState(seed)
    ee_mask = np.zeros((N_E, N_E), dtype=bool)
    cf      = np.zeros((N, N))
    for j in range(N_E):
        m = rng.rand(N_E) < P_EE;  m[j] = False
        ee_mask[m, j] = True
        mi = rng.rand(N_I) < P_EI
        cf[I_idx[mi], j] = W_EI
    for ji, j in enumerate(I_idx):
        m = rng.rand(N_E) < P_IE
        cf[E_idx[m], j] = W_IE
        mi = rng.rand(N_I) < P_II;  mi[ji] = False
        cf[I_idx[mi], j] = W_II
    return ee_mask, cf


print("=" * 60)
print("구조적 가소성 테스트 — 기다리는 뉴런, 죽지 않는 뉴런")
print(f"생성 기준: {CONSISTENCY_FORM_THRESH}트라이얼 일관성 | 제거 기준: {CONSISTENCY_PRUNE_MAX}트라이얼 이하")
print("=" * 60)

print("\n연결 행렬 생성 중...")
ee, cf = build_circuit(seed=0)
initial_conn = int(ee.sum())
print(f"  초기 E→E 연결 수: {initial_conn}개")

# ── 단일 트라이얼 시뮬레이션 ──────────────────────────────────
def run_trial(ee_mask, cf_mat, stim_mask, duration_ms=300, seed=42):
    """
    한 트라이얼 실행.
    반환: 자극 구간(50~200ms) 동안 활성화된 E 뉴런 집합 (bool 벡터)
    """
    steps = int(duration_ms * 1e-3 / DT)
    rng   = np.random.RandomState(seed)

    v   = np.full(N, V_REST) + rng.randn(N) * 1e-3
    s   = np.zeros(N_E)
    If  = np.zeros(N)
    ref = np.zeros(N, dtype=int)

    # 자극 구간 동안 발화한 E 뉴런 기록
    active_during_stim = np.zeros(N_E, dtype=bool)

    for step in range(steps):
        t_ms = step * DT * 1000

        Iext = np.zeros(N)
        if 50 <= t_ms < 200:
            Iext[stim_mask] = STIM

        Inmda        = np.zeros(N)
        Inmda[:N_E]  = ee_mask.dot(s) * G_EE

        n    = rng.randn(N) * NOISE * np.sqrt(DT / TAU_VEC)
        inr  = ref > 0
        dv   = (V_REST - v + Iext + Inmda + If + IBG_VEC) * (DT / TAU_VEC) + n
        v    = np.where(inr, V_RESET, v + dv)
        ref  = np.maximum(ref - 1, 0)

        fired = (v >= V_THRESH) & ~inr
        fi    = np.where(fired)[0]
        ef    = fi[fi < N_E]

        if len(fi) > 0:
            v[fired] = V_RESET;  ref[fired] = REF_STEPS
            if len(ef) > 0:
                s[ef] += 0.5 * (1.0 - s[ef])
                # 자극 구간 활성 기록
                if 50 <= t_ms < 200:
                    active_during_stim[ef] = True
            If += cf_mat[:, fi].sum(axis=1)

        s  *= DECAY_S
        If *= DECAY_F

    return active_during_stim   # (N_E,) bool


# ── 구조적 가소성 업데이트 ────────────────────────────────────
def apply_plasticity(ee_mask, consistency, n_trials, rng):
    """
    일관성 기반 연결 생성/제거.
    consistency[i,j] = i와 j가 함께 활성화된 트라이얼 수
    죽음 없음 — 연결 못 한 뉴런은 기다린다.
    """
    new_formed = 0
    new_pruned = 0
    max_new    = int(ee_mask.sum() * MAX_NEW_CONN_RATIO)

    # 벡터화: 조건 충족 쌍 찾기
    np.fill_diagonal(consistency, 0)   # 자기 자신 제외

    # 연결 생성 후보: 연결 없고 일관성 높음
    can_form = (~ee_mask) & (consistency >= CONSISTENCY_FORM_THRESH)
    # 연결 제거 후보: 연결 있고 일관성 낮음
    can_prune = ee_mask & (consistency <= CONSISTENCY_PRUNE_MAX)

    form_candidates = np.argwhere(can_form)
    prune_candidates = np.argwhere(can_prune)

    # 생성
    for idx in form_candidates:
        if new_formed >= max_new:
            break
        i, j = idx
        if rng.rand() < P_FORM:
            ee_mask[i, j] = True
            new_formed += 1

    # 제거 (가지치기)
    for idx in prune_candidates:
        i, j = idx
        if rng.rand() < P_PRUNE:
            ee_mask[i, j] = False
            new_pruned += 1

    return ee_mask, new_formed, new_pruned


# ── 메인 실험: 반복 훈련 ─────────────────────────────────────
N_TRIALS = 10   # 훈련 트라이얼 수

# 자극 패턴: E 뉴런 앞쪽 40% (일관된 패턴 반복)
stim_mask = np.zeros(N, dtype=bool)
stim_neurons = np.arange(int(N_E * 0.4))   # 뉴런 0~159번
stim_mask[stim_neurons] = True

print(f"\n자극 패턴: E 뉴런 {len(stim_neurons)}개 (0~{len(stim_neurons)-1}번) 반복 자극")
print(f"훈련 트라이얼: {N_TRIALS}회\n")

rng_plasticity = np.random.RandomState(77)
ee_current     = ee.copy()

history = []   # 트라이얼별 연결 수 기록

print("훈련 진행 중...")
print(f"{'트라이얼':>6} | {'연결 수':>8} | {'새 연결':>6} | {'제거':>6}")
print("-" * 36)

# 일관성 행렬: consistency[i,j] = i와 j가 함께 활성화된 트라이얼 수
consistency = np.zeros((N_E, N_E), dtype=np.int16)

for trial in range(N_TRIALS):
    # 트라이얼 실행 → 자극 구간 활성 뉴런 집합
    active = run_trial(ee_current, cf, stim_mask, duration_ms=300, seed=trial * 13)

    # 일관성 업데이트: 이번 트라이얼에서 같이 활성화된 뉴런 쌍
    # active는 (N_E,) bool → 외적으로 쌍 계산
    coact_this = np.outer(active, active).astype(np.int16)
    np.fill_diagonal(coact_this, 0)
    consistency += coact_this

    active_count = int(active.sum())

    # 구조적 가소성 업데이트 (매 트라이얼마다)
    ee_current, formed, pruned = apply_plasticity(
        ee_current, consistency.copy(), trial + 1, rng_plasticity
    )

    conn_now = int(ee_current.sum())
    history.append({'trial': trial + 1, 'connections': conn_now,
                    'active_neurons': active_count,
                    'formed': formed, 'pruned': pruned})
    print(f"  {trial+1:>4}회  | {conn_now:>8} | {formed:>+6} | {pruned:>6}  "
          f"(활성 {active_count}개)")

# ── 결과 분석 ─────────────────────────────────────────────────
final_conn    = int(ee_current.sum())
total_formed  = sum(h['formed'] for h in history)
total_pruned  = sum(h['pruned'] for h in history)
net_change    = final_conn - initial_conn

print(f"\n{'='*60}")
print(f"훈련 전 연결: {initial_conn}개")
print(f"훈련 후 연결: {final_conn}개  (순증가 {net_change:+d}개)")
print(f"총 생성: {total_formed}개 / 총 제거: {total_pruned}개")

# 새 연결이 자극 뉴런들 사이에 집중됐는가?
stim_set = set(stim_neurons.tolist())
new_conn_in_stim  = 0
new_conn_out_stim = 0

# ee_current(현재) XOR ee(초기) = 새로 생긴 연결
new_connections = ee_current & ~ee
for i in range(N_E):
    for j in range(N_E):
        if new_connections[i, j]:
            if i in stim_set and j in stim_set:
                new_conn_in_stim += 1
            else:
                new_conn_out_stim += 1

total_new = new_conn_in_stim + new_conn_out_stim
stim_concentration = new_conn_in_stim / (total_new + 1e-9)

print(f"\n새 연결 위치 분석:")
print(f"  자극 뉴런들 사이: {new_conn_in_stim}개")
print(f"  그 외:           {new_conn_out_stim}개")
print(f"  자극 뉴런 집중도: {stim_concentration:.1%}")

# 기준: 자극 뉴런이 전체의 40%이므로 랜덤이면 ~16% (0.4²)
# 집중도가 16%보다 훨씬 높으면 → 경험에 따라 연결이 선택적으로 생긴 것
random_expected = (len(stim_neurons) / N_E) ** 2
print(f"  무작위 기대값:   {random_expected:.1%}  "
      f"→ {stim_concentration / random_expected:.1f}배 집중")

passed1 = bool(total_formed > 0)
passed2 = bool(stim_concentration > random_expected * 1.5)

print(f"\n검증 결과:")
print(f"  새 연결 생성:        {'✅' if passed1 else '❌'}  ({total_formed}개)")
print(f"  경험 기반 집중:      {'✅' if passed2 else '❌'}  "
      f"(기대 {random_expected:.1%} → 실제 {stim_concentration:.1%})")

all_passed = bool(passed1 and passed2)
verdict = "PASS — 경험 기반 구조 변화 확인" if all_passed else "PARTIAL"
print(f"\n최종 판정: {verdict}")

# ── 저장 ─────────────────────────────────────────────────────
def to_python(obj):
    if isinstance(obj, (np.bool_,)):  return bool(obj)
    if isinstance(obj, np.integer):   return int(obj)
    if isinstance(obj, np.floating):  return float(obj)
    if isinstance(obj, dict):         return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):         return [to_python(v) for v in obj]
    return obj

output = to_python({
    'experiment': '구조적 가소성 (죽음 없는 버전)',
    'timestamp':  datetime.now().isoformat(),
    'design_note': '연결 못 한 뉴런은 사라지지 않고 기다린다. 기회는 모두에게.',
    'params': {
        'N_E': N_E, 'N_I': N_I,
        'n_trials': N_TRIALS,
        'consistency_form_thresh': CONSISTENCY_FORM_THRESH,
        'p_form': P_FORM,
        'p_prune': P_PRUNE,
        'stim_neurons': len(stim_neurons),
    },
    'results': {
        'initial_connections': initial_conn,
        'final_connections':   final_conn,
        'net_change':          net_change,
        'total_formed':        total_formed,
        'total_pruned':        total_pruned,
        'new_conn_in_stim':    new_conn_in_stim,
        'new_conn_out_stim':   new_conn_out_stim,
        'stim_concentration':  round(float(stim_concentration), 4),
        'random_expected':     round(float(random_expected), 4),
        'concentration_ratio': round(float(stim_concentration / (random_expected + 1e-9)), 2),
        'history':             history,
        'passed_new_formed':   passed1,
        'passed_concentrated': passed2,
    },
    'all_passed': all_passed,
    'verdict':    verdict,
})

out_path = os.path.join(RESULTS_DIR, 'V4-구조적가소성-결과.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n결과 저장: {out_path}")
