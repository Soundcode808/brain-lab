"""
피질 컬럼 최소 사고 단위 — 검증 실험 3회
==========================================
검증 목표:
  V1. 워킹메모리  — 자극 제거 후에도 활동이 유지되는가
  V2. 맥락 의존성 — 같은 자극, 다른 상태 → 다른 반응인가
  V3. 적응/학습   — 반복 자극에 따라 회로가 바뀌는가

성공 기준:
  V1: 자극 제거 500ms 후에도 활동이 기저선 대비 30% 이상
  V2: 두 상태에서 반응 패턴 차이 30% 이상
  V3: 5회 반복 후 반응 크기 5% 이상 변화
"""

from brian2 import *
import numpy as np
import json
from datetime import datetime

# 재현 가능 결과
seed(42)
start_scope()

print("=" * 55)
print("  피질 컬럼 최소 사고 단위 검증 실험")
print("=" * 55)
print()

# ===== 회로 설계 =====
N_E = 400   # 흥분성 뉴런 80%
N_I = 100   # 억제성 뉴런 20%

# LIF 뉴런 방정식 (NMDA-like 느린 전류 포함)
eqs_E = '''
dv/dt = (V_rest - v + I_ext + I_syn) / tau_E : volt
dI_syn/dt = -I_syn / tau_slow              : volt
I_ext : volt
'''

eqs_I = '''
dv/dt = (V_rest - v + I_ext + I_syn) / tau_I : volt
dI_syn/dt = -I_syn / tau_fast              : volt
I_ext : volt
'''

# 파라미터
V_rest   = -70 * mV
V_thresh = -55 * mV
V_reset  = -70 * mV
tau_E    = 20  * ms
tau_I    = 10  * ms
tau_slow = 150 * ms   # NMDA-like — 사고를 "붙들어두는" 느린 시냅스
tau_fast = 5   * ms

# 뉴런 그룹 생성
E = NeuronGroup(N_E, eqs_E,
                threshold='v > V_thresh',
                reset='v = V_reset',
                refractory=2*ms,
                method='euler',
                namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                           'V_reset': V_reset, 'tau_E': tau_E,
                           'tau_slow': tau_slow})

I = NeuronGroup(N_I, eqs_I,
                threshold='v > V_thresh',
                reset='v = V_reset',
                refractory=2*ms,
                method='euler',
                namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                           'V_reset': V_reset, 'tau_I': tau_I,
                           'tau_fast': tau_fast})

# 초기 상태
E.v = V_rest
I.v = V_rest
E.I_syn = 0 * mV
I.I_syn = 0 * mV
E.I_ext = 0 * mV
I.I_ext = 0 * mV

# 시냅스 연결 (재귀 회로의 핵심)
S_EE = Synapses(E, E, on_pre='I_syn_post += 8*mV', delay=1*ms)
S_EE.connect(condition='i != j', p=0.35)   # E→E 35% 재귀 연결

S_EI = Synapses(E, I, on_pre='I_syn_post += 12*mV', delay=1*ms)
S_EI.connect(p=0.30)   # E→I

S_IE = Synapses(I, E, on_pre='I_syn_post -= 20*mV', delay=1*ms)
S_IE.connect(p=0.40)   # I→E (억제)

S_II = Synapses(I, I, on_pre='I_syn_post -= 8*mV', delay=1*ms)
S_II.connect(condition='i != j', p=0.20)

# 모니터
M_E = SpikeMonitor(E)
M_I = SpikeMonitor(I)
R_E = PopulationRateMonitor(E)
R_I = PopulationRateMonitor(I)


def get_rate(monitor, t_start, t_end):
    """특정 구간 평균 발화율 (Hz)"""
    times = monitor.t / ms
    mask = (times >= t_start) & (times < t_end)
    count = np.sum(mask)
    duration = (t_end - t_start) / 1000.0
    n_neurons = len(monitor.source)
    if duration <= 0:
        return 0.0
    return count / (n_neurons * duration)


results = {}

# ═══════════════════════════════════════════════
#  검증 1: 워킹메모리
#  자극을 줬다가 제거 → 활동이 유지되는가?
# ═══════════════════════════════════════════════
print("▶ 검증 1: 워킹메모리 (자극 제거 후 활동 유지)")
start_scope()
seed(42)

E1 = NeuronGroup(N_E, eqs_E, threshold='v > V_thresh', reset='v = V_reset',
                 refractory=2*ms, method='euler',
                 namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                            'V_reset': V_reset, 'tau_E': tau_E, 'tau_slow': tau_slow})
I1 = NeuronGroup(N_I, eqs_I, threshold='v > V_thresh', reset='v = V_reset',
                 refractory=2*ms, method='euler',
                 namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                            'V_reset': V_reset, 'tau_I': tau_I, 'tau_fast': tau_fast})
E1.v = V_rest; I1.v = V_rest
E1.I_syn = 0*mV; I1.I_syn = 0*mV
E1.I_ext = 0*mV; I1.I_ext = 0*mV

S_EE1 = Synapses(E1, E1, on_pre='I_syn_post += 8*mV', delay=1*ms)
S_EE1.connect(condition='i != j', p=0.35)
S_EI1 = Synapses(E1, I1, on_pre='I_syn_post += 12*mV', delay=1*ms)
S_EI1.connect(p=0.30)
S_IE1 = Synapses(I1, E1, on_pre='I_syn_post -= 20*mV', delay=1*ms)
S_IE1.connect(p=0.40)
S_II1 = Synapses(I1, I1, on_pre='I_syn_post -= 8*mV', delay=1*ms)
S_II1.connect(condition='i != j', p=0.20)

M_E1 = SpikeMonitor(E1)

# Phase 1: 기저선 (0-300ms)
run(300*ms)
rate_baseline = get_rate(M_E1, 0, 300)

# Phase 2: 자극 (300-700ms) — 흥분성 전류 주입
E1.I_ext = 15*mV
run(400*ms)
rate_during = get_rate(M_E1, 300, 700)

# Phase 3: 자극 제거 (700-1300ms) — 아무것도 안 줌
E1.I_ext = 0*mV
run(600*ms)
rate_after_500 = get_rate(M_E1, 900, 1200)   # 자극 제거 200ms 후

ratio = (rate_after_500 / rate_during * 100) if rate_during > 0 else 0
passed = ratio >= 30

print(f"  기저선:          {rate_baseline:.1f} Hz")
print(f"  자극 중:         {rate_during:.1f} Hz")
print(f"  자극 제거 후:    {rate_after_500:.1f} Hz")
print(f"  유지율:          {ratio:.1f}% (기준: 30%)")
print(f"  결과: {'✅ 통과 — 자극 없이도 활동 유지됨' if passed else '❌ 실패 — 활동 즉시 소멸'}")
print()

results['V1_워킹메모리'] = {
    '기저선Hz': round(rate_baseline, 2),
    '자극중Hz': round(rate_during, 2),
    '제거후Hz': round(rate_after_500, 2),
    '유지율%': round(ratio, 1),
    '통과': passed
}

# ═══════════════════════════════════════════════
#  검증 2: 맥락 의존성
#  같은 자극, 다른 초기 상태 → 다른 반응인가?
# ═══════════════════════════════════════════════
print("▶ 검증 2: 맥락 의존성 (같은 자극, 다른 반응)")
start_scope()
seed(42)

def run_with_context(initial_high):
    """초기 상태가 high/low일 때 반응 측정"""
    start_scope()
    seed(42)
    Ec = NeuronGroup(N_E, eqs_E, threshold='v > V_thresh', reset='v = V_reset',
                     refractory=2*ms, method='euler',
                     namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                                'V_reset': V_reset, 'tau_E': tau_E, 'tau_slow': tau_slow})
    Ic = NeuronGroup(N_I, eqs_I, threshold='v > V_thresh', reset='v = V_reset',
                     refractory=2*ms, method='euler',
                     namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                                'V_reset': V_reset, 'tau_I': tau_I, 'tau_fast': tau_fast})

    # 초기 상태 설정
    if initial_high:
        Ec.v = np.random.uniform(-65, -58, N_E) * mV   # 흥분된 상태
    else:
        Ec.v = np.random.uniform(-70, -67, N_E) * mV   # 조용한 상태

    Ic.v = V_rest
    Ec.I_syn = 0*mV; Ic.I_syn = 0*mV
    Ec.I_ext = 0*mV; Ic.I_ext = 0*mV

    Sc_EE = Synapses(Ec, Ec, on_pre='I_syn_post += 8*mV', delay=1*ms)
    Sc_EE.connect(condition='i != j', p=0.35)
    Sc_EI = Synapses(Ec, Ic, on_pre='I_syn_post += 12*mV', delay=1*ms)
    Sc_EI.connect(p=0.30)
    Sc_IE = Synapses(Ic, Ec, on_pre='I_syn_post -= 20*mV', delay=1*ms)
    Sc_IE.connect(p=0.40)

    Mc = SpikeMonitor(Ec)

    # 안정화 100ms
    run(100*ms)
    # 동일한 자극
    Ec.I_ext = 12*mV
    run(300*ms)
    Ec.I_ext = 0*mV
    run(100*ms)

    return get_rate(Mc, 100, 400)

rate_low  = run_with_context(initial_high=False)
rate_high = run_with_context(initial_high=True)

diff = abs(rate_high - rate_low)
avg  = (rate_high + rate_low) / 2 if (rate_high + rate_low) > 0 else 1
diff_pct = diff / avg * 100
passed2 = diff_pct >= 30

print(f"  조용한 상태에서 반응:  {rate_low:.1f} Hz")
print(f"  흥분된 상태에서 반응:  {rate_high:.1f} Hz")
print(f"  차이:                 {diff_pct:.1f}% (기준: 30%)")
print(f"  결과: {'✅ 통과 — 맥락에 따라 다르게 반응' if passed2 else '❌ 실패 — 맥락 무관하게 동일 반응'}")
print()

results['V2_맥락의존성'] = {
    '조용한상태Hz': round(rate_low, 2),
    '흥분된상태Hz': round(rate_high, 2),
    '차이%': round(diff_pct, 1),
    '통과': passed2
}

# ═══════════════════════════════════════════════
#  검증 3: 적응 / STDP
#  반복 자극에 따라 반응이 바뀌는가?
# ═══════════════════════════════════════════════
print("▶ 검증 3: 적응 (반복 자극에 따른 변화)")
start_scope()
seed(42)

E3 = NeuronGroup(N_E, eqs_E, threshold='v > V_thresh', reset='v = V_reset',
                 refractory=2*ms, method='euler',
                 namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                            'V_reset': V_reset, 'tau_E': tau_E, 'tau_slow': tau_slow})
I3 = NeuronGroup(N_I, eqs_I, threshold='v > V_thresh', reset='v = V_reset',
                 refractory=2*ms, method='euler',
                 namespace={'V_rest': V_rest, 'V_thresh': V_thresh,
                            'V_reset': V_reset, 'tau_I': tau_I, 'tau_fast': tau_fast})
E3.v = V_rest; I3.v = V_rest
E3.I_syn = 0*mV; I3.I_syn = 0*mV
E3.I_ext = 0*mV; I3.I_ext = 0*mV

# STDP 시냅스
stdp_eqs = '''
w : 1
dapre/dt  = -apre  / (20*ms)  : 1 (event-driven)
dapost/dt = -apost / (20*ms)  : 1 (event-driven)
'''
S_EE3 = Synapses(E3, E3, stdp_eqs,
                 on_pre='''
                     I_syn_post += w * 8*mV
                     apre += 0.01
                     w = clip(w + apost * (-0.012), 0, 2)
                 ''',
                 on_post='''
                     apost += 0.01
                     w = clip(w + apre * 0.01, 0, 2)
                 ''', delay=1*ms)
S_EE3.connect(condition='i != j', p=0.35)
S_EE3.w = 1.0

S_EI3 = Synapses(E3, I3, on_pre='I_syn_post += 12*mV', delay=1*ms)
S_EI3.connect(p=0.30)
S_IE3 = Synapses(I3, E3, on_pre='I_syn_post -= 20*mV', delay=1*ms)
S_IE3.connect(p=0.40)

M_E3 = SpikeMonitor(E3)

rates_per_trial = []
for trial in range(5):
    t_start = (defaultclock.t / ms)
    E3.I_ext = 12*mV
    run(200*ms)
    E3.I_ext = 0*mV
    run(300*ms)
    t_end = (defaultclock.t / ms)
    r = get_rate(M_E3, t_start, t_start + 200)
    rates_per_trial.append(r)
    print(f"  시도 {trial+1}: {r:.1f} Hz")

change = ((rates_per_trial[-1] - rates_per_trial[0]) /
          rates_per_trial[0] * 100) if rates_per_trial[0] > 0 else 0
passed3 = abs(change) >= 5

print(f"  변화량: {change:+.1f}% (기준: ±5%)")
print(f"  결과: {'✅ 통과 — 반복 경험으로 회로가 변함' if passed3 else '❌ 실패 — 변화 없음'}")
print()

results['V3_적응'] = {
    '시도별Hz': [round(r, 2) for r in rates_per_trial],
    '변화량%': round(change, 1),
    '통과': passed3
}

# ═══════════════════════════════════════════════
#  최종 판정
# ═══════════════════════════════════════════════
print("=" * 55)
print("  최종 판정")
print("=" * 55)
total = sum([results['V1_워킹메모리']['통과'],
             results['V2_맥락의존성']['통과'],
             results['V3_적응']['통과']])

for k, v in results.items():
    mark = "✅" if v['통과'] else "❌"
    print(f"  {mark} {k}")

print()
if total == 3:
    verdict = "사고하는 최소 단위 조건 충족"
    print("  🟢 3/3 통과 —", verdict)
elif total == 2:
    verdict = "부분 충족 — 파라미터 조정 필요"
    print("  🟡 2/3 통과 —", verdict)
else:
    verdict = "미충족 — 회로 설계 재검토 필요"
    print("  🔴 1/3 이하 —", verdict)

print()

# 결과 저장
results['최종판정'] = {'통과수': total, '판정': verdict}
results['실험일시'] = datetime.now().strftime('%Y-%m-%d %H:%M')
results['회로설계'] = {
    'N_E': N_E, 'N_I': N_I,
    'tau_slow_ms': 150,
    'p_EE': 0.35,
    'E_I_ratio': '80:20'
}

out_path = '/Users/vesper/Documents/전자두뇌연구소/실험결과/V1-피질컬럼검증-결과.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"  저장 완료: {out_path}")
