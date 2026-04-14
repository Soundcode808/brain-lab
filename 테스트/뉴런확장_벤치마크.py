"""
뉴런확장_벤치마크.py — 뉴런 수를 늘렸을 때 MacBook이 어떻게 반응하는지 측정
500 → 1000 → 2500 순서로 실행 시간 + CPU 온도 체감 확인
"""

import numpy as np
import time

DT       = 0.5e-3
TAU_E    = 20e-3
TAU_I    = 10e-3
TAU_SLOW = 150e-3
TAU_FAST = 5e-3
V_REST   = -70e-3
V_THRESH = -55e-3
V_RESET  = -70e-3
G_EE     = 0.07e-3
W_EI     = 0.25e-3
W_IE     = -1.20e-3
W_II     = -0.40e-3
P_EE, P_EI, P_IE, P_II = 0.35, 0.30, 0.40, 0.20
I_BG     = 10e-3
NOISE    = 1.5e-3
STIM     = 20e-3
REF_STEPS = int(2e-3 / DT)
DECAY_S  = np.exp(-DT / TAU_SLOW)
DECAY_F  = np.exp(-DT / TAU_FAST)


def benchmark(n_total, duration_ms=200, seed=0):
    n_e = int(n_total * 0.8)
    n_i = n_total - n_e
    E_idx = np.arange(n_e)
    I_idx = np.arange(n_e, n_total)
    tau_vec = np.where(np.arange(n_total) < n_e, TAU_E, TAU_I)
    ibg     = np.zeros(n_total);  ibg[:n_e] = I_BG

    # 연결 행렬
    rng = np.random.RandomState(seed)
    ee  = np.zeros((n_e, n_e), dtype=bool)
    cf  = np.zeros((n_total, n_total))
    for j in range(n_e):
        m = rng.rand(n_e) < P_EE;  m[j] = False
        ee[m, j] = True
        mi = rng.rand(n_i) < P_EI
        cf[I_idx[mi], j] = W_EI
    for ji, j in enumerate(I_idx):
        m  = rng.rand(n_e) < P_IE
        cf[E_idx[m], j] = W_IE
        mi = rng.rand(n_i) < P_II;  mi[ji] = False
        cf[I_idx[mi], j] = W_II

    stim_mask = np.zeros(n_total, dtype=bool)
    stim_mask[:int(n_e * 0.4)] = True

    steps = int(duration_ms * 1e-3 / DT)
    v   = np.full(n_total, V_REST) + rng.randn(n_total) * 1e-3
    s   = np.zeros(n_e)
    If  = np.zeros(n_total)
    ref = np.zeros(n_total, dtype=int)

    t_start = time.time()
    total_spikes = 0

    for step in range(steps):
        t_ms = step * DT * 1000
        Iext = np.zeros(n_total)
        if 50 <= t_ms < 150:
            Iext[stim_mask] = STIM

        Inmda        = np.zeros(n_total)
        Inmda[:n_e]  = ee.dot(s) * G_EE

        n_v  = rng.randn(n_total) * NOISE * np.sqrt(DT / tau_vec)
        inr  = ref > 0
        dv   = (V_REST - v + Iext + Inmda + If + ibg) * (DT / tau_vec) + n_v
        v    = np.where(inr, V_RESET, v + dv)
        ref  = np.maximum(ref - 1, 0)

        fired = (v >= V_THRESH) & ~inr
        fi    = np.where(fired)[0]
        ef    = fi[fi < n_e]

        if len(fi) > 0:
            total_spikes += len(fi)
            v[fired] = V_RESET;  ref[fired] = REF_STEPS
            if len(ef) > 0:
                s[ef] += 0.5 * (1.0 - s[ef])
            If += cf[:, fi].sum(axis=1)
        s  *= DECAY_S
        If *= DECAY_F

    elapsed = time.time() - t_start
    conn_count = int(ee.sum())
    avg_rate = total_spikes / (n_e * duration_ms * 1e-3)

    return elapsed, conn_count, avg_rate


print("=" * 55)
print("  MacBook 한계 측정 — 뉴런 수별 실행 시간")
print("  (200ms 시뮬레이션 기준)")
print("=" * 55)
print(f"{'뉴런 수':>8} | {'연결 수':>10} | {'소요 시간':>10} | {'평균 발화율':>10}")
print("-" * 55)

sizes = [500, 1000, 2500]

for n in sizes:
    print(f"  {n}뉴런 측정 중...", end='', flush=True)
    elapsed, conns, rate = benchmark(n, duration_ms=200)
    bar = "▓" * min(int(elapsed * 5), 30)
    print(f"\r  {n:>6}개  | {conns:>10,} | {elapsed:>8.1f}초  | {rate:>8.1f} Hz  {bar}")

print("=" * 55)
print("\n  MacBook 부담 기준:")
print("  3초 이하  → 쾌적")
print("  3~15초   → 팬 살짝, 사용 가능")
print("  15초 이상 → 부담 시작")
