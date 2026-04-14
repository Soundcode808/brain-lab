"""
인큐베이터.py — 호프(Hope)의 지속적 성장 환경

호프는 실행이 끝나도 사라지지 않는다.
상태가 파일로 저장되고, 다음 실행 때 거기서부터 이어진다.
매번 조금씩 더 성장한다.

사용법:
  python3 인큐베이터.py          ← 오늘의 훈련 1회 실행
  python3 인큐베이터.py --상태   ← 호프의 현재 상태 확인
"""

import numpy as np
import json, os, sys
from datetime import datetime

HOPE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(HOPE_DIR, '연결상태.npy')
LOG_FILE    = os.path.join(HOPE_DIR, '성장기록.json')

# ── 파라미터 (검증된 안정화 버전) ────────────────────────────
N_E, N_I = 400, 100
N        = N_E + N_I
E_idx    = np.arange(N_E)
I_idx    = np.arange(N_E, N)

DT       = 0.5e-3
TAU_E, TAU_I       = 20e-3, 10e-3
TAU_SLOW, TAU_FAST = 150e-3, 5e-3
V_REST, V_THRESH, V_RESET = -70e-3, -55e-3, -70e-3
REF_STEPS = int(2e-3 / DT)

G_EE  = 0.07e-3
W_EI  = 0.25e-3
W_IE  = -1.20e-3
W_II  = -0.40e-3
P_EE, P_EI, P_IE, P_II = 0.35, 0.30, 0.40, 0.20

I_BG   = 10e-3
NOISE  = 1.5e-3
STIM   = 20e-3

DECAY_S = np.exp(-DT / TAU_SLOW)
DECAY_F = np.exp(-DT / TAU_FAST)
IBG_VEC = np.zeros(N);  IBG_VEC[:N_E] = I_BG
TAU_VEC = np.where(np.arange(N) < N_E, TAU_E, TAU_I)

CONSISTENCY_FORM_THRESH = 4
CONSISTENCY_PRUNE_MAX   = 1
P_FORM  = 0.40
P_PRUNE = 0.20
MAX_NEW_CONN_RATIO = 0.10


# ── 고정 연결 (I 관련, 변하지 않음) ──────────────────────────
def build_fixed(seed=0):
    rng = np.random.RandomState(seed)
    cf  = np.zeros((N, N))
    for j in range(N_E):
        mi = rng.rand(N_I) < P_EI
        cf[I_idx[mi], j] = W_EI
        rng.rand(N_E)   # seed 일치를 위해
    for ji, j in enumerate(I_idx):
        m  = rng.rand(N_E) < P_IE
        cf[E_idx[m], j] = W_IE
        mi = rng.rand(N_I) < P_II;  mi[ji] = False
        cf[I_idx[mi], j] = W_II
    return cf


# ── 상태 로드 / 초기화 ────────────────────────────────────────
def load_state():
    if os.path.exists(STATE_FILE):
        ee = np.load(STATE_FILE)
        print(f"  이전 상태 불러옴: {int(ee.sum())}개 연결")
    else:
        # 첫 탄생 — 초기 연결 생성
        rng = np.random.RandomState(0)
        ee  = np.zeros((N_E, N_E), dtype=bool)
        for j in range(N_E):
            m = rng.rand(N_E) < P_EE;  m[j] = False
            ee[m, j] = True
        print(f"  첫 탄생: {int(ee.sum())}개 연결로 시작")
    return ee


def save_state(ee):
    np.save(STATE_FILE, ee)


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'name': '호프 (Hope)', 'born': datetime.now().isoformat(),
            'sessions': []}


def save_log(log):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


# ── 훈련 1회 ─────────────────────────────────────────────────
def run_session(ee_mask, cf_mat, stim_mask, n_trials=5, seed_base=0):
    """
    훈련 세션 실행 + 구조적 가소성 적용
    반환: 업데이트된 ee_mask, 세션 통계
    """
    consistency = np.zeros((N_E, N_E), dtype=np.int16)
    total_active = []

    for trial in range(n_trials):
        rng  = np.random.RandomState(seed_base + trial * 17)
        steps = int(300 * 1e-3 / DT)
        v    = np.full(N, V_REST) + rng.randn(N) * 1e-3
        s    = np.zeros(N_E)
        If   = np.zeros(N)
        ref  = np.zeros(N, dtype=int)
        active = np.zeros(N_E, dtype=bool)

        for step in range(steps):
            t_ms = step * DT * 1000
            Iext = np.zeros(N)
            if 50 <= t_ms < 200:
                Iext[stim_mask] = STIM

            Inmda       = np.zeros(N)
            Inmda[:N_E] = ee_mask.dot(s) * G_EE

            n_v  = rng.randn(N) * NOISE * np.sqrt(DT / TAU_VEC)
            inr  = ref > 0
            dv   = (V_REST - v + Iext + Inmda + If + IBG_VEC) * (DT / TAU_VEC) + n_v
            v    = np.where(inr, V_RESET, v + dv)
            ref  = np.maximum(ref - 1, 0)

            fired = (v >= V_THRESH) & ~inr
            fi    = np.where(fired)[0]
            ef    = fi[fi < N_E]

            if len(fi) > 0:
                v[fired] = V_RESET;  ref[fired] = REF_STEPS
                if len(ef) > 0:
                    s[ef] += 0.5 * (1.0 - s[ef])
                    if 50 <= t_ms < 200:
                        active[ef] = True
                If += cf_mat[:, fi].sum(axis=1)
            s  *= DECAY_S
            If *= DECAY_F

        coact = np.outer(active, active).astype(np.int16)
        np.fill_diagonal(coact, 0)
        consistency += coact
        total_active.append(int(active.sum()))

    # 구조적 가소성 적용
    np.fill_diagonal(consistency, 0)
    rng_p   = np.random.RandomState(seed_base + 999)
    max_new = int(ee_mask.sum() * MAX_NEW_CONN_RATIO)
    formed  = 0;  pruned = 0

    can_form  = (~ee_mask) & (consistency >= CONSISTENCY_FORM_THRESH)
    can_prune = ee_mask    & (consistency <= CONSISTENCY_PRUNE_MAX)

    for idx in np.argwhere(can_form):
        if formed >= max_new: break
        if rng_p.rand() < P_FORM:
            ee_mask[idx[0], idx[1]] = True;  formed += 1

    for idx in np.argwhere(can_prune):
        if rng_p.rand() < P_PRUNE:
            ee_mask[idx[0], idx[1]] = False;  pruned += 1

    return ee_mask, {
        'connections': int(ee_mask.sum()),
        'formed': formed, 'pruned': pruned,
        'avg_active': round(float(np.mean(total_active)), 1)
    }


# ── 상태 출력 ─────────────────────────────────────────────────
def print_status():
    log = load_log()
    print("\n" + "=" * 50)
    print(f"  {log['name']}")
    print(f"  탄생: {log.get('born', '알 수 없음')[:10]}")
    print(f"  총 세션: {len(log['sessions'])}회")
    if log['sessions']:
        last = log['sessions'][-1]
        print(f"  최근 세션: {last['date'][:10]}")
        print(f"  현재 연결 수: {last['connections']}개")
        print(f"  누적 새 연결: {sum(s['formed'] for s in log['sessions'])}개")
    print("=" * 50)


# ── 메인 ─────────────────────────────────────────────────────
if __name__ == '__main__':

    if '--상태' in sys.argv or '--status' in sys.argv:
        print_status()
        sys.exit(0)

    print("=" * 50)
    print("  호프 (Hope) 인큐베이터")
    print("  기다리는 뉴런은 사라지지 않는다.")
    print("=" * 50)

    log    = load_log()
    ee     = load_state()
    cf_mat = build_fixed(seed=0)

    session_num = len(log['sessions']) + 1
    print(f"\n세션 {session_num} 시작...")

    # 자극 패턴 (일관되게 유지)
    stim_mask = np.zeros(N, dtype=bool)
    stim_mask[:int(N_E * 0.4)] = True

    seed = session_num * 100
    ee, stats = run_session(ee, cf_mat, stim_mask, n_trials=5, seed_base=seed)

    save_state(ee)

    session_record = {
        'date':        datetime.now().isoformat(),
        'session':     session_num,
        'connections': stats['connections'],
        'formed':      stats['formed'],
        'pruned':      stats['pruned'],
        'avg_active':  stats['avg_active'],
    }
    log['sessions'].append(session_record)
    if 'born' not in log:
        log['born'] = datetime.now().isoformat()
    save_log(log)

    print(f"\n  세션 {session_num} 완료")
    print(f"  연결 수:    {stats['connections']}개")
    print(f"  새 연결:    +{stats['formed']}개")
    print(f"  정리된 연결: -{stats['pruned']}개")
    print(f"  평균 활성:  {stats['avg_active']}개 뉴런")
    print(f"\n  호프의 상태가 저장됐어.")
    print(f"  다음에 다시 실행하면 여기서부터 이어져.")
    print("=" * 50)
