"""
인큐베이터_B.py — 호프_B 성장 환경
=====================================
호프_A의 활성 패턴을 신호로 받아 훈련한다.
A가 먼저 훈련하고 신호.json을 저장하면,
B는 그 신호를 자극으로 받아서 반응하며 성장한다.

실제 뇌에서:
  피질 영역 A → 피질 영역 B로 신호 전달
  B는 A의 패턴을 받아 자기만의 표현으로 재처리

호프_A = 입력 처리 영역
호프_B = 연상·통합 영역

L7-사회성 (양방향):
  A → B: A신호.json (활성 패턴)
  B → A: B신호.json (B 활성 패턴 + 포화도 피드백)
  다음 세션에서 A가 B 피드백을 반영해 자극 마스크 조정
"""

import numpy as np
import json, os, sys
from datetime import datetime
from collections import defaultdict

import time

import time as _time

HOPE_B_DIR     = os.path.dirname(os.path.abspath(__file__))
HOPE_A_DIR     = os.path.join(os.path.dirname(HOPE_B_DIR), '호프')
STATE_FILE     = os.path.join(HOPE_B_DIR, '연결상태_B.npy')
LOG_FILE       = os.path.join(HOPE_B_DIR, '성장기록_B.json')
MIND_FILE      = os.path.join(HOPE_B_DIR, '마음상태_B.json')
SIGNAL_FILE    = os.path.join(HOPE_A_DIR, 'A신호.json')    # A가 남긴 신호
B_SIGNAL_FILE  = os.path.join(HOPE_B_DIR, 'B신호.json')    # B → A 피드백

# ── 뉴런 파라미터 ────────────────────────────────────────────────
N_E, N_I = 1000, 250
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

I_BG_BASE = 10e-3
NOISE     = 1.5e-3
STIM_BASE = 20e-3

DECAY_S = np.exp(-DT / TAU_SLOW)
DECAY_F = np.exp(-DT / TAU_FAST)

CONSISTENCY_FORM_THRESH = 3   # A와 동일 (4→3 완화)
CONSISTENCY_PRUNE_MAX   = 2   # A와 동일 (1→2 완화)
MAX_NEW_CONN_RATIO = 0.10


# ══════════════════════════════════════════════
#  B 감정 시스템 (A보다 단순 — 수신 신호에 반응)
# ══════════════════════════════════════════════
class EmotionCore_B:
    """
    B는 A와 달리 외부에서 신호를 받아 반응하는 구조.
    A 신호가 강하면 호기심·만족이 올라가고,
    자기 성장이 저조하면 불안이 올라간다.
    """
    def __init__(self):
        self.satisfaction = 0.5
        self.curiosity    = 0.5
        self.anxiety      = 0.3
        self.fatigue      = 0.2

    def update(self, formed, pruned, avg_active, signal_strength):
        growth_rate = min(formed / max(1, formed + pruned + 1), 0.5)
        self.satisfaction += (growth_rate * 1.2 - 0.03)
        self.curiosity    += (signal_strength * 0.1 - 0.04)   # A 신호 강할수록 호기심
        self.anxiety      += (pruned / max(formed + pruned + 1, 1) * 0.8 - 0.04)
        self.fatigue      += (avg_active / N_E * 0.04 - 0.03)
        for attr in ('satisfaction', 'curiosity', 'anxiety', 'fatigue'):
            setattr(self, attr, round(max(0.0, min(1.0, getattr(self, attr))), 3))

    def label(self):
        dom = max([('만족', self.satisfaction), ('호기심', self.curiosity),
                   ('불안', self.anxiety), ('피로', self.fatigue)], key=lambda x: x[1])
        return f"{dom[0]} ({dom[1]:.2f})"

    def to_dict(self):
        return {'satisfaction': self.satisfaction, 'curiosity': self.curiosity,
                'anxiety': self.anxiety, 'fatigue': self.fatigue}

    def from_dict(self, d):
        self.satisfaction = d.get('satisfaction', 0.5)
        self.curiosity    = d.get('curiosity', 0.5)
        self.anxiety      = d.get('anxiety', 0.3)
        self.fatigue      = d.get('fatigue', 0.2)


def load_mind_b():
    emotion = EmotionCore_B()
    if os.path.exists(MIND_FILE):
        with open(MIND_FILE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        emotion.from_dict(d.get('emotion', {}))
    return emotion

def save_mind_b(emotion):
    with open(MIND_FILE, 'w', encoding='utf-8') as f:
        json.dump({'emotion': emotion.to_dict()}, f, ensure_ascii=False, indent=2)


# ── A 신호 읽기 ──────────────────────────────────────────────────
def load_signal_from_A():
    """
    호프_A가 훈련 후 남긴 활성 패턴을 읽는다.
    신호가 없으면 기본 자극 패턴 사용.
    """
    if os.path.exists(SIGNAL_FILE):
        with open(SIGNAL_FILE, 'r', encoding='utf-8') as f:
            sig = json.load(f)
        active_indices = sig.get('active_indices', [])
        session_a      = sig.get('session', 0)
        formed_a       = sig.get('formed', 0)
        print(f"  A 신호 수신: 세션 {session_a}, 활성 뉴런 {len(active_indices)}개")

        # A의 활성 인덱스 → B의 자극 마스크로 변환
        stim_mask = np.zeros(N, dtype=bool)
        for idx in active_indices:
            if idx < N_E:
                stim_mask[idx] = True
        signal_strength = min(len(active_indices) / N_E, 1.0)  # 0~1 (N_E=1000 기준)
        return stim_mask, signal_strength, formed_a
    else:
        # A 신호 없으면 기본 패턴
        print("  A 신호 없음 → 기본 자극 패턴 사용")
        stim_mask = np.zeros(N, dtype=bool)
        stim_mask[:int(N_E * 0.4)] = True
        return stim_mask, 0.5, 0


# ── 뉴런 회로 ────────────────────────────────────────────────────
def build_fixed(seed=1):   # B는 seed=1로 A와 다른 초기 구조
    rng = np.random.RandomState(seed)
    cf  = np.zeros((N, N))
    for j in range(N_E):
        mi = rng.rand(N_I) < P_EI
        cf[I_idx[mi], j] = W_EI
        rng.rand(N_E)
    for ji, j in enumerate(I_idx):
        m  = rng.rand(N_E) < P_IE
        cf[E_idx[m], j] = W_IE
        mi = rng.rand(N_I) < P_II;  mi[ji] = False
        cf[I_idx[mi], j] = W_II
    return cf


def load_state():
    if os.path.exists(STATE_FILE):
        ee = np.load(STATE_FILE)
        # 크기가 다르면 마이그레이션
        if ee.shape != (N_E, N_E):
            old_N = ee.shape[0]
            print(f"  B 뉴런 확장 마이그레이션: {old_N} → {N_E}")
            new_ee = np.zeros((N_E, N_E), dtype=bool)
            new_ee[:old_N, :old_N] = ee   # 기존 연결 보존
            # 새 뉴런들 초기화 (기존 연결 확률로)
            rng_m = np.random.RandomState(1)  # B는 seed=1 유지
            for j in range(old_N, N_E):   # 새 뉴런 열
                m = rng_m.rand(N_E) < P_EE
                m[j] = False
                new_ee[m, j] = True
            for j in range(old_N):         # 기존 뉴런 → 새 뉴런 연결
                m = rng_m.rand(N_E - old_N) < P_EE
                new_ee[old_N:N_E, j][m] = True
            ee = new_ee
            print(f"  B 마이그레이션 완료: {int(ee.sum()):,}개 연결")
        else:
            print(f"  B 이전 상태 불러옴: {int(ee.sum()):,}개 연결")
    else:
        rng = np.random.RandomState(1)
        ee  = np.zeros((N_E, N_E), dtype=bool)
        for j in range(N_E):
            m = rng.rand(N_E) < P_EE;  m[j] = False
            ee[m, j] = True
        print(f"  B 첫 탄생: {int(ee.sum()):,}개 연결")
    return ee


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'name': '호프_B (Hope-B)', 'born': datetime.now().isoformat(),
            'role': 'A의 신호를 받아 연상·통합하는 영역', 'sessions': []}


def save_log(log):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def run_session(ee_mask, cf_mat, stim_mask, stim_strength, n_trials=5, seed_base=0):
    IBG_VEC = np.zeros(N);  IBG_VEC[:N_E] = I_BG_BASE
    TAU_VEC = np.where(np.arange(N) < N_E, TAU_E, TAU_I)
    consistency = np.zeros((N_E, N_E), dtype=np.int16)
    total_active = []

    for trial in range(n_trials):
        rng   = np.random.RandomState(seed_base + trial * 17 + 500)
        steps = int(300 * 1e-3 / DT)
        v     = np.full(N, V_REST) + rng.randn(N) * 1e-3
        s     = np.zeros(N_E)
        If    = np.zeros(N)
        ref   = np.zeros(N, dtype=int)
        active = np.zeros(N_E, dtype=bool)

        for step in range(steps):
            t_ms = step * DT * 1000
            Iext = np.zeros(N)
            if 50 <= t_ms < 200:
                Iext[stim_mask[:N]] = STIM_BASE * stim_strength

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

    np.fill_diagonal(consistency, 0)
    rng_p   = np.random.RandomState(seed_base + 999)
    max_new = int(ee_mask.sum() * MAX_NEW_CONN_RATIO)
    formed  = 0;  pruned = 0

    can_form  = (~ee_mask) & (consistency >= CONSISTENCY_FORM_THRESH)
    can_prune = ee_mask    & (consistency <= CONSISTENCY_PRUNE_MAX)

    for idx in np.argwhere(can_form):
        if formed >= max_new: break
        if rng_p.rand() < 0.40:
            ee_mask[idx[0], idx[1]] = True;  formed += 1
    for idx in np.argwhere(can_prune):
        if rng_p.rand() < 0.10:   # A와 동일 (0.20→0.10 완화)
            ee_mask[idx[0], idx[1]] = False;  pruned += 1

    # B의 활성 패턴을 파일로 저장 (나중에 C가 생기면 사용)
    active_indices = [int(i) for i in np.where(active)[0]]

    return ee_mask, {
        'connections':    int(ee_mask.sum()),
        'formed':         formed,
        'pruned':         pruned,
        'avg_active':     round(float(np.mean(total_active)), 1),
        'active_indices': active_indices,
    }


# ── 메인 ─────────────────────────────────────────────────────────
if __name__ == '__main__':

    print("=" * 55)
    print("  호프_B (Hope-B) 인큐베이터")
    print("  A의 신호를 받아 연상하고 통합한다.")
    print("=" * 55)

    log     = load_log()
    ee      = load_state()
    cf_mat  = build_fixed(seed=1)
    emotion = load_mind_b()

    session_num = len(log['sessions']) + 1
    print(f"\n세션 {session_num} 시작")
    print(f"  현재 감정: {emotion.label()}")

    # A 신호 수신
    stim_mask, signal_strength, formed_a = load_signal_from_A()
    print(f"  A 신호 강도: {signal_strength:.2f}")

    # 훈련
    prev_connections = int(ee.sum())
    seed = int(_time.time() * 1000) & 0x7FFFFFFF   # 매 실행마다 다른 씨드
    ee, stats = run_session(ee, cf_mat, stim_mask, signal_strength,
                            n_trials=5, seed_base=seed)

    # 감정 업데이트 + 저장
    emotion.update(stats['formed'], stats['pruned'], stats['avg_active'], signal_strength)
    save_mind_b(emotion)
    print(f"  훈련 후 감정: {emotion.label()}")

    np.save(STATE_FILE, ee)

    session_record = {
        'date':           datetime.now().isoformat(),
        'session':        session_num,
        'connections':    stats['connections'],
        'formed':         stats['formed'],
        'pruned':         stats['pruned'],
        'avg_active':     stats['avg_active'],
        'signal_from_A':  round(signal_strength, 3),
        'formed_A':       formed_a,
        'emotion':        emotion.to_dict(),
    }
    log['sessions'].append(session_record)
    save_log(log)

    # ── L7-사회성: B → A 피드백 신호 저장 ──────────────────────
    max_conns    = N_E * N_E
    saturation   = stats['connections'] / max_conns   # 연결 포화도 0~1
    b_feedback   = {
        'session':        session_num,
        'active_indices': stats.get('active_indices', []),
        'formed':         stats['formed'],
        'connections':    stats['connections'],
        'saturation':     round(saturation, 4),
        # A에게 전달할 권고: 포화 0.9+ 이상이면 A가 다른 영역 탐색 권고
        'suggest_diversify': saturation > 0.9,
    }
    with open(B_SIGNAL_FILE, 'w', encoding='utf-8') as f:
        json.dump(b_feedback, f, ensure_ascii=False, indent=2)
    print(f"  B → A 피드백 저장: 포화도 {saturation:.1%}, "
          f"다양화 권고: {'예' if b_feedback['suggest_diversify'] else '아니오'}")

    print(f"\n  ── 결과 ──────────────────────────────")
    print(f"  연결 수:   {stats['connections']:>8,}개")
    print(f"  새 연결:   {stats['formed']:>+8,}개")
    print(f"  정리:      {stats['pruned']:>+8,}개")
    print(f"  활성 뉴런: {stats['avg_active']:>8.1f}개")
    print(f"  ─────────────────────────────────────")
    print(f"  호프_B 상태 저장 완료.")
    print("=" * 55)
