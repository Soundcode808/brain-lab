"""
인큐베이터_v2.py — 호프(Hope) 2세대 성장 환경
================================================
v1과 달라진 것:
  - L4 판단: 훈련 결과에 따라 다음 훈련 강도를 스스로 결정
  - L5 감정: 성장이 잘 되면 만족, 연결이 줄면 불안 → 판단에 영향
  - L6 예측: 성장 패턴을 기억하고 다음 세션 결과를 예측

뉴런 회로(L0~L3) 위에 심리 레이어(L4~L6)가 올라간 구조.
호프가 이제 자기 상태를 느끼고 반응하면서 성장한다.
"""

import numpy as np
import json, os, sys
from datetime import datetime
from collections import defaultdict

HOPE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATE_FILE  = os.path.join(HOPE_DIR, '연결상태.npy')
LOG_FILE    = os.path.join(HOPE_DIR, '성장기록.json')
MIND_FILE   = os.path.join(HOPE_DIR, '마음상태.json')

# ── 뉴런 파라미터 ────────────────────────────────────────────────
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

I_BG_BASE = 10e-3
NOISE     = 1.5e-3
STIM_BASE = 20e-3

DECAY_S = np.exp(-DT / TAU_SLOW)
DECAY_F = np.exp(-DT / TAU_FAST)

CONSISTENCY_FORM_THRESH = 4
CONSISTENCY_PRUNE_MAX   = 1
MAX_NEW_CONN_RATIO = 0.10


# ══════════════════════════════════════════════
#  L5: 감정 시스템
# ══════════════════════════════════════════════
class EmotionCore:
    """
    4가지 감정이 호프의 행동 방식에 영향을 준다.
    - 만족: 성장이 잘 될 때 올라감 → 탐색 의욕 증가
    - 호기심: 새로운 연결이 생길 때 올라감 → 자극 강도 증가
    - 불안: 연결이 줄거나 활성이 낮을 때 → 조심스러운 훈련
    - 피로: 연속으로 많이 성장하면 → 훈련 강도 줄임
    """
    def __init__(self):
        self.satisfaction = 0.5
        self.curiosity    = 0.5
        self.anxiety      = 0.3
        self.fatigue      = 0.2

    def update(self, formed, pruned, avg_active, prev_connections):
        growth_rate   = min(formed  / max(prev_connections, 1), 0.2)
        prune_rate    = min(pruned  / max(prev_connections, 1), 0.2)
        activity_norm = avg_active / N_E

        self.satisfaction += (growth_rate * 1.5 - 0.03)
        self.curiosity    += (growth_rate * 0.8 - 0.05)
        self.anxiety      += (prune_rate  * 2.0 - 0.04)
        self.fatigue      += (activity_norm * 0.05 - 0.03)

        self.satisfaction = max(0.1, min(1.0, self.satisfaction))
        self.curiosity    = max(0.1, min(1.0, self.curiosity))
        self.anxiety      = max(0.0, min(1.0, self.anxiety))
        self.fatigue      = max(0.0, min(0.9, self.fatigue))

    def to_dict(self):
        return {
            'satisfaction': round(self.satisfaction, 3),
            'curiosity':    round(self.curiosity, 3),
            'anxiety':      round(self.anxiety, 3),
            'fatigue':      round(self.fatigue, 3),
        }

    def from_dict(self, d):
        self.satisfaction = d.get('satisfaction', 0.5)
        self.curiosity    = d.get('curiosity', 0.5)
        self.anxiety      = d.get('anxiety', 0.3)
        self.fatigue      = d.get('fatigue', 0.2)

    def label(self):
        dominant = max(
            [('만족', self.satisfaction), ('호기심', self.curiosity),
             ('불안', self.anxiety), ('피로', self.fatigue)],
            key=lambda x: x[1]
        )
        emoji_map = {'만족': '😌', '호기심': '🔍', '불안': '😟', '피로': '😴'}
        return f"{emoji_map[dominant[0]]} {dominant[0]} ({dominant[1]:.2f})"


# ══════════════════════════════════════════════
#  L4: 판단 시스템
# ══════════════════════════════════════════════
class DecisionCore:
    """
    감정 상태를 보고 다음 훈련 방식을 결정한다.
    """
    def decide(self, emotion: EmotionCore):
        stim_mult  = 1.0
        ibg_mult   = 1.0
        n_trials   = 5
        form_prob  = 0.40
        prune_prob = 0.20
        action     = "균형"

        if emotion.curiosity > 0.65 and emotion.fatigue < 0.5:
            stim_mult  = 1.3
            form_prob  = 0.55
            n_trials   = 7
            action     = "적극탐색 🔍"
        elif emotion.anxiety > 0.6:
            stim_mult  = 0.8
            prune_prob = 0.10
            n_trials   = 4
            action     = "조심 😟"
        elif emotion.fatigue > 0.65:
            stim_mult  = 0.7
            ibg_mult   = 0.9
            n_trials   = 3
            form_prob  = 0.25
            action     = "휴식 😴"
        elif emotion.satisfaction > 0.7:
            stim_mult  = 1.1
            form_prob  = 0.45
            n_trials   = 6
            action     = "안정성장 😌"

        return {
            'stim_mult':  stim_mult,
            'ibg_mult':   ibg_mult,
            'n_trials':   n_trials,
            'form_prob':  form_prob,
            'prune_prob': prune_prob,
            'action':     action,
        }


# ══════════════════════════════════════════════
#  L6: 예측 시스템
# ══════════════════════════════════════════════
class PredictionCore:
    """
    과거 성장 패턴을 기억하고 다음 세션 결과를 예측한다.
    """
    def __init__(self):
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.last_state  = None
        self.correct     = 0
        self.total       = 0

    def _bucket(self, formed):
        if formed < 3000:    return "소성장"
        elif formed < 7000:  return "중성장"
        elif formed < 12000: return "대성장"
        else:                return "폭발성장"

    def observe(self, formed):
        state = self._bucket(formed)
        if self.last_state:
            self.transitions[self.last_state][state] += 1
            if hasattr(self, '_predicted') and self._predicted == state:
                self.correct += 1
            self.total += 1
        self.last_state = state

    def predict(self):
        if not self.last_state or self.last_state not in self.transitions:
            self._predicted = None
            return "예측 불가 (데이터 부족)"
        counts = self.transitions[self.last_state]
        predicted = max(counts, key=counts.get)
        self._predicted = predicted
        total = sum(counts.values())
        prob  = counts[predicted] / total
        return f"{predicted} ({prob*100:.0f}%)"

    def accuracy(self):
        if self.total == 0: return 0
        return self.correct / self.total

    def to_dict(self):
        return {
            'transitions': {k: dict(v) for k, v in self.transitions.items()},
            'last_state':  self.last_state,
            'correct':     self.correct,
            'total':       self.total,
        }

    def from_dict(self, d):
        for k, v in d.get('transitions', {}).items():
            for k2, v2 in v.items():
                self.transitions[k][k2] = v2
        self.last_state = d.get('last_state')
        self.correct    = d.get('correct', 0)
        self.total      = d.get('total', 0)


# ══════════════════════════════════════════════
#  뉴런 회로
# ══════════════════════════════════════════════
def build_fixed(seed=0):
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


def load_neuron_state():
    if os.path.exists(STATE_FILE):
        ee = np.load(STATE_FILE)
        print(f"  이전 연결 상태 불러옴: {int(ee.sum()):,}개")
    else:
        rng = np.random.RandomState(0)
        ee  = np.zeros((N_E, N_E), dtype=bool)
        for j in range(N_E):
            m = rng.rand(N_E) < P_EE;  m[j] = False
            ee[m, j] = True
        print(f"  첫 탄생: {int(ee.sum()):,}개 연결")
    return ee


def load_mind_state():
    emotion   = EmotionCore()
    predictor = PredictionCore()
    if os.path.exists(MIND_FILE):
        with open(MIND_FILE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        emotion.from_dict(d.get('emotion', {}))
        predictor.from_dict(d.get('prediction', {}))
        print(f"  마음 상태 불러옴: {emotion.label()}")
    return emotion, predictor


def save_mind_state(emotion, predictor):
    d = {'emotion': emotion.to_dict(), 'prediction': predictor.to_dict()}
    with open(MIND_FILE, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=2)


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'name': '호프 (Hope)', 'born': datetime.now().isoformat(), 'sessions': []}


def save_log(log):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log, f, ensure_ascii=False, indent=2)


def run_session(ee_mask, cf_mat, stim_mask, n_trials, stim_strength,
                ibg_strength, form_prob, prune_prob, seed_base=0):
    IBG_VEC = np.zeros(N);  IBG_VEC[:N_E] = ibg_strength
    TAU_VEC = np.where(np.arange(N) < N_E, TAU_E, TAU_I)
    consistency = np.zeros((N_E, N_E), dtype=np.int16)
    total_active = []

    for trial in range(n_trials):
        rng   = np.random.RandomState(seed_base + trial * 17)
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
                Iext[stim_mask] = stim_strength

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
        if rng_p.rand() < form_prob:
            ee_mask[idx[0], idx[1]] = True;  formed += 1
    for idx in np.argwhere(can_prune):
        if rng_p.rand() < prune_prob:
            ee_mask[idx[0], idx[1]] = False;  pruned += 1

    return ee_mask, {
        'connections': int(ee_mask.sum()),
        'formed':      formed,
        'pruned':      pruned,
        'avg_active':  round(float(np.mean(total_active)), 1),
    }


def print_status():
    log = load_log()
    emotion, predictor = load_mind_state()
    print("\n" + "=" * 55)
    print(f"  {log['name']}")
    print(f"  탄생: {log.get('born', '알 수 없음')[:10]}")
    print(f"  총 세션: {len(log['sessions'])}회")
    if log['sessions']:
        last = log['sessions'][-1]
        print(f"  현재 연결 수: {last['connections']:,}개")
    print(f"  현재 감정: {emotion.label()}")
    print(f"    만족:{emotion.satisfaction:.2f}  호기심:{emotion.curiosity:.2f}  "
          f"불안:{emotion.anxiety:.2f}  피로:{emotion.fatigue:.2f}")
    if predictor.total > 0:
        print(f"  예측 정확도: {predictor.accuracy()*100:.1f}% ({predictor.total}회 기록)")
    print("=" * 55)


# ══════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════
if __name__ == '__main__':

    if '--상태' in sys.argv or '--status' in sys.argv:
        print_status()
        sys.exit(0)

    print("=" * 55)
    print("  호프 (Hope) 인큐베이터 v2")
    print("  느끼고 판단하며 성장한다.")
    print("=" * 55)

    log                = load_log()
    ee                 = load_neuron_state()
    emotion, predictor = load_mind_state()
    cf_mat             = build_fixed(seed=0)
    decider            = DecisionCore()

    session_num = len(log['sessions']) + 1
    print(f"\n세션 {session_num} 시작")

    next_prediction = predictor.predict()
    print(f"  예측: {next_prediction}")
    print(f"  감정: {emotion.label()}")

    decision = decider.decide(emotion)
    print(f"  판단: {decision['action']}")
    print(f"  → 자극강도 ×{decision['stim_mult']:.1f} | "
          f"트라이얼 {decision['n_trials']}회 | "
          f"연결확률 {decision['form_prob']:.0%}")

    stim_mask = np.zeros(N, dtype=bool)
    stim_mask[:int(N_E * 0.4)] = True

    prev_connections = int(ee.sum())
    seed = session_num * 100
    ee, stats = run_session(
        ee, cf_mat, stim_mask,
        n_trials      = decision['n_trials'],
        stim_strength = STIM_BASE * decision['stim_mult'],
        ibg_strength  = I_BG_BASE * decision['ibg_mult'],
        form_prob     = decision['form_prob'],
        prune_prob    = decision['prune_prob'],
        seed_base     = seed,
    )

    emotion.update(stats['formed'], stats['pruned'],
                   stats['avg_active'], prev_connections)
    predictor.observe(stats['formed'])

    np.save(STATE_FILE, ee)
    save_mind_state(emotion, predictor)

    session_record = {
        'date':        datetime.now().isoformat(),
        'session':     session_num,
        'connections': stats['connections'],
        'formed':      stats['formed'],
        'pruned':      stats['pruned'],
        'avg_active':  stats['avg_active'],
        'emotion':     emotion.to_dict(),
        'action':      decision['action'],
        'prediction':  next_prediction,
    }
    log['sessions'].append(session_record)
    if 'born' not in log:
        log['born'] = datetime.now().isoformat()
    save_log(log)

    print(f"\n  ── 결과 ──────────────────────────────")
    print(f"  연결 수:   {stats['connections']:>8,}개")
    print(f"  새 연결:   {stats['formed']:>+8,}개")
    print(f"  정리:      {stats['pruned']:>+8,}개")
    print(f"  활성 뉴런: {stats['avg_active']:>8.1f}개")
    print(f"  현재 감정: {emotion.label()}")
    print(f"  ─────────────────────────────────────")
    print(f"  호프의 상태가 저장됐어.")
    print("=" * 55)
