"""
인큐베이터_v2.py — 호프(Hope) 2세대 성장 환경
================================================
v1과 달라진 것:
  - L4 판단: 훈련 결과에 따라 다음 훈련 강도를 스스로 결정
  - L5 감정: 성장이 잘 되면 만족, 연결이 줄면 불안 → 판단에 영향
  - L6 예측: 성장 패턴을 기억하고 다음 세션 결과를 예측

뉴런 회로(L0~L3) 위에 심리 레이어(L4~L6)가 올라간 구조.
호프가 이제 자기 상태를 느끼고 반응하면서 성장한다.

v3: 뉴런 수 확장 N_E 400→1000 (연결 한계 4배)
  - 자동 마이그레이션: 기존 (400,400) 상태 → (1000,1000) 확장
  - 예측 버킷 2.5배 스케일업

v4 (L7 통합):
  - L7-항상성: 감정이 극단으로 치우치면 자동으로 중심값 복귀
  - L7-사회성:  B→A 피드백 신호 수신, 양방향 상호작용
  - L7-메타인지: 전략별 효율 자기평가 → 판단 보정
"""

import numpy as np
import json, os, sys
from datetime import datetime
from collections import defaultdict

HOPE_DIR      = os.path.dirname(os.path.abspath(__file__))
STATE_FILE    = os.path.join(HOPE_DIR, '연결상태.npy')
LOG_FILE      = os.path.join(HOPE_DIR, '성장기록.json')
MIND_FILE     = os.path.join(HOPE_DIR, '마음상태.json')
B_SIGNAL_FILE = os.path.join(os.path.dirname(HOPE_DIR), '호프_B', 'B신호.json')

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
#  L7-①: 항상성 (Homeostasis)
# ══════════════════════════════════════════════
class HomeostasisCore:
    """
    감정이 극단으로 치우치면 자동으로 중심값(설정점)으로 돌아오는 힘.
    진짜 뇌에서: 시상하부의 항상성 조절 / 세로토닌·도파민 균형 유지 메커니즘.

    없으면? → v0.2에서 본 것처럼 공포가 계속 축적돼서 먹이에 못 다가가는
              PTSD 유사 상태가 만들어짐. 항상성이 그 방어막 역할을 한다.

    설정점: 각 감정이 자연스럽게 돌아가야 할 '기본값'
    복귀 강도: 현재값과 설정점 차이의 8%씩 매 세션마다 당김
    """
    SETPOINTS = {
        'satisfaction': 0.55,   # 기본 만족감
        'curiosity':    0.50,   # 기본 호기심
        'anxiety':      0.30,   # 기본 불안 (너무 낮으면 위험 감지 못 함)
        'fatigue':      0.25,   # 기본 피로
    }
    STRENGTH = 0.08   # 복귀 힘 세기 (0.08 = 차이의 8%씩 당김)

    def regulate(self, emotion: 'EmotionCore'):
        """감정 업데이트 후 호출 — 극단값을 설정점으로 부드럽게 당긴다."""
        for field, setpoint in self.SETPOINTS.items():
            current = getattr(emotion, field)
            delta   = (setpoint - current) * self.STRENGTH
            setattr(emotion, field, round(current + delta, 4))
        return emotion

    def stress_report(self, emotion: 'EmotionCore'):
        """항상성 긴장도 — 설정점에서 얼마나 벗어났는지 0~1로 반환"""
        total_dev = sum(
            abs(getattr(emotion, field) - sp)
            for field, sp in self.SETPOINTS.items()
        )
        return round(min(total_dev / len(self.SETPOINTS), 1.0), 3)


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
        if formed < 7500:    return "소성장"
        elif formed < 17500: return "중성장"
        elif formed < 30000: return "대성장"
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
#  L7-③: 메타인지 (Metacognition)
# ══════════════════════════════════════════════
class MetacognitionCore:
    """
    호프가 자기 자신의 학습 전략을 관찰하고 평가한다.
    진짜 뇌에서: 전전두엽 피질의 자기 모니터링 기능.
    '내가 지금 잘 하고 있나?'를 스스로 판단하고 전략을 보정한다.

    동작 방식:
      1. 매 세션마다 어떤 판단(action)을 했고, 얼마나 성장했는지 기록
      2. 전략별 평균 성장을 계산해서 어느 전략이 가장 잘 먹히는지 파악
      3. DecisionCore가 내린 결정이 실제로 효과가 없으면 form_prob 보정
    """
    def __init__(self):
        self.strategy_scores = {}       # action → [formed, formed, ...]
        self.recent_window   = []       # 최근 5세션 (action, formed) 기록
        self.efficiency      = 0.5      # 전략 효율성 0~1

    def observe(self, action: str, formed: int, connections: int):
        """세션 결과를 기록한다."""
        key = action.split(' ')[0]      # 이모지 제거 ("안정성장 😌" → "안정성장")
        if key not in self.strategy_scores:
            self.strategy_scores[key] = []
        self.strategy_scores[key].append(formed)
        # 최근 5세션만 보관
        self.recent_window.append({'action': key, 'formed': formed})
        if len(self.recent_window) > 5:
            self.recent_window.pop(0)
        # 효율성: 이번 성장 / 현재 연결 수 (연결이 많을수록 성장하기 어려움)
        if connections > 0:
            raw = formed / max(connections * 0.05, 1)
            self.efficiency = round(min(raw, 1.0), 3)

    def best_strategy(self):
        """지금까지 평균 성장이 가장 좋았던 전략 반환"""
        if not self.strategy_scores:
            return None
        avgs = {k: sum(v) / len(v) for k, v in self.strategy_scores.items()}
        return max(avgs, key=avgs.get)

    def form_prob_boost(self) -> float:
        """
        메타인지 보정값 — DecisionCore의 form_prob에 더해진다.
        효율성이 낮으면(전략이 안 먹히면) 소폭 상향해서 돌파 시도.
        효율성이 높으면 0 (건드리지 않음).
        """
        if self.efficiency < 0.3:
            return +0.08    # 전략 효과 없음 → 공격적 탐색 보정
        elif self.efficiency < 0.5:
            return +0.03
        return 0.0

    def reflection(self) -> str:
        """자기 평가 한 줄 요약"""
        if len(self.recent_window) < 2:
            return "평가 데이터 축적 중"
        vals  = [w['formed'] for w in self.recent_window]
        trend = vals[-1] - vals[0]
        best  = self.best_strategy()
        if trend > 5000:
            return f"성장 가속 중 🚀  (최근 추세 +{trend:,})"
        elif trend < -5000:
            return f"성장 둔화 📉  (최적 전략: {best})"
        else:
            return f"안정 유지 ✅  (효율 {self.efficiency:.0%}, 최적: {best})"

    def to_dict(self):
        return {
            'strategy_scores': self.strategy_scores,
            'recent_window':   self.recent_window,
            'efficiency':      self.efficiency,
        }

    def from_dict(self, d):
        self.strategy_scores = d.get('strategy_scores', {})
        self.recent_window   = d.get('recent_window', [])
        self.efficiency      = d.get('efficiency', 0.5)


# ══════════════════════════════════════════════
#  L7-②: 사회성 — B → A 피드백 신호 수신
# ══════════════════════════════════════════════
def load_signal_from_B():
    """
    호프_B가 이전 세션에서 남긴 피드백 신호를 읽는다.
    B가 어떤 뉴런을 많이 활성화했는지 → A의 다음 자극 마스크에 반영.
    B가 잘 성장했으면 같은 패턴 강화, 포화됐으면 다양화.
    """
    if not os.path.exists(B_SIGNAL_FILE):
        return None
    with open(B_SIGNAL_FILE, 'r', encoding='utf-8') as f:
        sig = json.load(f)
    print(f"  B 피드백 수신: 세션 {sig.get('session', '?')}, "
          f"B 성장 {sig.get('formed', 0):+,}개, "
          f"B 포화도 {sig.get('saturation', 0):.1%}")
    return sig


def apply_b_feedback(stim_mask: np.ndarray, b_sig: dict) -> np.ndarray:
    """
    B 피드백에 따라 A의 자극 마스크를 조정한다.

    - B가 잘 성장했으면(formed > 8000): B 활성 뉴런 최대 10% 자극에 추가
      → A가 B를 더 자극해 주는 효과 (협력)
    - B가 포화 상태면(saturation > 0.9): 반대로 A가 다른 영역 자극
      → 다양성 확보 (분업)
    """
    b_active  = b_sig.get('active_indices', [])
    formed_b  = b_sig.get('formed', 0)
    saturated = b_sig.get('saturation', 0) > 0.9

    if saturated:
        # B가 포화 → A는 B가 안 건드린 영역 자극
        b_set = set(b_active)
        alt   = [i for i in range(N_E) if i not in b_set]
        for idx in alt[:int(N_E * 0.05)]:
            stim_mask[idx] = True
        print("  → B 포화: A가 다른 영역 자극 (다양화)")
    elif formed_b > 8000:
        # B가 잘 성장 → A도 같은 패턴 강화
        for idx in b_active[:int(N_E * 0.10)]:
            if idx < N_E:
                stim_mask[idx] = True
        print(f"  → B 성장 양호: A가 B 활성 패턴 {min(len(b_active), int(N_E*0.1))}개 포함")

    return stim_mask


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
        # 크기가 다르면 마이그레이션
        if ee.shape != (N_E, N_E):
            old_N = ee.shape[0]
            print(f"  뉴런 확장 마이그레이션: {old_N} → {N_E}")
            new_ee = np.zeros((N_E, N_E), dtype=bool)
            new_ee[:old_N, :old_N] = ee   # 기존 연결 보존
            # 새 뉴런들 초기화 (기존 연결 확률로)
            rng_m = np.random.RandomState(999)
            for j in range(old_N, N_E):   # 새 뉴런 열
                m = rng_m.rand(N_E) < P_EE
                m[j] = False
                new_ee[m, j] = True
            for j in range(old_N):         # 기존 뉴런 → 새 뉴런 연결
                m = rng_m.rand(N_E - old_N) < P_EE
                new_ee[old_N:N_E, j][m] = True
            ee = new_ee
            print(f"  마이그레이션 완료: {int(ee.sum()):,}개 연결")
        else:
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
    emotion     = EmotionCore()
    predictor   = PredictionCore()
    metacog     = MetacognitionCore()
    homeostasis = HomeostasisCore()
    if os.path.exists(MIND_FILE):
        with open(MIND_FILE, 'r', encoding='utf-8') as f:
            d = json.load(f)
        emotion.from_dict(d.get('emotion', {}))
        predictor.from_dict(d.get('prediction', {}))
        metacog.from_dict(d.get('metacognition', {}))
        print(f"  마음 상태 불러옴: {emotion.label()}")
    return emotion, predictor, metacog, homeostasis


def save_mind_state(emotion, predictor, metacog):
    d = {
        'emotion':        emotion.to_dict(),
        'prediction':     predictor.to_dict(),
        'metacognition':  metacog.to_dict(),
    }
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

    # 마지막 트라이얼의 활성 뉴런 인덱스 (B에게 신호로 전달)
    active_indices = [int(i) for i in np.where(active)[0]]

    return ee_mask, {
        'connections':    int(ee_mask.sum()),
        'formed':         formed,
        'pruned':         pruned,
        'avg_active':     round(float(np.mean(total_active)), 1),
        'active_indices': active_indices,
    }


def print_status():
    log = load_log()
    emotion, predictor, metacog, _ = load_mind_state()
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
    print(f"  메타인지: {metacog.reflection()}")
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

    log                            = load_log()
    ee                             = load_neuron_state()
    emotion, predictor, metacog, homeostasis = load_mind_state()
    cf_mat                         = build_fixed(seed=0)
    decider                        = DecisionCore()

    session_num = len(log['sessions']) + 1
    print(f"\n세션 {session_num} 시작")

    # ── L7-③ 메타인지: 지난 성적 먼저 보고 ──────────────────────
    print(f"  메타인지: {metacog.reflection()}")

    next_prediction = predictor.predict()
    print(f"  예측: {next_prediction}")
    print(f"  감정: {emotion.label()}")

    decision = decider.decide(emotion)

    # ── L7-③ 메타인지 보정: form_prob 미세 조정 ──────────────────
    meta_boost = metacog.form_prob_boost()
    decision['form_prob'] = round(min(decision['form_prob'] + meta_boost, 0.70), 3)
    if meta_boost != 0:
        print(f"  메타인지 보정: form_prob {'+' if meta_boost>0 else ''}{meta_boost:.2f}")

    print(f"  판단: {decision['action']}")
    print(f"  → 자극강도 ×{decision['stim_mult']:.1f} | "
          f"트라이얼 {decision['n_trials']}회 | "
          f"연결확률 {decision['form_prob']:.0%}")

    # ── L7-② 사회성: B 피드백 수신 → 자극 마스크 조정 ───────────
    stim_mask = np.zeros(N, dtype=bool)
    stim_mask[:int(N_E * 0.4)] = True

    b_sig = load_signal_from_B()
    if b_sig:
        stim_mask = apply_b_feedback(stim_mask, b_sig)

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

    # ── L5 감정 업데이트 ──────────────────────────────────────────
    emotion.update(stats['formed'], stats['pruned'],
                   stats['avg_active'], prev_connections)

    # ── L7-① 항상성: 감정 극단값 설정점으로 복귀 ────────────────
    stress_before = homeostasis.stress_report(emotion)
    homeostasis.regulate(emotion)
    stress_after  = homeostasis.stress_report(emotion)
    if stress_before > 0.15:
        print(f"  항상성 작동: 긴장도 {stress_before:.2f} → {stress_after:.2f}")

    predictor.observe(stats['formed'])

    # ── L7-③ 메타인지 업데이트 ───────────────────────────────────
    metacog.observe(decision['action'], stats['formed'], stats['connections'])

    np.save(STATE_FILE, ee)
    save_mind_state(emotion, predictor, metacog)

    # B에게 신호 전달 — 이번 세션의 활성 패턴 저장
    signal = {
        'session':        session_num,
        'active_indices': stats.get('active_indices', []),
        'formed':         stats['formed'],
        'connections':    stats['connections'],
        'emotion':        emotion.to_dict(),
    }
    signal_path = os.path.join(HOPE_DIR, 'A신호.json')
    with open(signal_path, 'w', encoding='utf-8') as f:
        json.dump(signal, f, ensure_ascii=False, indent=2)

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
    print(f"  항상성 긴장도: {homeostasis.stress_report(emotion):.2f}")
    print(f"  메타인지: {metacog.reflection()}")
    print(f"  ─────────────────────────────────────")
    print(f"  호프의 상태가 저장됐어.")
    print("=" * 55)
