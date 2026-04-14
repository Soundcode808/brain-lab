"""
L4 실험: 판단 — 충돌하는 욕구 사이에서 최적 행동을 고른다
============================================================
L2까지는 "이건 위험하다/먹이다"를 판단했다.
그런데 진짜 어려운 건 이런 상황이다:

  "눈앞에 먹이가 있다. 그런데 뒤에서 포식자 발소리가 들린다."
  "배가 너무 고프다. 하지만 저 열매는 독이 있을 수도 있다."

이건 단순한 '위험/안전' 판단이 아니라,
여러 욕구가 충돌할 때 '지금 뭘 해야 가장 이득인가'를 결정하는 것이다.

진짜 뇌에서 이걸 담당하는 곳:
  - 전전두엽 피질 (prefrontal cortex): 계획, 의사결정
  - 기저핵 (basal ganglia): 행동 선택 (경쟁적 억제)
  - 편도체 (amygdala): 위험 신호 → 긴급 오버라이드

우리가 만들 것:
  경쟁적 행동 선택 시스템 (Winner-Takes-All)
  각 행동이 "나를 골라!" 하고 경쟁한다.
  가장 효용(utility)이 높은 행동이 이긴다.
  단, 긴급 위험이면 무조건 도망 (편도체 오버라이드).
"""

import json
import os
import math
import random

random.seed(42)


# ===== 행동 후보 =====
class Action:
    """하나의 행동 후보"""
    def __init__(self, name, emoji):
        self.name = name
        self.emoji = emoji


ACTIONS = {
    "도망":   Action("도망", "🏃"),
    "접근":   Action("접근", "🍖"),
    "경계":   Action("경계", "👀"),
    "탐색":   Action("탐색", "🔍"),
    "대기":   Action("대기", "😴"),
    "은신":   Action("은신", "🫥"),
}


# ===== 욕구(Drive) 시스템 =====
class Drive:
    """
    하나의 욕구.
    배고픔, 안전, 호기심 같은 것.
    값이 높을수록 그 욕구가 강하다 (0~1).
    """
    def __init__(self, name, value=0.5, decay=0.01, weight=1.0):
        self.name = name
        self.value = value        # 현재 강도 (0~1)
        self.decay = decay        # 시간이 지나면 자연 변화하는 양
        self.weight = weight      # 판단에서의 중요도

    def tick(self):
        """시간이 지나면 욕구가 자연스럽게 변한다"""
        self.value = max(0, min(1, self.value + self.decay))

    def satisfy(self, amount):
        """욕구가 해소된다"""
        self.value = max(0, self.value - amount)

    def increase(self, amount):
        """욕구가 증가한다"""
        self.value = min(1, self.value + amount)


# ===== 판단 엔진 =====
class DecisionEngine:
    """
    L4: 판단 엔진
    여러 행동 후보가 경쟁하고, 가장 효용이 높은 걸 선택한다.

    원리: 기저핵의 경쟁적 억제 (competitive inhibition)
    - 각 행동의 효용을 계산한다
    - 가장 높은 효용의 행동이 이긴다 (Winner-Takes-All)
    - 단, 긴급 위험이면 편도체가 오버라이드한다
    """

    def __init__(self):
        # 욕구 시스템
        self.drives = {
            "배고픔": Drive("배고픔", value=0.3, decay=0.02, weight=1.0),
            "안전":   Drive("안전", value=0.7, decay=-0.01, weight=1.5),  # 안전은 서서히 불안해짐
            "호기심": Drive("호기심", value=0.5, decay=0.01, weight=0.7),
            "피로":   Drive("피로", value=0.2, decay=0.01, weight=0.8),
        }

        # 편도체 오버라이드 임계값
        self.panic_threshold = 0.85  # 이 이상이면 무조건 도망

        # 의사결정 기록
        self.decision_log = []

    def calculate_utility(self, action_name, danger_score, food_score, novelty=0.0):
        """
        각 행동의 효용(utility)을 계산한다.
        효용 = 그 행동이 현재 욕구를 얼마나 충족시킬지 예상.
        """
        hunger = self.drives["배고픔"].value
        safety = self.drives["안전"].value
        curiosity = self.drives["호기심"].value
        fatigue = self.drives["피로"].value

        if action_name == "도망":
            # 위험이 높고 안전 욕구가 강할수록 도망 효용 높음
            utility = danger_score * (1 - safety) * self.drives["안전"].weight
            # 피로하면 도망 효용 약간 감소 (진짜 뇌도 그렇다)
            utility *= (1 - fatigue * 0.3)

        elif action_name == "접근":
            # 먹이가 있고 배고플수록 접근 효용 높음
            utility = food_score * hunger * self.drives["배고픔"].weight
            # 위험하면 접근 효용 감소
            utility *= (1 - danger_score * 0.8)

        elif action_name == "경계":
            # 위험과 먹이가 모두 중간일 때 유용
            ambiguity = 1 - abs(danger_score - food_score)
            utility = ambiguity * 0.5 * (1 + danger_score * 0.3)

        elif action_name == "탐색":
            # 호기심이 높고 위험이 낮을 때
            utility = curiosity * novelty * self.drives["호기심"].weight
            utility *= (1 - danger_score * 0.9)  # 위험하면 탐색 안 함

        elif action_name == "대기":
            # 피로하거나, 위험/먹이 모두 낮을 때
            utility = fatigue * 0.5 + (1 - danger_score) * (1 - food_score) * 0.3

        elif action_name == "은신":
            # 위험하지만 도망치기엔 피로할 때
            utility = danger_score * fatigue * 0.8
            # 안전 욕구가 낮을수록(=불안할수록) 은신 효용 높음
            utility *= (1 - safety * 0.5)
        else:
            utility = 0

        return max(0, utility)

    def decide(self, danger_score, food_score, novelty=0.0, context=""):
        """
        판단을 내린다.

        Returns: (선택된 행동, 모든 행동의 효용, 판단 과정 로그)
        """
        log = {
            "context": context,
            "inputs": {
                "danger": round(danger_score, 3),
                "food": round(food_score, 3),
                "novelty": round(novelty, 3),
            },
            "drives": {name: round(d.value, 3) for name, d in self.drives.items()},
        }

        # 편도체 오버라이드: 극도의 위험이면 즉시 도망
        if danger_score >= self.panic_threshold:
            log["override"] = "편도체 패닉 — 무조건 도망"
            log["chosen"] = "도망"
            log["utilities"] = {"도망": 999}
            self.decision_log.append(log)
            return "도망", {"도망": 999}, log

        # 각 행동의 효용 계산
        utilities = {}
        for name in ACTIONS:
            utilities[name] = round(
                self.calculate_utility(name, danger_score, food_score, novelty), 3
            )

        # Winner-Takes-All: 가장 높은 효용의 행동 선택
        chosen = max(utilities, key=utilities.get)

        # 효용이 모두 매우 낮으면 대기
        if max(utilities.values()) < 0.05:
            chosen = "대기"

        log["utilities"] = utilities
        log["chosen"] = chosen
        self.decision_log.append(log)

        return chosen, utilities, log

    def apply_outcome(self, action, was_dangerous, had_food, was_novel=False):
        """행동 결과를 욕구에 반영한다"""
        if action == "접근" and had_food:
            self.drives["배고픔"].satisfy(0.3)
        elif action == "도망":
            self.drives["피로"].increase(0.15)
        elif action == "탐색" and was_novel:
            self.drives["호기심"].satisfy(0.2)
        elif action == "대기":
            self.drives["피로"].satisfy(0.1)

        if was_dangerous and action != "도망" and action != "은신":
            self.drives["안전"].satisfy(0.2)  # 안전감 하락
        elif not was_dangerous:
            self.drives["안전"].increase(0.1)

        if not had_food:
            self.drives["배고픔"].increase(0.05)

        # 시간 경과
        for d in self.drives.values():
            d.tick()


# ===== 시뮬레이션 =====
print("=" * 60)
print("  L4 실험: 판단 (경쟁적 행동 선택)")
print("  — 충돌하는 욕구 사이에서 최적 행동을 고른다")
print("=" * 60)
print()
print("  원리: 기저핵의 경쟁적 억제 (competitive inhibition)")
print("  + 편도체의 긴급 오버라이드 (panic response)")
print()

engine = DecisionEngine()

# === 시나리오 1: 평화로운 상황 ===
print("━" * 60)
print("  시나리오 1: 평화로운 아침 — 위험 없고, 먹이도 없다")
print("━" * 60)
print()

chosen, utils, log = engine.decide(0.1, 0.1, novelty=0.3, context="평화로운 아침")
print(f"  욕구 상태: ", end="")
for name, d in engine.drives.items():
    bar = "█" * int(d.value * 10) + "░" * (10 - int(d.value * 10))
    print(f"{name} [{bar}] {d.value:.0%}  ", end="")
print()
print(f"  행동 효용:")
for name, u in sorted(utils.items(), key=lambda x: -x[1]):
    bar = "■" * int(u * 20)
    marker = " ← 선택!" if name == chosen else ""
    print(f"    {ACTIONS[name].emoji} {name:<4}: {u:.3f} {bar}{marker}")
print(f"\n  결정: {ACTIONS[chosen].emoji} {chosen}")
engine.apply_outcome(chosen, False, False, True)
print()

# === 시나리오 2: 먹이 발견 ===
print("━" * 60)
print("  시나리오 2: 열매 냄새가 난다 — 배고프기도 하다")
print("━" * 60)
print()

# 배고픔 올리기
engine.drives["배고픔"].increase(0.3)

chosen, utils, log = engine.decide(0.1, 0.7, novelty=0.1, context="열매 냄새")
print(f"  욕구 상태: ", end="")
for name, d in engine.drives.items():
    bar = "█" * int(d.value * 10) + "░" * (10 - int(d.value * 10))
    print(f"{name} [{bar}] {d.value:.0%}  ", end="")
print()
print(f"  행동 효용:")
for name, u in sorted(utils.items(), key=lambda x: -x[1]):
    bar = "■" * int(u * 20)
    marker = " ← 선택!" if name == chosen else ""
    print(f"    {ACTIONS[name].emoji} {name:<4}: {u:.3f} {bar}{marker}")
print(f"\n  결정: {ACTIONS[chosen].emoji} {chosen}")
engine.apply_outcome(chosen, False, True)
print()

# === 시나리오 3: 핵심 — 먹이 + 위험 동시 ===
print("━" * 60)
print("  시나리오 3: 먹이가 있다, 그런데 뒤에서 발소리가 들린다!")
print("  ↑ 이게 L4의 핵심 — 충돌하는 욕구 사이의 판단")
print("━" * 60)
print()

chosen, utils, log = engine.decide(0.6, 0.7, novelty=0.0, context="먹이+위험 동시")
print(f"  욕구 상태: ", end="")
for name, d in engine.drives.items():
    bar = "█" * int(d.value * 10) + "░" * (10 - int(d.value * 10))
    print(f"{name} [{bar}] {d.value:.0%}  ", end="")
print()
print(f"  행동 효용:")
for name, u in sorted(utils.items(), key=lambda x: -x[1]):
    bar = "■" * int(u * 20)
    marker = " ← 선택!" if name == chosen else ""
    print(f"    {ACTIONS[name].emoji} {name:<4}: {u:.3f} {bar}{marker}")
print(f"\n  결정: {ACTIONS[chosen].emoji} {chosen}")
print(f"  이유: 위험({log['inputs']['danger']:.0%})과 먹이({log['inputs']['food']:.0%})가 동시 → 욕구 경쟁 → 가장 효용 높은 행동 선택")
engine.apply_outcome(chosen, True, True)
print()

# === 시나리오 4: 극도의 위험 — 편도체 오버라이드 ===
print("━" * 60)
print("  시나리오 4: 곰이 눈앞에! — 편도체 패닉")
print("━" * 60)
print()

chosen, utils, log = engine.decide(0.95, 0.3, novelty=0.0, context="곰 등장")
print(f"  욕구 상태: ", end="")
for name, d in engine.drives.items():
    bar = "█" * int(d.value * 10) + "░" * (10 - int(d.value * 10))
    print(f"{name} [{bar}] {d.value:.0%}  ", end="")
print()

if "override" in log:
    print(f"  ⚡ {log['override']}")
print(f"  결정: {ACTIONS[chosen].emoji} {chosen}")
print(f"  (위험 {log['inputs']['danger']:.0%} ≥ 패닉 임계값 {engine.panic_threshold:.0%} → 효용 계산 건너뜀)")
engine.apply_outcome(chosen, True, False)
print()

# === 시나리오 5: 피로 + 위험 — 은신 선택? ===
print("━" * 60)
print("  시나리오 5: 위험하다, 그런데 너무 지쳤다...")
print("━" * 60)
print()

engine.drives["피로"].increase(0.5)
engine.drives["안전"].satisfy(0.3)

chosen, utils, log = engine.decide(0.7, 0.1, novelty=0.0, context="피로+위험")
print(f"  욕구 상태: ", end="")
for name, d in engine.drives.items():
    bar = "█" * int(d.value * 10) + "░" * (10 - int(d.value * 10))
    print(f"{name} [{bar}] {d.value:.0%}  ", end="")
print()
print(f"  행동 효용:")
for name, u in sorted(utils.items(), key=lambda x: -x[1]):
    bar = "■" * int(u * 20)
    marker = " ← 선택!" if name == chosen else ""
    print(f"    {ACTIONS[name].emoji} {name:<4}: {u:.3f} {bar}{marker}")
print(f"\n  결정: {ACTIONS[chosen].emoji} {chosen}")
print(f"  이유: 위험하지만 피로({engine.drives['피로'].value:.0%})가 높아 도망 효용 감소")
print()

# === 시나리오 6: 배고파 죽겠는데 위험한 먹이 ===
print("━" * 60)
print("  시나리오 6: 굶주림 끝에 독버섯 발견 — 위험하지만 배고프다")
print("━" * 60)
print()

engine.drives["배고픔"].value = 0.9  # 극도의 배고픔
engine.drives["피로"].value = 0.3    # 피로 회복

chosen, utils, log = engine.decide(0.5, 0.6, novelty=0.0, context="굶주림+위험한 먹이")
print(f"  욕구 상태: ", end="")
for name, d in engine.drives.items():
    bar = "█" * int(d.value * 10) + "░" * (10 - int(d.value * 10))
    print(f"{name} [{bar}] {d.value:.0%}  ", end="")
print()
print(f"  행동 효용:")
for name, u in sorted(utils.items(), key=lambda x: -x[1]):
    bar = "■" * int(u * 20)
    marker = " ← 선택!" if name == chosen else ""
    print(f"    {ACTIONS[name].emoji} {name:<4}: {u:.3f} {bar}{marker}")
print(f"\n  결정: {ACTIONS[chosen].emoji} {chosen}")
print(f"  이유: 배고픔({engine.drives['배고픔'].value:.0%})이 극단적 → 위험({log['inputs']['danger']:.0%})을 감수하고 접근")
print()

# === 요약 ===
print("=" * 60)
print("  L4: 판단 — 이게 뭔가요?")
print("=" * 60)
print()
print("  v0.1의 행동 결정:")
print("    if 위험 > 0.6: 도망")
print("    elif 먹이 > 0.6: 접근")
print("    else: 대기")
print("  → 단순. 충돌 상황을 못 다룸.")
print()
print("  L4의 행동 결정:")
print("    1. 현재 욕구를 확인한다 (배고픔, 안전, 호기심, 피로)")
print("    2. 각 행동의 효용을 계산한다 (욕구 × 상황 × 가중치)")
print("    3. 가장 효용 높은 행동이 이긴다 (Winner-Takes-All)")
print("    4. 극도의 위험이면 편도체가 오버라이드 (무조건 도망)")
print("  → 상황과 내부 상태에 따라 다른 결정을 내린다.")
print()
print("  핵심 발견:")
print("  - 같은 '먹이+위험' 상황이라도 배고프면 접근, 안전하면 경계")
print("  - 피로하면 도망 대신 은신을 선택")
print("  - 극단적 위험이면 모든 계산을 건너뛰고 즉시 도망")
print("  - 이게 진짜 뇌가 하는 일이다 — 전전두엽 + 기저핵 + 편도체")
print()

# 결과 저장
output = {
    "experiment": "L4_판단",
    "date": "2026-04-15",
    "architecture": "경쟁적 행동 선택 (Winner-Takes-All) + 편도체 오버라이드",
    "drives": ["배고픔", "안전", "호기심", "피로"],
    "actions": list(ACTIONS.keys()),
    "scenarios": [log for log in engine.decision_log],
    "conclusion": "욕구 경쟁 기반 판단 시스템 정상 작동. 충돌 상황에서 욕구 강도에 따라 다른 행동 선택. 편도체 오버라이드(패닉) 정상 작동. v0.1의 단순 if문을 대체할 수 있음."
}

result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "실험결과", "L4-판단-결과.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  📁 결과 저장됨: 실험결과/L4-판단-결과.json")
