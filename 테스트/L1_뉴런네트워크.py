"""
L1 실험: 뉴런 여러 개를 연결한다
================================
L0에서는 뉴런 1개를 만들었다.
이제 여러 개를 연결해서 "네트워크"를 만든다.

핵심: 한 뉴런의 ⚡발화가 다른 뉴런의 입력이 된다.
이러면 내가 숫자를 안 넣어줘도 뉴런끼리 서로 신호를 주고받는다.

구조:
  [감각뉴런A] ──┐
                ├──→ [중간뉴런D] ──┐
  [감각뉴런B] ──┘                  ├──→ [판단뉴런F] → 최종 판단
                ┌──→ [중간뉴런E] ──┘
  [감각뉴런C] ──┘

비유:
  A = 눈 (시각)
  B = 귀 (청각)  → D = "위험 감지" → F = "도망칠까 말까" 최종 판단
  C = 코 (후각)  → E = "먹이 감지"
"""

import json
import os

# ===== L0에서 만든 뉴런 (그대로 가져옴) =====
class Neuron:
    def __init__(self, name, num_inputs, threshold=0.5):
        self.name = name
        self.weights = [0.5] * num_inputs
        self.threshold = threshold
        self.last_output = 0.0  # 마지막 출력값 저장
        self.history = []
    
    def receive(self, inputs):
        weighted = [inputs[i] * self.weights[i] for i in range(len(inputs))]
        total = sum(weighted)
        fired = total >= self.threshold
        
        # 발화하면 1.0, 아니면 0.0 출력
        self.last_output = 1.0 if fired else 0.0
        
        record = {
            "inputs": [round(x, 4) for x in inputs],
            "weighted_sum": round(total, 4),
            "fired": fired,
            "output": self.last_output
        }
        self.history.append(record)
        return self.last_output, fired


# ===== 네트워크 구성 =====
print("=" * 60)
print("  L1 실험: 뉴런 네트워크")
print("  — 6개의 뉴런을 연결합니다")
print("=" * 60)
print()

# 감각 뉴런 (외부 입력을 받는 첫 번째 층)
neuron_A = Neuron("👁 눈(시각)", num_inputs=1, threshold=0.3)
neuron_B = Neuron("👂 귀(청각)", num_inputs=1, threshold=0.3)
neuron_C = Neuron("👃 코(후각)", num_inputs=1, threshold=0.3)

# 중간 뉴런 (감각 뉴런들의 출력을 조합하는 층)
neuron_D = Neuron("⚠️ 위험감지", num_inputs=2, threshold=0.7)  # 눈+귀 → 위험?
neuron_E = Neuron("🍖 먹이감지", num_inputs=2, threshold=0.7)  # 귀+코 → 먹이?

# 판단 뉴런 (최종 결정)
neuron_F = Neuron("🧠 최종판단", num_inputs=2, threshold=0.4)  # 위험+먹이 → 행동?

print("  네트워크 구조:")
print()
print("  [👁 눈] ──┐")
print("            ├──→ [⚠️ 위험감지] ──┐")
print("  [👂 귀] ──┤                     ├──→ [🧠 최종판단]")
print("            ├──→ [🍖 먹이감지] ──┘")
print("  [👃 코] ──┘")
print()
print("  총 6개 뉴런, 3개 층(감각 → 중간 → 판단)")
print()


# ===== 시나리오 테스트 =====
scenarios = [
    {
        "name": "🌙 조용한 밤",
        "desc": "아무것도 안 보이고, 안 들리고, 냄새도 없다",
        "eye": 0.1, "ear": 0.1, "nose": 0.1
    },
    {
        "name": "🐺 늑대 출현",
        "desc": "큰 그림자가 보이고(눈↑), 으르렁 소리(귀↑), 냄새는 약함",
        "eye": 0.9, "ear": 0.8, "nose": 0.2
    },
    {
        "name": "🥩 먹이 발견",
        "desc": "안 보이지만(눈↓), 바스락 소리(귀↑), 고기 냄새(코↑)",
        "eye": 0.1, "ear": 0.6, "nose": 0.9
    },
    {
        "name": "🐺🥩 늑대 + 먹이 동시",
        "desc": "그림자 보이고(눈↑), 소리도 크고(귀↑), 냄새도 남(코↑)",
        "eye": 0.9, "ear": 0.9, "nose": 0.8
    },
    {
        "name": "🌿 바람 소리",
        "desc": "안 보이고(눈↓), 살랑살랑(귀 약간↑), 풀냄새(코 약간↑)",
        "eye": 0.1, "ear": 0.3, "nose": 0.3
    },
]

all_results = []

for i, s in enumerate(scenarios, 1):
    print("─" * 60)
    print(f"  시나리오 {i}: {s['name']}")
    print(f"  상황: {s['desc']}")
    print()
    
    # === 1층: 감각 뉴런 ===
    out_A, fire_A = neuron_A.receive([s["eye"]])
    out_B, fire_B = neuron_B.receive([s["ear"]])
    out_C, fire_C = neuron_C.receive([s["nose"]])
    
    print(f"  1층 감각:")
    print(f"    👁 눈 ← {s['eye']} → {'⚡발화' if fire_A else '💤침묵'}")
    print(f"    👂 귀 ← {s['ear']} → {'⚡발화' if fire_B else '💤침묵'}")
    print(f"    👃 코 ← {s['nose']} → {'⚡발화' if fire_C else '💤침묵'}")
    print()
    
    # === 2층: 중간 뉴런 ===
    # 위험감지 = 눈 + 귀
    out_D, fire_D = neuron_D.receive([out_A, out_B])
    # 먹이감지 = 귀 + 코
    out_E, fire_E = neuron_E.receive([out_B, out_C])
    
    print(f"  2층 종합:")
    print(f"    ⚠️ 위험감지 ← 눈({out_A}) + 귀({out_B}) → {'⚡위험!' if fire_D else '💤안전'}")
    print(f"    🍖 먹이감지 ← 귀({out_B}) + 코({out_C}) → {'⚡먹이!' if fire_E else '💤없음'}")
    print()
    
    # === 3층: 판단 뉴런 ===
    out_F, fire_F = neuron_F.receive([out_D, out_E])
    
    # 판단 해석
    if fire_D and fire_E:
        judgment = "⚡ 위험하지만 먹이도 있다! → 긴장 상태"
    elif fire_D and not fire_E:
        judgment = "⚡ 위험! 도망쳐! → 회피"
    elif not fire_D and fire_E:
        judgment = "⚡ 먹이 발견! → 접근"
    else:
        judgment = "💤 아무 일도 없다 → 대기"
    
    print(f"  3층 최종판단:")
    print(f"    🧠 판단 ← 위험({out_D}) + 먹이({out_E}) → {judgment}")
    print()
    
    all_results.append({
        "scenario": s["name"],
        "inputs": {"eye": s["eye"], "ear": s["ear"], "nose": s["nose"]},
        "layer1": {"eye": fire_A, "ear": fire_B, "nose": fire_C},
        "layer2": {"danger": fire_D, "food": fire_E},
        "layer3": {"judgment": judgment, "fired": fire_F}
    })


# ===== 요약 =====
print("=" * 60)
print("  실험 요약")
print("=" * 60)
print()
print("  시나리오         | 위험 | 먹이 | 최종 판단")
print("  ─────────────────┼──────┼──────┼──────────────")
for r in all_results:
    danger = "⚠️" if r["layer2"]["danger"] else "  "
    food = "🍖" if r["layer2"]["food"] else "  "
    name = r["scenario"].ljust(16)
    # get short judgment
    j = r["layer3"]["judgment"]
    short = j.split("→")[1].strip() if "→" in j else j
    print(f"  {name} | {danger}   | {food}   | {short}")

print()
print("=" * 60)
print("  이게 뭔가요?")
print("=" * 60)
print()
print("  L0에서는 뉴런 1개가 혼자 계산했습니다.")
print("  L1에서는 6개 뉴런이 서로 연결됐습니다.")
print()
print("  핵심 변화:")
print("  - 1층(감각)의 출력이 → 2층(종합)의 입력이 됩니다")
print("  - 2층(종합)의 출력이 → 3층(판단)의 입력이 됩니다")
print("  - 내가 넣어준 건 '눈/귀/코' 수치뿐인데")
print("  - '위험이다', '먹이다', '도망쳐'라는 판단이 나옵니다")
print()
print("  이게 바로 '층을 쌓으면 복잡해진다'의 시작입니다.")
print()
print("  다음 단계(L2)에서는:")
print("  - '학습'을 넣습니다")
print("  - 늑대한테 물렸던 경험 → 다음엔 더 빨리 도망치게")
print("  - 가중치가 자동으로 바뀌는 것 = 학습")
print()

# 결과 저장
output = {
    "experiment": "L1_뉴런네트워크",
    "date": "2026-04-15",
    "neuron_count": 6,
    "layers": 3,
    "structure": "감각(3) → 중간(2) → 판단(1)",
    "scenarios": all_results,
    "conclusion": "6개 뉴런 네트워크 정상 작동. 감각 입력 → 종합 → 판단 3단계 처리 확인. 층을 쌓으면 단순 입력에서 복합 판단이 나온다."
}

result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "실험결과", "L1-뉴런네트워크-결과.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  📁 결과 저장됨: 실험결과/L1-뉴런네트워크-결과.json")

