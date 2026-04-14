"""
L0 실험: 인공 뉴런 1개
=====================
뇌의 가장 작은 단위 = 뉴런.
뉴런은 이렇게 작동한다:
  1. 여러 입력을 받는다 (수상돌기)
  2. 각 입력에 "가중치"를 곱한다 (어떤 입력이 더 중요한가)
  3. 전부 더한다
  4. 임계값을 넘으면 "발화" (신호를 보낸다)
  5. 안 넘으면 침묵

이것이 뇌의 860억 뉴런 중 "딱 1개"가 하는 일이다.
"""

import json
import os

# ===== 뉴런 클래스 =====
class Neuron:
    """인공 뉴런 1개"""
    
    def __init__(self, name, num_inputs, threshold=0.5):
        """
        name: 이 뉴런의 이름
        num_inputs: 입력 몇 개를 받는가
        threshold: 임계값 (이 값을 넘어야 발화)
        """
        self.name = name
        self.weights = [0.5] * num_inputs  # 초기 가중치 (전부 0.5)
        self.threshold = threshold
        self.history = []  # 발화 기록
    
    def receive(self, inputs):
        """
        입력을 받아서 처리한다.
        inputs: 숫자 리스트 (예: [0.3, 0.8, 0.1])
        """
        # 1단계: 각 입력 × 가중치
        weighted = []
        for i in range(len(inputs)):
            w = inputs[i] * self.weights[i]
            weighted.append(w)
        
        # 2단계: 전부 더한다
        total = sum(weighted)
        
        # 3단계: 임계값 비교 → 발화 여부
        fired = total >= self.threshold
        
        # 기록 저장
        record = {
            "inputs": inputs,
            "weights": [round(w, 4) for w in self.weights],
            "weighted_sum": round(total, 4),
            "threshold": self.threshold,
            "fired": fired,
            "signal": "⚡ 발화!" if fired else "💤 침묵"
        }
        self.history.append(record)
        
        return record
    
    def status(self):
        """현재 뉴런 상태"""
        return {
            "name": self.name,
            "weights": [round(w, 4) for w in self.weights],
            "threshold": self.threshold,
            "total_signals": len(self.history),
            "fires": sum(1 for h in self.history if h["fired"]),
            "silences": sum(1 for h in self.history if not h["fired"])
        }


# ===== 실험 실행 =====
print("=" * 60)
print("  L0 실험: 인공 뉴런 1개")
print("  — 뇌의 가장 작은 단위를 만들었습니다")
print("=" * 60)
print()

# 뉴런 생성: 입력 3개, 임계값 0.5
neuron = Neuron(name="뉴런-알파", num_inputs=3, threshold=0.5)

print(f"🧠 [{neuron.name}] 생성됨")
print(f"   입력 수: 3개")
print(f"   초기 가중치: {neuron.weights}")
print(f"   임계값: {neuron.threshold}")
print()

# 테스트 시나리오들
tests = [
    {"name": "약한 자극", "inputs": [0.1, 0.1, 0.1], "설명": "세 입력 모두 약함"},
    {"name": "중간 자극", "inputs": [0.5, 0.5, 0.3], "설명": "적당한 입력"},
    {"name": "강한 자극", "inputs": [0.9, 0.8, 0.7], "설명": "세 입력 모두 강함"},
    {"name": "한쪽만 강함", "inputs": [1.0, 0.0, 0.0], "설명": "하나만 최대, 나머지 0"},
    {"name": "두 개 강함", "inputs": [0.8, 0.7, 0.0], "설명": "두 개 강하고 하나 없음"},
]

print("─" * 60)
print("  실험 시작: 5가지 다른 자극을 줘본다")
print("─" * 60)
print()

for i, test in enumerate(tests, 1):
    result = neuron.receive(test["inputs"])
    print(f"  테스트 {i}: {test['name']}")
    print(f"  설명: {test['설명']}")
    print(f"  입력값:    {test['inputs']}")
    print(f"  × 가중치:  {result['weights']}")
    print(f"  = 합계:    {result['weighted_sum']}")
    print(f"  임계값:    {result['threshold']}")
    print(f"  결과:      {result['signal']}")
    print()

# 최종 상태
print("─" * 60)
print("  최종 상태")
print("─" * 60)
status = neuron.status()
print(f"  뉴런 이름: {status['name']}")
print(f"  총 신호 수: {status['total_signals']}")
print(f"  발화 횟수: {status['fires']}회")
print(f"  침묵 횟수: {status['silences']}회")
print()

# 비유 설명
print("=" * 60)
print("  이게 뭔가요?")
print("=" * 60)
print()
print("  지금 만든 건 뇌세포(뉴런) 딱 1개입니다.")
print("  이 뉴런은:")
print("  - 3개의 입력을 받습니다 (눈, 귀, 코처럼)")
print("  - 각 입력의 중요도(가중치)를 곱합니다")
print("  - 다 더해서 임계값을 넘으면 '발화' 합니다")
print("  - 안 넘으면 조용히 있습니다")
print()
print("  실제 뇌에는 이게 860억 개 있습니다.")
print("  그리고 각 뉴런이 평균 7,000개의 다른 뉴런과 연결됩니다.")
print()
print("  다음 단계(L1)에서는:")
print("  - 뉴런 여러 개를 만들어서 연결합니다")
print("  - 한 뉴런의 '발화'가 다른 뉴런의 '입력'이 됩니다")
print("  - 이때부터 '네트워크'가 됩니다")
print()

# 결과 저장
output = {
    "experiment": "L0_뉴런하나",
    "date": "2026-04-15",
    "neuron_status": status,
    "test_results": neuron.history,
    "conclusion": "인공 뉴런 1개가 정상 작동. 입력 × 가중치 → 합계 → 임계값 비교 → 발화/침묵 구조 확인됨."
}

# JSON 결과 파일
result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "실험결과", "L0-뉴런하나-결과.json")
os.makedirs(os.path.dirname(result_path), exist_ok=True)
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  📁 결과 저장됨: 실험결과/L0-뉴런하나-결과.json")

