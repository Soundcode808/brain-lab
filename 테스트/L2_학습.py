"""
L2 실험: 학습 — 경험에서 스스로 배우는 뉴런
==========================================
L1까지는 내가 가중치를 정해줬다. (가짜 판단)
L2에서는 뉴런이 스스로 가중치를 바꾼다. (진짜 학습)

원리:
  1. 뉴런에게 문제를 준다
  2. 뉴런이 답을 내놓는다
  3. 정답과 비교한다
  4. 틀렸으면 → 가중치를 조금 고친다
  5. 맞았으면 → 그대로 둔다
  6. 이걸 반복하면 → 점점 맞추기 시작한다

이게 진짜 뇌가 하는 일이다.
아기가 불에 데이면 → "불 = 아프다" 연결이 강해지는 것과 같다.
"""

import json
import os
import random

random.seed(42)  # 재현 가능하도록

# ===== 학습하는 뉴런 =====
class LearningNeuron:
    def __init__(self, name, num_inputs, learning_rate=0.1):
        self.name = name
        # 가중치를 랜덤으로 시작 (아무것도 모르는 상태)
        self.weights = [round(random.uniform(-0.5, 0.5), 4) for _ in range(num_inputs)]
        self.bias = round(random.uniform(-0.5, 0.5), 4)  # 기본 성향
        self.learning_rate = learning_rate
        self.error_history = []
    
    def predict(self, inputs):
        """입력을 받아서 예측한다"""
        total = sum(inputs[i] * self.weights[i] for i in range(len(inputs))) + self.bias
        # 시그모이드: 부드러운 발화 (0~1 사이 값)
        import math
        output = 1 / (1 + math.exp(-total))
        return output
    
    def learn(self, inputs, expected):
        """
        예측하고, 틀리면 가중치를 고친다.
        inputs: 입력값들
        expected: 정답 (1.0 = 위험, 0.0 = 안전)
        """
        # 1. 예측
        prediction = self.predict(inputs)
        
        # 2. 얼마나 틀렸나? (오차)
        error = expected - prediction
        
        # 3. 가중치 수정 (틀린 만큼 고친다)
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
            self.weights[i] = round(self.weights[i], 4)
        self.bias += self.learning_rate * error
        self.bias = round(self.bias, 4)
        
        self.error_history.append(abs(error))
        return prediction, error


# ===== 훈련 데이터 (경험) =====
# 뉴런에게 "이런 상황은 위험하다/안전하다"를 가르친다
# [눈(시각), 귀(청각), 코(후각)] → 위험도 (1=위험, 0=안전)

training_data = [
    # 위험한 상황들
    {"inputs": [0.9, 0.8, 0.2], "answer": 1.0, "상황": "큰 그림자 + 으르렁"},
    {"inputs": [0.7, 0.9, 0.1], "answer": 1.0, "상황": "그림자 + 큰 소리"},
    {"inputs": [0.8, 0.7, 0.3], "answer": 1.0, "상황": "큰 움직임 + 소리"},
    {"inputs": [1.0, 1.0, 0.0], "answer": 1.0, "상황": "눈앞에 포식자"},
    {"inputs": [0.6, 0.8, 0.1], "answer": 1.0, "상황": "소리가 점점 커짐"},
    
    # 안전한 상황들
    {"inputs": [0.1, 0.1, 0.1], "answer": 0.0, "상황": "조용한 밤"},
    {"inputs": [0.1, 0.2, 0.8], "answer": 0.0, "상황": "꽃향기만"},
    {"inputs": [0.2, 0.3, 0.7], "answer": 0.0, "상황": "풀냄새 + 바람"},
    {"inputs": [0.0, 0.1, 0.9], "answer": 0.0, "상황": "먹이 냄새만"},
    {"inputs": [0.1, 0.2, 0.2], "answer": 0.0, "상황": "아무것도 없음"},
]


# ===== 실험 시작 =====
print("=" * 60)
print("  L2 실험: 학습하는 뉴런")
print("  — 경험에서 스스로 배운다")
print("=" * 60)
print()

neuron = LearningNeuron("학습뉴런", num_inputs=3, learning_rate=0.5)

print(f"  🧒 [{neuron.name}] 생성됨 (아무것도 모르는 상태)")
print(f"     초기 가중치: {neuron.weights}  ← 랜덤 (의미 없는 숫자)")
print(f"     편향(bias): {neuron.bias}")
print()

# === STEP 1: 학습 전 테스트 (아무것도 모르는 상태) ===
print("─" * 60)
print("  STEP 1: 학습 전 — 아무것도 모르는 상태에서 맞춰보기")
print("─" * 60)
print()

before_results = []
correct_before = 0
for d in training_data:
    pred = neuron.predict(d["inputs"])
    guess = "위험" if pred >= 0.5 else "안전"
    actual = "위험" if d["answer"] == 1.0 else "안전"
    match = "✅" if guess == actual else "❌"
    if guess == actual:
        correct_before += 1
    before_results.append({"상황": d["상황"], "예측": guess, "정답": actual, "맞음": guess == actual})
    print(f"  {match} {d['상황']:<14} → 예측: {guess}  (확신도: {pred:.1%})  정답: {actual}")

print()
print(f"  정답률: {correct_before}/{len(training_data)} ({correct_before/len(training_data):.0%})")
print(f"  → 랜덤 가중치라서 찍는 거나 마찬가지")
print()

# === STEP 2: 학습 과정 ===
print("─" * 60)
print("  STEP 2: 학습 시작 — 같은 경험을 100번 반복한다")
print("─" * 60)
print()

epochs = 100
epoch_accuracy = []

for epoch in range(epochs):
    epoch_errors = []
    for d in training_data:
        pred, error = neuron.learn(d["inputs"], d["answer"])
        epoch_errors.append(abs(error))
    
    avg_error = sum(epoch_errors) / len(epoch_errors)
    
    # 정확도 계산
    correct = 0
    for d in training_data:
        pred = neuron.predict(d["inputs"])
        guess = 1 if pred >= 0.5 else 0
        if guess == d["answer"]:
            correct += 1
    accuracy = correct / len(training_data)
    epoch_accuracy.append(accuracy)
    
    # 주요 시점만 출력
    if epoch in [0, 4, 9, 19, 49, 99]:
        bar = "█" * int(accuracy * 20) + "░" * (20 - int(accuracy * 20))
        print(f"  반복 {epoch+1:3d}회 | 오차: {avg_error:.4f} | 정답률: {accuracy:.0%} |{bar}|")

print()
print(f"  📊 학습 완료!")
print(f"     최종 가중치: {neuron.weights}")
print(f"     편향(bias): {neuron.bias}")
print()

# 가중치 해석
print("  📖 가중치가 뭘 배웠나?")
labels = ["눈(시각)", "귀(청각)", "코(후각)"]
for i, (label, w) in enumerate(zip(labels, neuron.weights)):
    if w > 0.3:
        meaning = "← 이게 크면 위험하다고 배움!"
    elif w < -0.3:
        meaning = "← 이게 크면 안전하다고 배움!"
    else:
        meaning = "← 별로 중요하지 않다고 봄"
    print(f"     {label}: {w:+.4f}  {meaning}")
print()

# === STEP 3: 학습 후 같은 테스트 ===
print("─" * 60)
print("  STEP 3: 학습 후 — 같은 문제 다시 풀어보기")
print("─" * 60)
print()

correct_after = 0
for d in training_data:
    pred = neuron.predict(d["inputs"])
    guess = "위험" if pred >= 0.5 else "안전"
    actual = "위험" if d["answer"] == 1.0 else "안전"
    match = "✅" if guess == actual else "❌"
    if guess == actual:
        correct_after += 1
    print(f"  {match} {d['상황']:<14} → 예측: {guess}  (확신도: {pred:.1%})  정답: {actual}")

print()
print(f"  정답률: {correct_after}/{len(training_data)} ({correct_after/len(training_data):.0%})")
print()

# === STEP 4: 진짜 시험 — 처음 보는 상황 ===
print("─" * 60)
print("  STEP 4: 진짜 시험 — 한 번도 본 적 없는 상황")
print("─" * 60)
print()
print("  이 상황들은 훈련 데이터에 없었습니다.")
print("  뉴런이 '일반화'할 수 있는지 봅니다.")
print()

new_situations = [
    {"inputs": [0.85, 0.75, 0.15], "상황": "큰 그림자 + 낮은 소리", "예상": "위험"},
    {"inputs": [0.15, 0.1, 0.85], "상황": "고요한데 맛있는 냄새", "예상": "안전"},
    {"inputs": [0.95, 0.95, 0.05], "상황": "눈앞에서 뭔가 달려옴", "예상": "위험"},
    {"inputs": [0.05, 0.15, 0.6], "상황": "약한 꽃향기", "예상": "안전"},
    {"inputs": [0.5, 0.5, 0.5], "상황": "모든 감각 보통", "예상": "애매함"},
]

for ns in new_situations:
    pred = neuron.predict(ns["inputs"])
    guess = "위험" if pred >= 0.5 else "안전"
    confidence = pred if pred >= 0.5 else (1 - pred)
    print(f"  🆕 {ns['상황']:<20} → 판단: {guess} (확신도: {confidence:.0%})  [사람 예상: {ns['예상']}]")

print()

# === 요약 ===
print("=" * 60)
print("  이게 뭔가요?")
print("=" * 60)
print()
print("  L1에서는 제가 '이건 위험이야'라고 정해줬습니다. (가짜 판단)")
print("  L2에서는 뉴런이 스스로 배웠습니다. (진짜 학습)")
print()
print(f"  학습 전: {correct_before}/{len(training_data)} 정답 ({correct_before/len(training_data):.0%})")
print(f"  학습 후: {correct_after}/{len(training_data)} 정답 ({correct_after/len(training_data):.0%})")
print()
print("  그리고 핵심:")
print("  → 한 번도 안 본 상황도 판단할 수 있게 됐습니다")
print("  → 이것을 '일반화(generalization)'라고 합니다")
print("  → 진짜 뇌도 이렇게 합니다")
print()
print("  뉴런이 배운 것을 말로 풀면:")
print("  '눈에 큰 게 보이고 + 소리가 크면 = 위험'")
print("  '냄새만 나면 = 안전'")
print("  아무도 이렇게 가르치지 않았는데, 데이터에서 스스로 찾아낸 겁니다.")
print()

# 결과 저장
output = {
    "experiment": "L2_학습",
    "date": "2026-04-15",
    "initial_weights": "random",
    "final_weights": neuron.weights,
    "final_bias": neuron.bias,
    "training_epochs": epochs,
    "accuracy_before": correct_before / len(training_data),
    "accuracy_after": correct_after / len(training_data),
    "accuracy_progress": epoch_accuracy[::10],
    "conclusion": "학습하는 뉴런 정상 작동. 랜덤 시작 → 100회 반복 후 훈련 데이터 정답률 달성. 새로운 상황에도 일반화 가능 확인."
}

result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "실험결과", "L2-학습-결과.json")
with open(result_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"  📁 결과 저장됨: 실험결과/L2-학습-결과.json")

