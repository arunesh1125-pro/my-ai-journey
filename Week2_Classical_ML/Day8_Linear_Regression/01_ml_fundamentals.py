"""
MACHINE LEARNING FUNDAMENTALS
==============================
Understanding how machines actually learn from data
"""

import numpy as np
import matplotlib.pyplot as plt

print("="*70)
print("MACHINE LEARNING: FROM FIRST PRINCIPLES")
print("="*70)

# WHAT IS LEARNING?

print("""
╔══════════════════════════════════════════════════════════════╗
║                  WHAT IS MACHINE LEARNING?                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  Traditional Programming:                                    ║
║  ┌─────────┐    ┌──────┐    ┌────────┐                      ║
║  │  Rules  │ +  │ Data │ →  │ Output │                      ║
║  └─────────┘    └──────┘    └────────┘                      ║
║  Example: if age > 18 then "Adult"                           ║
║                                                              ║
║  Machine Learning:                                           ║
║  ┌──────┐    ┌────────┐    ┌─────────┐                      ║
║  │ Data │ +  │ Output │ →  │  Rules  │                      ║
║  └──────┘    └────────┘    └─────────┘                      ║
║  Example: Given 10,000 people + labels,                      ║
║           discover the age pattern automatically             ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

# LEARNING PROCESS

print("\n" + "="*70)
print("HOW A MACHINE LEARNS: SIMPLE EXAMPLE")
print("="*70)

print("""
Business Problem: Predict Sales from Ad Spend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Historical Data (what we know):
┌──────────────┬─────────────┐
│  Ad Spend    │   Sales     │
│  (₹ lakhs)   │  (₹ lakhs)  │
├──────────────┼─────────────┤
│      1       │      3      │
│      2       │      5      │
│      3       │      7      │
│      4       │      9      │
│      5       │     11      │
└──────────────┴─────────────┘

Pattern: Sales ≈ 2 × Ad_Spend + 1

Machine Learning Goal: 
Discover this pattern automatically from data!
""")

# Real data
ad_spend = np.array([1, 2, 3, 4, 5])
sales = np.array([3, 5, 7, 9, 11])

# STEP 1: MAKE A GUESS (Random Model)

print("\nSTEP 1: START WITH RANDOM GUESS")
print("-"*50)

# Random initial guess: Sales = 1.5 x Ad_Spend + 0.5
m_guess = 1.5 # slope
b_guess = 0.5 # intercept

predictions_guess = m_guess * ad_spend + b_guess

print(f"Random Model: Sales = {m_guess} × Ad_Spend + {b_guess}")
print("\nPredictions vs Reality: ")
print(f"{'Ad Spend':>10} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
for spend, pred, actual in zip(ad_spend, predictions_guess, sales):
    error = abs(pred - actual)
    print(f"{spend:>10} {pred:>10.1f} {actual:>10} {error:>10.1f}")

total_error_guess = np.sum((predictions_guess - sales)**2)
print(f"\nTotal Error (squared): {total_error_guess:.2f}")

# STEP 2: IMPROVE THE GUESS (Learning!)


print("\n" + "="*70)
print("STEP 2: LEARN FROM MISTAKES (Gradient Descent)")
print("="*70)

print("""
How Machine Learning Works:
1. Make prediction with current model
2. Calculate error (how wrong we were)
3. Adjust parameters to reduce error
4. Repeat until error is minimized

This is called GRADIENT DESCENT!
""")

# Better model (learned)
m_learned = 2.0
b_learned = 1.0

predictions_learned = m_learned * ad_spend + b_learned

print(f"\nLearned Model: Sales = {m_learned} × Ad_Spend + {b_learned}")
print(f"\nPredictions vs Reality:")
print(f"{'Ad Spend':>10} {'Predicted':>10} {'Actual':>10} {'Error':>10}")
print("-"*45)
for spend, pred, actual in zip(ad_spend, predictions_learned, sales):
    error = abs(pred - actual)
    print(f"{spend:>10} {pred:>10.1f} {actual:>10} {error:10.1f}")

total_error_learned = np.sum((predictions_learned-sales)**2)
print(f"\nTotal Error (squared): {total_error_learned:.2f}")

print(f"\n✅ Error reduced from {total_error_guess:.2f} → {total_error_learned:.2f}")
print("✅ This is MACHINE LEARNING in action!")

# VISUALIZE THE  LEARNING

plt.figure(figsize=(12, 5))

# Before learning
plt.subplot(1, 2, 1)
plt.scatter(ad_spend, sales, color='red', s=100, label='Actual sales', zorder=3)
plt.plot(ad_spend, predictions_guess, 'b--', linewidth=2, label='Random Guess')
for x, y_pred, y_actual in zip(ad_spend, predictions_guess, sales):
    plt.plot([x, x], [y_pred, y_actual], 'k:', linewidth=1, alpha=0.5)
plt.xlabel('Ad Spend (₹ lakhs)', fontweight='bold')
plt.ylabel('Sales (₹ lakhs)', fontweight='bold')
plt.title('Before Learning (High Error)', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)


# After Learning
plt.subplot(1, 2, 2)
plt.scatter(ad_spend, sales, color='red', s=100, label='Actual Sales', zorder=3)
plt.plot(ad_spend, predictions_learned, 'g-', linewidth=2, label='Learned Model')
for x, y_pred, y_actual in zip(ad_spend, predictions_learned, sales):
    plt.plot([x, x], [y_pred, y_actual], 'k:', linewidth=1, alpha=0.5)
plt.xlabel('Ad Spend (₹ lakhs)', fontweight='bold')
plt.ylabel('Sales (₹ lakhs)', fontweight='bold')
plt.title('After Learning (Low Error)', fontweight='bold', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('01_Learning_process.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved visualization: 01_learning_process.png")

# KEY ML CONCEPTS

print("\n" + "="*70)
print("KEY MACHINE LEARNING CONCEPTS")
print("="*70)

concepts = """
1. FEATURES (X): Input variables
   Example: Ad Spend, Years of Experience, Temperature
   -> What we use to make predictions

2. TARGET (y): Output variable
   Example: Sales, Salary, Ice Cream Sales
   -> What we're trying to predict

3. MODEL: Mathematical function that maps X to y
   Example: y = 2X+ 1
   -> The "brain" that learned the pattern

4. PARAMETERS: Values the model learns
   Example: m=2, b=1 (slope and intercept)
   -> What changes during training

5. TRAINING: Process of learning parameters from data
   Example: Adjusting m and b to minimize error
   -> How the machine "learns"

6. PREDICTION: Using trained model on new data
   Example: If Ad_Spend = 10, predict_Sales = 2(10) + 1 = 21
   -> Applying what was learned

7. ERROR/LOSS: How wrong predictions are
   Example: Predicted=8, Actual=7 -> Error=1

8. GRADIENT DESCENT: Algorithm to minimize error
   Example: Taking small steps downhill to find minimum
   -> The "learning" algorithm
"""

print(concepts)

print("="*70)
print("FUNDAMENTALS COMPLETE!")
print("="*70)