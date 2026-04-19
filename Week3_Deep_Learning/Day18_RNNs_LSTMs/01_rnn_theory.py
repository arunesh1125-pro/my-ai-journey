"""
RECURRENT NEURAL NETWORKS (RNNs) & LSTMs
========================================
Teaching AI to understand sequences and remember context
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

print("="*80)
print("RECURRENT NEURAL NETWORKS (RNNs) & LSTMs")
print("="*80)

# ============================================
# WHY RNNs? THE SEQUENTIAL DATA PROBLEM
# ============================================

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                  WHY RNNs? THE SEQUENCE PROBLEM                        ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  YESTERDAY'S ARCHITECTURES (Dense & CNN):                              ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  Dense Networks:                                                 │ ║
║  │  • Fixed-size input (e.g., 784 pixels)                          │ ║
║  │  • Each input independent                                        │ ║
║  │  • No memory of previous inputs                                  │ ║
║  │  ✅ Great for: Tabular data                                     │ ║
║  │  ❌ Bad for: Sequences                                          │ ║
║  │                                                                  │ ║
║  │  CNNs:                                                           │ ║
║  │  • Fixed-size 2D input (e.g., 32×32×3)                          │ ║
║  │  • Spatial relationships                                         │ ║
║  │  • No temporal memory                                            │ ║
║  │  ✅ Great for: Images                                           │ ║
║  │  ❌ Bad for: Time series, text                                  │ ║
║  │                                                                  │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  THE PROBLEM: SEQUENTIAL DATA                                          ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  Examples of sequences where ORDER matters:                      │ ║
║  │                                                                  │ ║
║  │  1. TIME SERIES (Stock prices, weather, sensor data)            │ ║
║  │     Today's price depends on yesterday's price                   │ ║
║  │     [100, 102, 98, 103, 105, ?]                                 │ ║
║  │                                                                  │ ║
║  │  2. TEXT (Language, sentiment)                                   │ ║
║  │     "I'm not happy" ≠ "I'm happy"                               │ ║
║  │     Word order changes meaning!                                  │ ║
║  │                                                                  │ ║
║  │  3. AUDIO (Speech recognition, music)                            │ ║
║  │     Sound at t=5s depends on sound at t=4s                      │ ║
║  │                                                                  │ ║
║  │  4. VIDEO (Action recognition)                                   │ ║
║  │     Frame 100 depends on frames 1-99                            │ ║
║  │                                                                  │ ║
║  │  Key insight: CONTEXT from the PAST matters! ⏰                  │ ║
║  │                                                                  │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
║  RNN SOLUTION: MEMORY                                                  ║
║  ┌──────────────────────────────────────────────────────────────────┐ ║
║  │                                                                  │ ║
║  │  RNNs maintain a "hidden state" (memory):                        │ ║
║  │                                                                  │ ║
║  │  At each time step:                                              │ ║
║  │  • Process current input                                         │ ║
║  │  • Combine with previous hidden state (memory)                   │ ║
║  │  • Update hidden state for next step                             │ ║
║  │  • Make prediction                                               │ ║
║  │                                                                  │ ║
║  │  Benefits:                                                        │ ║
║  │  ✅ Handles variable-length sequences                           │ ║
║  │  ✅ Learns temporal patterns                                     │ ║
║  │  ✅ Shares weights across time steps                             │ ║
║  │  ✅ Models dependencies in data                                  │ ║
║  │                                                                  │ ║
║  └──────────────────────────────────────────────────────────────────┘ ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝
""")

# ============================================
# HOW RNNs WORK
# ============================================

print("\n" + "="*80)
print("HOW RNNs WORK: THE MEMORY MECHANISM")
print("="*80)

print("""
VANILLA RNN (Simple Recurrent Network):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ARCHITECTURE (Unrolled Over Time):

        Input:    x₁        x₂        x₃        x₄
                   ↓         ↓         ↓         ↓
    Hidden:   → [RNN] →  → [RNN] →  → [RNN] →  → [RNN] →
                   ↓         ↓         ↓         ↓
    Output:      y₁        y₂        y₃        y₄

Key insight: Same RNN cell used at EVERY time step!
Arrow → represents "hidden state" flowing through time.


MATHEMATICAL FORMULATION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At each time step t:

h_t = tanh(W_hh × h_{t-1} + W_xh × x_t + b_h)
y_t = W_hy × h_t + b_y

Where:
- x_t = Input at time t
- h_t = Hidden state (memory) at time t
- h_{t-1} = Hidden state from previous time step
- y_t = Output at time t
- W_hh = Weights for hidden-to-hidden (memory update)
- W_xh = Weights for input-to-hidden
- W_hy = Weights for hidden-to-output
- b_h, b_y = Biases


EXAMPLE: Predicting Next Word
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sentence: "The cat sat on the ___"

Step 1: Input "The"
        h₁ = process("The")
        Remember: "We're starting a sentence"

Step 2: Input "cat"
        h₂ = process("cat", h₁)
        Remember: "The cat... we're talking about a cat"

Step 3: Input "sat"
        h₃ = process("sat", h₂)
        Remember: "The cat sat... cat is doing something"

Step 4: Input "on"
        h₄ = process("on", h₃)
        Remember: "The cat sat on... on what?"

Step 5: Input "the"
        h₅ = process("the", h₄)
        Predict: "mat" (common completion!)

Each step adds context to memory! 🧠


KEY ADVANTAGES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ VARIABLE LENGTH INPUT
   • "Hello" (1 word) or "This is a long sentence" (6 words)
   • Dense/CNN need fixed size!

✅ PARAMETER SHARING
   • Same weights used at every time step
   • Like CNNs share filters across space, RNNs share across time

✅ TEMPORAL DEPENDENCIES
   • Captures relationships across time
   • Yesterday affects today affects tomorrow
""")

# ============================================
# THE VANISHING GRADIENT PROBLEM
# ============================================

print("\n" + "="*80)
print("VANILLA RNN PROBLEM: VANISHING GRADIENTS")
print("="*80)

print("""
THE PROBLEM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vanilla RNNs struggle with LONG sequences.

Example: "The cat, which was sitting on the mat near the window 
overlooking the garden, finally ___"

To predict "meowed", the RNN needs to remember "cat" from 15 words ago!

What happens:
- Gradient flows backward through time
- At each step, gradient is multiplied by weights
- After many steps: gradient → 0 (vanishes) or → ∞ (explodes)
- Can't learn long-term dependencies! ❌


MATHEMATICAL INTUITION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Gradient flow: ∂L/∂h₁ = ∂L/∂h₂ × ∂h₂/∂h₁ × ... × ∂h_T/∂h_{T-1}

If ∂h_t/∂h_{t-1} < 1 (usually true with tanh):
After T steps: gradient ≈ (0.9)^T

T=10: gradient ≈ 0.35 (OK)
T=50: gradient ≈ 0.005 (vanishing!)
T=100: gradient ≈ 0.000027 (completely gone!)

RNN forgets long-term context! 😔


REAL-WORLD IMPACT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Short sequences (< 10 steps): Vanilla RNN works ✅
Long sequences (> 20 steps): Vanilla RNN fails ❌

Examples:
- "I am happy" → "positive" ✅ (short, RNN OK)
- Long movie review → sentiment ❌ (long, RNN forgets)
- Stock prices (last 5 days) → predict ✅ (short, OK)
- Stock prices (last 200 days) → predict ❌ (long, forgets)

We need something better for long sequences! 🤔
""")

# ============================================
# LSTM: THE SOLUTION
# ============================================

print("\n" + "="*80)
print("LSTM: LONG SHORT-TERM MEMORY")
print("="*80)

print("""
╔════════════════════════════════════════════════════════════════════════╗
║                    LSTM: THE BREAKTHROUGH                              ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║  Invented: 1997 by Hochreiter & Schmidhuber                           ║
║  Purpose: Solve vanishing gradient problem                            ║
║  Key idea: EXPLICIT memory cell with gates                            ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

LSTM ARCHITECTURE (The "Smart Memory"):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LSTM has 4 components:

1. CELL STATE (C_t) - The "conveyor belt" of memory
   • Runs through entire sequence
   • Information flows with minimal changes
   • Like a highway for memory! 🛣️

2. FORGET GATE (f_t) - Decides what to forget
   • "Is this information still relevant?"
   • σ(W_f × [h_{t-1}, x_t] + b_f)
   • Output: 0 (forget completely) to 1 (keep completely)

3. INPUT GATE (i_t) - Decides what to add to memory
   • "Is this new information important?"
   • σ(W_i × [h_{t-1}, x_t] + b_i)
   • Combined with candidate values

4. OUTPUT GATE (o_t) - Decides what to output
   • "What should I output from my memory?"
   • σ(W_o × [h_{t-1}, x_t] + b_o)
   • Filters cell state for output


LSTM STEP-BY-STEP:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

At time step t:

Step 1: FORGET - What to throw away?
        f_t = σ(W_f × [h_{t-1}, x_t] + b_f)
        
Step 2: INPUT - What new info to store?
        i_t = σ(W_i × [h_{t-1}, x_t] + b_i)
        C̃_t = tanh(W_c × [h_{t-1}, x_t] + b_c)
        
Step 3: UPDATE - Update cell state
        C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t
        (⊙ means element-wise multiplication)
        
Step 4: OUTPUT - What to output?
        o_t = σ(W_o × [h_{t-1}, x_t] + b_o)
        h_t = o_t ⊙ tanh(C_t)


INTUITIVE EXAMPLE: Reading a Story
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sentence: "Alice went to the store. Bob stayed home. Alice bought milk."

Processing "Alice went to store":
- FORGET: Nothing to forget yet
- INPUT: Remember "Alice is the subject"
- CELL STATE: C = ["Alice is doing something"]
- OUTPUT: h = "Alice at store"

Processing "Bob stayed home":
- FORGET: Keep Alice info (might be relevant later)
- INPUT: Add "Bob stayed home"
- CELL STATE: C = ["Alice at store", "Bob at home"]
- OUTPUT: h = "Two people, different locations"

Processing "Alice bought milk":
- FORGET: Keep Alice, forget Bob (not relevant now)
- INPUT: Add "bought milk"
- CELL STATE: C = ["Alice at store", "bought milk"]
- OUTPUT: h = "Alice purchased item"

LSTM remembered "Alice" across 3 sentences! 🧠✨


WHY LSTM SOLVES VANISHING GRADIENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cell state equation:
C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t

Notice: Addition (+), not just multiplication!
- Vanilla RNN: h_t = tanh(W × h_{t-1}) → repeated multiplication
- LSTM: C_t = f × C_{t-1} + ... → addition preserves gradient!

Result: Gradients flow easily through hundreds of time steps! ✅


VANILLA RNN vs LSTM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Vanilla RNN:
- Simple (one hidden state)
- Fast to compute
- Good for short sequences (< 10 steps)
- Forgets long-term context ❌
- Parameters: ~4K per layer

LSTM:
- Complex (4 gates + cell state)
- 4x slower than RNN
- Great for long sequences (100+ steps) ✅
- Remembers long-term context ✅
- Parameters: ~16K per layer (4x more)

Trade-off: Complexity for capability!
""")

# ============================================
# KERAS IMPLEMENTATION
# ============================================

print("\n" + "="*80)
print("BUILDING RNNs & LSTMs IN KERAS")
print("="*80)

print("""
KERAS MAKES IT EASY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Simple RNN:
    layers.SimpleRNN(units, return_sequences=True/False)

LSTM:
    layers.LSTM(units, return_sequences=True/False)

GRU (Gated Recurrent Unit - LSTM variant):
    layers.GRU(units, return_sequences=True/False)


KEY PARAMETERS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

units: Number of neurons in hidden state
    • 32, 64, 128 typical for simple tasks
    • 256, 512 for complex tasks

return_sequences:
    • True: Return full sequence of outputs (for stacked RNNs)
    • False: Return only last output (for final prediction)

input_shape:
    • (timesteps, features)
    • timesteps: Length of sequence
    • features: Dimensions at each step


EXAMPLE ARCHITECTURES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SINGLE LSTM (Simple):
   Input → LSTM(64) → Dense → Output
   
2. STACKED LSTM (Deep):
   Input → LSTM(128, return_sequences=True) → LSTM(64) → Dense → Output
   
3. BIDIRECTIONAL LSTM (Reads both directions):
   Input → Bidirectional(LSTM(64)) → Dense → Output
""")

# Build example models
print("\n🔨 Example 1: Simple LSTM for Sequence Prediction")

simple_lstm = keras.Sequential([
    layers.LSTM(64, input_shape=(10, 1)),  # 10 timesteps, 1 feature
    layers.Dense(1)
])

print("✅ Simple LSTM created")
print("\n📊 Model Summary:")
simple_lstm.summary()

print("\n" + "-"*80)
print("\n🔨 Example 2: Stacked LSTM (Deep)")

stacked_lstm = keras.Sequential([
    layers.LSTM(128, return_sequences=True, input_shape=(10, 1)),
    layers.Dropout(0.2),
    layers.LSTM(64),
    layers.Dropout(0.2),
    layers.Dense(1)
])

print("✅ Stacked LSTM created")
print("\n📊 Model Summary:")
stacked_lstm.summary()

print("\n" + "-"*80)
print("\n🔨 Example 3: Bidirectional LSTM")

bidirectional_lstm = keras.Sequential([
    layers.Bidirectional(layers.LSTM(64), input_shape=(10, 1)),
    layers.Dense(1)
])

print("✅ Bidirectional LSTM created")
print("\n📊 Model Summary:")
bidirectional_lstm.summary()

# ============================================
# COMPARISON TABLE
# ============================================

print("\n" + "="*80)
print("ARCHITECTURE COMPARISON")
print("="*80)

comparison = """
╔═══════════════════════════════════════════════════════════════════════╗
║               DENSE vs CNN vs RNN/LSTM COMPARISON                     ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  Architecture │ Input Type      │ Best For           │ Memory        ║
║  ─────────────┼─────────────────┼────────────────────┼─────────────  ║
║  Dense        │ 1D fixed vector │ Tabular data       │ None          ║
║               │ (e.g., 784)     │ Simple patterns    │               ║
║               │                 │                    │               ║
║  CNN          │ 2D fixed image  │ Images             │ None          ║
║               │ (e.g., 32×32×3) │ Spatial patterns   │               ║
║               │                 │                    │               ║
║  RNN          │ Variable seq    │ Short sequences    │ Short-term    ║
║               │ (e.g., [t,f])   │ < 10 timesteps     │ (forgets)     ║
║               │                 │                    │               ║
║  LSTM         │ Variable seq    │ Long sequences     │ Long-term ✅  ║
║               │ (e.g., [t,f])   │ 100+ timesteps     │ (remembers)   ║
║                                                                       ║
╠═══════════════════════════════════════════════════════════════════════╣
║                                                                       ║
║  USE CASES BY ARCHITECTURE:                                           ║
║  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ║
║                                                                       ║
║  Dense:                                                               ║
║  • House price prediction (bedrooms, sqft, location)                 ║
║  • Customer churn (age, purchases, tenure)                           ║
║  • Iris classification (sepal, petal dimensions)                     ║
║                                                                       ║
║  CNN:                                                                 ║
║  • Image classification (CIFAR-10, ImageNet)                         ║
║  • Object detection (YOLO, R-CNN)                                    ║
║  • Medical imaging (X-ray diagnosis)                                 ║
║                                                                       ║
║  RNN/LSTM:                                                            ║
║  • Stock price prediction (time series)                              ║
║  • Text generation (GPT, language models)                            ║
║  • Sentiment analysis (movie reviews)                                ║
║  • Speech recognition (audio → text)                                 ║
║  • Machine translation (English → Tamil)                             ║
║  • Video analysis (action recognition)                               ║
║                                                                       ║
╚═══════════════════════════════════════════════════════════════════════╝
"""

print(comparison)

# ============================================
# KEY INSIGHTS
# ============================================

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

insights = """
🎓 WHAT WE LEARNED:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SEQUENCES NEED MEMORY
   • Order matters in sequential data
   • Context from past affects future
   • Dense/CNN can't model this → RNN can! ✅

2. VANILLA RNN LIMITATION
   • Works for short sequences (< 10 steps)
   • Vanishing gradients for long sequences
   • Forgets long-term context ❌

3. LSTM BREAKTHROUGH
   • Explicit memory cell with gates
   • Forget, Input, Output gates control info flow
   • Solves vanishing gradient problem
   • Remembers 100+ timesteps! ✅

4. PARAMETER SHARING ACROSS TIME
   • Like CNNs share across space, RNNs share across time
   • Same weights at every timestep
   • Efficient for variable-length sequences

5. BIDIRECTIONAL = BETTER CONTEXT
   • Read sequence forward AND backward
   • "The cat sat on the ___" + "mat was soft"
   • Future context helps current prediction
   • 2x parameters but often worth it!


💡 PRACTICAL TIPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Use LSTM (not vanilla RNN) for most tasks
✅ Start with 64-128 units
✅ Stack 2-3 LSTM layers for complex tasks
✅ Use Dropout (0.2-0.3) between layers
✅ Bidirectional for text (reads both ways)
✅ return_sequences=True for stacking
✅ return_sequences=False for final layer

🎯 WHEN TO USE WHAT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Stock Prices → LSTM (time series)
Movie Review → LSTM (text with context)
Sentence → Bidirectional LSTM (context both ways)
Real-time Sensor → Simple RNN (fast, short context)
"""

print(insights)

print("\n" + "="*80)
print("SESSION 1 COMPLETE: RNN/LSTM Theory Mastered!")
print("="*80)
print("\n☕ Take a 15-minute break before building stock predictor!")