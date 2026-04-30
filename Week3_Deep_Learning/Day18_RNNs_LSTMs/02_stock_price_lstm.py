"""
STOCK PRICE PREDICTION WITH LSTM
=================================
Time series forecasting using LSTM networks
Predicting next day's price from historical data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers
import time

print("="*80)
print("STOCK PRICE PREDICTION WITH LSTM")
print("="*80)

# ============================================
# GENERATE SYNTHETIC STOCK DATA
# ============================================

print("\n" + "="*80)
print("GENERATING SYNTHETIC STOCK DATA")
print("="*80)

print("""
SYNTHETIC STOCK PRICE GENERATOR:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

We'll generate realistic stock-like data with:
- Trend component (gradual increase/decrease)
- Seasonal component (weekly patterns)
- Random walk (daily volatility)
- Occasional jumps (news events)

This mimics real stock behavior without needing API access!
""")

# Generate synthetic stock data
np.random.seed(42)

def generate_stock_data(n_days=1000, start_price=100):
    """Generate synthetic stock price data"""
    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='D')
    
    # Trend component (gradual increase)
    trend = np.linspace(0, 50, n_days)
    
    # Seasonal component (weekly pattern)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Random walk (daily volatility)
    random_walk = np.cumsum(np.random.randn(n_days) * 2)
    
    # Occasional jumps (news events)
    jumps = np.zeros(n_days)
    jump_days = np.random.choice(n_days, size=10, replace=False)
    jumps[jump_days] = np.random.randn(10) * 15
    
    # Combine all components
    prices = start_price + trend + seasonal + random_walk + np.cumsum(jumps)
    
    # Ensure prices don't go negative
    prices = np.maximum(prices, 10)
    
    return pd.DataFrame({
        'Date': dates,
        'Price': prices
    })

# Generate data
stock_data = generate_stock_data(n_days=1000)

print(f"✅ Generated {len(stock_data)} days of stock data")
print(f"\n📊 Data Statistics:")
print(f"   Start price: ${stock_data['Price'].iloc[0]:.2f}")
print(f"   End price: ${stock_data['Price'].iloc[-1]:.2f}")
print(f"   Min price: ${stock_data['Price'].min():.2f}")
print(f"   Max price: ${stock_data['Price'].max():.2f}")
print(f"   Mean price: ${stock_data['Price'].mean():.2f}")
print(f"   Std dev: ${stock_data['Price'].std():.2f}")

# Visualize raw data
plt.figure(figsize=(15, 6))
plt.plot(stock_data['Date'], stock_data['Price'], linewidth=2, color='#2E86AB')
plt.xlabel('Date', fontweight='bold', fontsize=12)
plt.ylabel('Price ($)', fontweight='bold', fontsize=12)
plt.title('Synthetic Stock Price Data (1000 Days)', fontweight='bold', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_stock_data_raw.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_stock_data_raw.png")

# ============================================
# DATA PREPROCESSING
# ============================================

print("\n" + "="*80)
print("DATA PREPROCESSING FOR LSTM")
print("="*80)

print("""
TIME SERIES PREPROCESSING STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NORMALIZATION
   • Scale prices to [0, 1]
   • Why: LSTMs learn better with normalized inputs
   • Use MinMaxScaler (preserves shape of distribution)

2. CREATE SEQUENCES
   • Convert to supervised learning problem
   • Input: Last 60 days → Output: Next day
   • Sliding window approach:
     [day1, day2, ..., day60] → day61
     [day2, day3, ..., day61] → day62
     ...

3. TRAIN/TEST SPLIT
   • Use first 80% for training
   • Last 20% for testing
   • IMPORTANT: No shuffle! (time series must be in order)

4. RESHAPE FOR LSTM
   • Input shape: (samples, timesteps, features)
   • Our case: (samples, 60, 1)
   • 60 timesteps, 1 feature (price)
""")

# Extract prices
prices = stock_data['Price'].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
prices_scaled = scaler.fit_transform(prices)

print(f"✅ Data normalized to range [0, 1]")

# Create sequences
def create_sequences(data, seq_length=60):
    """Convert time series to supervised learning sequences"""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  # Use last 60 days to predict next day
X, y = create_sequences(prices_scaled, SEQ_LENGTH)

print(f"✅ Created sequences")
print(f"   Sequence length: {SEQ_LENGTH} days")
print(f"   Total sequences: {len(X)}")
print(f"   X shape: {X.shape} (samples, timesteps, features)")
print(f"   y shape: {y.shape}")

# Train/test split (80/20)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"\n✅ Train/test split")
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# ============================================
# BUILD LSTM MODEL
# ============================================

print("\n" + "="*80)
print("BUILDING LSTM MODEL")
print("="*80)

print("""
LSTM ARCHITECTURE FOR STOCK PREDICTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Input:           (60, 1)      60 days of prices
                    ↓
LSTM(128):       (60, 128)    First LSTM layer (return sequences)
Dropout(0.2):    (60, 128)    Regularization
                    ↓
LSTM(64):        (64,)        Second LSTM layer (last output only)
Dropout(0.2):    (64,)        Regularization
                    ↓
Dense(32):       (32,)        Fully connected
                    ↓
Dense(1):        (1,)         Output: Next day's price

Why this architecture:
✅ Stacked LSTMs (2 layers) for complex patterns
✅ Decreasing units (128→64) - funnel architecture
✅ Dropout (0.2) prevents overfitting
✅ Dense layer for final non-linear combination
""")

# Build model
model = keras.Sequential([
    # First LSTM layer (return sequences for stacking)
    layers.LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    layers.Dropout(0.2),
    
    # Second LSTM layer
    layers.LSTM(64, return_sequences=False),
    layers.Dropout(0.2),
    
    # Dense layers
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

print("✅ LSTM model created\n")
print("📊 Model Summary:")
model.summary()

# ============================================
# COMPILE & TRAIN
# ============================================

print("\n" + "="*80)
print("COMPILING & TRAINING MODEL")
print("="*80)

# Compile
model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error
)

print("""
✅ Model compiled with:
   • Optimizer: Adam
   • Loss: MSE (Mean Squared Error)
   • Metrics: MAE (Mean Absolute Error)
""")

# Train
print("🚀 Starting training...")
print("   Epochs: 50")
print("   Batch size: 32")
print("   Validation split: 20%")
print()

start_time = time.time()

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

train_time = time.time() - start_time

print(f"\n✅ Training complete in {train_time:.2f} seconds ({train_time/60:.1f} minutes)")

# ============================================
# EVALUATION & PREDICTION
# ============================================

print("\n" + "="*80)
print("MODEL EVALUATION")
print("="*80)

# Make predictions
y_train_pred = model.predict(X_train, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# Inverse transform (scale back to original prices)
y_train_actual = scaler.inverse_transform(y_train)
y_train_pred_actual = scaler.inverse_transform(y_train_pred)
y_test_actual = scaler.inverse_transform(y_test)
y_test_pred_actual = scaler.inverse_transform(y_test_pred)

# Calculate metrics
train_mse = mean_squared_error(y_train_actual, y_train_pred_actual)
train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
train_r2 = r2_score(y_train_actual, y_train_pred_actual)

test_mse = mean_squared_error(y_test_actual, y_test_pred_actual)
test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
test_r2 = r2_score(y_test_actual, y_test_pred_actual)

print(f"\n📊 TRAINING SET PERFORMANCE:")
print(f"   MSE:  {train_mse:.2f}")
print(f"   MAE:  ${train_mae:.2f}")
print(f"   R²:   {train_r2:.4f}")

print(f"\n📊 TEST SET PERFORMANCE:")
print(f"   MSE:  {test_mse:.2f}")
print(f"   MAE:  ${test_mae:.2f}")
print(f"   R²:   {test_r2:.4f}")

print(f"\n💡 Interpretation:")
print(f"   • On average, predictions are off by ${test_mae:.2f}")
print(f"   • R² = {test_r2:.4f} means model explains {test_r2*100:.1f}% of variance")

# VISUALIZATIONS
# ============================================

print("\n" + "="*80)
print("CREATING VISUALIZATIONS")
print("="*80)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

# Plot 1: Training History - Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#3498db')
ax1.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax1.set_ylabel('Loss (MSE)', fontweight='bold', fontsize=11)
ax1.set_title('Training & Validation Loss', fontweight='bold', fontsize=13)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Training History - MAE
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(history.history['mae'], label='Training MAE', linewidth=2, color='#2ecc71')
ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2, color='#9b59b6')
ax2.set_xlabel('Epoch', fontweight='bold', fontsize=11)
ax2.set_ylabel('MAE', fontweight='bold', fontsize=11)
ax2.set_title('Training & Validation MAE', fontweight='bold', fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Training Set Predictions
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(y_train_actual, label='Actual Price', linewidth=2, color='#2E86AB', alpha=0.7)
ax3.plot(y_train_pred_actual, label='Predicted Price', linewidth=2, color='#A23B72', alpha=0.7)
ax3.set_xlabel('Time (Days)', fontweight='bold', fontsize=11)
ax3.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
ax3.set_title(f'Training Set: Actual vs Predicted (MAE: ${train_mae:.2f})', 
              fontweight='bold', fontsize=13)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Test Set Predictions
ax4 = fig.add_subplot(gs[2, :])
ax4.plot(y_test_actual, label='Actual Price', linewidth=2, color='#2E86AB', alpha=0.7)
ax4.plot(y_test_pred_actual, label='Predicted Price', linewidth=2, color='#A23B72', alpha=0.7)
ax4.set_xlabel('Time (Days)', fontweight='bold', fontsize=11)
ax4.set_ylabel('Price ($)', fontweight='bold', fontsize=11)
ax4.set_title(f'Test Set: Actual vs Predicted (MAE: ${test_mae:.2f}, R²: {test_r2:.3f})', 
              fontweight='bold', fontsize=13, color='green')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)

plt.suptitle('LSTM STOCK PRICE PREDICTION RESULTS', 
             fontsize=16, fontweight='bold', y=0.995)

plt.savefig('02_stock_lstm_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_stock_lstm_results.png")

# Error distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Training errors
train_errors = y_train_actual.flatten() - y_train_pred_actual.flatten()
axes[0].hist(train_errors, bins=50, color='#3498db', alpha=0.7, edgecolor='black')
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Prediction Error ($)', fontweight='bold', fontsize=11)
axes[0].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[0].set_title('Training Error Distribution', fontweight='bold', fontsize=13)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Test errors
test_errors = y_test_actual.flatten() - y_test_pred_actual.flatten()
axes[1].hist(test_errors, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
axes[1].set_xlabel('Prediction Error ($)', fontweight='bold', fontsize=11)
axes[1].set_ylabel('Frequency', fontweight='bold', fontsize=11)
axes[1].set_title('Test Error Distribution', fontweight='bold', fontsize=13)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.suptitle('PREDICTION ERROR ANALYSIS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('02_stock_error_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_stock_error_distribution.png")

# FUTURE PREDICTION
# ============================================

print("\n" + "="*80)
print("PREDICTING NEXT 30 DAYS")
print("="*80)

print("""
MULTI-STEP PREDICTION:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Strategy: Iterative prediction
1. Use last 60 days to predict day 1001
2. Append prediction to sequence
3. Use days 2-61 + prediction to predict day 1002
4. Repeat for 30 days

Note: Uncertainty compounds over time!
""")

# Get last 60 days
last_sequence = prices_scaled[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

# Predict next 30 days
future_predictions = []
current_sequence = last_sequence.copy()

for i in range(30):
    # Predict next day
    next_pred = model.predict(current_sequence, verbose=0)
    future_predictions.append(next_pred[0, 0])
    
    # Update sequence (remove oldest, add prediction)
    current_sequence = np.append(current_sequence[:, 1:, :], 
                                 next_pred.reshape(1, 1, 1), 
                                 axis=1)

# Inverse transform predictions
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create dates for future predictions
last_date = stock_data['Date'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq='D')

# Visualize future predictions
plt.figure(figsize=(15, 6))

# Plot historical data (last 200 days)
plt.plot(stock_data['Date'].iloc[-200:], stock_data['Price'].iloc[-200:], 
         label='Historical Price', linewidth=2, color='#2E86AB')

# Plot future predictions
plt.plot(future_dates, future_predictions, 
         label='Future Predictions (30 days)', linewidth=2, color='#e74c3c', linestyle='--')

plt.axvline(x=last_date, color='green', linestyle='--', linewidth=2, 
            alpha=0.5, label='Today')
plt.xlabel('Date', fontweight='bold', fontsize=12)
plt.ylabel('Price ($)', fontweight='bold', fontsize=12)
plt.title('Stock Price: Historical + 30-Day Forecast', fontweight='bold', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('02_stock_future_prediction.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 02_stock_future_prediction.png")

print(f"\n📊 30-Day Forecast:")
print(f"   Current price: ${stock_data['Price'].iloc[-1]:.2f}")
print(f"   Day 1 prediction: ${future_predictions[0][0]:.2f}")
print(f"   Day 7 prediction: ${future_predictions[6][0]:.2f}")
print(f"   Day 30 prediction: ${future_predictions[29][0]:.2f}")
print(f"   Predicted change: ${future_predictions[29][0] - stock_data['Price'].iloc[-1]:.2f}")

# ============================================
# SAVE MODEL
# ============================================

print("\n" + "="*80)
print("SAVING MODEL")
print("="*80)

model.save('stock_price_lstm.keras')
print("✅ Model saved: stock_price_lstm.keras")

# ============================================
# KEY INSIGHTS
# ============================================

print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

insights = f"""
🎓 WHAT WE LEARNED ABOUT TIME SERIES WITH LSTM:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. SEQUENCE CREATION IS CRITICAL
   • Sliding window approach (60 days → 1 day)
   • Each sequence has temporal context
   • LSTM learns patterns across time

2. PERFORMANCE METRICS
   • MAE: ${test_mae:.2f} average error
   • R²: {test_r2:.3f} ({test_r2*100:.1f}% variance explained)
   • Good fit for stock prediction!

3. LSTM CAPTURES PATTERNS
   • Trends (gradual increase/decrease)
   • Seasonality (weekly patterns)
   • Momentum (recent price movements)
   • Not magic - can't predict random events!

4. MULTI-STEP PREDICTION UNCERTAINTY
   • 1-day ahead: Most accurate
   • 7-day ahead: Less accurate
   • 30-day ahead: Uncertainty compounds
   • Use with caution for long-term forecasts

5. NORMALIZATION ESSENTIAL
   • LSTMs sensitive to input scale
   • MinMaxScaler to [0, 1]
   • Always inverse transform for interpretation


💡 REAL-WORLD CONSIDERATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

⚠️ Stock Prediction Limitations:
- Market is partially random (efficient market hypothesis)
- News events unpredictable (earnings, Fed decisions, wars)
- This model only uses price - real models use:
  - Volume, moving averages, RSI, MACD
  - News sentiment, social media
  - Economic indicators

✅ Where LSTM Time Series Works Well:
- Weather forecasting (more predictable patterns)
- Energy demand prediction (seasonal patterns)
- Website traffic forecasting (regular patterns)
- Inventory demand (business cycles)


🚀 PRODUCTION IMPROVEMENTS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To improve accuracy:
1. Add more features (volume, technical indicators)
2. Use longer sequences (90-120 days)
3. Ensemble multiple models
4. Attention mechanisms (Transformers!)
5. External data (news sentiment, economic data)
"""

print(insights)

print("\n" + "="*80)
print("SESSION 2 COMPLETE: STOCK PRICE LSTM BUILT!")
print("="*80)
print(f"\n🎉 Achieved R² = {test_r2:.3f} and MAE = ${test_mae:.2f}!")
print(f"   LSTM successfully learned temporal patterns!")
print("\n☕ Take a 15-minute break before sentiment analysis!")