"""
SENTIMENT ANALYSIS WITH LSTM
=============================
Text classification using LSTM networks
Classifying movie reviews as positive/negative
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import time

print("="*80)
print("SENTIMENT ANALYSIS WITH LSTM")
print("="*80)

# LOAD IMDB DATASET

print("\n" + "="*80)
print("LOADING IMDB MOVIE REVIEW DATASET")
print("="*80)

# Load data
print("📥 Loading IMDB dataset...")
(X_train_raw, y_train), (X_test_raw, y_test) = keras.datasets.imdb.load_data(num_words=10000)

print(f"✅ Dataset loaded!")
print(f"   Training samples: {len(X_train_raw):,}")
print(f"   Test samples: {len(X_test_raw):,}")
print(f"   Vocabulary size: 10,000 most common words")

# Check class distribution
unique, counts = np.unique(y_train, return_counts=True)
print(f"\n📊 Class Distribution:")
for label, count in zip(unique, counts):
  sentiment = "Negative" if label == 0 else "Positive"
  print(f"    {sentiment}: {count:,} samples ({count/len(y_train)*100:.1f}%)")

# Examine a sample review
print(f"\n📝 Example Review (as word indices):")
print(f"   {X_train_raw[0][:20]}...")
print(f"   Length: {len(X_train_raw[0])} words")
print(f"   Label: {'Positive' if y_train[0] == 1 else 'Negative'}")

# Decode sample review to text
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {v: k for k, v in word_index.items()}

def decode_review(encoded_review):
  """Convert word indices back to text"""
  return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

sample_review = decode_review(X_train_raw[0])
print(f"\n📝 Decoded Sample Review:")
print(f"   {sample_review[:200]}...")

# DATA PREPROCESSING

print("\n" + "="*80)
print("DATA PREPROCESSING FOR TEXT")
print("="*80)

print("""
TEXT PREPROCESSING STEPS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. VOCABULARY LIMITATION
   • Using 10,000 most common words
   • Rare words replaced with <UNK> (unknown)
   • Why: Reduces dimensionality, focuses on important words

2. SEQUENCE PADDING
   • Reviews have different lengths (50-2000 words)
   • Pad/truncate to fixed length (250 words)
   • Why: LSTM needs fixed input shape for batching
   
   Short review: "Great movie" (2 words)
   → Pad: [0, 0, 0, ..., 0, 245, 89] (250 words)
   
   Long review: 500 words
   → Truncate: Keep last 250 words

3. TRAIN/VAL SPLIT
   • Use 20% of training for validation
   • Monitor overfitting during training

Strategy: Padding/truncating to 250 words
""")

# Pad sequence to same length
MAX_LEN = 250 # Maximum review length

X_train = pad_sequences(X_train_raw, maxlen=MAX_LEN, padding='post', truncating='post')
X_test = pad_sequences(X_test_raw, maxlen=MAX_LEN, padding='post', truncating='post')

print(f"✅ Sequences padded/truncated to {MAX_LEN} words")
print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Analyze review lengths
review_lengths = [len(review) for review in X_train_raw]

# Analyze review lengths
review_lengths = [len(review) for review in X_train_raw]

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(review_lengths, bins=50, color='#3498db', edgecolor='black', alpha=0.7)
plt.axvline(x=MAX_LEN, color='red', linestyle='--', linewidth=2, 
            label=f'Cutoff ({MAX_LEN} words)')
plt.xlabel('Review Length (words)', fontweight='bold', fontsize=11)
plt.ylabel('Frequency', fontweight='bold', fontsize=11)
plt.title('Distribution of Review Lengths', fontweight='bold', fontsize=13)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(review_lengths, bins=50, color='#e74c3c', edgecolor='black', alpha=0.7)
plt.axvline(x=MAX_LEN, color='red', linestyle='--', linewidth=2, 
            label=f'Cutoff ({MAX_LEN} words)')
plt.xlabel('Review Length (words)', fontweight='bold', fontsize=11)
plt.ylabel('Frequency', fontweight='bold', fontsize=11)
plt.title('Distribution (Zoomed)', fontweight='bold', fontsize=13)
plt.xlim(0, 500)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

plt.suptitle('REVIEW LENGTH ANALYSIS', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('03_review_length_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Saved: 03_review_length_distribution.png")
