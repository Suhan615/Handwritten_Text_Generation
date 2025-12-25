import numpy as np
import tensorflow as tf

# Load text data
with open("handwritten_text.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Create character mappings
chars = sorted(list(set(text)))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}

# Convert text to numbers
text_as_int = np.array([char_to_idx[c] for c in text])

# Create sequences
seq_length = 40
examples = []
targets = []

for i in range(len(text_as_int) - seq_length):
    examples.append(text_as_int[i:i+seq_length])
    targets.append(text_as_int[i+seq_length])

X = np.array(examples)
y = np.array(targets)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(len(chars), 64),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(len(chars))
])

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Train model
model.fit(X, y, epochs=10, batch_size=64)

# Text generation function
def generate_text(start_string, length=200):
    input_eval = [char_to_idx[c] for c in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    generated_text = start_string

    for _ in range(length):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)

        # Make logits 2D
        predictions = tf.expand_dims(predictions, 0)

        predicted_id = tf.random.categorical(predictions, 1)[0, 0].numpy()

        generated_text += idx_to_char[predicted_id]
        input_eval = tf.expand_dims([predicted_id], 0)

    return generated_text

# Generate text
print("\nGenerated Text:\n")
print(generate_text("Dear "))
