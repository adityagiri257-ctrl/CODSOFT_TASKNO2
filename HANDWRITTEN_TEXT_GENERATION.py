import tensorflow as tf
import numpy as np

# Load dataset
with open("handwritten_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Create character vocabulary
vocab = sorted(set(text))
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# Convert text to integer IDs
text_as_int = np.array([char2idx[c] for c in text])

# Sequence length
seq_length = 100

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = dataset.batch(seq_length + 1, drop_remainder=True)

# Split into input and target
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# Shuffle and batch
BUFFER_SIZE = 10000
BATCH_SIZE = 64

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# Model parameters
vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 512

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.SimpleRNN(rnn_units, return_sequences=True),
    tf.keras.layers.Dense(vocab_size)
])

# Compile model
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
)

# Train model
model.fit(dataset, epochs=20)

# Text generation function
def generate_text(model, start_string, num_generate=300, temperature=1.0):

    input_eval = [char2idx[c] for c in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    generated_text = []

    for i in range(num_generate):

        predictions = model(input_eval)

        predictions = predictions[:, -1, :] / temperature

        predicted_id = tf.random.categorical(predictions, 1)[0, 0].numpy()

        generated_text.append(idx2char[predicted_id])

        input_eval = tf.expand_dims([predicted_id], 0)

    return start_string + ''.join(generated_text)

# Generate sample text
print(generate_text(model, start_string="The "))
