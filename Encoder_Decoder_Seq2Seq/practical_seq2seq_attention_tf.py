import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 1. Data Preparation (Synthetic for simplicity)
# Reverse a sequence of characters, e.g., "abc" -> "cba"

def generate_data(num_samples=10000, max_len=10):
    chars = 'abcdefghijklmnopqrstuvwxyz'
    input_texts = []
    target_texts = []
    for _ in range(num_samples):
        length = np.random.randint(1, max_len + 1)
        input_seq = ''.join(np.random.choice(list(chars), length))
        target_seq = input_seq[::-1]
        input_texts.append(input_seq)
        target_texts.append(target_seq)
    return input_texts, target_texts

input_texts, target_texts = generate_data()

# Add special tokens and create vocabulary
input_vocab = sorted(list(set("".join(input_texts))))
target_vocab = sorted(list(set("".join(target_texts))))

input_vocab_size = len(input_vocab) + 3  # +3 for <pad>, <start>, and <end>
target_vocab_size = len(target_vocab) + 3  # +3 for <pad>, <start>, and <end>

input_token_index = dict([(char, i + 2) for i, char in enumerate(input_vocab)])
input_token_index["<pad>"] = 0
input_token_index["<start>"] = 1
input_token_index["<end>"] = len(input_vocab) + 2

target_token_index = dict([(char, i + 2) for i, char in enumerate(target_vocab)])
target_token_index["<pad>"] = 0
target_token_index["<start>"] = 1
target_token_index["<end>"] = len(target_vocab) + 2

reverse_target_token_index = dict((i, char) for char, i in target_token_index.items())

max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts]) + 1 # +1 for <end>

# Vectorize the data
def vectorize_sequences(texts, token_index, max_len):
    vectorized_data = np.zeros((len(texts), max_len), dtype='int32')
    for i, text in enumerate(texts):
        for t, char in enumerate(text):
            vectorized_data[i, t] = token_index[char]
    return vectorized_data

encoder_input_data = vectorize_sequences(input_texts, input_token_index, max_encoder_seq_length)
decoder_input_data = np.zeros((len(target_texts), max_decoder_seq_length), dtype='int32')
decoder_target_data = np.zeros((len(target_texts), max_decoder_seq_length, target_vocab_size), dtype='float32')

for i, target_text in enumerate(target_texts):
    decoder_input_data[i, 0] = target_token_index['<start>']
    for t, char in enumerate(target_text):
        decoder_input_data[i, t + 1] = target_token_index[char]
        decoder_target_data[i, t, target_token_index[char]] = 1.
    decoder_target_data[i, len(target_text), target_token_index['<end>']] = 1.

# 2. Build the Encoder-Decoder with Attention Model

# Encoder
class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(enc_units, return_sequences=True, return_state=True)

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.gru.units))

# Attention (Bahdanau Attention)
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query hidden state shape == (batch_size, hidden size)
        # values encoder output shape == (batch_size, max_len, hidden size)

        # expand_dims to add time axis to query
        query_with_time_axis = tf.expand_dims(query, 1) # (batch_size, 1, hidden size)

        # score shape == (batch_size, max_len, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_len, units)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # attention_weights shape == (batch_size, max_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

# Decoder
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(dec_units, return_sequences=True, return_state=True)
        self.fc = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(dec_units)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_len, hidden_size)
        context_vector, attention_weights = self.attention(hidden, enc_output)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size, vocab)
        x = self.fc(output)

        return x, state, attention_weights

# Model Parameters
embedding_dim = 256
units = 512
batch_size = 64
epochs = 10

encoder = Encoder(input_vocab_size, embedding_dim, units)
decoder = Decoder(target_vocab_size, embedding_dim, units)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

# 3. Training Step
@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([target_token_index['<start>']] * batch_size, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# 4. Training Loop

# Prepare dataset for training
BUFFER_SIZE = len(encoder_input_data)
dataset = tf.data.Dataset.from_tensor_slices((encoder_input_data, decoder_input_data)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(batch_size, drop_remainder=True)

for epoch in range(epochs):
    enc_hidden = encoder.initialize_hidden_state(batch_size)
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(len(encoder_input_data) // batch_size)):
        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

        if batch % 100 == 0:
            print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')

    print(f'Epoch {epoch+1} Loss {total_loss.numpy() / (len(encoder_input_data) // batch_size):.4f}')

# 5. Inference (Translation)
def evaluate(sentence):
    attention_plot = np.zeros((max_decoder_seq_length, max_encoder_seq_length))

    sentence = sentence.lower()
    inputs = [input_token_index[i] for i in sentence]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_encoder_seq_length,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''
    hidden = [tf.zeros((1, units))]
    enc_output, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_token_index['<start>']], 0)

    for t in range(max_decoder_seq_length):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_output)

        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += reverse_target_token_index[predicted_id]

        if reverse_target_token_index[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # The predicted ID is fed back into the model as the next input
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + [i for i in sentence], fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + [i for i in predicted_sentence], fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.FixedLocator(range(len(sentence) + 1)))
    ax.yaxis.set_major_locator(ticker.FixedLocator(range(len(predicted_sentence) + 1)))

    plt.show()

# Test the model
example_sentences = [
    "hello",
    "world",
    "tensorflow",
    "deeplearning",
    "attention"
]

for s in example_sentences:
    result, sentence, attention_plot = evaluate(s)
    print(f'Input: {sentence}')
    print(f'Predicted: {result}')
    # plot_attention(attention_plot, sentence, result)


