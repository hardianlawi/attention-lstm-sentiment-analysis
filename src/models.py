import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Embedding, Input, Layer
from tensorflow.keras.models import Model


def load_model(model_path):
    return tf.keras.models.load_model(model_path)


def get_model(model_type, *args, **kwargs):
    if model_type == "lstm":
        return _lstm_based_model(*args, **kwargs)
    if model_type == "attention":
        return _attention_based_model(*args, **kwargs)
    raise NotImplementedError


def _lstm_based_model(maxlen, vocab_size, emb_size, hidden_size=32, mask_zero=True):
    inp = Input(shape=[maxlen])
    emb = Embedding(vocab_size, emb_size, mask_zero=mask_zero,)
    x = emb(inp)
    x = LSTM(hidden_size)(x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    return model


def _attention_based_model(
    maxlen,
    vocab_size,
    emb_size,
    hidden_size=32,
    attention_hs=16,
    mask_zero=True,
    return_attn_weights=False,
):
    inp = Input(shape=[maxlen])
    emb = Embedding(vocab_size, emb_size, mask_zero=mask_zero,)
    x = emb(inp)
    x, hs, cs = LSTM(hidden_size, return_sequences=True, return_state=True)(x)
    x, weights = BahdanauAttention(attention_hs)(hs, x)
    out = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=out)
    if return_attn_weights:
        model_attention = Model(inputs=inp, outputs=weights)
        return model, model_attention
    return model


class BahdanauAttention(Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    # TODO: Add masking
    def call(self, query, values):
        # query : [batch_size, hidden_size]
        # values: [batch_size, maxlen, hidden_size]

        # (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # (batch_size, maxlen, units) + (batch_size, 1, units) = (batch_size, maxlen, units)
        score = self.W1(values) + self.W2(hidden_with_time_axis)

        # (batch_size, maxlen, 1)
        score = self.V(tf.nn.tanh(score))

        # attention_weights shape == (batch_size, maxlen, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
