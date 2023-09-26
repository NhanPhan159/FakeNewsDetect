import tensorflow as tf
import numpy as np

def positional_encoding(len_seq, embed_dim):
  ### lenght :  so chu trong doan van
  ### depht : chieu sau vector
  # ?
  embed_dim = embed_dim/2 

  positions = np.arange(len_seq )[:, np.newaxis]     # (seq, 1)
  embed_dims = np.arange(embed_dim)[np.newaxis, :]/embed_dim   # (1, embed_dim)

  # nhan 2 vector
  angle_rates = 1 / (10000**embed_dims)         # (1, embed_dim)
  angle_rads = positions * angle_rates      # (pos, embed_dim)

  pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

  return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, embed_dim, len_seq=500):
    super().__init__()
    self.embed_dim = embed_dim
    self.embedding = tf.keras.layers.Embedding(vocab_size, embed_dim)
    self.pos_encoding = positional_encoding(len_seq=len_seq, embed_dim=embed_dim)

  def call(self, x):
    """
    input :  batch x sequen_len
    output :  batch x sequen_len x embed_dim
    """
    length_seq = tf.shape(x)[1]
    x = self.embedding(x)
    # chuan hoa x
    x *= tf.math.sqrt(tf.cast(self.embed_dim, tf.float32)) ## tf.cast ep kieu 
    x = x + self.pos_encoding[tf.newaxis, :length_seq, :]
    return x

class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class LstmBlock(tf.keras.layers.Layer):
  def __init__(self,units_lstm,dropout_rate,units_ff):
    super().__init__()
    self.lstm = tf.keras.Sequential([
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units_lstm,return_sequences=True)),
      tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units_lstm,return_sequences=True)),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    # self.linear = tf.keras.layers.Dense(units_ff,activation='relu')
  def call(self,x):
    x = self.lstm(x)
    # x = self.linear(x)
    return x

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, embed_dim, units_ff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(units_ff, activation='relu'),
      tf.keras.layers.Dense(units_ff),
      tf.keras.layers.Dropout(dropout_rate)
    ])
  def call(self, x):
    x = self.seq(x)
    return x
  
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, embed_dim, num_heads, units_lstm,units_ff, dropout_rate=0.1):
    super().__init__()

    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=embed_dim,
        dropout=dropout_rate)
    self.ffn = FeedForward(embed_dim, units_ff)
    # self.lstm_block = LstmBlock(units_lstm,dropout_rate,units_ff)
    self.norm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()

  def call(self, x):
    x_atten = self.self_attention(x)
    x_ff = self.ffn(x_atten)
    x = self.norm(self.add([x_atten,x_ff]))
    # x_lstm = self.lstm_block(x)
    # x = self.norm(x_ff)
    return x

class HighLayer(tf.keras.layers.Layer):
  def __init__(self,units,stride=[1,1],kernel=[3,3]):
    super().__init__()
    self.seq = tf.keras.Sequential([
       tf.keras.layers.Conv2D(units,kernel,activation='relu'),
       tf.keras.layers.MaxPooling2D(2,2),
      #  tf.keras.layers.Conv2D(units*2,kernel,activation='relu'),
      #  tf.keras.layers.MaxPooling2D(2,2)
    ])
  def call(self,x):
    return self.seq(x)
  
class TranBM(tf.keras.Model):
  def __init__(self, *, num_layers, embed_dim, num_heads,len_seq,
               units_ff,units_lstm, vocab_size,units_cnn,stride = [1,1],
               kernel=[3,3],dropout_rate=0.1):
    super().__init__()
    self.num_layers = num_layers

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size,len_seq=len_seq, embed_dim=embed_dim)

    self.enc_layers = [
        EncoderLayer(embed_dim=embed_dim,
                     num_heads=num_heads,
                     units_ff=units_ff,
                     units_lstm=units_lstm,
                     dropout_rate=dropout_rate)
        for _ in range(num_layers)]
    self.lstm_block = LstmBlock(units_lstm,dropout_rate,units_ff)
    self.high_layer = HighLayer(units_cnn,kernel,stride)
    self.fc = tf.keras.Sequential([
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dropout(dropout_rate),
      tf.keras.layers.Dense(32, activation='relu'),
      tf.keras.layers.Dropout(dropout_rate),
      tf.keras.layers.Dense(1,activation='sigmoid')
    ])


  def call(self, x):
    # `x` is token-IDs shape: (batch, len_seq)
    x = self.pos_embedding(x)  # Shape `(batch_size, len_seq, embed_dim)`.
    for i in range(self.num_layers):
      x = self.enc_layers[i](x)
    x = self.lstm_block(x)
    x = tf.expand_dims(x, -1)
    x = self.high_layer(x)
    x = tf.keras.layers.Flatten()(x)
    x = self.fc(x)
    return x  # Shape `(batch_size, 1)`.
  

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=470):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps
  def get_config(self):
    config = {
    'd_model': self.d_model,
    'warmup_steps': self.warmup_steps,
     }
    return config
  def __call__(self, step):
    step = tf.cast(step, tf.float32)
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
  
class CustomModule(tf.Module):
  def __init__(self,tran_lstm):
    super(CustomModule, self).__init__()
    self.model = tran_lstm

  @tf.function(input_signature=[tf.TensorSpec([None,None], tf.int64)])
  def __call__(self, x):
    return self.model(x)