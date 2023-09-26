import pandas as pd
import numpy as np
from tran_lstm import TranBM,CustomSchedule,CustomModule
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger

## tham so de tokenizer
num_words = 28841
oov_token = '<UK>'
pad_type = 'post'
trunc_type = 'post'
maxlen = 500

### tham so cho model
len_seq = 500
embed_dim = 16
vocal_size = 28841
units_lstm = 32
dropout_rate = 0.1
units_ff = 16
num_heads = 3
num_layers = 2
units_cnn = 16
kernel = None
stride = None

### doc data
train = pd.read_csv("../dataset/preprocessing/train_vn.csv")
validate = pd.read_csv("../dataset/preprocessing/validate_vn.csv")
test = pd.read_csv("../dataset/preprocessing/test_vn.csv")

x_train = train['noi_dung'].tolist()
y_train = train['nhan'].tolist()
x_validate = validate['noi_dung'].tolist()
y_validate = validate['nhan'].tolist()
x_test = test['noi_dung'].tolist()
y_test = test['nhan'].values



### tokenizer data
tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
tokenizer.fit_on_texts(x_train)
# luu token
json_token = tokenizer.to_json()
with open("../save_model/word_index_ver3.json", "w") as outfile:
    outfile.write(json_token)
print("word index xong !!!")

train_sequences = tokenizer.texts_to_sequences(x_train)
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

valid_sequences = tokenizer.texts_to_sequences(x_validate)
valid_padded = pad_sequences(valid_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

test_sequences = tokenizer.texts_to_sequences(x_test)
test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

y_validate = np.array(y_validate,dtype='int')
y_train = np.array(y_train,dtype='int')

### xay dung model
model =  TranBM(num_layers=num_layers,
                 embed_dim=embed_dim,
                 num_heads=num_heads,
                 units_ff=units_ff,
                 vocab_size=vocal_size,
                 units_lstm=units_lstm,
                 len_seq = len_seq,
                 units_cnn = units_cnn,
                 dropout_rate=dropout_rate)
learning_rate = CustomSchedule(embed_dim)

## define loss & optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

num_epochs = 15
# model.build(input_shape=(None, 500))
# model.summary()

# Train the model
print("Training models ...")
##callback
result_train = CSVLogger("../save_model/result_train_tranlstm.csv",append=True)
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('val_accuracy') >=0.97:
      print("\nReached 97% validate accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()

his = model.fit(train_padded, y_train, epochs=num_epochs,batch_size=64, validation_data=(valid_padded, y_validate),callbacks=[result_train,callbacks])

score = model.evaluate(test_padded,y_test,verbose=0,batch_size=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
#save model
custom_model = CustomModule(model)
tf.saved_model.save(custom_model, "../save_model/modelFin")
