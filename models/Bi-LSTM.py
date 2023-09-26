from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.callbacks import CSVLogger


## tham so
model_name = 'GRU'
vocab_size = 28814
maxlen = 500
output_dim = 16
pad_type = 'post'
trunc_type = 'post'

## load data
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
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<UK>")
tokenizer.fit_on_texts(x_train)
## luu token
# json_token = tokenizer.to_json()
# with open("../save_model/word_index.json", "w") as outfile:
#     outfile.write(json_token)
# print("word index xong !!!")

train_sequences = tokenizer.texts_to_sequences(x_train)
train_padded = pad_sequences(train_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

valid_sequences = tokenizer.texts_to_sequences(x_validate)
valid_padded = pad_sequences(valid_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

test_sequences = tokenizer.texts_to_sequences(x_test)
test_padded = pad_sequences(test_sequences, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

y_validate = np.array(y_validate,dtype='int')
y_train = np.array(y_train,dtype='int')
## define model
result_train = CSVLogger("../save_model/result_train_bilstm.csv",append=True)

num_epochs = 15
model = Sequential([
                Embedding(input_dim=vocab_size, output_dim=output_dim, input_length=maxlen),
                Bidirectional(LSTM(units=output_dim,return_sequences=True)),
                Bidirectional(LSTM(units=output_dim)),
                Dense(64, activation='relu'),
                Dropout(0.1),
                Dense(32, activation='relu'),
                Dropout(0.1),
                Dense(units=1, activation='sigmoid')
            ])
## define loss & optimizer
loss_fn = BinaryCrossentropy()
optimizer = Adam(learning_rate=0.01, epsilon=1e-6)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])
his = model.fit(train_padded, y_train, epochs=num_epochs,batch_size=64, validation_data=(valid_padded, y_validate),callbacks=[result_train])

score = model.evaluate(test_padded,y_test,verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

##save
model.save("../save_model/BiLstm.h5")
