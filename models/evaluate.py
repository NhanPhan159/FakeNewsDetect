import tensorflow as tf
import pandas as pd
import numpy as np
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn import metrics

# plot training
result = pd.read_csv("../save_model/result_train_tranlstm.csv")
fig = plt.figure(figsize=(6, 8))
#
fig.add_subplot(1, 2, 1)
plt.plot(result['accuracy'].values)
plt.plot(result['val_accuracy'].values)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
#
fig.add_subplot(1, 2, 2)
plt.plot(result['loss'].values)
plt.plot(result['val_loss'].values)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# ##
thresh = 0.65
## load test
test = pd.read_csv("../dataset/preprocessing/test_vn.csv")
x_test = test['noi_dung'].tolist()
y_test = test['nhan'].values
total_fake = np.sum(y_test==1)
total_real = np.sum(y_test==0)
total = len(y_test)
y_test = tf.expand_dims(y_test,axis=-1)

# load token and use token
f = open('../save_model/word_index_ver3.json')
word_index_json = json.load(f)
word_index_json = json.dumps(word_index_json)
token = tokenizer_from_json(word_index_json)
words_index = token.texts_to_sequences(x_test)
padding = pad_sequences(words_index,padding='post', truncating='post', maxlen=500)

#predict
model_load = tf.saved_model.load("../save_model/modelFin") ### câu lệnh load mô hình đề xuất
# model_load = load_model("../save_model/BiLstm.h5") ### câu lệnh load mô hình Bi-LSTM và GRU
predict = model_load(padding).numpy() ### sử dụng dòng này nếu load mô hình đề xuất
# predict = model_load.predict(padding) ### sử dụng dòng này nếu load mô hình GRU và Bi-LSTM

# evalue fakenews
predict_fake = np.where(predict > thresh,1,-1)
evaluate =  predict_fake == y_test
right_value = np.sum(evaluate==True)
print("Accuracy for predict fake news : {}%".format(round(right_value/total_fake,3)*100))

## evalue realnews
predict_real = np.where(predict > thresh,-1,0)
evaluate =  predict_real == y_test
right_value = np.sum(evaluate==True)
print("Accuracy for predict real news : {}%".format(round(right_value/total_real,3)*100))

## evaluate overall
predict_over = np.where(predict > thresh,1,0)
evaluate =  predict_over == y_test
right_value = np.sum(evaluate==True)
print("Accuracy for predict overall news : {}%".format(round(right_value/total,3)*100))

### ma trận nhầm lẫn
confusion_matrix = metrics.confusion_matrix(y_test, predict_over)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Real", "Fake"])
cm_display.plot()
plt.show()



