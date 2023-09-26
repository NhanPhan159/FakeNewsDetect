import streamlit as st
from prepare_data.data_clean import DataClean
from prepare_data.stop_word_vn import STOP_WORDS
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from models.tran_lstm import PositionalEmbedding, EncoderLayer, LstmBlock, HighLayer


def preprocessing(text):
    d = DataClean(stopword_vn=STOP_WORDS)
    text = d.preprocessingVN(text)
    return text

def wordIndex(text):
    f = open('./save_model/word_index_ver3.json')
    word_index_json = json.load(f)
    word_index_json = json.dumps(word_index_json)
    token = tokenizer_from_json(word_index_json)
    words_index = token.texts_to_sequences(text)
    print("Kết quả cho word index: ", words_index)
    padding = pad_sequences(words_index,padding='post', truncating='post', maxlen=500)
    return padding

def customRound(values): ### tính độ tin cậy
    values = round(values[0][0],3)
    if values > 0.5:
        return "giả",round(values,3) * 100
    else:
        return "thật",round((1-values),3) * 100
    
@st.cache_resource
def loadModel(path):
    model = tf.saved_model.load(path)
    return model


# load model
model = loadModel("./save_model/modelFin")
# model = tf.keras.models.load_model("./save_model/Gru.h5")

# Giao diện web
st.title("Công cụ AI hỗ trợ bạn đánh giá tính xác thực của bài báo")
form = st.form("my_form")
form.text_area("Hãy nhập một bài báo và còn lại để AI lo : ", key="text")
if form.form_submit_button("Dự đoán") :
    text = st.session_state["text"]
    text = preprocessing(text)
    encoder_input = wordIndex([text])
    result = model(encoder_input).numpy()
    real_or_fake,rate = customRound(result)
    st.write("Kết quả dự đoán : "+real_or_fake)
    st.write("Với độ tin cậy : {}%".format(str(rate)))