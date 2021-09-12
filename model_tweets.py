import pandas as pd
import numpy as np
import tensorflow as tf
from google.protobuf.descriptor import MethodDescriptor
import pickle
import requests
from flask import Flask,request,jsonify,render_template 
#from tensorflow.keras.preprocessing.sequence import pad_sequences


#app = Flask("Category Model")
app = Flask("Sentiment Model")


global category
global label

category = {'neutral':0,'positive':1,'negative':2}
label = list(category.keys())


# max_len = 80
# embeding_dimension = 300
# trunc_type = "post"
# pad_type="post"


new_model = tf.keras.models.load_model('./model/Deep_Learning.h5')



with open('./model/tfidfVect.pickle','rb') as pr:   #save it in a notepad
    loaded_tokenizer = pickle.load(pr)


def predict_category(text):
    x_test_idf = loaded_tokenizer.transform(text)
    X_test = scipy.sparse.csr_matrix.todense(x_test_idf)
    pred = new_model.predict(X_test)
    return label[np.argmax(pred)]

@app.route('/')
def home():
    return render_template('form.html')


@app.route('/result',methods=["POST"])
def result():
    if request.method == "POST":
        text = request.form['input']
        predicted_category = predict_category(text)
        return render_template('result.html',text=text,predicted_category=predicted_category)



if __name__ == '__main__':
    app.run(host="localhost",port=5000,debug=True)
    


