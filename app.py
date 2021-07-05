import numpy as np
import pandas as pd
import re
import os 
from flask import Flask, render_template, flash, request, url_for, redirect, session
import tensorflow as tf
from numpy import array
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
#tf.compat.v1.get_default_graph()

IMAGE_FOLDER = os.path.join('static', 'img')

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

def init():
    global model, graph
    
    model = load_model('movie_rating.h5')
    #graph = tf.get_default_graph()
    
@app.route('/', methods = ['POST', 'GET'])
def home():
    return render_template('index.html')


@app.route('/movie_rating', methods = ['POST', "GET"])
def movie_rating():
    if request.method == 'POST':  
        text = request.form['text']
        Sentiment = ''
        max_review_length = 500
        word_to_id = imdb.get_word_index()
        strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
        text = text.lower().replace("<br />", " ")
        text = re.sub(strip_special_chars, "", text.lower())
        
        words = text.split()
        x_text = [[word_to_id[word] if (word in word_to_id and word_to_id[word] <= 20000) else 0 for word in words ]]
        x_test = sequence.pad_sequences(x_test, maxlen = 500)
        vector = np.array([x_test.flatten()])
        with graph.as_default():
            probability = model.predict(array([vector][0]))[0][0]
            class1 = model.predict_classes(array([vector][0]))[0][0]
        if class1 == 0:
            sentiment = 'Negative'
            img_name = os.path.join(app.config['UPLOAD_FOLDER'], 'sad.png')
        else:
            sentiment = 'Positive'
            img_name = os.path.join(app.config['UPLOAD_FOLDER',], 'smile.png')
    return render_template('index.html', text = text, sentiment = sentiment, probability = probability, image = img_name)
    
    
if __name__ == '__main__':
    init()
    app.run()