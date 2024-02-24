from flask import Flask, render_template, request
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)

# Load the models
with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('vect.pkl', 'rb') as file:
    loaded_vectorizer = pickle.load(file)
# Preprocess text function
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(words)

# Function to output label
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not Fake News"

# Define routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_input = request.form['news']
        processed_news = preprocess_text(news_input)
        vectorized_news = loaded_vectorizer.transform([ processed_news])
        prediction = loaded_model.predict(vectorized_news)
        label = output_label(prediction[0])
        return render_template('result.html', news=news_input, prediction=label)
if __name__ == '__main__':
    app.run(debug=True)
