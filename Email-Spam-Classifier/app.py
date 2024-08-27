from flask import Flask, request, jsonify, send_from_directory
import pickle
import os

app = Flask(__name__)

# Load the model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

@app.route('/classify', methods=['POST'])
def classify_text():
    data = request.json
    message = data['message']
    vectorized_message = vectorizer.transform([message])
    prediction = model.predict(vectorized_message)[0]
    probability = model.predict_proba(vectorized_message)[0]


    return jsonify({
        'category': 'spam' if prediction == 1 else 'ham',
        'probability': {
            'spam': probability[1],
            'ham': probability[0]
        }
    })

@app.route('/')
def serve_html():
    return send_from_directory(os.path.dirname(__file__), 'index.html')

if __name__ == '__main__':
    app.run(debug=True)
