from flask import Flask, request, render_template
from model import predict_query

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    query = request.form['query']
    result = predict_query(query)
    return render_template('index.html', query=query, result=result)

if __name__ == '__main__':
    app.run(debug=True)