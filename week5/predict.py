import pickle

from flask import Flask, request, jsonify

with open("./model_C=1.0.bin", "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask('churn')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    customer = request.get_json()

    # Transform the input data using the DictVectorizer
    X = dv.transform([customer])

    # Predict the churn probability using the model
    y_pred = model.predict_proba(X)[0, 1]
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred),
        'churn': bool(churn)
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
