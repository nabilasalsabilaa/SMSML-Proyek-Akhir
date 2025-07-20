import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

MLFLOW_ENDPOINT = "http://127.0.0.1:1234/invocations"  

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    payload = {
        "dataframe_records": [data]  
    }

    response = requests.post(
        MLFLOW_ENDPOINT,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    if response.status_code == 200:
        return jsonify(response.json())
    else:
        return jsonify({"error": response.text}), response.status_code

if __name__ == "__main__":
    app.run(debug=True)