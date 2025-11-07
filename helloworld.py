from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

# allow your local Vite dev origin(s)
CORS(app,
     resources={r"/*": {"origins": [
         "http://localhost:5173",
         "http://127.0.0.1:5173",
         "https://spoofygoofy.xyz"
     ]}},
     supports_credentials=True)  # set to True only if you send cookies/auth

@app.route("/")
def home():
    return "Welcome to the ML API!"

@app.route("/predict", methods=["POST"])
def predict():
    """
    Example input JSON:
    {
        "feature_1": 1.2,
        "feature_2": 3.4
    }
    """
    try:
        data = request.get_json(force=True)

        # Extract features safely with defaults
        x1 = float(data.get("feature_1", 0))
        x2 = float(data.get("feature_2", 0))

        # Simple example model: weighted sum
        result = 0.3 * x1 + 0.7 * x2

        return jsonify({"prediction": result})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
