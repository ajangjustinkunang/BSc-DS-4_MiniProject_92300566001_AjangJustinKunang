"""
SpamShield Flask Application
============================
A professional SaaS web application for detecting spam messages using a
text-based machine learning pipeline.
"""

import os
import sys
import pickle
from flask import Flask, jsonify, render_template, request

# Add the model directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'model'))

# Import model components
from model import SpamDetectionModel

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max request size

# Global service instance
spam_service = None

MODEL_PATH = 'trained_model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'
SELECTOR_PATH = 'feature_selector.pkl'
DATA_PATH = 'data/spam.csv'


def load_model_components():
    """Load or train model components from disk."""
    global spam_service
    try:
        spam_service = SpamDetectionModel(
            model_path=MODEL_PATH,
            vectorizer_path=VECTORIZER_PATH,
            selector_path=SELECTOR_PATH,
        )

        if not spam_service.load_model():
            print("Model components not found. Training a new model...")
            X, y = spam_service.load_data(DATA_PATH)

            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            X_train_tfidf = spam_service.preprocess_data(X_train)
            X_test_tfidf = spam_service.vectorizer.transform(X_test)

            X_train_selected = spam_service.select_features(X_train_tfidf, y_train, n_features=2000)
            X_test_selected = spam_service.feature_selector.transform(X_test_tfidf)

            spam_service.train_model(X_train_selected, y_train)
            spam_service.evaluate_model(X_test_selected, y_test)
            spam_service.save_model()

        else:
            print("Loaded saved model components.")

        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False


def predict_spam(message_text):
    """Predict whether a message is spam based on raw text."""
    if spam_service is None:
        raise RuntimeError("Model not loaded. Please restart the application.")

    if not message_text or not isinstance(message_text, str):
        raise ValueError("Please provide a valid message text to analyze.")

    prediction = spam_service.predict(message_text)
    proba = spam_service.predict_proba(message_text)
    confidence = max(proba) * 100
    prediction_text = "Spam" if prediction == 1 else "Not Spam"
    return f"{prediction_text} (Confidence: {confidence:.1f}%)"


# Routes
@app.route("/", methods=["GET", "POST"])
def home():
    """Home page with spam detector form."""
    result = ""
    error = ""
    message_text = ""

    if request.method == "POST":
        message_text = request.form.get("message", "").strip()
        try:
            result = predict_spam(message_text)
        except ValueError as exc:
            error = str(exc)
        except Exception as exc:
            error = f"An error occurred: {str(exc)}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        message=message_text,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for spam prediction."""
    try:
        data = request.get_json(silent=True)
        if not data or "message" not in data:
            return jsonify({"error": "Missing JSON body with a 'message' field."}), 400

        message = data["message"]
        if not isinstance(message, str) or not message.strip():
            return jsonify({"error": "The 'message' field must be a non-empty string."}), 400

        result = predict_spam(message)
        return jsonify({"prediction": result, "status": "success"}), 200
    except ValueError as exc:
        return jsonify({"error": str(exc), "status": "error"}), 400
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}", "status": "error"}), 500


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "model_loaded": spam_service is not None}), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("SPAMSHIELD - SPAM DETECTION SERVICE")
    print("=" * 60)

    if load_model_components():
        print("\n✅ Application ready! Starting Flask server...\n")
        print("🌐 Access the application at: http://127.0.0.1:5000")
        print("📊 API available at: http://127.0.0.1:5000/api/predict")
        print("\nPress Ctrl+C to stop the server.\n")
        app.run(debug=True, host='127.0.0.1', port=5000)
    else:
        print("\n❌ Failed to load model components. Exiting.")
        sys.exit(1)