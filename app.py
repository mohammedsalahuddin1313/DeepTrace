"""
DeepTrace Flask API: exposes POST /predict for hybrid spatial-frequency deepfake detection.

Prediction logic lives in inference.predict_deepfake(); the model is loaded once at startup
via inference.load_model() for performance (not on every HTTP request).
"""

from __future__ import annotations

import os
import sys
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from config import cfg
from inference import load_model, predict_deepfake

# --- Paths: temp uploads cleared per request after prediction ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".mp4",
    ".avi",
    ".mov",
    ".webm",
    ".mkv",
)


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100 MB uploads
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model_path": cfg.BEST_MODEL_PATH})

    @app.route("/predict", methods=["POST"])
    def predict():
        """
        Accept multipart/form-data with field name "file" (image or video).
        Returns JSON: {"result": "real"|"fake", "confidence": float in [0,1]}.

        Frontend connects with FormData.append("file", fileBlob) to http://127.0.0.1:5000/predict
        """
        if "file" not in request.files:
            return jsonify({"error": 'Missing form field "file".'}), 400

        f = request.files["file"]
        if not f or f.filename == "":
            return jsonify({"error": "No file selected."}), 400

        ext = os.path.splitext(f.filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return jsonify(
                {
                    "error": f"Unsupported extension '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                }
            ), 400

        safe_base = secure_filename(f.filename) or "upload"
        unique_name = f"{uuid.uuid4().hex}_{safe_base}"
        save_path = os.path.join(UPLOAD_DIR, unique_name)

        try:
            f.save(save_path)
            # --- Prediction happens in inference.predict_deepfake (same ML path as test.py) ---
            result, confidence = predict_deepfake(save_path)
            return jsonify({"result": result, "confidence": round(confidence, 6)})
        except FileNotFoundError as e:
            return jsonify({"error": str(e)}), 400
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500
        finally:
            if os.path.isfile(save_path):
                try:
                    os.remove(save_path)
                except OSError:
                    pass

    return app


def main():
    try:
        load_model()
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    app = create_app()
    print("DeepTrace API running at http://127.0.0.1:5000")
    print('POST /predict with form-data key "file" (image or video)')
    app.run(host="127.0.0.1", port=5000, debug=False)


if __name__ == "__main__":
    main()
