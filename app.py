"""
GrainPalette – Rice Type Classification web app.

Run locally:
    python app.py

Run in production (used by the Procfile on Render):
    gunicorn app:app
"""
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(BASE_DIR / "models" / "rice.h5"))).expanduser().resolve()
UPLOAD_DIR = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CLASS_NAMES = ["arborio", "basmati", "ipsala", "jasmine", "karacadag"]
IMG_SIZE = (224, 224)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grainpalette")

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload cap
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")

# --------------------------------------------------------------------------
# Model loading (lazy + defensive, so the app can start and serve pages
# even if the .h5 file is missing or TensorFlow fails to import it).
# --------------------------------------------------------------------------
_model = None
_model_load_error = None


def get_model():
    """Load and cache the Keras model. Returns None if it can't be loaded."""
    global _model, _model_load_error
    if _model is not None:
        return _model
    if _model_load_error is not None:
        return None
    try:
        import tensorflow as tf

        model_path = str(MODEL_PATH)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Train it with train_model.py or place rice.h5 in the models/ folder."
            )
        _model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully from %s", model_path)
    except Exception as exc:  # noqa: BLE001 - we want to surface any load error
        _model_load_error = str(exc)
        logger.error("Could not load model: %s", exc)
    return _model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath):
    """Match exactly the preprocessing used during training:
    BGR->RGB, resize to 224x224, scale to [0, 1]."""
    import cv2

    img = cv2.imread(filepath)
    if img is None:
        raise ValueError("The uploaded file is not a readable image.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)


# Try to warm up the model at startup (also runs under gunicorn on Render).
# Failure here is logged but never crashes the app.
get_model()


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/details")
def details():
    return render_template("details.html")


@app.route("/contact")
def contact():
    return render_template("contact.html")


@app.route("/result", methods=["POST"])
def predict():
    model = get_model()
    if model is None:
        return (
            render_template(
                "error.html",
                error_message="The prediction model isn't available right now. "
                + (_model_load_error or "Please try again later."),
            ),
            503,
        )

    if "image" not in request.files or request.files["image"].filename == "":
        return render_template("error.html", error_message="No image was selected."), 400

    file = request.files["image"]
    if not allowed_file(file.filename):
        return (
            render_template(
                "error.html",
                error_message="Unsupported file type. Please upload a PNG or JPG image.",
            ),
            400,
        )

    filename = secure_filename(file.filename)
    stamped_name = f"{datetime.utcnow():%Y%m%d%H%M%S}_{filename}"
    filepath = UPLOAD_DIR / stamped_name
    file.save(str(filepath))

    try:
        batch = preprocess_image(str(filepath))
        preds = model.predict(batch)
        prediction = CLASS_NAMES[int(np.argmax(preds))]
        confidence = float(np.max(preds)) * 100
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed")
        return render_template("error.html", error_message=str(exc)), 500
    finally:
        # Uploaded images are only needed for the single prediction.
        try:
            filepath.unlink(missing_ok=True)
        except OSError:
            pass

    return render_template(
        "results.html", prediction_text=prediction, confidence=f"{confidence:.1f}"
    )


@app.errorhandler(404)
def not_found(_e):
    return render_template("error.html", error_message="Page not found."), 404


@app.errorhandler(500)
def server_error(_e):
    return render_template("error.html", error_message="Something went wrong on our end."), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
