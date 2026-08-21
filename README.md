# GrainPalette

GrainPalette is a Flask web application that classifies an uploaded rice
grain image as Arborio, Basmati, Ipsala, Jasmine, or Karacadag. It uses a
MobileNetV2 transfer-learning model with a five-class softmax output.

## Features

- Upload PNG, JPG, or JPEG images up to 16 MB.
- Resize and normalize images to 224 x 224 pixels before prediction.
- Display the predicted rice type and confidence score.
- Provide informative errors when an image or model is unavailable.

## Requirements

- Python 3.11
- pip
- Windows, macOS, or Linux

Python 3.11 is recommended because the project pins TensorFlow 2.15.0.

## Install and run locally

From the project directory:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

The repository includes a trained model at `models/rice.h5`. Start the app:

```bash
python app.py
```

Open http://127.0.0.1:5000 in a browser.

## Train the model

The bundled `Data/` folder contains a small sample dataset. To regenerate
the model with five quick epochs:

```bash
python quick_train.py
```

For a configurable run:

```bash
python train_model.py --data-dir Data --epochs 10 --output models/rice.h5
```

The sample set contains only about 10 images per class and is intended to
verify the pipeline, not to provide production-level accuracy. For better
results, use a larger labelled dataset with one folder per class:

```text
Data/
├── Arborio/
├── Basmati/
├── Ipsala/
├── Jasmine/
└── final-rice-class/
```

The training scripts accept either `Karakadag` or `Karacadag` for the final
class folder. Put PNG, JPG, or JPEG images inside the class folders; the
folder name supplies each image label. Images are resized to 224 x 224
pixels and scaled to the range 0 to 1.

## Project structure

```text
app.py                 Flask application and prediction routes
train_model.py         Configurable training script
quick_train.py         Five-epoch training shortcut
requirements.txt       Python dependencies
models/rice.h5         Trained Keras model
Data/                  Labelled training images
templates/             HTML templates
static/                CSS, JavaScript, images, and upload files
Procfile               Production start command
runtime.txt            Recommended Python version
```

## Configuration

The app supports these environment variables:

```text
PORT        Port used by Flask; defaults to 5000
MODEL_PATH  Optional path to a different rice.h5 model
SECRET_KEY  Flask secret key
FLASK_DEBUG Set to true to enable development debug mode
```

## Deployment

The included `Procfile` starts the app with Gunicorn on platforms such as
Render:

```bash
gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120
```

Ensure `models/rice.h5` is included in the deployment or provide its path
through `MODEL_PATH`.
