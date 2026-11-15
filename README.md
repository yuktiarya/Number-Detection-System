# Digit Studio

A Streamlit interface for drawing or uploading a handwritten digit and predicting it with the MNIST CNN. Streamlit is the user-facing frontend; FastAPI provides the prediction API.

## 1. Install dependencies

Open a terminal in this folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Use an environment with a TensorFlow build compatible with your Python version. If TensorFlow cannot be installed for that interpreter, use a supported Python version such as Python 3.11 or 3.12.

## 2. Train and save the CNN

The notebook defines the model but does not save it. The training script uses the same CNN architecture and the included CSV data:

```powershell
python train_model.py
```

It trains for five epochs and writes `cnn_mnist.keras` in this folder.

## 3. Start the FastAPI backend

In the first terminal, from this folder:

```powershell
python -m uvicorn app:app --reload
```

FastAPI serves the prediction API at `http://127.0.0.1:8000`. Its interactive API documentation is at <http://127.0.0.1:8000/docs>.

## 4. Start the Streamlit frontend

In a second terminal, from this folder:

```powershell
python -m streamlit run streamlit_app.py
```

Open the local Streamlit URL shown in the terminal (usually <http://localhost:8501>). Draw a digit or upload a PNG, JPEG, WEBP, or BMP image. The Streamlit app calls FastAPI's `POST /predict` endpoint and displays the result and class probabilities.

If FastAPI runs at a different address, set `FASTAPI_URL` before starting Streamlit:

```powershell
$env:FASTAPI_URL = "http://127.0.0.1:8000"
python -m streamlit run streamlit_app.py
```

If you retrain the notebook model instead, save it as `cnn_mnist.keras` in this folder with `cnn.save("cnn_mnist.keras")`.
