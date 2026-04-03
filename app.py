from io import StringIO
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from src.utils import SPECTRAL_BASE_FEATURES, resolve_feature_columns


APP_TITLE = "Urban Land Cover Predictor"
MODEL_PATH = Path("artifacts/rf_model.joblib")
TRAIN_DATA_PATH = Path("data/training.csv")
TARGET_COL = "class"


app = FastAPI(title=APP_TITLE)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


def get_expected_features() -> list[str]:
    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found at {TRAIN_DATA_PATH}. Cannot infer expected features."
        )

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    return resolve_feature_columns(
        SPECTRAL_BASE_FEATURES,
        train_df.drop(columns=[TARGET_COL], errors="ignore").columns,
    )


MODEL = load_model()
EXPECTED_FEATURES = get_expected_features()


def render_home(message: str = "") -> HTMLResponse:
    feature_preview = ", ".join(EXPECTED_FEATURES[:10])
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{APP_TITLE}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 900px;
                margin: 40px auto;
                padding: 0 20px;
                line-height: 1.5;
            }}
            .card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 24px;
                background: #fafafa;
            }}
            .message {{
                margin-bottom: 16px;
                color: #b00020;
                font-weight: bold;
            }}
            button {{
                margin-top: 12px;
                padding: 10px 16px;
                border: none;
                border-radius: 6px;
                background: #0b5ed7;
                color: white;
                cursor: pointer;
            }}
            code {{
                background: #eee;
                padding: 2px 4px;
                border-radius: 4px;
            }}
        </style>
    </head>
    <body>
        <h1>{APP_TITLE}</h1>
        <div class="card">
            {f'<div class="message">{message}</div>' if message else ''}
            <p>Upload a CSV file containing the spectral feature columns expected by the trained model.</p>
            <p><strong>Expected feature count:</strong> {len(EXPECTED_FEATURES)}</p>
            <p><strong>First few expected columns:</strong> <code>{feature_preview}</code></p>
            <form action="/predict" method="post" enctype="multipart/form-data">
                <input type="file" name="file" accept=".csv" required />
                <br />
                <button type="submit">Predict</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/", response_class=HTMLResponse)
async def home():
    return render_home()


@app.post("/predict", response_class=HTMLResponse)
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        return render_home("Please upload a CSV file.")

    content = await file.read()
    try:
        input_df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read CSV file: {exc}") from exc

    missing_columns = [column for column in EXPECTED_FEATURES if column not in input_df.columns]
    if missing_columns:
        preview = ", ".join(missing_columns[:10])
        return render_home(
            "Missing required columns. "
            f"Example missing columns: {preview}"
        )

    prediction_df = input_df.copy()
    prediction_df["prediction"] = MODEL.predict(input_df[EXPECTED_FEATURES])

    preview_rows = prediction_df.head(20).to_html(index=False)
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{APP_TITLE} - Results</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 1100px;
                margin: 40px auto;
                padding: 0 20px;
            }}
            a {{
                display: inline-block;
                margin-bottom: 16px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #ccc;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <a href="/">Upload another file</a>
        <h1>Prediction Results</h1>
        <p><strong>Rows predicted:</strong> {len(prediction_df)}</p>
        {preview_rows}
    </body>
    </html>
    """
    return HTMLResponse(content=html)
