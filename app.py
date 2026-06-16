import pickle
import warnings
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  🎓  Student Score Predictor  — ML API v1.0
#  Model: Linear Regression (scikit-learn)
#  Features: Study Hours · Past Scores ·
#             Sleep · Practice Papers
# ─────────────────────────────────────────────

app = FastAPI(
    title="🎓 Student Score Predictor",
    description=(
        "Predict a student's exam score based on study habits.\n\n"
        "**Features used:**\n"
        "- `hours_studied` — Hours spent studying\n"
        "- `previous_scores` — Score in the previous exam\n"
        "- `sleep_hours` — Hours of sleep per night\n"
        "- `sample_question_papers_practiced` — Number of practice papers completed\n\n"
        "_Model: Linear Regression | Framework: scikit-learn_"
    ),
    version="1.0.0",
)

# ── Load model ──────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)


# ── Schemas ─────────────────────────────────
class PredictRequest(BaseModel):
    hours_studied: float = Field(..., ge=0, le=24, example=7.0,
                                  description="Hours spent studying (0–24)")
    previous_scores: float = Field(..., ge=0, le=100, example=78.0,
                                    description="Score in previous exam (0–100)")
    sleep_hours: float = Field(..., ge=0, le=24, example=8.0,
                                description="Sleep hours per night (0–24)")
    sample_question_papers_practiced: float = Field(..., ge=0, example=5.0,
                                                     description="Number of practice papers done")


class PredictResponse(BaseModel):
    predicted_score: float = Field(..., description="Predicted exam score (0–100)")
    message: str = Field(..., description="Performance message based on prediction")


# ── Helpers ─────────────────────────────────
def score_message(score: float) -> str:
    if score >= 90:
        return "🏆 Outstanding! Top of the class material."
    elif score >= 75:
        return "🌟 Great job! Strong performance expected."
    elif score >= 60:
        return "📈 Decent score — a little more effort goes a long way!"
    elif score >= 40:
        return "📚 Average — consider more study hours and practice papers."
    else:
        return "⚠️ Needs improvement — focus on consistency and sleep!"


# ── Routes ──────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root():
    return """
    <html>
      <head><title>Student Score Predictor</title></head>
      <body style="font-family:sans-serif;text-align:center;padding:60px;background:#f0f4ff;">
        <h1>🎓 Student Score Predictor API</h1>
        <p style="font-size:18px;color:#555;">ML-powered exam score prediction</p>
        <a href="/docs" style="
          display:inline-block;margin-top:20px;padding:12px 28px;
          background:#4f46e5;color:white;border-radius:8px;
          text-decoration:none;font-size:16px;">
          📖 Open API Docs
        </a>
      </body>
    </html>
    """


@app.post("/predict", response_model=PredictResponse, summary="Predict Exam Score")
def predict(data: PredictRequest):
    """
    Submit student details and get a **predicted exam score** along with
    a performance message.
    """
    features = [[
        data.hours_studied,
        data.previous_scores,
        data.sleep_hours,
        data.sample_question_papers_practiced,
    ]]
    try:
        score = round(float(model.predict(features)[0]), 2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        predicted_score=score,
        message=score_message(score),
    )


@app.get("/health", summary="Health Check")
def health():
    """Returns API status — useful for uptime monitors and deployment checks."""
    return {"status": "✅ healthy", "model": "LinearRegression", "version": "1.0.0"}
