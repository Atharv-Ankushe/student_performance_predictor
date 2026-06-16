import pickle
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ─────────────────────────────
st.set_page_config(
    page_title="🎓 Student Score Predictor",
    page_icon="🎓",
    layout="centered"
)

# ── Load model ──────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ── Styling ─────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f0f4ff; }
    .stButton>button {
        width: 100%;
        background-color: #4f46e5;
        color: white;
        font-size: 18px;
        padding: 12px;
        border-radius: 10px;
        border: none;
        margin-top: 10px;
    }
    .stButton>button:hover { background-color: #4338ca; }
    .result-box {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        padding: 30px;
        border-radius: 16px;
        text-align: center;
        margin-top: 20px;
    }
    .score-number { font-size: 64px; font-weight: 800; }
    .score-label { font-size: 18px; opacity: 0.85; }
    .message-box {
        background: #ffffff;
        border-left: 5px solid #4f46e5;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 16px;
        font-size: 16px;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────
st.markdown("## 🎓 Student Score Predictor")
st.markdown("Fill in the details below to predict your exam score.")
st.markdown("---")

# ── Inputs ──────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.number_input(
        "📖 Hours Studied", min_value=0.0, max_value=24.0, value=6.0, step=0.5
    )
    sleep_hours = st.number_input(
        "😴 Sleep Hours", min_value=0.0, max_value=24.0, value=7.0, step=0.5
    )

with col2:
    previous_scores = st.number_input(
        "📊 Previous Exam Score", min_value=0.0, max_value=100.0, value=70.0, step=1.0
    )
    practice_papers = st.number_input(
        "📝 Practice Papers Done", min_value=0.0, max_value=50.0, value=4.0, step=1.0
    )

# ── Predict ─────────────────────────────────
if st.button("🔮 Predict My Score"):
    features = [[hours_studied, previous_scores, sleep_hours, practice_papers]]
    score = round(float(model.predict(features)[0]), 2)
    score = max(0.0, min(100.0, score))  # clamp to 0–100

    # Performance message
    if score >= 90:
        emoji, msg = "🏆", "Outstanding! Top of the class material."
    elif score >= 75:
        emoji, msg = "🌟", "Great job! Strong performance expected."
    elif score >= 60:
        emoji, msg = "📈", "Decent score — a little more effort goes a long way!"
    elif score >= 40:
        emoji, msg = "📚", "Average — consider more study hours and practice papers."
    else:
        emoji, msg = "⚠️", "Needs improvement — focus on consistency and sleep!"

    st.markdown(f"""
    <div class="result-box">
        <div class="score-label">Predicted Exam Score</div>
        <div class="score-number">{score}</div>
        <div class="score-label">out of 100</div>
    </div>
    <div class="message-box">{emoji} {msg}</div>
    """, unsafe_allow_html=True)
