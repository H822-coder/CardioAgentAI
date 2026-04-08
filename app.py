import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import anthropic

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────
# 1. Load & Prepare Data
# ──────────────────────────────────────────────
df = pd.read_csv("C:/Users/23btr/OneDrive/Desktop/coding/Python/Ai agent mini project/framingham.csv")

if "Sex" in df.columns:
    df["Sex"] = df["Sex"].map({"male": 1, "Male": 1, "female": 0, "Female": 0})
if "currentSmoker" in df.columns:
    df["currentSmoker"] = df["currentSmoker"].map({"Yes": 1, "yes": 1, "No": 0, "no": 0, 1: 1, 0: 0})
if "diabetes" in df.columns:
    df["diabetes"] = df["diabetes"].map({"Yes": 1, "yes": 1, "No": 0, "no": 0, 1: 1, 0: 0})

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()

FEATURES = ["age", "Sex", "sysBP", "totChol", "glucose", "currentSmoker", "BMI"]
from imblearn.over_sampling import SMOTE

X = df[FEATURES]
y = df["TenYearCHD"]

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# ──────────────────────────────────────────────
# 2. Train Model
# ──────────────────────────────────────────────
print("⏳ Training AI model …")
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X, y)

cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")
model_auc = float(np.mean(cv_scores))
print(f"✅ Model trained — AUC: {model_auc:.3f}")

pop_stats = {
    "age":     {"mean": float(df["age"].mean()),     "std": float(df["age"].std()),     "min": float(df["age"].min()),     "max": float(df["age"].max())},
    "sysBP":   {"mean": float(df["sysBP"].mean()),   "std": float(df["sysBP"].std()),   "min": float(df["sysBP"].min()),   "max": float(df["sysBP"].max())},
    "totChol": {"mean": float(df["totChol"].mean()), "std": float(df["totChol"].std()), "min": float(df["totChol"].min()), "max": float(df["totChol"].max())},
    "glucose": {"mean": float(df["glucose"].mean()), "std": float(df["glucose"].std()), "min": float(df["glucose"].min()), "max": float(df["glucose"].max())},
    "BMI":     {"mean": float(df["BMI"].mean()),     "std": float(df["BMI"].std()),     "min": float(df["BMI"].min()),     "max": float(df["BMI"].max())},
}

feature_importances = dict(zip(FEATURES, [float(x) for x in model.feature_importances_]))

# ──────────────────────────────────────────────
# 3. Anthropic client + Agent tools
# ──────────────────────────────────────────────

# Set ANTHROPIC_API_KEY in your environment before running.
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ── Tool definitions exposed to the agent ──
AGENT_TOOLS = [
    {
        "name": "get_risk_prediction",
        "description": (
            "Run the cardiovascular risk model for a patient and return the numeric "
            "risk probability (0-100), risk category (LOW/MODERATE/HIGH), individual "
            "risk factors with their severities, and health tips."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "age":     {"type": "number", "description": "Patient age in years"},
                "gender":  {"type": "string", "enum": ["Male", "Female"]},
                "sysBP":   {"type": "number", "description": "Systolic blood pressure (mmHg)"},
                "totChol": {"type": "number", "description": "Total cholesterol (mg/dL)"},
                "glucose": {"type": "number", "description": "Fasting glucose (mg/dL)"},
                "smoking": {"type": "string", "enum": ["Yes", "No"]},
                "bmi":     {"type": "number", "description": "Body Mass Index"},
            },
            "required": ["age", "gender", "sysBP", "totChol", "glucose", "smoking", "bmi"],
        },
    },
    {
        "name": "get_whatif_comparison",
        "description": (
            "Run the risk model twice — once with the patient's current values and "
            "once with modified 'what-if' values — and return both risk scores plus "
            "the absolute risk change. Use this to evaluate the impact of lifestyle "
            "improvements (e.g. quitting smoking, losing weight, lowering BP)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "current": {
                    "type": "object",
                    "description": "Current patient vitals (same fields as get_risk_prediction)",
                },
                "improved": {
                    "type": "object",
                    "description": "Hypothetical improved vitals to compare against",
                },
            },
            "required": ["current", "improved"],
        },
    },
    {
        "name": "get_population_stats",
        "description": (
            "Return population-level statistics from the Framingham dataset: "
            "mean/std for each vital, overall CHD risk distribution, and "
            "age-group risk profiles. Useful for contextualising a patient's values."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


# ── Tool execution helpers ──

def _run_prediction(params: dict) -> dict:
    """Core prediction logic (shared by tool + original /api/predict route)."""
    age      = float(params.get("age", 50))
    gender   = params.get("gender", "Male")
    sys_bp   = float(params.get("sysBP", 120))
    tot_chol = float(params.get("totChol", 200))
    glucose  = float(params.get("glucose", 90))
    smoking  = params.get("smoking", "No")
    bmi      = float(params.get("bmi", 24.0))

    male_binary   = 1 if gender == "Male" else 0
    smoker_binary = 1 if smoking == "Yes" else 0

    user_array = np.array([[age, male_binary, sys_bp, tot_chol, glucose, smoker_binary, bmi]])
    risk_prob  = float(model.predict_proba(user_array)[0][1]) * 100

    if risk_prob >= 50:
        category = "HIGH"
        emoji = "🔴"
        advice = "Please consult a cardiologist immediately."

    elif risk_prob >= 20:
        category = "MODERATE"
        emoji = "🟡"
        advice = "Consider lifestyle modifications."
    
    else:
        category = "LOW"
        emoji = "🟢"
        advice = "No immediate high risk detected."
    safe_ranges = {
        "age": (0, 45), "sysBP": (90, 120),
        "totChol": (125, 200), "glucose": (70, 100), "BMI": (18.5, 25.0),
    }
    user_values = {"age": age, "sysBP": sys_bp, "totChol": tot_chol, "glucose": glucose, "BMI": bmi}
    risk_factors = []
    for feat, val in user_values.items():
        lo, hi = safe_ranges[feat]
        if val > hi:
            severity = min((val - hi) / (hi * 0.5), 1.0)
            risk_factors.append({"factor": feat, "value": val, "safe_range": [lo, hi], "severity": round(severity, 2), "status": "elevated"})
        elif val < lo:
            severity = min((lo - val) / (lo * 0.5), 1.0)
            risk_factors.append({"factor": feat, "value": val, "safe_range": [lo, hi], "severity": round(severity, 2), "status": "low"})
        else:
            risk_factors.append({"factor": feat, "value": val, "safe_range": [lo, hi], "severity": 0.0, "status": "normal"})

    if smoker_binary:
        risk_factors.append({"factor": "Smoking", "value": "Yes", "safe_range": "No", "severity": 0.8, "status": "elevated"})

    tips = []
    if sys_bp > 120:
        tips.append({"icon": "🧂", "title": "Reduce Sodium Intake",   "desc": "Limit salt to < 2300 mg/day."})
        tips.append({"icon": "🏃", "title": "Regular Cardio Exercise","desc": "150 min/week of moderate-intensity exercise."})
    if tot_chol > 200:
        tips.append({"icon": "🥑", "title": "Heart-Healthy Diet",     "desc": "Increase fibre, omega-3. Reduce saturated fats."})
        tips.append({"icon": "💊", "title": "Cholesterol Check-up",   "desc": "Discuss statin therapy with your doctor."})
    if glucose > 100:
        tips.append({"icon": "🍬", "title": "Monitor Blood Sugar",    "desc": "Reduce refined sugars; test for pre-diabetes."})
    if bmi > 25:
        tips.append({"icon": "⚖️", "title": "Weight Management",      "desc": "5-10% weight reduction lowers cardiovascular risk."})
    if smoker_binary:
        tips.append({"icon": "🚭", "title": "Quit Smoking",           "desc": "Smoking doubles heart-attack risk."})
    if age > 55:
        tips.append({"icon": "🩺", "title": "Regular Screenings",     "desc": "Annual cardiac check-ups after age 55."})
    if not tips:
        tips.append({"icon": "✅", "title": "Keep It Up!",            "desc": "Your vitals look great."})

    comparison = {}
    for feat in ["age", "sysBP", "totChol", "glucose", "BMI"]:
        val = user_values[feat]
        percentile = float(np.mean(df[feat].values <= val) * 100)
        comparison[feat] = {
            "user_value": val,
            "population_mean": round(pop_stats[feat]["mean"], 1),
            "percentile": round(percentile, 1),
        }

    return {
        "risk_probability": round(risk_prob, 1),
        "category": category,
        "emoji": emoji,
        "advice": advice,
        "risk_factors": risk_factors,
        "health_tips": tips,
        "comparison": comparison,
        "feature_importances": feature_importances,
    }


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Dispatch a tool call and return a JSON string result."""
    if tool_name == "get_risk_prediction":
        result = _run_prediction(tool_input)
        return json.dumps(result)

    if tool_name == "get_whatif_comparison":
        current  = _run_prediction(tool_input["current"])
        improved = _run_prediction(tool_input["improved"])
        delta    = round(improved["risk_probability"] - current["risk_probability"], 1)
        return json.dumps({
            "current_risk":  current["risk_probability"],
            "improved_risk": improved["risk_probability"],
            "risk_change":   delta,
            "interpretation": (
                f"Risk would {'decrease' if delta < 0 else 'increase'} by "
                f"{abs(delta):.1f} percentage points with the proposed changes."
            ),
        })

    if tool_name == "get_population_stats":
        age_groups = pd.cut(
            df["age"], bins=[20, 35, 45, 55, 65, 100],
            labels=["20-35", "36-45", "46-55", "56-65", "65+"]
        )
        age_risk = df.groupby(age_groups, observed=True)["TenYearCHD"].mean().to_dict()
        age_risk = {k: round(float(v) * 100, 1) for k, v in age_risk.items()}
        return json.dumps({
            "population_stats":   pop_stats,
            "risk_distribution": {
                "total":     int(len(df)),
                "high_risk": int((df["TenYearCHD"] == 1).sum()),
                "low_risk":  int((df["TenYearCHD"] == 0).sum()),
            },
            "age_risk_profile":   age_risk,
            "feature_importances": feature_importances,
            "model_auc":          model_auc,
        })

    return json.dumps({"error": f"Unknown tool: {tool_name}"})


def run_agent(user_message: str, conversation_history: list) -> dict:
    """
    Agentic loop:
      1. Send conversation + tools to Claude.
      2. If Claude returns tool_use blocks, execute them and feed results back.
      3. Repeat until Claude returns a final text response.
    Returns {"reply": str, "tools_used": list[str]}.
    """
    SYSTEM_PROMPT = """You are CardioGuard AI, a helpful and empathetic cardiovascular health assistant.
You have access to a validated machine-learning model trained on the Framingham Heart Study dataset.

Your responsibilities:
- Assess 10-year coronary heart disease (CHD) risk from patient vitals.
- Explain results clearly to both patients and clinicians.
- Simulate what-if scenarios (lifestyle improvements, medication effects).
- Compare a patient's vitals against population statistics.
- Provide evidence-based, actionable health recommendations.

Always use the available tools to fetch real model predictions rather than guessing.
Be empathetic, clear, and never alarmist. Always recommend consulting a doctor for
clinical decisions. Format your final response in clean plain text (no markdown tables).
"""

    messages = conversation_history + [{"role": "user", "content": user_message}]
    tools_used = []

    while True:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
            messages=messages,
        )

        # Collect any tool-use blocks
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if not tool_use_blocks:
            # No more tool calls — extract final text reply
            text_blocks = [b.text for b in response.content if b.type == "text"]
            return {"reply": "\n".join(text_blocks), "tools_used": tools_used}

        # Execute every tool Claude requested and build the results message
        tool_results = []
        for block in tool_use_blocks:
            tools_used.append(block.name)
            result_str = execute_tool(block.name, block.input)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_str,
            })

        # Append Claude's assistant turn + our tool results, then loop
        messages.append({"role": "assistant", "content": response.content})
        messages.append({"role": "user",      "content": tool_results})


# ──────────────────────────────────────────────
# 4. API Routes
# ──────────────────────────────────────────────
@app.route("/")
def home():
    return "🚀 CardioGuard API is running!"

@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model_auc": model_auc, "dataset_rows": len(df)})


@app.route("/api/stats", methods=["GET"])
def dataset_stats():
    age_groups = pd.cut(
        df["age"], bins=[20, 35, 45, 55, 65, 100],
        labels=["20-35", "36-45", "46-55", "56-65", "65+"]
    )
    age_risk = df.groupby(age_groups, observed=True)["TenYearCHD"].mean().to_dict()
    age_risk = {k: round(float(v) * 100, 1) for k, v in age_risk.items()}
    return jsonify({
        "population":          pop_stats,
        "risk_distribution": {
            "total":     int(len(df)),
            "high_risk": int((df["TenYearCHD"] == 1).sum()),
            "low_risk":  int((df["TenYearCHD"] == 0).sum()),
        },
        "age_risk_profile":    age_risk,
        "feature_importances": feature_importances,
        "model_auc":           model_auc,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Original ML-only prediction endpoint (unchanged behaviour)."""
    data = request.get_json(force=True)
    result = _run_prediction({
        "age":     data.get("age", 50),
        "gender":  data.get("gender", "Male"),
        "sysBP":   data.get("sysBP", 120),
        "totChol": data.get("totChol", 200),
        "glucose": data.get("glucose", 90),
        "smoking": data.get("smoking", "No"),
        "bmi":     data.get("bmi", 24.0),
    })
    return jsonify(result)


# ── NEW: Agent-powered endpoints ──────────────

@app.route("/api/agent/chat", methods=["POST"])
def agent_chat():
    """
    Conversational agent endpoint.

    Request body:
      {
        "message": "What is my heart disease risk?",
        "history": [          ← optional: previous turns
          {"role": "user",      "content": "..."},
          {"role": "assistant", "content": "..."}
        ]
      }

    Response:
      {
        "reply":       "...",   ← agent's natural-language answer
        "tools_used":  ["get_risk_prediction"],
        "history":     [...]    ← updated history to pass back next turn
      }

    Example questions the agent handles autonomously:
      - "What is my 10-year CHD risk? Age 55, male, BP 145, chol 230, glucose 105, smoker, BMI 28"
      - "How much would quitting smoking reduce my risk?"
      - "How do my numbers compare to the general population?"
      - "Explain what systolic blood pressure means for heart health"
    """
    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    history = data.get("history", [])

    if not message:
        return jsonify({"error": "message is required"}), 400

    result = run_agent(message, history)

    # Build updated history for the client to pass back on the next turn
    updated_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": result["reply"]},
    ]

    return jsonify({
        "reply":      result["reply"],
        "tools_used": result["tools_used"],
        "history":    updated_history,
    })


@app.route("/api/agent/assess", methods=["POST"])
def agent_assess():
    """
    One-shot full assessment: takes patient vitals, runs the ML model,
    and returns both the raw prediction AND an AI-generated narrative report.

    Request body: same fields as /api/predict
    Response adds:  { "ai_report": "...", "tools_used": [...] }
    """
    data = request.get_json(force=True)

    # Run raw prediction first
    prediction = _run_prediction({
        "age":     data.get("age", 50),
        "gender":  data.get("gender", "Male"),
        "sysBP":   data.get("sysBP", 120),
        "totChol": data.get("totChol", 200),
        "glucose": data.get("glucose", 90),
        "smoking": data.get("smoking", "No"),
        "bmi":     data.get("bmi", 24.0),
    })

    # Ask the agent to narrate it
    prompt = (
        f"A patient has the following vitals: age {data.get('age')}, "
        f"gender {data.get('gender')}, systolic BP {data.get('sysBP')} mmHg, "
        f"total cholesterol {data.get('totChol')} mg/dL, "
        f"fasting glucose {data.get('glucose')} mg/dL, "
        f"smoker: {data.get('smoking')}, BMI {data.get('bmi')}. "
        "Please: (1) calculate their 10-year CHD risk, "
        "(2) compare their vitals to the population, "
        "(3) suggest the top 3 lifestyle changes that would reduce risk the most, "
        "and (4) provide a brief, empathetic summary suitable for the patient to read."
    )

    agent_result = run_agent(prompt, [])

    return jsonify({
        **prediction,
        "ai_report":  agent_result["reply"],
        "tools_used": agent_result["tools_used"],
    })


@app.route("/api/agent/whatif", methods=["POST"])
def agent_whatif():
    """
    What-if simulation agent.

    Request body:
      {
        "current":  { age, gender, sysBP, totChol, glucose, smoking, bmi },
        "scenario": "I want to quit smoking and lose 5 kg over the next year"
      }

    The agent interprets the scenario, infers modified vitals,
    runs the comparison tool, and narrates the projected risk change.
    """
    data     = request.get_json(force=True)
    current  = data.get("current", {})
    scenario = data.get("scenario", "")

    if not current or not scenario:
        return jsonify({"error": "current vitals and scenario are required"}), 400

    prompt = (
        f"Current patient vitals: {json.dumps(current)}. "
        f"The patient wants to know: '{scenario}'. "
        "Please: (1) infer what the improved vitals would look like after this change, "
        "(2) use the what-if comparison tool to calculate how much their risk would change, "
        "and (3) provide an encouraging, concrete explanation of the projected benefit."
    )

    result = run_agent(prompt, [])

    return jsonify({
        "scenario":   scenario,
        "reply":      result["reply"],
        "tools_used": result["tools_used"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)