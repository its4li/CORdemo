import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ── 1. DATA GENERATION ───────────────────────────────────────────────────────

@st.cache_data
def generate_data() -> pd.DataFrame:
    random = np.random.default_rng(seed=42)
    employees = ["EMP_01", "EMP_02", "EMP_03", "EMP_04", "EMP_05"]
    start_date = datetime(2026, 4, 30)
    records = []

    for day_idx in range(30):
        date = (start_date + timedelta(days=day_idx)).strftime("%Y-%m-%d")
        is_burnout_window = day_idx >= 20  # last 10 days

        for emp in employees:
            if emp == "EMP_04" and is_burnout_window:
                first_hour  = round(float(random.uniform(6.0, 7.5)),  1)
                last_hour   = round(float(random.uniform(23.0, 26.0)), 1)
                off_msgs    = int(random.integers(15, 31))
                jira_bounces = int(random.integers(10, 21))
            else:
                first_hour  = round(float(random.uniform(8.0, 9.5)),  1)
                last_hour   = round(float(random.uniform(17.0, 18.5)), 1)
                off_msgs    = int(random.integers(0, 5))
                jira_bounces = int(random.integers(0, 4))

            records.append({
                "date":                    date,
                "employee_id":             emp,
                "first_activity_hour":     first_hour,
                "last_activity_hour":      last_hour,
                "off_hours_messages_count": off_msgs,
                "jira_task_bounces":       jira_bounces,
            })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    return df

# ── 2. ANOMALY DETECTION (IsolationForest) ────────────────────────────────────

@st.cache_data
def compute_burnout_scores(df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "first_activity_hour",
        "last_activity_hour",
        "off_hours_messages_count",
        "jira_task_bounces",
    ]

    scaler  = StandardScaler()
    X       = scaler.fit_transform(df[features])

    model   = IsolationForest(
        n_estimators=200,
        contamination=0.08,
        random_state=42,
    )
    model.fit(X)

    # raw_scores: more negative → more anomalous
    raw_scores = model.score_samples(X)

    # Normalise to [0, 100] where 100 = most anomalous (highest burnout risk)
    min_s, max_s = raw_scores.min(), raw_scores.max()
    normalised   = 1 - (raw_scores - min_s) / (max_s - min_s)   # invert
    df = df.copy()
    df["burnout_risk_score"] = (normalised * 100).round(1)
    return df

# ── 3. UI HELPERS ─────────────────────────────────────────────────────────────

RISK_HIGH_THRESHOLD   = 70.0
RISK_MEDIUM_THRESHOLD = 45.0

def risk_label(score: float) -> tuple[str, str]:
    if score >= RISK_HIGH_THRESHOLD:
        return "🔴 HIGH RISK",   "#FF4B4B"
    if score >= RISK_MEDIUM_THRESHOLD:
        return "🟠 MEDIUM RISK", "#FFA500"
    return "🟢 LOW RISK",        "#21C45D"

def metric_card(label: str, value: str, colour: str = "#ffffff") -> None:
    st.markdown(
        f"""
        <div style="
            background: #1E1E2E;
            border-radius: 12px;
            padding: 18px 22px;
            border-left: 4px solid {colour};
            margin-bottom: 4px;
        ">
            <p style="color:#A0A0B0;font-size:13px;margin:0">{label}</p>
            <p style="color:{colour};font-size:26px;font-weight:700;margin:4px 0 0">{value}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── 4. MAIN APP ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Burnout Audit · Team Dashboard",
        page_icon="🧠",
        layout="wide",
    )

    # Global dark-mode style tweaks
    st.markdown(
        """
        <style>
        body, .stApp { background-color: #13131F; color: #E0E0F0; }
        [data-testid="stSidebar"] { background-color: #1A1A2E; }
        h1, h2, h3 { color: #E0E0F0; }
        hr { border-color: #2E2E4E; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Data pipeline ──────────────────────────────────────────────────────────
    df_raw    = generate_data()
    df_scored = compute_burnout_scores(df_raw)

    employees = sorted(df_scored["employee_id"].unique())

    # ── Sidebar ────────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/fluency/96/brain.png",
            width=64,
        )
        st.title("Burnout Audit")
        st.caption("Unsupervised anomaly detection on Slack & Jira metadata.")
        st.divider()

        selected_emp = st.selectbox(
            "👤  Select Employee",
            employees,
            index=employees.index("EMP_04"),
        )

        st.divider()
        st.markdown(
            """
            **Model:** IsolationForest  
            **Features:** working hours, off-hours messages, Jira bounces  
            **Risk threshold:** ≥ 70 = High
            """
        )

    # ── Filter for selected employee ──────────────────────────────────────────
    emp_df = df_scored[df_scored["employee_id"] == selected_emp].sort_values("date")

    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(f"## 🧠 Burnout Risk Dashboard — `{selected_emp}`")
    st.caption(f"30-day window · {emp_df['date'].min().date()} → {emp_df['date'].max().date()}")
    st.divider()

    # ── Key Metrics ────────────────────────────────────────────────────────────
    avg_risk    = emp_df["burnout_risk_score"].mean()
    peak_risk   = emp_df["burnout_risk_score"].max()
    avg_last_hr = emp_df["last_activity_hour"].mean()
    avg_off_msg = emp_df["off_hours_messages_count"].mean()
    label, colour = risk_label(avg_risk)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Avg Burnout Risk Score", f"{avg_risk:.1f} / 100", colour)
    with col2:
        metric_card("Peak Risk Score", f"{peak_risk:.1f} / 100", "#A78BFA")
    with col3:
        metric_card("Avg Last Activity Hour", f"{avg_last_hr:.1f}:00", "#60A5FA")
    with col4:
        metric_card("Avg Off-Hours Messages / Day", f"{avg_off_msg:.1f}", "#F472B6")

    st.markdown("&nbsp;")

    # ── HR Alert ───────────────────────────────────────────────────────────────
    if avg_risk >= RISK_HIGH_THRESHOLD:
        st.error(
            f"⚠️  **HR ALERT — {selected_emp}**  \n"
            f"This employee's average burnout risk score is **{avg_risk:.1f}/100**, "
            f"exceeding the HIGH RISK threshold ({RISK_HIGH_THRESHOLD}).  \n"
            "Immediate managerial check-in is recommended. Observed signals: "
            "late-night activity, elevated off-hours Slack messages, and high Jira task churn.",
            icon="🚨",
        )
    elif avg_risk >= RISK_MEDIUM_THRESHOLD:
        st.warning(
            f"🟠  **{selected_emp}** is showing moderate burnout signals "
            f"(avg score {avg_risk:.1f}/100). Consider a proactive 1:1 conversation.",
        )
    else:
        st.success(
            f"✅  **{selected_emp}** appears healthy with an average risk score of {avg_risk:.1f}/100.",
        )

    st.divider()

    # ── Charts ─────────────────────────────────────────────────────────────────
    left_col, right_col = st.columns([3, 2], gap="large")

    with left_col:
        # Daily working hours chart
        st.subheader("📅 Daily Working Hours")
        fig_hours = go.Figure()

        fig_hours.add_trace(go.Scatter(
            x=emp_df["date"], y=emp_df["first_activity_hour"],
            mode="lines+markers", name="First Activity",
            line=dict(color="#60A5FA", width=2),
            marker=dict(size=6),
        ))
        fig_hours.add_trace(go.Scatter(
            x=emp_df["date"], y=emp_df["last_activity_hour"],
            mode="lines+markers", name="Last Activity",
            line=dict(color="#F472B6", width=2),
            marker=dict(size=6),
            fill="tonexty", fillcolor="rgba(244,114,182,0.08)",
        ))
        # Reference line: 18:00 normal end
        fig_hours.add_hline(
            y=18.0, line_dash="dash",
            line_color="#A0A0B0", opacity=0.5,
            annotation_text="Normal EOD (18:00)",
            annotation_font_color="#A0A0B0",
        )

        fig_hours.update_layout(
            paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
            font_color="#E0E0F0",
            legend=dict(bgcolor="#1E1E2E", bordercolor="#2E2E4E"),
            xaxis=dict(gridcolor="#2E2E4E"),
            yaxis=dict(gridcolor="#2E2E4E", title="Hour of Day"),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
        )
        st.plotly_chart(fig_hours, use_container_width=True)

    with right_col:
        # Burnout risk score over time
        st.subheader("🔥 Burnout Risk Score Over Time")
        fig_risk = go.Figure()

        fig_risk.add_trace(go.Scatter(
            x=emp_df["date"], y=emp_df["burnout_risk_score"],
            mode="lines+markers", name="Risk Score",
            line=dict(color="#FF4B4B", width=2.5),
            marker=dict(size=7, color=emp_df["burnout_risk_score"],
                        colorscale="RdYlGn_r", cmin=0, cmax=100),
            fill="tozeroy", fillcolor="rgba(255,75,75,0.1)",
        ))
        fig_risk.add_hline(
            y=RISK_HIGH_THRESHOLD, line_dash="dot",
            line_color="#FF4B4B", opacity=0.7,
            annotation_text="High Risk Threshold",
            annotation_font_color="#FF4B4B",
        )
        fig_risk.update_layout(
            paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
            font_color="#E0E0F0",
            xaxis=dict(gridcolor="#2E2E4E"),
            yaxis=dict(gridcolor="#2E2E4E", title="Risk Score (0-100)", range=[0, 105]),
            margin=dict(l=0, r=0, t=10, b=0),
            height=320,
            showlegend=False,
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.divider()

    # ── Off-hours & Jira Bounces ───────────────────────────────────────────────
    st.subheader("📊 Stress Signals — Off-Hours Messages & Jira Task Bounces")
    fig_signals = px.bar(
        emp_df, x="date",
        y=["off_hours_messages_count", "jira_task_bounces"],
        barmode="group",
        color_discrete_map={
            "off_hours_messages_count": "#A78BFA",
            "jira_task_bounces":        "#34D399",
        },
        labels={
            "value":                    "Count",
            "off_hours_messages_count": "Off-Hours Messages",
            "jira_task_bounces":        "Jira Task Bounces",
        },
    )
    fig_signals.update_layout(
        paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
        font_color="#E0E0F0",
        legend=dict(bgcolor="#1E1E2E", bordercolor="#2E2E4E", title=""),
        xaxis=dict(gridcolor="#2E2E4E"),
        yaxis=dict(gridcolor="#2E2E4E"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=280,
    )
    st.plotly_chart(fig_signals, use_container_width=True)

    st.divider()

    # ── Team Comparison Heatmap ────────────────────────────────────────────────
    st.subheader("👥 Team Risk Heatmap (All Employees × 30 Days)")
    pivot = df_scored.pivot(index="employee_id", columns="date", values="burnout_risk_score")
    pivot.columns = [str(c.date()) for c in pivot.columns]

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="RdYlGn_r",
        zmin=0, zmax=100,
        aspect="auto",
        labels=dict(color="Risk Score"),
    )
    fig_heat.update_layout(
        paper_bgcolor="#1E1E2E", plot_bgcolor="#1E1E2E",
        font_color="#E0E0F0",
        coloraxis_colorbar=dict(
            tickfont=dict(color="#E0E0F0"),
            title=dict(font=dict(color="#E0E0F0")),
        ),
        xaxis=dict(showticklabels=False, title="Date (30 days →)"),
        margin=dict(l=0, r=0, t=10, b=0),
        height=240,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Raw Data Table ─────────────────────────────────────────────────────────
    with st.expander("🗃️  Raw Data for Selected Employee"):
        st.dataframe(
            emp_df.set_index("date")[[
                "first_activity_hour", "last_activity_hour",
                "off_hours_messages_count", "jira_task_bounces",
                "burnout_risk_score",
            ]].style
              .background_gradient(subset=["burnout_risk_score"], cmap="RdYlGn_r")
              .format(precision=1),
            use_container_width=True,
        )

if __name__ == "__main__":
    main()
