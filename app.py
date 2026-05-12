import streamlit as st
import pdfplumber
import requests
from dotenv import load_dotenv
import os
import json
import re
from groq import Groq

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="AI Career Intelligence Agent",
    page_icon="🚀",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------

st.markdown("""
<style>

/* =========================================
GLOBAL APP
========================================= */

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.main {
    background-color: #0B1120;
    color: #F8FAFC;
}

/* Main Layout */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* =========================================
HEADINGS
========================================= */

h1, h2, h3 {
    font-weight: 700;
    letter-spacing: -0.5px;
}

h1 {
    font-size: 64px !important;
}

h2 {
    font-size: 34px !important;
}

h3 {
    font-size: 24px !important;
}

/* =========================================
METRIC CARDS
========================================= */

[data-testid="metric-container"] {

    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;

    background: linear-gradient(
        180deg,
        #111827,
        #0F172A
    );

    border: 1px solid #1E293B;

    padding: 28px 20px;

    border-radius: 18px;

    text-align: center;

    min-height: 180px;

    box-shadow:
        0px 8px 24px rgba(0,0,0,0.25);

    transition: all 0.3s ease;
}

[data-testid="metric-container"]:hover {

    transform: translateY(-4px);

    border: 1px solid #2563EB;

    box-shadow:
        0px 12px 28px rgba(37,99,235,0.20);
}

/* =========================================
METRIC LABELS
========================================= */

[data-testid="stMetricLabel"] {

    display: flex;

    justify-content: center;

    align-items: center;

    width: 100%;

    text-align: center;

    font-size: 18px;

    font-weight: 600;

    color: #CBD5E1;

    margin-bottom: 10px;
}

/* =========================================
METRIC VALUES
========================================= */

[data-testid="stMetricValue"] {

    display: flex;

    justify-content: center;

    align-items: center;

    width: 100%;

    text-align: center;

    font-size: 42px;

    font-weight: 800;

    color: #FFFFFF;

    line-height: 1.1;
}

/* =========================================
BUTTONS
========================================= */

div.stButton {

    display: flex;

    justify-content: center;

    align-items: center;
}

div.stButton > button {

    width: 100%;

    max-width: 420px;

    height: 64px;

    border: none;

    border-radius: 16px;

    background: linear-gradient(
        135deg,
        #2563EB,
        #1D4ED8
    );

    color: white;

    font-size: 20px;

    font-weight: 700;

    letter-spacing: 0.3px;

    transition: all 0.3s ease;

    box-shadow:
        0px 8px 20px rgba(37,99,235,0.35);
}

div.stButton > button:hover {

    transform: translateY(-3px);

    background: linear-gradient(
        135deg,
        #3B82F6,
        #2563EB
    );

    box-shadow:
        0px 10px 24px rgba(37,99,235,0.45);
}

/* =========================================
INPUT FIELDS
========================================= */

textarea,
input {

    background-color: #111827 !important;

    border: 1px solid #1E293B !important;

    border-radius: 12px !important;

    color: #F8FAFC !important;

    font-size: 15px !important;
}

/* =========================================
TEXT AREAS
========================================= */

textarea {

    line-height: 1.7 !important;
}

/* =========================================
TABS
========================================= */

button[data-baseweb="tab"] {

    font-size: 16px;

    font-weight: 600;

    border-radius: 12px;

    padding: 10px 18px;

    transition: 0.2s ease;
}

button[data-baseweb="tab"]:hover {

    background-color: #111827;
}

/* =========================================
CONTAINERS
========================================= */

[data-testid="stVerticalBlockBorderWrapper"] {

    border-radius: 18px;

    border: 1px solid #1E293B;

    background-color: #0F172A;

    padding: 12px;
}

/* =========================================
CODE BLOCKS
========================================= */

pre {

    border-radius: 14px !important;

    padding: 16px !important;

    background-color: #111827 !important;

    border: 1px solid #1E293B !important;

    font-size: 14px !important;
}

/* =========================================
SUCCESS / WARNING / INFO
========================================= */

div[data-baseweb="notification"] {

    border-radius: 14px;
}

/* =========================================
SCROLLBAR
========================================= */

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #0F172A;
}

::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 12px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD VARIABLES
# -----------------------------

# Load local .env file

load_dotenv()

# =========================================
# OPENROUTER API KEY
# =========================================

try:

    OPENROUTER_API_KEY = st.secrets[
        "OPENROUTER_API_KEY"
    ]

except Exception:

    OPENROUTER_API_KEY = os.getenv(
        "OPENROUTER_API_KEY"
    )

# =========================================
# GROQ API KEY
# =========================================

try:

    GROQ_API_KEY = st.secrets[
        "GROQ_API_KEY"
    ]

except Exception:

    GROQ_API_KEY = os.getenv(
        "GROQ_API_KEY"
    )


# =========================================
# GROQ AI REQUEST
# =========================================

def analyze_with_groq(prompt):

    try:

        client = Groq(
            api_key=GROQ_API_KEY
        )

        completion = client.chat.completions.create(

            model="llama-3.3-70b-versatile",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            temperature=0.2,

            max_tokens=4000
        )

        return completion.choices[0].message.content

    except Exception as e:

        print("Groq failed:", e)

        return None


# =========================================
# OPENROUTER FALLBACK
# =========================================

def analyze_with_openrouter(prompt):

    try:

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        data = {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data
        )

        result = response.json()

        if "choices" in result:

            return result["choices"][0]["message"]["content"]

        return None

    except Exception as e:

        print("OpenRouter failed:", e)

        return None


# =========================================
# MASTER AI FUNCTION
# =========================================

def generate_ai_response(prompt):

    # PRIMARY → GROQ

    result = analyze_with_groq(prompt)

    if result:

        return result

    print("Groq unavailable. Switching to fallback.")

    # FALLBACK → OPENROUTER

    result = analyze_with_openrouter(prompt)

    if result:

        return result

    return None


# -----------------------------
# HERO SECTION
# -----------------------------

with st.container(border=True):

    st.markdown(
        """
<div style="text-align:center; padding:50px 20px;">

<h1 style="font-size:68px; margin-bottom:20px;">
🚀 AI Career Intelligence Agent
</h1>

<p style="font-size:26px; color:#A1A1AA; margin-bottom:0px;">
Professional resume optimization, ATS analysis,
and interview preparation powered by AI.
</p>

</div>
""",
        unsafe_allow_html=True
    )

# ------------------------------------------------
# PLATFORM VALUE SECTION
# ------------------------------------------------

col1, col2, col3 = st.columns(3)

# ================================================
# CARD 1
# ================================================

with col1:

    with st.container(border=True):

        st.markdown("### 📄 ATS Optimization")

        st.caption(
            "Detects missing keywords, weak positioning, and recruiter visibility issues."
        )

        st.success("Keyword Match Analysis")

        st.info("ATS-Friendly Resume Structuring")

# ================================================
# CARD 2
# ================================================

with col2:

    with st.container(border=True):

        st.markdown("### ✍ Resume Tailoring")

        st.caption(
            "Generates recruiter-ready resume rewrites tailored to the target role."
        )

        st.success("Professional Summary Rewrites")

        st.info("Achievement-Focused Bullet Optimization")

# ================================================
# CARD 3
# ================================================

with col3:

    with st.container(border=True):

        st.markdown("### 🎤 Interview Q & A")

        st.caption(
            "Creates realistic interview preparation based on job requirements."
        )

        st.success("Role-Specific Interview Questions")

        st.info("Professional Sample Answers")



st.markdown("")

st.markdown("---")




# -----------------------------
# USER INPUTS
# -----------------------------

with st.container(border=True):

    st.subheader("📥 Candidate Information")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("👤 Full Name")

    with col2:
        target_role = st.text_input("🎯 Target Role")

    job_description = st.text_area(
        "📄 Paste Job Description",
        height=250,
        placeholder="Paste the full job description here..."
    )

    uploaded_file = st.file_uploader(
        "📎 Upload Resume (PDF)",
        type=["pdf"]
    )

# -----------------------------
# CENTERED ANALYZE BUTTON
# -----------------------------

left_space, center_col = st.columns([2, 3])

with center_col:

    analyze_clicked = st.button(
        "🚀 Analyze Resume"
    )

# -----------------------------
# ANALYSIS LOGIC
# -----------------------------

if analyze_clicked:

    # -----------------------------
    # VALIDATION
    # -----------------------------

    if uploaded_file is None:

        st.warning("Please upload your resume.")

    elif not job_description:

        st.warning("Please paste the job description.")

    else:

        with st.spinner("Analyzing resume with AI..."):

            # -----------------------------
            # PDF EXTRACTION
            # -----------------------------

            extracted_text = ""

            try:

                with pdfplumber.open(uploaded_file) as pdf:

                    for page in pdf.pages:

                        text = page.extract_text()

                        if text:

                            extracted_text += text + "\n"

            except Exception as e:

                st.error("Failed to extract PDF text.")

                st.write(e)

            prompt = f"""
You are an elite recruiter, ATS evaluator,
career strategist, executive resume writer,
and interview coach.

Your task is to deeply analyze and professionally
tailor the candidate resume for the target role.

You must produce HIGHLY PERSONALIZED,
PRACTICAL, and RECRUITER-LEVEL output.

Your response must feel like:
- a professional resume consultant
- a hiring manager
- a recruiter
- an ATS optimization expert

NOT like a generic AI chatbot.

--------------------------------------------------
IMPORTANT RULES
--------------------------------------------------

1. Be highly specific.

2. Mention exact candidate experiences,
internships, projects, tools, responsibilities,
and achievements from the resume.

3. Avoid generic career advice.

4. Tailor ALL recommendations to:
- the target role
- job responsibilities
- required skills
- seniority level

5. Simulate realistic recruiter thinking.

6. The rewritten sections MUST feel:
- professionally written
- ATS optimized
- recruiter friendly
- modern
- polished
- immediately usable

7. The rewritten content must be realistic
for the candidate experience level.

8. DO NOT exaggerate experience.

9. DO NOT invent fake achievements.

10. The output must preserve the ORIGINAL
resume structure.

--------------------------------------------------
STRUCTURE PRESERVATION RULES
--------------------------------------------------

PROFESSIONAL SUMMARY:
- Preserve the SAME number of sentences
as the original summary.
- Rewrite professionally for the target role.
- Improve ATS optimization.
- Improve recruiter positioning.
- Make the summary stronger and more modern.
- Keep it realistic and natural.

WORK EXPERIENCE:
- Preserve the SAME number of work experiences.
- Preserve the SAME number of bullet points
under each role.
- Rewrite EVERY bullet professionally.
- Use stronger action verbs.
- Improve business language.
- Improve financial/operational positioning.
- Make bullets achievement-oriented.
- Make bullets recruiter-friendly.
- Tailor bullets to the job description.
- Keep all rewritten bullets realistic.

PROJECTS SECTION:
- DO NOT rewrite project titles.
- Preserve ALL original project titles exactly.
- Preserve the SAME number of bullets
under each project.
- ONLY rewrite project bullets.
- Rewrite bullets professionally.
- Make projects sound commercially valuable.
- Highlight analytical, operational,
financial, or business impact.

SKILLS SECTION:
- Suggest missing ATS keywords.
- Suggest important skills to prioritize.
- Suggest skills to remove if weak/redundant.
- Focus on relevance for the target role.

INTERVIEW QUESTIONS:
Generate highly realistic interview questions:
- technical
- behavioral
- operational
- scenario-based
- HR questions

Questions must match:
- candidate experience level
- target role
- industry expectations

SAMPLE ANSWERS:
Provide realistic, professional,
and well-structured answers.

Answers must:
- sound natural
- avoid robotic wording
- avoid exaggerated claims
- align with candidate background
- sound interview-ready

--------------------------------------------------
CANDIDATE NAME:
{name}

TARGET ROLE:
{target_role}

--------------------------------------------------
CANDIDATE RESUME:
{extracted_text}

--------------------------------------------------
JOB DESCRIPTION:
{job_description}

--------------------------------------------------

Return ONLY VALID JSON.

DO NOT include markdown.
DO NOT include explanations outside JSON.
DO NOT include ```json formatting.

Use this EXACT JSON structure:

{{
  "match_score": "0-100",

  "ats_score": "0-100",

  "interview_readiness": "0-100",

  "strengths": [
    "",
    ""
  ],

  "weaknesses": [
    "",
    ""
  ],

  "recruiter_insights": [
    "",
    ""
  ],

  "missing_keywords": [
    "",
    ""
  ],

  "ats_keywords_to_add": [
    "",
    ""
  ],

  "professional_summary": {{
    "original_summary_analysis": "",
    "rewritten_version": ""
  }},

  "work_experience_rewrites": [
    {{
      "job_title": "",

      "original_bullets": [
        ""
      ],

      "rewritten_bullets": [
        ""
      ]
    }}
  ],

  "project_rewrites": [
    {{
      "project_title": "",

      "original_bullets": [
        ""
      ],

      "rewritten_bullets": [
        ""
      ]
    }}
  ],

  "skills_optimization": {{
    "add": [
      ""
    ],

    "remove": [
      ""
    ],

    "prioritize": [
      ""
    ]
  }},

  "interview_questions": {{
    "technical": [
      ""
    ],

    "behavioral": [
      ""
    ],

    "scenario_based": [
      ""
    ],

    "hr_questions": [
      ""
    ]
  }},

  "sample_answers": [
    {{
      "question": "",

      "answer": ""
    }}
  ],

  "final_assessment": {{
    "shortlist_decision": "",

    "concerns": [
      ""
    ],

    "recommendations": [
      ""
    ]
  }}
}}
"""

            # -----------------------------
            # AI REQUEST
            # -----------------------------

            try:

                ai_output = generate_ai_response(prompt)

                if not ai_output:

                    st.error(
                        "All AI providers are temporarily unavailable."
                    )

                else:

                    # -----------------------------
                    # CLEAN JSON
                    # -----------------------------

                    cleaned_output = re.sub(
                        r"```json|```",
                        "",
                        ai_output
                    ).strip()

                    parsed = json.loads(cleaned_output)

                    # -----------------------------
                    # SUCCESS MESSAGE
                    # -----------------------------

                    st.success(
                        "Your CV is optimized"
                    )

                    # -----------------------------
                    # METRICS DASHBOARD
                    # -----------------------------

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "🎯 Match Score",
                        f"{parsed['match_score']}%"
                    )

                    col2.metric(
                        "📄 ATS Score",
                        f"{parsed['ats_score']}%"
                    )

                    col3.metric(
                        "🎤 Interview Readiness",
                        f"{parsed['interview_readiness']}%"
                    )
                    st.markdown("---")


                    st.subheader("📈 Recruiter Confidence Level")

                    confidence = int(parsed["match_score"]) / 100

                    st.progress(confidence)

                    if confidence >= 0.75:
                        st.success("High recruiter confidence")

                    elif confidence >= 0.5:
                        st.warning("Moderate recruiter confidence")

                    else:
                        st.error("Low recruiter confidence")


                    # -----------------------------
                    # TABS
                    # -----------------------------

                    tab1, tab2, tab3, tab4 = st.tabs([
                        "📊 Match Analysis",
                        "✍ Resume Rewrite",
                        "🎯 ATS Optimization",
                        "🎤 Interview Prep"
                    ])

                    # ==================================================
                    # TAB 1 — MATCH ANALYSIS
                    # ==================================================

                    with tab1:

                        # ------------------------------------------
                        # TOP RECRUITER RISKS
                        # ------------------------------------------

                        with st.container(border=True):

                            st.markdown("## 🚨 Top Recruiter Risks")

                            st.caption(
                                "These are the biggest concerns recruiters may notice immediately."
                            )

                            for item in parsed["weaknesses"]:
                                st.error(item)

                        st.markdown("")

                        # ------------------------------------------
                        # STRENGTHS
                        # ------------------------------------------

                        with st.container(border=True):

                            st.markdown("## ✅ Candidate Strengths")

                            st.caption(
                                "Areas where the profile aligns well with the target role."
                            )

                            for item in parsed["strengths"]:
                                st.success(item)

                        st.markdown("")

                        # ------------------------------------------
                        # RECRUITER INSIGHTS
                        # ------------------------------------------

                        with st.container(border=True):

                            st.markdown("## 🧠 Recruiter Insights")

                            st.caption(
                                "How hiring managers and recruiters may interpret this profile."
                            )

                            for item in parsed["recruiter_insights"]:
                                st.info(item)

                        st.markdown("")

                        # ------------------------------------------
                        # FINAL HIRING ASSESSMENT
                        # ------------------------------------------

                        with st.container(border=True):

                            st.markdown("## 🏁 Final Hiring Assessment")

                            decision = parsed["final_assessment"]["shortlist_decision"]

                            if decision.lower() == "yes":

                                st.success(
                                    f"✅ Recruiter Shortlist Decision: {decision}"
                                )

                            elif decision.lower() == "maybe":

                                st.warning(
                                    f"⚠ Recruiter Shortlist Decision: {decision}"
                                )

                            else:

                                st.error(
                                    f"❌ Recruiter Shortlist Decision: {decision}"
                                )

                            st.markdown("")

                            st.markdown("### 🚨 Recruiter Concerns")

                            for item in parsed["final_assessment"]["concerns"]:
                                st.warning(item)

                            st.markdown("")

                            st.markdown("### 📈 Improvement Recommendations")

                            for item in parsed["final_assessment"]["recommendations"]:
                                st.success(item)



                    # ==================================================
                    # TAB 2 — RESUME REWRITE
                    # ==================================================

                    with tab2:

                        st.subheader("✍ Resume Tailoring Studio")

                        # ------------------------------------------------
                        # PROFESSIONAL SUMMARY
                        # ------------------------------------------------

                        with st.container(border=True):

                            st.markdown("## 📝 Optimized Professional Summary")

                            st.text_area(
                                "📋 Copy-Paste Ready Summary",
                                parsed["professional_summary"]["rewritten_version"],
                                height=160
                            )

                        st.markdown("")

                        # ------------------------------------------------
                        # WORK EXPERIENCE
                        # ------------------------------------------------

                        with st.container(border=True):

                            st.markdown("## 💼 Optimized Work Experience")

                            for item in parsed["work_experience_rewrites"]:

                                st.markdown(
                                    f"### 💼 {item['job_title']}"
                                )

                                # SIDE-BY-SIDE LAYOUT
                                col1, col2 = st.columns(2)

                                # ----------------------------------------
                                # ORIGINAL BULLETS
                                # ----------------------------------------

                                with col1:

                                    st.markdown("#### Original")

                                    for bullet in item["original_bullets"]:

                                        st.markdown(
                                            f"""
                    <div style="
                    padding:10px;
                    border-radius:10px;
                    background-color:#1E1E1E;
                    margin-bottom:8px;
                    font-size:14px;
                    ">
                    • {bullet}
                    </div>
                    """,
                                            unsafe_allow_html=True
                                        )

                                # ----------------------------------------
                                # OPTIMIZED BULLETS
                                # ----------------------------------------

                                with col2:

                                    st.markdown("#### Optimized")

                                    for bullet in item["rewritten_bullets"]:

                                        st.markdown(
                                            f"""
                    <div style="
                    padding:10px;
                    border-radius:10px;
                    background-color:#0F2E1D;
                    border-left:4px solid #22C55E;
                    margin-bottom:8px;
                    font-size:14px;
                    ">
                    • {bullet}
                    </div>
                    """,
                                            unsafe_allow_html=True
                                        )

                                # ----------------------------------------
                                # COPY SECTION
                                # ----------------------------------------

                                optimized_text = "\n".join(
                                    [f"• {b}" for b in item["rewritten_bullets"]]
                                )

                                st.text_area(
                                    "📋 Copy Optimized Experience",
                                    optimized_text,
                                    height=160
                                )

                                st.markdown("---")

                        st.markdown("")

                        # ------------------------------------------------
                        # PROJECTS
                        # ------------------------------------------------

                        with st.container(border=True):

                            st.markdown("## 📚 Optimized Projects")

                            for item in parsed["project_rewrites"]:

                                st.markdown(
                                    f"### 📚 {item['project_title']}"
                                )

                                # SIDE-BY-SIDE LAYOUT
                                col1, col2 = st.columns(2)

                                # ----------------------------------------
                                # ORIGINAL PROJECT BULLETS
                                # ----------------------------------------

                                with col1:

                                    st.markdown("#### Original")

                                    for bullet in item["original_bullets"]:

                                        st.markdown(
                                            f"""
                    <div style="
                    padding:10px;
                    border-radius:10px;
                    background-color:#1E1E1E;
                    margin-bottom:8px;
                    font-size:14px;
                    ">
                    • {bullet}
                    </div>
                    """,
                                            unsafe_allow_html=True
                                        )

                                # ----------------------------------------
                                # OPTIMIZED PROJECT BULLETS
                                # ----------------------------------------

                                with col2:

                                    st.markdown("#### Optimized")

                                    for bullet in item["rewritten_bullets"]:

                                        st.markdown(
                                            f"""
                    <div style="
                    padding:10px;
                    border-radius:10px;
                    background-color:#0F2E1D;
                    border-left:4px solid #22C55E;
                    margin-bottom:8px;
                    font-size:14px;
                    ">
                    • {bullet}
                    </div>
                    """,
                                            unsafe_allow_html=True
                                        )

                                # ----------------------------------------
                                # COPY SECTION
                                # ----------------------------------------

                                optimized_project = "\n".join(
                                    [f"• {b}" for b in item["rewritten_bullets"]]
                                )

                                st.text_area(
                                    "📋 Copy Optimized Project",
                                    optimized_project,
                                    height=160
                                )

                                st.markdown("---")

                    # ==================================================
                    # TAB 3 — ATS OPTIMIZATION
                    # ==================================================

                    with tab3:

                        st.subheader("🔑 Missing ATS Keywords")

                        st.code(
                            "\n".join(parsed["missing_keywords"])
                        )

                        st.subheader("➕ Keywords To Add")

                        st.code(
                            "\n".join(parsed["ats_keywords_to_add"])
                        )

                        st.subheader("🛠 Skills To Add")

                        st.success(
                            "\n".join(parsed["skills_optimization"]["add"])
                        )

                        st.subheader("❌ Skills To Remove")

                        st.warning(
                            "\n".join(parsed["skills_optimization"]["remove"])
                        )

                        st.subheader("⭐ Skills To Prioritize")

                        st.info(
                            "\n".join(parsed["skills_optimization"]["prioritize"])
                        )

                    # ==================================================
                    # TAB 4 — INTERVIEW PREP
                    # ==================================================

                    with tab4:

                        categories = parsed["interview_questions"]

                        st.subheader("🎤 Personalized Interview Preparation")

                        st.markdown("""
                    Prepare for realistic recruiter and hiring manager questions
                    tailored to your target role and experience level.
                    """)

                        # ---------------------------------------------
                        # TECHNICAL QUESTIONS
                        # ---------------------------------------------

                        with st.container(border=True):

                            st.subheader("🧠 Technical Questions")

                            for q in categories["technical"]:

                                with st.expander(q):

                                    st.info(
                                        "Focus on structured, role-specific answers with practical examples."
                                    )

                                    st.write(
                                        "Use operational, analytical, or finance-related examples where possible."
                                    )

                        # ---------------------------------------------
                        # BEHAVIORAL QUESTIONS
                        # ---------------------------------------------

                        with st.container(border=True):

                            st.subheader("🤝 Behavioral Questions")

                            for q in categories["behavioral"]:

                                with st.expander(q):

                                    st.info(
                                        "Use the STAR method: Situation, Task, Action, Result."
                                    )

                                    st.write(
                                        "Emphasize communication, ownership, teamwork, and decision-making."
                                    )

                        # ---------------------------------------------
                        # SCENARIO-BASED QUESTIONS
                        # ---------------------------------------------

                        with st.container(border=True):

                            st.subheader("🎯 Scenario-Based Questions")

                            for q in categories["scenario_based"]:

                                with st.expander(q):

                                    st.info(
                                        "Interviewers want to evaluate your judgment and business thinking."
                                    )

                                    st.write(
                                        "Explain your thought process step-by-step before giving the final action."
                                    )

                        # ---------------------------------------------
                        # HR QUESTIONS
                        # ---------------------------------------------

                        with st.container(border=True):

                            st.subheader("💬 HR Questions")

                            for q in categories["hr_questions"]:

                                with st.expander(q):

                                    st.info(
                                        "Keep answers professional, concise, and aligned with your career goals."
                                    )

                                    st.write(
                                        "Avoid overly generic answers and connect your response to the target role."
                                    )

                        # ---------------------------------------------
                        # SAMPLE ANSWERS
                        # ---------------------------------------------

                        with st.container(border=True):

                            st.subheader("🎤 Sample High-Quality Answers")

                            for item in parsed["sample_answers"]:

                                with st.expander(item["question"]):

                                    st.success("Recommended Answer")

                                    st.text_area(
                                        "Copy & Practice",
                                        item["answer"],
                                        height=220
                                    )

                    # -----------------------------
                    # PREMIUM FOOTER
                    # -----------------------------

                    st.markdown("---")

                    with st.container():

                        st.markdown(
                            """
                    <div style="
                    text-align:center;
                    padding: 10px 0px 30px 0px;
                    ">

                    <h3 style="
                    margin-bottom:8px;
                    color:#F8FAFC;
                    font-weight:700;
                    ">
                    🚀 AI Career Intelligence Agent
                    </h3>

                    <p style="
                    font-size:16px;
                    color:#94A3B8;
                    margin-bottom:18px;
                    ">
                    Recruiter-level resume optimization and interview preparation powered by AI.
                    </p>

                    <div style="
                    display:flex;
                    justify-content:center;
                    gap:12px;
                    flex-wrap:wrap;
                    margin-bottom:18px;
                    ">

                    <span style="
                    background:#111827;
                    padding:8px 14px;
                    border-radius:999px;
                    font-size:14px;
                    border:1px solid #1E293B;
                    ">
                    ATS Optimization
                    </span>

                    <span style="
                    background:#111827;
                    padding:8px 14px;
                    border-radius:999px;
                    font-size:14px;
                    border:1px solid #1E293B;
                    ">
                    Resume Tailoring
                    </span>

                    <span style="
                    background:#111827;
                    padding:8px 14px;
                    border-radius:999px;
                    font-size:14px;
                    border:1px solid #1E293B;
                    ">
                    Interview Preparation
                    </span>

                    <span style="
                    background:#111827;
                    padding:8px 14px;
                    border-radius:999px;
                    font-size:14px;
                    border:1px solid #1E293B;
                    ">
                    Recruiter Insights
                    </span>

                    </div>

                    <p style="
                    font-size:13px;
                    color:#64748B;
                    ">
                    Designed for professionals who want stronger positioning in competitive hiring markets.
                    </p>

                    </div>
                    """,
                            unsafe_allow_html=True
                        )

            except Exception as e:

                st.error("Something went wrong.")

                st.write(e)