
from hashlib import sha1
from html import escape

import streamlit as st

from utils.parser import MAX_SIZE_MB, extract_text
from utils.screening import (
    EXPERIENCE_WEIGHT,
    SKILL_WEIGHT,
    TEXT_WEIGHT,
    screen_resume,
)
from utils.recommendations import generate_recommendations


st.set_page_config(
    page_title="Resume Studio",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# --------------------------------------------------
# CUSTOM STYLING
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');

    :root { color-scheme: dark; }

    .stApp {
        background: #090b0d;
        color: #edf0ee;
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }

    .block-container {
        max-width: 1100px;
        padding: 3.2rem 2rem 4rem;
    }

    .eyebrow {
        color: #9ac8ac;
        font: 500 11px 'DM Mono', monospace;
        letter-spacing: .17em;
        text-transform: uppercase;
    }

    h1 {
        letter-spacing: -.055em !important;
        font-size: clamp(2.5rem, 6vw, 4.3rem) !important;
        line-height: 1.02 !important;
        margin: .45rem 0 .8rem !important;
    }

    .intro {
        color: #989f9b;
        font-size: 1.04rem;
        max-width: 590px;
        line-height: 1.65;
    }

    .panel {
        background: #101315;
        border: 1px solid #242a27;
        border-radius: 14px;
        padding: 1.35rem 1.45rem;
    }

    .panel-title {
        color: #e9edea;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: .25rem;
    }

    .panel-sub {
        color: #868e89;
        font-size: .88rem;
        margin-bottom: 1rem;
    }

    .metric-label {
        color: #919994;
        font: 500 10px 'DM Mono', monospace;
        text-transform: uppercase;
        letter-spacing: .13em;
    }

    .metric-value {
        color: #eaf2ec;
        font-size: 2rem;
        font-weight: 600;
        letter-spacing: -.05em;
        margin-top: .15rem;
    }

    .chip {
        display: inline-block;
        padding: .38rem .65rem;
        margin: .25rem .28rem .15rem 0;
        border: 1px solid #33483c;
        border-radius: 999px;
        background: #142019;
        color: #b8e4c6;
        font-size: .82rem;
    }

    .chip-missing {
        border-color: #55423a;
        background: #211815;
        color: #edb79e;
    }

    .recommendation {
        color: #c6ccc8;
        border-left: 2px solid #86c39b;
        padding: .12rem 0 .12rem .85rem;
        margin: .8rem 0;
        line-height: 1.55;
    }

    .stButton > button {
        background: #a8e2b8;
        color: #0b140e;
        border: 0;
        border-radius: 8px;
        font-weight: 700;
        min-height: 2.8rem;
    }

    .stButton > button:hover {
        background: #c4f0d0;
        color: #0b140e;
    }

    [data-testid="stFileUploader"] section {
        background: #0c0f10;
        border: 1px dashed #39433d;
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: #8fcea2;
    }

    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"] input {
        background: #0c0f10;
        border-color: #333b36;
    }

    hr {
        border-color: #242a27;
        margin: 2rem 0;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def render_chips(items, missing=False):

    if not items:
        st.caption("None identified")
        return

    cls = "chip chip-missing" if missing else "chip"

    st.markdown(
        "".join(
            f'<span class="{cls}">{escape(str(item))}</span>'
            for item in items
        ),
        unsafe_allow_html=True,
    )


def result_card(label, value, detail=""):

    st.markdown(
        f'<div class="panel">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="panel-sub" style="margin: .15rem 0 0">'
        f'{detail}</div></div>',
        unsafe_allow_html=True,
    )


def render_score_component(label, score, contribution, weight):

    st.markdown(f"**{label}**")

    st.progress(
        max(0.0, min(float(score) / 100, 1.0))
    )

    st.caption(
        f"{score:.1f}% score · "
        f"{contribution:.2f} / {weight} points"
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="eyebrow">✳ &nbsp; CAREER TOOLKIT / 01</div>',
    unsafe_allow_html=True,
)

st.title("Resume Studio")

st.markdown(
    '<div class="intro">'
    'A clearer read on how your experience lines up. '
    'Upload a resume, add a role description, and get '
    'a focused skills and fit review.'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")


# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

left, right = st.columns([0.88, 1.12], gap="large")

with left:

    st.markdown(
        '<div class="panel-title">01 / Your resume</div>'
        '<div class="panel-sub">'
        'PDF, DOCX, or TXT · up to 10 MB'
        '</div>',
        unsafe_allow_html=True,
    )

    resume_file = st.file_uploader(
        "Drop your resume here",
        type=["pdf", "docx", "txt"],
        label_visibility="collapsed",
    )

    use_ocr = st.checkbox(
        "Try OCR for scanned PDFs",
        value=False,
        help="OCR requires Tesseract to be installed on the server.",
    )

    if resume_file:

        st.caption(
            f"Ready · {resume_file.name} · "
            f"{resume_file.size / 1024:.0f} KB"
        )


with right:

    st.markdown(
        '<div class="panel-title">02 / Target role</div>'
        '<div class="panel-sub">'
        'Paste the job description for a tailored comparison'
        '</div>',
        unsafe_allow_html=True,
    )

    job_description = st.text_area(
        "Job description",
        height=226,
        label_visibility="collapsed",
        placeholder=(
            "Paste the job description here… "
            "Include the responsibilities and skills "
            "that matter most."
        ),
    )


# Clear stale results whenever either analysis input changes.
resume_signature = (
    sha1(resume_file.getvalue()).hexdigest()
    if resume_file is not None
    else ""
)
job_signature = sha1(job_description.encode("utf-8")).hexdigest()
input_signature = (resume_signature, job_signature)

if st.session_state.get("screening_input_signature") != input_signature:
    st.session_state.pop("screening_results", None)
    st.session_state.pop("resume_filename", None)
    st.session_state["screening_input_signature"] = input_signature


# --------------------------------------------------
# ANALYZE RESUME
# --------------------------------------------------

analyze = st.button(
    "Analyze resume  →",
    use_container_width=True,
    disabled=not (
        resume_file and job_description.strip()
    ),
)


if analyze:

    parsed = extract_text(
        resume_file,
        filename=resume_file.name,
        ocr=use_ocr,
    )

    if parsed["status"] != "ok":

        st.error(parsed["message"])

    else:

        resume_text = parsed["text"]

        try:

            with st.spinner(
                "Reading your resume against the role…"
            ):

                # Central hybrid screening pipeline
                results = screen_resume(
                    resume_text,
                    job_description,
                )

            st.session_state["screening_results"] = results

            st.session_state["resume_filename"] = (
                resume_file.name
            )

        except Exception as exc:

            st.error(
                f"Resume analysis failed: {exc}"
            )


# --------------------------------------------------
# DISPLAY RESULTS
# --------------------------------------------------

if "screening_results" in st.session_state:

    data = st.session_state["screening_results"]

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(
        '<div class="eyebrow">YOUR REVIEW</div>',
        unsafe_allow_html=True,
    )

    st.subheader(
        f"Analysis · "
        f"{st.session_state.get('resume_filename', 'Resume')}"
    )

    # Main result cards
    a, b, c = st.columns(3, gap="medium")

    with a:

        result_card(
            "Resume / role fit",
            f"{data['match_score']:.0f}%",
            "Hybrid score · skills, text and experience",
        )

    with b:

        result_card(
            "Skills matched",
            str(len(data["matched_skills"])),
            "Identified in both documents",
        )

    with c:

        if data["predicted_role"] == "Unavailable":
            role_detail = (
                data.get("model_error")
                or "Train the role model to enable this signal"
            )
        elif data.get("confidence_source") == "profile":
            role_detail = (
                f"{data['model_confidence']:.0f}% profile match"
            )
        else:
            role_detail = (
                f"{data['model_confidence']:.0f}% model confidence"
            )

        result_card(
            "Predicted profile",
            data["predicted_role"],
            role_detail,
        )

    st.write("")

    # Hybrid score breakdown
    st.markdown(
        '<div class="panel-title">Score breakdown</div>'
        '<div class="panel-sub">'
        'How the overall match score was calculated'
        '</div>',
        unsafe_allow_html=True,
    )

    breakdown = data["score_breakdown"]

    x, y, z = st.columns(3, gap="medium")

    with x:

        render_score_component(
            f"Technical skills · {SKILL_WEIGHT:.0%}",
            data["skill_coverage"],
            breakdown["skill_score"],
            SKILL_WEIGHT * 100,
        )

    with y:

        render_score_component(
            f"Text similarity · {TEXT_WEIGHT:.0%}",
            data["text_similarity"],
            breakdown["text_score"],
            TEXT_WEIGHT * 100,
        )

    with z:

        render_score_component(
            f"Experience relevance · {EXPERIENCE_WEIGHT:.0%}",
            data["experience_relevance"],
            breakdown["experience_score"],
            EXPERIENCE_WEIGHT * 100,
        )

    st.caption(
        "The component percentages measure different signals. "
        "Their weighted contributions add up to the overall score."
    )

    st.write("")

    # Skills section
    skill_col, gaps_col = st.columns(2, gap="large")

    with skill_col:

        st.markdown(
            '<div class="panel-title">Skills already showing</div>',
            unsafe_allow_html=True,
        )

        render_chips(data["matched_skills"])

    with gaps_col:

        st.markdown(
            '<div class="panel-title">Skills to consider</div>',
            unsafe_allow_html=True,
        )

        render_chips(
            data["missing_skills"],
            missing=True,
        )

    st.write("")

    # Recommendations
    with st.expander(
        "Personalized next steps",
        expanded=True,
    ):

        for suggestion in generate_recommendations(data):

            st.markdown(
                f'<div class="recommendation">'
                f'{escape(suggestion)}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.caption(
        "Use this review as a guide. Similarity and role "
        "classification are model signals, not a hiring decision."
    )


# --------------------------------------------------
# EMPTY STATE
# --------------------------------------------------

if (
    not (resume_file and job_description.strip())
    and "screening_results" not in st.session_state
):

    st.markdown("<hr>", unsafe_allow_html=True)

    st.caption(
        "Your analysis stays in this session. "
        "Add both inputs above to get started."
    )
