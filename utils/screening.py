
"""Combines skill, text, experience, and role signals into a resume fit score."""


from utils.model import (
    load_model,
    predict_role,
    calculate_text_similarity
)

from utils.skills import analyze_skills, extract_skills


# --------------------------------------------------
# HYBRID SCORING WEIGHTS
# --------------------------------------------------

SKILL_WEIGHT = 0.75
TEXT_WEIGHT = 0.15
EXPERIENCE_WEIGHT = 0.10


# --------------------------------------------------
# SECTION EXTRACTION
# --------------------------------------------------

def normalize_heading(line):

    # Normalized headings make matching independent of capitalization,
    # Markdown markers, and trailing colons.
    heading = line.strip().lower()

    heading = heading.lstrip("#").strip()
    heading = heading.strip("*").strip()
    heading = heading.rstrip(":").strip()

    return heading


def extract_relevant_sections(resume_text):

    # Only experience and project sections are used for experience relevance;
    # skill, education, and summary sections can add noisy repeated terms.
    relevant_sections = []

    current_section = None

    experience_headings = {
        "experience",
        "work experience",
        "professional experience",
        "employment history",
        "work history",
        "employment",
        "internship",
        "internships",
        "career history",
        "project experience"
    }

    project_headings = {
        "projects",
        "academic projects",
        "personal projects",
        "technical projects",
        "project experience"
    }

    other_headings = {
        "education",
        "technical skills",
        "skills",
        "profile",
        "summary",
        "additional",
        "certifications"
    }

    for line in resume_text.splitlines():

        heading = normalize_heading(line)

        if heading in experience_headings:

            current_section = "experience"
            continue

        elif heading in project_headings:

            current_section = "projects"
            continue

        elif heading in other_headings:

            current_section = None
            continue

        if current_section and line.strip():

            relevant_sections.append(line.strip())

    return " ".join(relevant_sections)


# --------------------------------------------------
# JOB RESPONSIBILITY EXTRACTION
# --------------------------------------------------

def extract_responsibilities(job_description):

    # If responsibility headings are present, only that section is compared.
    # Otherwise, the fallback skips explicit requirements and qualifications.
    lines = job_description.splitlines()

    responsibility_headings = {
        "responsibilities",
        "key responsibilities",
        "what you'll do",
        "what you will do",
        "duties",
        "role responsibilities"
    }

    stopping_headings = {
        "required skills",
        "technical skills",
        "requirements",
        "qualifications",
        "preferred qualifications",
        "about you",
        "benefits",
        "what we offer",
        "education"
    }

    responsibility_lines = []

    inside_responsibilities = False

    for line in lines:

        heading = normalize_heading(line)

        if heading in responsibility_headings:

            inside_responsibilities = True
            continue

        if heading in stopping_headings:

            inside_responsibilities = False
            continue

        if inside_responsibilities and line.strip():

            responsibility_lines.append(line.strip())

    # Fallback if no responsibility heading is found
    if not responsibility_lines:
        fallback_lines = []
        inside_stopping_section = False

        for line in lines:
            heading = normalize_heading(line)

            if heading in stopping_headings:
                inside_stopping_section = True
                continue

            if heading in responsibility_headings:
                inside_stopping_section = False
                continue

            if (
                not inside_stopping_section
                and line.strip()
                and ":" not in line
            ):
                fallback_lines.append(line.strip())

        if fallback_lines:
            return " ".join(fallback_lines)

        return job_description

    return " ".join(responsibility_lines)


# --------------------------------------------------
# SKILL COVERAGE
# --------------------------------------------------

def calculate_skill_coverage(
    matched_skills,
    missing_skills
):

    # Skill coverage is the percentage of recognized JD skills found in the
    # resume. It is zero when the job description has no recognized skills.
    total_required = (
        len(matched_skills) + len(missing_skills)
    )

    if total_required == 0:

        return 0.0

    coverage = (
        len(matched_skills) / total_required
    ) * 100

    return round(coverage, 2)


# --------------------------------------------------
# EXPERIENCE RELEVANCE
# --------------------------------------------------

def calculate_experience_relevance(
    resume_text,
    job_description
):

    # This uses two signals: wording overlap and demonstrated required skills.
    # Extract experience and project descriptions
    relevant_text = extract_relevant_sections(
        resume_text
    )

    if not relevant_text.strip():

        return {
            "score": 0.0,
            "responsibility_similarity": 0.0,
            "demonstrated_skill_coverage": 0.0
        }

    # Extract responsibilities from job description
    responsibilities = extract_responsibilities(
        job_description
    )

    # Signal 1: Responsibility similarity
    responsibility_similarity = calculate_text_similarity(
        relevant_text,
        responsibilities
    )

    # Signal 2: Demonstrated technical skills
    required_skills = extract_skills(
        job_description
    )

    demonstrated_skills = extract_skills(
        relevant_text
    )

    if required_skills:

        demonstrated_skill_coverage = (
            len(
                required_skills.intersection(
                    demonstrated_skills
                )
            )
            / len(required_skills)
        ) * 100

    else:

        demonstrated_skill_coverage = (
            responsibility_similarity
        )

    # Combine both experience signals
    experience_score = (
        0.60 * responsibility_similarity
        + 0.40 * demonstrated_skill_coverage
    )

    return {
        "score": round(experience_score, 2),

        "responsibility_similarity": round(
            responsibility_similarity, 2
        ),

        "demonstrated_skill_coverage": round(
            demonstrated_skill_coverage, 2
        )
    }


# --------------------------------------------------
# HYBRID SCORE CALCULATION
# --------------------------------------------------

def calculate_hybrid_score(
    skill_coverage,
    text_similarity,
    experience_relevance
):

    # Weighted components keep the final score in the range 0 to 100.
    score = (
        skill_coverage * SKILL_WEIGHT
        + text_similarity * TEXT_WEIGHT
        + experience_relevance * EXPERIENCE_WEIGHT
    )

    return round(
        min(max(score, 0), 100),
        2
    )


# --------------------------------------------------
# MAIN SCREENING PIPELINE
# --------------------------------------------------

def screen_resume(
    resume_text,
    job_description
):

    # Main orchestration function: validate inputs, analyze each signal,
    # classify the candidate, and return one display-ready result dictionary.
    # Step 1: Validate inputs
    if (
        not isinstance(resume_text, str)
        or not resume_text.strip()
    ):

        raise ValueError(
            "Resume text cannot be empty."
        )

    if (
        not isinstance(job_description, str)
        or not job_description.strip()
    ):

        raise ValueError(
            "Job description cannot be empty."
        )

    # Step 2: Extract and compare skills
    matched_skills, missing_skills = analyze_skills(
        resume_text,
        job_description
    )

    required_skill_count = len(
        extract_skills(job_description)
    )

    # Step 3: Calculate skill coverage
    skill_coverage = calculate_skill_coverage(
        matched_skills,
        missing_skills
    )

    # Step 4: Calculate overall textual similarity
    text_similarity = calculate_text_similarity(
        resume_text,
        job_description
    )

    # Step 5: Calculate experience relevance
    experience_analysis = calculate_experience_relevance(
        resume_text,
        job_description
    )

    experience_relevance = experience_analysis["score"]

    # Step 6: Calculate final hybrid score
    match_score = calculate_hybrid_score(
        skill_coverage,
        text_similarity,
        experience_relevance
    )

    # Step 7: Predict candidate profile
    try:

        model, vectorizer = load_model()

        role_result = predict_role(
            resume_text,
            model,
            vectorizer
        )

    except FileNotFoundError:

        role_result = {
            "predicted_role": "Unavailable",
            "model_confidence": 0,
            "model_error": (
                "Trained role model files are missing. "
                "Run train_model.py to enable role classification."
            )
        }

    except RuntimeError as exc:

        role_result = {
            "predicted_role": "Unavailable",
            "model_confidence": 0,
            "model_error": str(exc)
        }

    # Step 8: Return complete screening results
    return {

        "predicted_role": role_result[
            "predicted_role"
        ],
        "model_category": role_result.get(
            "model_category",
            role_result["predicted_role"]
        ),

        "model_confidence": role_result[
            "model_confidence"
        ],
        "confidence_source": role_result.get(
            "confidence_source",
            "model"
        ),
        "model_error": role_result.get(
            "model_error",
            ""
        ),

        "match_score": match_score,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "skill_coverage": skill_coverage,
        "required_skill_count": required_skill_count,

        "text_similarity": text_similarity,

        "experience_relevance": experience_relevance,

        "responsibility_similarity": (
            experience_analysis[
                "responsibility_similarity"
            ]
        ),

        "demonstrated_skill_coverage": (
            experience_analysis[
                "demonstrated_skill_coverage"
            ]
        ),

        "score_breakdown": {

            "skill_score": round(
                skill_coverage * SKILL_WEIGHT,
                2
            ),

            "text_score": round(
                text_similarity * TEXT_WEIGHT,
                2
            ),

            "experience_score": round(
                experience_relevance * EXPERIENCE_WEIGHT,
                2
            )
        }
    }
