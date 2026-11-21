
"""Trains and loads the role classifier and calculates text similarity."""


from pathlib import Path
import joblib
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import clean_text


ROLE_PROFILES = {
    # These focused profiles improve sample-role predictions before the
    # broad Kaggle-trained SVM classifier is used as a fallback.
    "SOFTWARE-DEVELOPER": {
        "title": ("software developer", "software engineer", "backend developer"),
        "signals": {
            "java": 3, "spring boot": 4, "rest api": 3, "sql": 2,
            "git": 1, "docker": 2, "linux": 1, "software engineering": 3,
            "postgresql": 1, "redis": 1, "agile": 2,
        },
    },
    "DATA-ANALYST": {
        "title": ("data analyst", "data analytics", "business analyst"),
        "signals": {
            "data analyst": 5, "excel": 3, "power bi": 4, "tableau": 4,
            "pandas": 3, "numpy": 2, "data analysis": 4, "statistics": 3,
            "data cleaning": 3, "dashboard": 3, "sql": 2,
        },
    },
    "MACHINE-LEARNING-ENGINEER": {
        "title": ("machine learning engineer", "ml engineer", "ai engineer"),
        "signals": {
            "machine learning": 5, "machine learning engineer": 6,
            "tensorflow": 4, "pytorch": 4, "scikit learn": 4,
            "scikitlearn": 4, "deep learning": 4, "nlp": 3,
            "computer vision": 3, "feature engineering": 3,
            "model evaluation": 3, "tensorflow": 4,
        },
    },
    "CYBERSECURITY-ANALYST": {
        "title": ("cybersecurity analyst", "security analyst", "soc analyst"),
        "signals": {
            "cybersecurity": 6, "network security": 5, "vulnerability": 4,
            "incident response": 5, "siem": 5, "firewall": 4,
            "penetration testing": 5, "wireshark": 4, "splunk": 4,
            "nmap": 4, "tcp ip": 3, "owasp": 4, "soc": 3,
        },
    },
    "FRONTEND-DEVELOPER": {
        "title": ("frontend developer", "front end developer", "ui developer"),
        "signals": {
            "frontend": 6, "javascript": 3, "typescript": 4, "react": 5,
            "html": 3, "css": 3, "tailwind": 4, "figma": 2,
            "nextjs": 4, "next js": 4, "responsive": 3, "accessibility": 3,
        },
    },
}


# Model folder path
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Create TF-IDF vectorizer
def create_vectorizer():
    return TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )


# Train SVM classifier
def train_classifier(resumes, categories):

    cleaned_resumes = [
        clean_text(text) for text in resumes
    ]

    if any(not text for text in cleaned_resumes):
        raise ValueError("Dataset contains empty resume text.")

    vectorizer = create_vectorizer()
    features = vectorizer.fit_transform(cleaned_resumes)

    model = SVC(
        kernel="linear",
        C=1.0,
        probability=True,
        class_weight="balanced",
        random_state=42
    )

    model.fit(features, categories)

    return model, vectorizer


# Predict job role
def predict_role(resume_text, model, vectorizer):

    # Profile matching gives specific roles a stronger signal than the
    # broad training categories; the trained SVM remains the fallback.
    cleaned = clean_text(resume_text)

    if not cleaned:
        raise ValueError("Resume text cannot be empty.")

    profile_scores = {}
    for role, profile in ROLE_PROFILES.items():
        text = cleaned.lower()
        score = 0

        for title in profile["title"]:
            if title in text:
                score += 12

        for signal, weight in profile["signals"].items():
            if signal in text:
                score += weight

        if score:
            profile_scores[role] = score

    if profile_scores:
        profile_role = max(profile_scores, key=profile_scores.get)
        profile_score = profile_scores[profile_role]
        ranked_scores = sorted(profile_scores.values(), reverse=True)

        if profile_score >= 12:
            runner_up = ranked_scores[1] if len(ranked_scores) > 1 else 0
            if profile_score > runner_up:
                confidence = round(
                    min(92.0, max(70.0, 72.0 + profile_score * 0.18)),
                    2,
                )
                return {
                    "predicted_role": profile_role,
                    "model_confidence": confidence,
                    "confidence_source": "profile",
                }

    features = vectorizer.transform([cleaned])

    predicted_role = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(probabilities.max() * 100)

    return {
        "predicted_role": str(predicted_role),
        "model_confidence": round(confidence, 2),
        "confidence_source": "model",
    }


# Calculate textual similarity between resume and job description
def calculate_text_similarity(resume_text, job_description):

    # Both texts receive the same cleaning rules before TF-IDF is compared.
    resume = clean_text(resume_text)
    job = clean_text(job_description)

    if not resume or not job:
        raise ValueError(
            "Resume and job description must not be empty."
        )

    vectorizer = create_vectorizer()

    vectors = vectorizer.fit_transform([
        resume,
        job
    ])

    similarity = cosine_similarity(
        vectors[0:1],
        vectors[1:2]
    )[0, 0]

    return round(float(similarity) * 100, 2)


# Backward-compatible function
def calculate_match(resume_text, job_description):

    return calculate_text_similarity(
        resume_text,
        job_description
    )


# Save trained SVM model and vectorizer
def save_model(model, vectorizer):

    joblib.dump(
        model,
        MODEL_DIR / "resume_model.pkl"
    )

    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.pkl"
    )


# Load saved model and vectorizer
def load_model():

    # Separate try/except handling distinguishes missing files from corrupt
    # or incompatible saved model data.
    model_path = MODEL_DIR / "resume_model.pkl"

    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    if not model_path.exists() or not vectorizer_path.exists():
        raise FileNotFoundError(
            "Trained model files not found. "
            "Run python train_model.py first."
        )

    try:
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except (
        OSError,
        EOFError,
        ValueError,
        AttributeError,
        ImportError,
        IndexError,
        KeyError,
        TypeError,
        pickle.UnpicklingError,
    ) as exc:
        raise RuntimeError(
            f"The saved role model could not be loaded: {exc}"
        ) from exc

    return model, vectorizer
