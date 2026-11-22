
"""Trains and loads the role classifier and calculates text similarity."""


from pathlib import Path
import joblib
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import clean_text


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


def create_detailed_role(resume_text, model_category):
    """Refine a broad SVM category into a specific resume profile."""
    category = str(model_category).strip().upper()
    text = clean_text(resume_text).lower()

    engineering_signals = [
        (
            "Machine Learning Engineer",
            (
                "machine learning",
                "deep learning",
                "tensorflow",
                "pytorch",
                "computer vision",
                "natural language processing",
                "nlp",
            ),
        ),
        (
            "Data Analyst",
            (
                "data analyst",
                "data analysis",
                "pandas",
                "tableau",
                "power bi",
                "excel",
                "data visualization",
            ),
        ),
        (
            "Data Engineer",
            ("spark", "hadoop", "data pipeline", "etl", "kafka"),
        ),
        (
            "Cybersecurity Engineer",
            (
                "cybersecurity",
                "network security",
                "penetration testing",
                "incident response",
                "siem",
            ),
        ),
        (
            "Frontend Developer",
            ("frontend", "react", "javascript", "typescript", "html", "css"),
        ),
        (
            "Backend Software Engineer",
            ("spring boot", "rest api", "java", "django", "node.js"),
        ),
        (
            "DevOps Engineer",
            ("docker", "kubernetes", "jenkins", "terraform", "ci cd"),
        ),
        (
            "Civil Engineer",
            ("civil engineering", "structural", "autocad", "construction"),
        ),
        (
            "Mechanical Engineer",
            ("mechanical engineering", "mechanical design", "cad", "solidworks"),
        ),
        (
            "Electrical Engineer",
            ("electrical engineering", "embedded", "circuit", "iot", "pcb"),
        ),
    ]

    matches = [
        (label, sum(signal in text for signal in signals))
        for label, signals in engineering_signals
    ]
    matches = [match for match in matches if match[1] > 0]
    if matches:
        return max(matches, key=lambda match: match[1])[0]

    category_labels = {
        "ACCOUNTANT": "Accounting Professional",
        "ADVOCATE": "Legal Professional",
        "AGRICULTURE": "Agriculture Professional",
        "APPAREL": "Apparel and Fashion Professional",
        "ARTS": "Creative Arts Professional",
        "AUTOMOBILE": "Automotive Engineer",
        "AVIATION": "Aviation Professional",
        "BANKING": "Banking Professional",
        "BPO": "Business Process Operations Professional",
        "BUSINESS-DEVELOPMENT": "Business Development Professional",
        "CONSULTANT": "Consulting Professional",
        "CONSTRUCTION": "Construction Professional",
        "DESIGNER": "Design Professional",
        "DIGITAL-MEDIA": "Digital Media Professional",
        "FITNESS": "Fitness Professional",
        "FINANCE": "Finance Professional",
        "HEALTHCARE": "Healthcare Professional",
        "HR": "Human Resources Professional",
        "PUBLIC-RELATIONS": "Public Relations Professional",
        "SALES": "Sales Professional",
        "TEACHER": "Education Professional",
    }

    return category_labels.get(category, category.replace("-", " ").title())


# Predict job role
def predict_role(resume_text, model, vectorizer):
    cleaned = clean_text(resume_text)

    if not cleaned:
        raise ValueError("Resume text cannot be empty.")

    features = vectorizer.transform([cleaned])

    predicted_role = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]

    confidence = float(probabilities.max() * 100)
    model_category = str(predicted_role)
    detailed_role = create_detailed_role(cleaned, model_category)

    return {
        "predicted_role": detailed_role,
        "model_category": model_category,
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
