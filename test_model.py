from utils.model import load_model, predict_role, calculate_match


# Replace this example with text extracted from a real resume.
resume = """
Python developer with experience in Python, SQL, pandas,
NumPy, data analysis and machine learning.
Built data processing scripts and analyzed datasets.
"""

job_description = """
We are looking for a data scientist with Python, SQL,
pandas, machine learning and data analysis skills.
"""


def main():
    model, vectorizer = load_model()

    result = predict_role(resume, model, vectorizer)
    match_score = calculate_match(resume, job_description)

    print("\n========== RESUME SCREENING RESULT ==========")
    print("Predicted Category :", result["predicted_role"])
    print("Model Confidence   :", f'{result["model_confidence"]}%')
    print("Resume-JD Match    :", f"{match_score}%")

    print("\nNote: This is a sample resume.")
    print("Replace it with actual extracted resume text to test.")


if __name__ == "__main__":
    main()
