
"""Generate short, practical suggestions from screening results."""


def generate_recommendations(screening_results):

    # Read values once so the recommendation rules remain easy to follow.
    recommendations = []

    match_score = screening_results.get("match_score", 0)
    matched_skills = screening_results.get("matched_skills", [])
    missing_skills = screening_results.get("missing_skills", [])

    predicted_role = screening_results.get(
        "predicted_role", "Unknown"
    )

    confidence = screening_results.get(
        "model_confidence", 0
    )

    skill_coverage = screening_results.get(
        "skill_coverage", 0
    )
    required_skill_count = screening_results.get(
        "required_skill_count", len(missing_skills) + len(matched_skills)
    )

    text_similarity = screening_results.get(
        "text_similarity", 0
    )

    experience_relevance = screening_results.get(
        "experience_relevance", 0
    )

    # Overall match assessment
    if match_score >= 80:
        recommendations.append(
            "Your resume shows strong alignment with the job description. "
            "Continue highlighting relevant achievements and projects."
        )

    elif match_score >= 65:
        recommendations.append(
            "Your resume shows good alignment with the target role. "
            "A few improvements could make your application stronger."
        )

    elif match_score >= 45:
        recommendations.append(
            "Your resume shows moderate alignment. "
            "Focus on improving relevant skills and demonstrating "
            "practical experience."
        )

    else:
        recommendations.append(
            "Your resume currently shows limited alignment with the "
            "job description. Review the requirements and identify "
            "areas where your experience can be strengthened."
        )

    # Skill gap recommendations
    if required_skill_count == 0:

        recommendations.append(
            "No recognized technical requirements were detected in the "
            "job description. Add specific tools or technologies to get "
            "a more accurate skill-gap analysis."
        )

    elif missing_skills:

        recommendations.append(
            "Skills to consider developing: "
            + ", ".join(missing_skills)
            + ". Add them to your resume only after gaining "
              "genuine practical experience."
        )

    else:

        recommendations.append(
            "All recognized skills from the job description "
            "were found in your resume. Consider demonstrating "
            "their application through practical projects."
        )

    # Skill coverage
    if required_skill_count > 0 and skill_coverage < 50:

        recommendations.append(
            "Your technical skill coverage is relatively low. "
            "Prioritize the essential skills mentioned in the "
            "job description."
        )

    elif required_skill_count > 0 and skill_coverage >= 80:

        recommendations.append(
            "Your technical skill coverage is strong. "
            "Make sure these skills are supported by concrete "
            "project or work examples."
        )

    # Text similarity
    if text_similarity < 40:

        recommendations.append(
            "The textual similarity between your resume and the "
            "job description is low. Improve the clarity of your "
            "project descriptions and use accurate, relevant "
            "technical terminology."
        )

    # Experience and project relevance
    if experience_relevance < 40:

        recommendations.append(
            "Your experience and project sections have limited "
            "textual overlap with the job description. Highlight "
            "relevant responsibilities, technologies and outcomes."
        )

    else:

        recommendations.append(
            "Your experience and project descriptions show "
            "some relevance to the target role. Include measurable "
            "results wherever possible."
        )

    # Predicted role confidence
    if confidence < 50:

        recommendations.append(
            "The role classifier has low confidence in its "
            "prediction. Treat the predicted profile as a tentative "
            "classification rather than a definitive result."
        )

    # Predicted role
    if predicted_role != "Unknown" and predicted_role != "Unavailable":

        recommendations.append(
            f"Your resume is currently classified under "
            f"{predicted_role}. Ensure your projects and "
            "experience clearly demonstrate your intended "
            "career direction."
        )

    # General improvements
    recommendations.append(
        "Use clear resume sections such as Profile, Technical Skills, "
        "Experience, Projects and Education."
    )

    recommendations.append(
        "Describe projects using your contribution, technologies "
        "used and measurable outcomes."
    )

    return recommendations
