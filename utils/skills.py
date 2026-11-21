
"""Skill dictionary, alias normalization, and resume/job skill comparison.

Skills are extracted by whole-word matching so short names such as "C" and
"R" do not match ordinary words. Aliases are converted to one canonical name.
"""

import re


# Master skill dictionary
SKILLS = {
    "Programming Languages": [
        "Python", "Java", "C", "C++", "C#", "JavaScript",
        "TypeScript", "PHP", "Ruby", "Go", "Rust", "SQL",
        "R", "MATLAB", "Kotlin", "Swift"
    ],

    "Web Development": [
        "HTML", "CSS", "React", "Angular", "Vue.js",
        "Node.js", "Express.js", "Django", "Flask",
        "FastAPI", "REST API", "Bootstrap", "Tailwind CSS"
    ],

    "Data Science & Machine Learning": [
        "Machine Learning", "Deep Learning", "NLP",
        "Natural Language Processing", "Computer Vision",
        "Data Science", "Data Analysis", "Pandas", "NumPy",
        "Scikit-learn", "TensorFlow", "PyTorch",
        "Keras", "Matplotlib", "Seaborn", "OpenCV",
        "Hugging Face", "Transformers"
    ],

    "Databases": [
        "MySQL", "PostgreSQL", "MongoDB", "SQLite",
        "Oracle", "Redis", "Microsoft SQL Server"
    ],

    "Cloud & DevOps": [
        "AWS", "Microsoft Azure", "Google Cloud",
        "Docker", "Kubernetes", "Git", "GitHub",
        "GitLab", "Linux", "CI/CD", "Jenkins"
    ],

    "Tools & Technologies": [
        "VS Code", "Jupyter Notebook", "Power BI",
        "Tableau", "Excel", "Apache Spark", "Hadoop",
        "Postman", "Figma"
    ],

    "Core Computer Science": [
        "Data Structures", "Algorithms", "OOP",
        "Object Oriented Programming", "Operating Systems",
        "Computer Networks", "DBMS", "Software Engineering",
        "System Design"
    ],

    "Cybersecurity": [
        "Cybersecurity", "Network Security", "SIEM",
        "Firewalls", "Penetration Testing", "Vulnerability Assessment",
        "Vulnerability Management", "Incident Response", "Threat Analysis",
        "Wireshark", "Splunk", "Nmap", "Burp Suite", "OWASP",
        "TCP/IP", "DNS", "Bash"
    ],

    "Analytics": [
        "Statistics", "Hypothesis Testing", "Data Cleaning",
        "Exploratory Data Analysis", "Dashboarding", "Reporting"
    ],

    "Frontend": [
        "Responsive Design", "Accessibility", "Next.js",
        "Jest", "React Testing Library", "Playwright",
        "Chrome DevTools", "Figma"
    ],

    "Soft Skills": [
        "Communication", "Leadership", "Teamwork",
        "Problem Solving", "Critical Thinking",
        "Time Management", "Adaptability"
    ]
}


# Different names that refer to the same skill
SKILL_ALIASES = {
    "HTML": [
        "HTML5",
        "HyperText Markup Language",
        "Hyper Text Markup Language"
    ],

    "CSS": [
        "CSS3",
        "Cascading Style Sheets"
    ],

    "NLP": [
        "Natural Language Processing"
    ],

    "OOP": [
        "Object Oriented Programming",
        "Object-Oriented Programming"
    ],

    "Scikit-learn": [
        "Scikit Learn",
        "sklearn"
    ],

    "JavaScript": [
        "JS"
    ],

    "Machine Learning": [
        "ML"
    ],

    "FastAPI": [
        "Fast API"
    ],

    "REST API": [
        "RESTful API",
        "REST APIs"
    ],

    "NumPy": [
        "Numpy"
    ],

    "Node.js": [
        "NodeJS",
        "Node JS"
    ],

    "Tailwind CSS": [
        "Tailwind"
    ],

    "PostgreSQL": [
        "Postgres"
    ],

    "Microsoft Azure": [
        "Azure"
    ],

    "GitHub": [
        "Git Hub"
    ],

    "CI/CD": [
        "Continuous Integration",
        "Continuous Deployment",
        "Continuous Integration and Continuous Deployment",
        "CICD"
    ],

    "TCP/IP": [
        "TCP IP"
    ],

    "Next.js": [
        "NextJS",
        "Next JS"
    ],

    "Chrome DevTools": [
        "Chrome Dev Tools"
    ],

    "React": [
        "ReactJS"
    ],

    "Vue.js": [
        "VueJS",
        "Vue JS"
    ],

    "Microsoft SQL Server": [
        "MSSQL",
        "MS SQL",
        "SQL Server"
    ],

    
}


# Normalize a skill name to its canonical form
def normalize_skill(skill):

    normalized = skill.strip().lower()

    for canonical, aliases in SKILL_ALIASES.items():

        if normalized == canonical.lower():
            return canonical

        if normalized in [alias.lower() for alias in aliases]:
            return canonical

    # Handle normal dictionary skills
    for category_skills in SKILLS.values():

        for known_skill in category_skills:

            if normalized == known_skill.lower():
                return normalize_skill_from_alias(known_skill)

    return skill.strip()


def normalize_skill_from_alias(skill):

    for canonical, aliases in SKILL_ALIASES.items():

        if skill.lower() == canonical.lower():
            return canonical

        if skill.lower() in [alias.lower() for alias in aliases]:
            return canonical

    return skill


# Match a skill using word boundaries
def contains_skill(text, skill):

    pattern = (
        r"(?<![a-zA-Z0-9_])"
        + re.escape(skill.lower())
        + r"(?![a-zA-Z0-9_])"
    )

    return bool(re.search(pattern, text))


# Extract skills from resume or job description
def extract_skills(text):

    if not isinstance(text, str) or not text.strip():
        return set()

    text = text.lower()

    found_skills = set()

    # Search canonical skills
    for category_skills in SKILLS.values():

        for skill in category_skills:

            if contains_skill(text, skill):

                canonical = normalize_skill(skill)

                found_skills.add(canonical)

    # Search alternative skill names
    for canonical, aliases in SKILL_ALIASES.items():

        for alias in aliases:

            if contains_skill(text, alias):

                found_skills.add(canonical)

    # Avoid treating C as a separate skill when C++ or C# is present
    if "C++" in found_skills or "C#" in found_skills:
        found_skills.discard("C")

    return found_skills


# Compare resume skills with required job skills
def compare_skills(resume_skills, required_skills):

    normalized_resume = {
        normalize_skill(skill).lower()
        for skill in resume_skills
    }

    matched_skills = []
    missing_skills = []

    # Remove duplicate requirements after normalization
    unique_requirements = {}

    for skill in required_skills:

        canonical = normalize_skill(skill)

        unique_requirements[canonical.lower()] = canonical

    for key, skill in unique_requirements.items():

        if key in normalized_resume:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    matched_skills.sort(key=str.casefold)
    missing_skills.sort(key=str.casefold)
    return matched_skills, missing_skills


# Main skill analysis function
def analyze_skills(resume_text, job_description):

    resume_skills = extract_skills(resume_text)

    required_skills = extract_skills(job_description)

    matched, missing = compare_skills(
        resume_skills,
        required_skills
    )

    return matched, missing
