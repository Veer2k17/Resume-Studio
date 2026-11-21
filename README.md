# Resume Studio

Resume Studio is a Streamlit application that analyzes a resume against a target
job description. It extracts PDF, DOCX, or TXT text, identifies matching skills,
calculates a hybrid fit score, predicts a role profile, and generates practical
next steps.

## Features

- PDF, DOCX, and TXT resume upload
- Optional OCR checkbox for scanned PDFs
- Resume and job-description skill comparison
- Hybrid fit score based on:
  - 75% recognized skill coverage
  - 15% text similarity
  - 10% experience/project relevance
- Specific profile detection for major technical roles
- Matching and missing skill displays
- Score breakdown and personalized recommendations
- Pretrained SVM role classifier
- Five representative PDF resumes for testing

## Project Structure

```text
.
|-- app.py                 # Streamlit interface
|-- train_model.py         # Trains and evaluates the role classifier
|-- test_model.py          # Runs a small fixed-resume test
|-- data/
|   `-- resumes_cleaned.csv
|-- models/
|   |-- resume_model.pkl
|   `-- tfidf_vectorizer.pkl
|-- samples/
|   `-- resumes/
|-- utils/
|   |-- parser.py          # Resume text extraction
|   |-- preprocessing.py   # Text cleaning and dataset preparation
|   |-- skills.py          # Skill dictionary and alias matching
|   |-- model.py           # TF-IDF, SVM, and role prediction
|   |-- screening.py       # Hybrid screening pipeline
|   `-- recommendations.py # Recommendation generation
`-- requirements.txt
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone https://github.com/Veer2k17/Resume-Studio.git
cd Resume-Studio
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit, upload a resume, and paste the target
job description.

## Train or Test the Model

Evaluate and retrain the classifier:

```bash
python train_model.py
```

Run the fixed sample:

```bash
python test_model.py
```

## How the Screening Score Works

The final fit score combines three signals:

1. **Skill coverage** - recognized JD skills found in the resume.
2. **Text similarity** - cosine similarity after TF-IDF text preprocessing.
3. **Experience relevance** - overlap between resume experience/projects and
   extracted job responsibilities.

The result is a screening guide, not a hiring decision. Profile detection and
text similarity cannot replace human resume review.

## Sample Resumes

The `samples/resumes` directory contains five test PDFs:

- Software Developer
- Data Analyst
- Machine Learning Engineer
- Cybersecurity Analyst
- Frontend Developer

## Dependencies

- Streamlit
- scikit-learn
- pandas
- joblib
- PyMuPDF
- python-docx

## License

This project is for educational use. No license is currently specified.
