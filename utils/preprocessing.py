"""Resume text preprocessing.

This file does two jobs:

1. clean_text()          The ONE cleaning function used everywhere: training data,
                         uploaded resumes, job descriptions, and skill lists.
                         It has no pandas dependency, so the app can import it freely.
2. preprocess_datasets() Builds the training CSV from one or more raw datasets:
                         column detection, label cleanup, job-title removal,
                         cleaning, de-duplication, train/test split, report.

Rule for the whole team: anything that gets compared with a resume (job
descriptions, skill lists) must go through clean_text() too. Otherwise
"C++" in a skill list will never match "cpp" in a cleaned resume.

Command line (from the project folder):
    python utils/preprocessing.py --input data/Resume.csv --output data/resumes_cleaned.csv
    python utils/preprocessing.py --input data/a.csv data/b.csv --label-map data/label_map.json
"""

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

# ---------------------------------------------------------------------------
# Rules for clean_text()
# ---------------------------------------------------------------------------

_BULLET_RE = re.compile("[•●▪■◦‣∙·○◆►▶✓✔➢➤❖★☆]")
_DASH_RE = re.compile("[–—−‐‑]")
_QUOTE_MAP = str.maketrans({"’": "'", "‘": "'", "‚": "'", "“": '"', "”": '"'})
_HYPHEN_WRAP_RE = re.compile(r"(?<=[a-z])-[ \t]*\n\s*(?=[a-z])")  # "develop-\nment"

_MONTHS = (
    r"(?:january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
)

# Things that carry no signal for matching: links, contact details, placeholders, dates.
_NOISE_RULES = [
    (re.compile(p), r)
    for p, r in [
        (r"https?://\S+|www\.\S+", " "),
        (r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " "),
        (r"\b(?:linkedin|github)\.com/\S*", " "),
        (r"(?<!\d)(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?!\d)", " "),
        (r"\bcompany\s*name\b", " "),  # anonymisation placeholders in the Kaggle data
        (r"\bcity\s*,?\s*state\b", " "),
        (rf"\b{_MONTHS}\.?\s*,?\s*(?:19|20)\d{{2}}\b", " "),
        (r"\b\d{1,2}\s*/\s*(?:19|20)\d{2}\b", " "),
    ]
]

# Technical terms that contain symbols. These MUST run before punctuation is
# stripped, otherwise "c++" and "c#" both collapse to "c" and disappear.
_PRE_RULES = [
    (re.compile(p), r)
    for p, r in [
        (r"(?<![a-z0-9])c\s*/\s*c\+\+", " clang cpp "),
        (r"(?<![a-z0-9])c\+\+", " cpp "),
        (r"(?<![a-z0-9])c#", " csharp "),
        (r"(?<![a-z0-9])f#", " fsharp "),
        (r"\bobjective[\s-]?c\b", " objectivec "),
        (r"(?<![a-z0-9])c\s+(?:programming|language)\b", " clang "),
        (r"(?<![a-z0-9])r\s+(?:programming|language|studio)\b", " rlang "),
        (r"\basp\.net\b", " aspnet "),
        (r"\bvb\.net\b", " vbnet "),
        (r"(?<![a-z0-9])\.net\b|\bdot\s?net\b", " dotnet "),
        (r"\b([a-z0-9]+)\.js\b", r" \1js "),  # node.js, chart.js, vue.js ...
        (r"\b(node|react|vue|angular|express|next|nuxt)\s+js\b", r" \1js "),
        (r"\bjs\b", " javascript "),
        (r"\br\s?&\s?d\b", " research and development "),
        (r"\bp\s?&\s?l\b", " profit and loss "),
    ]
]

# Aliases and abbreviations. These run on already-cleaned tokens.
_NOT_AFTER_NUMBER = r"(?<!\d)(?<!\d )"  # "500 ml" is millilitres, not machine learning
_POST_RULES = [
    (re.compile(p), r)
    for p, r in [
        (r"\bc plus plus\b", "cpp"),
        (r"\bc sharp\b", "csharp"),
        (r"\bjava script\b", "javascript"),
        (r"\btype script\b", "typescript"),
        (r"\breactjs\b", "react"),
        (r"\bvuejs\b", "vue"),
        (r"\bangularjs\b", "angular"),
        (r"\bd3js\b", "d3"),
        (r"\bpy torch\b", "pytorch"),
        (r"\btensor flow\b", "tensorflow"),
        (r"\b(?:scikit learn|sklearn)\b", "scikitlearn"),
        (r"\bpower bi\b", "powerbi"),
        (r"\bgit hub\b", "github"),
        (r"\bgit lab\b", "gitlab"),
        (r"\bnum py\b", "numpy"),
        (r"\bk8s\b", "kubernetes"),
        (r"\b(?:postgres|postgre sql)\b", "postgresql"),
        (r"\b(?:mongo db|mongo)\b", "mongodb"),
        (r"\bmy sql\b", "mysql"),
        (r"\b(?:ms )?sql server\b", "sqlserver"),
        (r"\bhtml ?5\b", "html"),
        (r"\bcss ?3\b", "css"),
        (r"\brestful\b", "rest"),
        (r"\bci cd\b", "cicd"),
        (r"\bms (excel|word|office|powerpoint|outlook|access|project)\b", r"microsoft \1"),
        (r"\be mail\b", "email"),
        (r"\be commerce\b", "ecommerce"),
        (r"\bb tech\b", "btech"),
        (r"\bm tech\b", "mtech"),
        (r"\bb sc\b", "bsc"),
        (r"\bm sc\b", "msc"),
        (r"\bph d\b", "phd"),
        (r"(?<=\d) ?ml\b", " milliliters"),  # unit, keeps the rule below idempotent
        (r"(?<=\d) ?dl\b", " deciliters"),
        (rf"{_NOT_AFTER_NUMBER}\bml\b", "machine learning"),
        (rf"{_NOT_AFTER_NUMBER}\bdl\b", "deep learning"),
        (r"\bnlp\b", "natural language processing"),
        (r"\bai\b", "artificial intelligence"),
        (r"\boops?\b", "object oriented programming"),
    ]
]


def _strip_control(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", " ")
    return "".join(
        ch for ch in text
        if ch == "\n" or unicodedata.category(ch) not in ("Cc", "Cf")
    )


def _strip_accents(text):
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def clean_text(text, keep_numbers=False, keep_lines=False):
    """Clean and normalise one piece of text (resume, job description, skill).

    keep_numbers: keep standalone numbers (default drops them: years, ids, ...).
    keep_lines:   keep line breaks (useful if you need section structure).

    Safe to call on already-cleaned text (running it twice gives the same result).
    """
    if text is None:
        return ""
    if isinstance(text, float) and text != text:  # NaN
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")

    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = _strip_control(text)
    text = text.translate(_QUOTE_MAP)
    text = _BULLET_RE.sub("\n", text)
    text = _DASH_RE.sub(" - ", text)
    text = _HYPHEN_WRAP_RE.sub("", text)

    for rx, rep in _NOISE_RULES:
        text = rx.sub(rep, text)
    for rx, rep in _PRE_RULES:
        text = rx.sub(rep, text)

    text = re.sub(r"(?<=\w)'s\b", "", text)  # possessives
    text = text.replace("'", "")
    text = text.replace("&", " and ")
    text = _strip_accents(text)

    if not keep_lines:
        text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9\n ]", " ", text)

    for rx, rep in _POST_RULES:
        text = rx.sub(rep, text)
    if not keep_numbers:
        text = re.sub(r"\b\d+\b", " ", text)

    if keep_lines:
        lines = (re.sub(r" +", " ", ln).strip() for ln in text.split("\n"))
        return "\n".join(ln for ln in lines if ln)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Job-title removal (dataset preparation only)
# ---------------------------------------------------------------------------

# Resumes in these datasets start with the job title, which is basically the label.
# We cut everything before the first section header. Strong headers are tried first.
_STRONG_HEADER = re.compile(
    r"\b(?:professional\s+summary|career\s+summary|executive\s+summary|"
    r"summary\s+of\s+qualifications|summary|career\s+objective|objective|"
    r"executive\s+profile|professional\s+profile|profile|career\s+overview|"
    r"overview|highlights|core\s+qualifications|qualifications|career\s+focus|"
    r"accomplishments|areas\s+of\s+expertise|core\s+competencies)\b",
    re.IGNORECASE,
)
_WEAK_HEADER = re.compile(
    r"\b(?:skills|work\s+history|professional\s+experience|experience|education)\b",
    re.IGNORECASE,
)


def strip_title(text, window=200):
    """Remove the job title at the top of a raw resume.

    Returns (text, was_stripped). Only strips when a section header appears
    within the first `window` characters, so resumes without a recognisable
    header are left untouched.
    """
    head = text[:window]
    for rx in (_STRONG_HEADER, _WEAK_HEADER):
        m = rx.search(head)
        if m:
            return (text[m.start():], True) if m.start() > 0 else (text, False)
    return text, False


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

_TEXT_COLUMNS = ["resumestr", "resume", "resumetext", "text", "content", "cvtext", "cv"]
_LABEL_COLUMNS = ["category", "label", "role", "jobrole", "jobcategory", "target", "class", "domain"]
_SMALL_CLASS = 30


def _load_table(path):
    import pandas as pd

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in (".json", ".jsonl"):
        return pd.read_json(path, lines=(suffix == ".jsonl"))
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported dataset format: {path.name}")


def _find_column(df, candidates, kind):
    names = {re.sub(r"[^a-z0-9]", "", str(c).lower()): c for c in df.columns}
    for cand in candidates:
        if cand in names:
            return names[cand]
    raise ValueError(
        f"Could not find a {kind} column. Columns found: {list(df.columns)}"
    )


def normalize_label(label):
    """'Information Technology', 'information_technology' -> 'INFORMATION-TECHNOLOGY'."""
    label = re.sub(r"[\s_/&]+", "-", str(label).strip().upper())
    return re.sub(r"-+", "-", label).strip("-")


def _fingerprint(text):
    return hashlib.sha1(text[:600].encode("utf-8")).hexdigest()


def _assign_split(df, test_size, seed):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=df.index)
    for _, idx in df.groupby("Category").groups.items():
        idx = np.array(list(idx))
        if len(idx) < 5:  # too small to split: keep everything in train
            continue
        rng.shuffle(idx)
        n_test = max(1, round(len(idx) * test_size))
        split.loc[idx[:n_test]] = "test"
    return split


def preprocess_datasets(
    input_files,
    output_file,
    label_map=None,
    strip_titles=True,
    test_size=0.2,
    seed=42,
    min_words=50,
):
    """Build a clean, merged, de-duplicated, split training CSV.

    input_files:  one path or a list of paths (csv, tsv, parquet, json, jsonl, xlsx).
                  Text and label columns are detected automatically.
    label_map:    optional dict to merge differently named labels, e.g.
                  {"JAVA-DEVELOPER": "INFORMATION-TECHNOLOGY"} (names are normalised first).
    strip_titles: remove the job title at the top of each resume (recommended).
    min_words:    drop resumes shorter than this after cleaning.

    Output columns: Resume, Category, source, split ("train" / "test").
    Fit TF-IDF and any model on split == "train" only.
    """
    import pandas as pd

    if isinstance(input_files, (str, Path)):
        input_files = [input_files]
    label_map = {normalize_label(k): normalize_label(v) for k, v in (label_map or {}).items()}

    frames, report = [], {"sources": {}}
    titles_stripped = 0

    for path in input_files:
        df = _load_table(path)
        text_col = _find_column(df, _TEXT_COLUMNS, "resume text")
        label_col = _find_column(df, _LABEL_COLUMNS, "label")
        df = df[[text_col, label_col]].copy()
        df.columns = ["Resume", "Category"]
        df["source"] = Path(path).stem
        report["sources"][Path(path).name] = {"rows": len(df), "text_column": text_col, "label_column": label_col}

        df = df.dropna(subset=["Resume", "Category"])
        df["Category"] = df["Category"].map(normalize_label).replace(label_map)
        df = df[(df["Resume"].astype(str).str.strip() != "") & (df["Category"] != "")]

        if strip_titles:
            stripped = df["Resume"].astype(str).map(strip_title)
            df["Resume"] = stripped.map(lambda r: r[0])
            titles_stripped += int(stripped.map(lambda r: r[1]).sum())

        df["Resume"] = df["Resume"].map(clean_text)
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    report["rows_loaded"] = len(data)
    report["titles_stripped"] = titles_stripped

    n = len(data)
    data = data[data["Resume"].str.split().str.len() >= min_words]
    report["dropped_too_short"] = n - len(data)

    # Same resume with different labels -> ambiguous, remove all copies.
    data = data.assign(_fp=data["Resume"].map(_fingerprint))
    label_counts = data.groupby("_fp")["Category"].nunique()
    conflicts = label_counts[label_counts > 1].index
    report["dropped_conflicting_labels"] = int(data["_fp"].isin(conflicts).sum())
    data = data[~data["_fp"].isin(conflicts)]

    n = len(data)
    data = data.drop_duplicates(subset="_fp", keep="first").drop(columns="_fp")
    report["dropped_duplicates"] = n - len(data)

    data = data.reset_index(drop=True)
    data["split"] = _assign_split(data, test_size, seed)

    counts = data["Category"].value_counts()
    report["rows_final"] = len(data)
    report["num_categories"] = int(counts.size)
    report["per_category"] = counts.to_dict()
    report["small_categories"] = counts[counts < _SMALL_CLASS].to_dict()
    report["split_sizes"] = data["split"].value_counts().to_dict()

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    report_path = output_path.with_name(output_path.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    _print_report(report, output_path)
    return data


def _print_report(report, output_path):
    print("\n========== PREPROCESSING SUMMARY ==========")
    for name, info in report["sources"].items():
        print(f"{name}: {info['rows']} rows (text='{info['text_column']}', label='{info['label_column']}')")
    print("Job titles removed from:", report["titles_stripped"], "resumes")
    print("Dropped (too short):", report["dropped_too_short"])
    print("Dropped (same resume, conflicting labels):", report["dropped_conflicting_labels"])
    print("Dropped (duplicates):", report["dropped_duplicates"])
    print("Final rows:", report["rows_final"], "| categories:", report["num_categories"])
    print("Split:", report["split_sizes"])
    if report["small_categories"]:
        print(f"Small categories (< {_SMALL_CLASS} resumes, expect weak results):", report["small_categories"])
    print("Saved to:", output_path)


if __name__ == "__main__":
    project = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Clean and prepare resume datasets.")
    parser.add_argument("--input", nargs="+", default=[str(project / "data" / "Resume.csv")])
    parser.add_argument("--output", default=str(project / "data" / "resumes_cleaned.csv"))
    parser.add_argument("--label-map", help="JSON file mapping label names to merge")
    parser.add_argument("--keep-title", action="store_true", help="do not remove job titles")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--min-words", type=int, default=50)
    args = parser.parse_args()

    mapping = json.loads(Path(args.label_map).read_text()) if args.label_map else None
    preprocess_datasets(
        args.input,
        args.output,
        label_map=mapping,
        strip_titles=not args.keep_title,
        test_size=args.test_size,
        min_words=args.min_words,
    )
