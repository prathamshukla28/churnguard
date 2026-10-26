"""Generate docs/ChurnGuard.pdf — a project explainer + interview prep deck.

Reads the real metrics from reports/metrics.json so the numbers never drift from
the committed evaluation artifacts. Run from the project root:

    python scripts/make_docs_pdf.py
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parent.parent
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text())
FIG = ROOT / "reports" / "figures"
OUT = ROOT / "docs" / "ChurnGuard.pdf"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---- palette -------------------------------------------------------------
INK = colors.HexColor("#0f172a")      # slate-900
BODY = colors.HexColor("#1e293b")     # slate-800
MUTED = colors.HexColor("#64748b")    # slate-500
ACCENT = colors.HexColor("#4f46e5")   # indigo-600
ACCENT_SOFT = colors.HexColor("#eef2ff")
LINE = colors.HexColor("#e2e8f0")
CODE_BG = colors.HexColor("#0f172a")
CODE_FG = colors.HexColor("#e2e8f0")
GOOD = colors.HexColor("#047857")

AUTHOR = "Pratham Shukla"
EMAIL = "prathamshukla2804@gmail.com"
REPO = "github.com/prathamshukla28/churnguard"

# ---- styles --------------------------------------------------------------
ss = getSampleStyleSheet()


def style(name, **kw):
    base = kw.pop("parent", ss["Normal"])
    return ParagraphStyle(name, parent=base, **kw)


H1 = style("H1", fontName="Helvetica-Bold", fontSize=19, leading=23,
           textColor=INK, spaceBefore=4, spaceAfter=10)
H2 = style("H2", fontName="Helvetica-Bold", fontSize=13, leading=17,
           textColor=ACCENT, spaceBefore=14, spaceAfter=6)
H3 = style("H3", fontName="Helvetica-Bold", fontSize=10.5, leading=14,
           textColor=INK, spaceBefore=9, spaceAfter=3)
BODYP = style("Body", fontName="Helvetica", fontSize=9.7, leading=14.5,
              textColor=BODY, alignment=TA_JUSTIFY, spaceAfter=6)
BULLET = style("Bullet", parent=BODYP, leftIndent=14, bulletIndent=3,
               spaceAfter=3, alignment=TA_LEFT)
LEAD = style("Lead", fontName="Helvetica", fontSize=11, leading=16,
             textColor=BODY, spaceAfter=8)
SMALL = style("Small", fontName="Helvetica", fontSize=8, leading=11,
              textColor=MUTED)
CODE = style("Code", fontName="Courier", fontSize=8.3, leading=12,
             textColor=CODE_FG, backColor=CODE_BG, leftIndent=8,
             rightIndent=8, spaceBefore=4, spaceAfter=8,
             borderPadding=(7, 8, 7, 8))
QBOLD = style("QBold", fontName="Helvetica-Bold", fontSize=9.8, leading=14,
              textColor=INK, spaceBefore=8, spaceAfter=2)
ABODY = style("ABody", parent=BODYP, textColor=BODY, spaceAfter=4)
CARDNUM = style("CardNum", fontName="Helvetica-Bold", fontSize=17,
                leading=19, textColor=ACCENT, alignment=TA_CENTER)
CARDLBL = style("CardLbl", fontName="Helvetica", fontSize=7.3, leading=9,
                textColor=MUTED, alignment=TA_CENTER)

story: list = []


def p(text, st=BODYP):
    story.append(Paragraph(text, st))


def h1(t):
    story.append(Paragraph(t, H1))
    story.append(HRFlowable(width="100%", thickness=1.4, color=ACCENT,
                            spaceBefore=1, spaceAfter=9))


def h2(t):
    story.append(Paragraph(t, H2))


def h3(t):
    story.append(Paragraph(t, H3))


def bullets(items, st=BULLET):
    for it in items:
        story.append(Paragraph(it, st, bulletText="•"))


def code(lines):
    txt = "<br/>".join(
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for s in lines
    )
    story.append(Paragraph(txt, CODE))


def gap(h=6):
    story.append(Spacer(1, h))


def qa(q, a_lines):
    flow = [Paragraph("Q&nbsp;&nbsp;" + q, QBOLD)]
    for a in a_lines:
        flow.append(Paragraph("A&nbsp;&nbsp;" + a if a == a_lines[0]
                              else a, ABODY))
    story.append(KeepTogether(flow))


# =========================================================================
# PAGE TEMPLATES
# =========================================================================
def cover(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setFillColor(INK)
    canvas.rect(0, 0, w, h, fill=1, stroke=0)
    canvas.setFillColor(ACCENT)
    canvas.rect(0, h - 6 * mm, w, 6 * mm, fill=1, stroke=0)
    # eyebrow
    canvas.setFillColor(colors.HexColor("#a5b4fc"))
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(2.4 * cm, h - 5.3 * cm, "MACHINE LEARNING  •  PROJECT DOSSIER")
    # title
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 44)
    canvas.drawString(2.4 * cm, h - 7.8 * cm, "ChurnGuard")
    canvas.setFont("Helvetica", 15)
    canvas.setFillColor(colors.HexColor("#cbd5e1"))
    canvas.drawString(2.4 * cm, h - 9.1 * cm,
                      "Customer Churn Prediction Service")
    # rule
    canvas.setStrokeColor(colors.HexColor("#334155"))
    canvas.setLineWidth(1)
    canvas.line(2.4 * cm, h - 10.1 * cm, w - 2.4 * cm, h - 10.1 * cm)
    # subtitle lines
    canvas.setFont("Helvetica", 10.5)
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    subs = [
        "An end-to-end ML service: data → features → training →",
        "evaluation → a served REST API, built with production practices.",
        "This dossier: what it is, how it works, and interview Q&A.",
    ]
    for i, s in enumerate(subs):
        canvas.drawString(2.4 * cm, h - 11.2 * cm - i * 0.62 * cm, s)
    # footer author block
    canvas.setFillColor(colors.white)
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawString(2.4 * cm, 3.4 * cm, AUTHOR)
    canvas.setFont("Helvetica", 9.5)
    canvas.setFillColor(colors.HexColor("#94a3b8"))
    canvas.drawString(2.4 * cm, 2.85 * cm, EMAIL)
    canvas.drawString(2.4 * cm, 2.4 * cm, REPO)
    canvas.setFillColor(colors.HexColor("#64748b"))
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(w - 2.4 * cm, 2.4 * cm,
                           "Python · scikit-learn · FastAPI · Docker · CI")
    canvas.restoreState()


def content_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    # header band
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.6)
    canvas.line(2 * cm, h - 1.5 * cm, w - 2 * cm, h - 1.5 * cm)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(ACCENT)
    canvas.drawString(2 * cm, h - 1.35 * cm, "CHURNGUARD")
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawRightString(w - 2 * cm, h - 1.35 * cm, "Project Dossier & Interview Prep")
    # footer
    canvas.line(2 * cm, 1.4 * cm, w - 2 * cm, 1.4 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(2 * cm, 1.05 * cm, AUTHOR)
    canvas.drawRightString(w - 2 * cm, 1.05 * cm, str(doc.page))
    canvas.restoreState()


doc = BaseDocTemplate(
    str(OUT), pagesize=A4,
    leftMargin=2 * cm, rightMargin=2 * cm,
    topMargin=2 * cm, bottomMargin=1.8 * cm,
    title="ChurnGuard — Project Dossier & Interview Prep",
    author=AUTHOR,
)
frame = Frame(doc.leftMargin, doc.bottomMargin,
              doc.width, doc.height, id="body")
doc.addPageTemplates([
    PageTemplate(id="Cover", frames=[frame], onPage=cover),
    PageTemplate(id="Content", frames=[frame], onPage=content_page),
])


# =========================================================================
# helpers for tables
# =========================================================================
def metric_cards():
    m = METRICS
    data = [
        (f"{m['roc_auc']:.2f}", "ROC-AUC"),
        (f"{m['brier']:.3f}", "Brier score"),
        (f"{m['recall']:.0%}", f"Recall @ {m['threshold']:.2f}"),
        (f"{int(m['cost_at_0.5'] - m['cost_at_threshold'])}", "Cost saved / 1k"),
    ]
    cells = [[Paragraph(n, CARDNUM) for n, _ in data],
             [Paragraph(lbl, CARDLBL) for _, lbl in data]]
    t = Table(cells, colWidths=[doc.width / 4.0] * 4)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_SOFT),
        ("TOPPADDING", (0, 0), (-1, 0), 11),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
        ("TOPPADDING", (0, 1), (-1, 1), 0),
        ("LINEAFTER", (0, 0), (-2, -1), 0.8, colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def results_table():
    m = METRICS
    rows = [
        ["Metric", "Value", "What it means"],
        ["ROC-AUC", f"{m['roc_auc']:.4f}",
         "Ranks a random churner above a random non-churner ~81% of the time"],
        ["PR-AUC", f"{m['pr_auc']:.4f}",
         f"Strong vs. the {m['churn_rate']:.0%} base churn rate"],
        ["Brier score", f"{m['brier']:.4f}",
         "Calibration quality of the probabilities (lower is better)"],
        ["Recall", f"{m['recall']:.2f}",
         f"Catches {m['recall']:.0%} of true churners at the chosen threshold"],
        ["Precision", f"{m['precision']:.2f}",
         f"Of flagged customers, {m['precision']:.0%} truly churn"],
        ["F1", f"{m['f1']:.2f}", "Harmonic mean of precision and recall"],
        ["Threshold", f"{m['threshold']:.2f}",
         "Chosen to minimise expected business cost (not 0.5)"],
        ["Expected cost",
         f"{int(m['cost_at_threshold'])} vs {int(m['cost_at_0.5'])}",
         "~37% lower than the naive 0.5 cutoff (cost_fn=5, cost_fp=1)"],
    ]
    t = Table(rows, colWidths=[2.8 * cm, 2.6 * cm, doc.width - 5.4 * cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (1, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 1), (1, -1), ACCENT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.6),
        ("TEXTCOLOR", (2, 1), (2, -1), BODY),
        ("FONTNAME", (2, 1), (2, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
        ("BOX", (0, 0), (-1, -1), 0.8, LINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
    ]))
    return t


def figure_grid():
    names = [
        ("roc_curve.png", "ROC curve"),
        ("calibration_curve.png", "Reliability / calibration"),
        ("cost_curve.png", "Cost vs. threshold"),
        ("feature_importance.png", "Permutation importance"),
    ]
    cw = doc.width / 2.0 - 4
    imgs, caps = [], []
    for fn, label in names:
        fp = FIG / fn
        if fp.exists():
            img = Image(str(fp))
            ar = img.imageHeight / float(img.imageWidth)
            img.drawWidth = cw
            img.drawHeight = cw * ar
            imgs.append(img)
            caps.append(Paragraph(label, SMALL))
    rows = []
    for i in range(0, len(imgs), 2):
        rows.append(imgs[i:i + 2])
        rows.append(caps[i:i + 2])
    t = Table(rows, colWidths=[doc.width / 2.0] * 2)
    t.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


# =========================================================================
# CONTENT
# =========================================================================
story.append(NextPageTemplate("Content"))
story.append(PageBreak())  # leave the cover page

# ---- 1. Executive summary
h1("1 &nbsp; What ChurnGuard Is")
p("<b>ChurnGuard</b> is a production-style machine-learning service that predicts "
  "which telecom customers are about to cancel their subscription, so a retention "
  "team can intervene <i>before</i> they leave. It is not a notebook — it is a packaged, "
  "tested, deployable service that covers the full ML lifecycle end to end.", LEAD)
gap(2)
story.append(metric_cards())
gap(8)
h3("The one-line pitch")
p("“I built an end-to-end churn-prediction service: a leak-free scikit-learn "
  "pipeline that is cross-validated, hyperparameter-tuned, probability-calibrated, "
  "and served behind a FastAPI REST API with per-prediction SHAP explanations, "
  "Docker packaging, a test suite, and CI.”")
h3("Why churn prediction?")
p("For a subscription business, keeping an existing customer is far cheaper than "
  "acquiring a new one. A model that ranks customers by churn risk lets a small "
  "retention budget be spent where it matters most. The project frames the problem "
  "the way a business actually experiences it — as a <b>cost trade-off</b> between "
  "chasing customers who were going to stay anyway (false positives) and missing "
  "customers who actually leave (false negatives).")

h2("The problem, precisely")
bullets([
    "<b>Task:</b> binary classification — will this customer churn? Output a "
    "calibrated probability, not just a yes/no.",
    "<b>Data:</b> ~7,043 Telco customers with ~21% base churn rate (class imbalance). "
    "Features include tenure, contract type, monthly/total charges, internet service, "
    "payment method and support add-ons.",
    "<b>Success metric:</b> ROC-AUC and PR-AUC for ranking quality, Brier score for "
    "probability calibration, and — crucially — <b>expected business cost</b> at the "
    "chosen decision threshold.",
])

# ---- 2. Architecture
h1("2 &nbsp; Architecture &amp; Data Flow")
p("Every stage is a small, single-responsibility module driven by one config file "
  "(<font face='Courier'>config/config.yaml</font>) — there are no magic numbers "
  "buried in the code.")
code([
    "config/config.yaml   (single source of truth)",
    "      |",
    "data.py -> features.py -> train.py ------------> evaluate.py",
    "(real or  (engineering +  (CV select ->          (threshold choice,",
    " synth)    ColumnTransformer) tune -> calibrate)   metrics, figures)",
    "      |",
    "      v",
    "models/churnguard.joblib  (ModelBundle: model + threshold + SHAP bg)",
    "      |",
    "      +--> explain.py (SHAP)   predict.py -> api.py",
    "                               (batch CLI)  (/health /metrics /stats /predict)",
    "      |",
    "      v",
    "Docker image -> Render / HF Spaces / anywhere",
])
h3("Key design choices")
bullets([
    "<b>No data leakage.</b> All scaling and encoding live inside a scikit-learn "
    "<font face='Courier'>Pipeline</font>, fitted on the training fold only and "
    "cross-validated as one unit.",
    "<b>Train/serve parity.</b> The same <font face='Courier'>engineer_features()</font> "
    "runs at training time and at request time, so the model never sees differently "
    "shaped inputs.",
    "<b>Self-describing artifact.</b> The saved <font face='Courier'>ModelBundle</font> "
    "carries the model, the business threshold, the feature schema and a SHAP background "
    "sample — the service is a single file.",
    "<b>Reproducible.</b> One seed flows through generation, splitting and modelling; "
    "<font face='Courier'>requirements.lock</font> pins the exact dependency tree.",
])

# ---- 3. Technical deep dive
h1("3 &nbsp; Technical Deep-Dive")

h2("3.1 &nbsp; Features &amp; preprocessing")
p("A <font face='Courier'>ColumnTransformer</font> applies "
  "<font face='Courier'>SimpleImputer + StandardScaler</font> to numeric columns and "
  "<font face='Courier'>SimpleImputer + OneHotEncoder(handle_unknown='ignore')</font> to "
  "categoricals. Wrapping it in a <font face='Courier'>Pipeline</font> guarantees the "
  "scaler's mean/variance are learned only from training data — the single most common "
  "source of leakage in tabular ML.")

h2("3.2 &nbsp; Model selection, tuning, calibration")
bullets([
    "<b>Compare:</b> logistic regression, random forest and gradient boosting are "
    "scored by 5-fold stratified cross-validated ROC-AUC. The winner here is "
    f"<b>logistic regression</b> (CV AUC {METRICS['cv_roc_auc']['logistic_regression']:.3f}).",
    "<b>Tune:</b> <font face='Courier'>RandomizedSearchCV</font> on the winner "
    f"(best <font face='Courier'>C={METRICS['best_params'].get('clf__C', 0):.3f}</font>).",
    "<b>Calibrate:</b> <font face='Courier'>CalibratedClassifierCV</font> (isotonic) so a "
    f"predicted “0.7” really means ~70% — Brier score {METRICS['brier']:.3f}.",
])

h2("3.3 &nbsp; The business threshold (the standout idea)")
p("The default 0.5 cutoff is arbitrary. ChurnGuard sweeps every threshold from 0.01 to "
  "0.99 and picks the one that <b>minimises expected cost</b>, given that a missed "
  "churner (false negative) costs 5× a wasted retention offer (false positive). "
  f"The chosen threshold of <b>{METRICS['threshold']:.2f}</b> cuts expected cost from "
  f"<b>{int(METRICS['cost_at_0.5'])}</b> to <b>{int(METRICS['cost_at_threshold'])}</b> per "
  "1,000 customers (~37% lower) versus the naive 0.5 cutoff. This is the detail that turns "
  "a model into a decision system.")

h2("3.4 &nbsp; Explainability (SHAP)")
p("Every prediction returns its top risk drivers. For the linear model, SHAP values have "
  "a closed form (<font face='Courier'>coef · (x − E[x])</font>); for tree models the "
  "code uses <font face='Courier'>shap.TreeExplainer</font>. Transformed one-hot columns "
  "are mapped back to business-readable names, so the API returns "
  "“<font face='Courier'>Contract = Month-to-month increases risk</font>”, not a "
  "cryptic encoded column.")

h2("3.5 &nbsp; Serving, MLOps &amp; quality")
bullets([
    "<b>FastAPI</b> service: <font face='Courier'>/</font> (interactive demo), "
    "<font face='Courier'>/health</font>, <font face='Courier'>/metrics</font>, "
    "<font face='Courier'>/stats</font>, <font face='Courier'>/predict</font> (with an "
    "<font face='Courier'>explain</font> flag).",
    "<b>Batch CLI:</b> "
    "<font face='Courier'>python -m churnguard.predict in.csv --out scored.csv</font>.",
    "<b>Docker</b> packaging with a container HEALTHCHECK; binds <font face='Courier'>$PORT</font> "
    "for Render / HF Spaces.",
    "<b>CI (GitHub Actions):</b> ruff + mypy + pytest with coverage on Python 3.10 and 3.12, "
    "plus an end-to-end train→evaluate smoke test.",
    "<b>Tracking:</b> best-effort MLflow logging of every training and evaluation run.",
    "<b>Quality gate:</b> 25 tests, ~89% coverage, clean ruff + mypy.",
])

# ---- 4. Results
h1("4 &nbsp; Results")
p(f"Trained on the Telco-style dataset ({METRICS['n_test']} held-out test customers, "
  f"{METRICS['churn_rate']:.0%} churn), the tuned and calibrated model reaches:")
gap(2)
story.append(results_table())
gap(10)
h3("Evaluation figures")
story.append(figure_grid())
p("Top churn drivers (permutation importance): " +
  ", ".join(f"<b>{d['feature']}</b>" for d in METRICS["top_drivers"]) +
  " — matching the known real-world Telco churn story (new, month-to-month, "
  "electronic-check customers churn most).", SMALL)

# ---- 5. Tech stack
h1("5 &nbsp; Tech Stack")
stack = [
    ["Layer", "Tools"],
    ["Language", "Python 3.10+"],
    ["ML / data", "scikit-learn, pandas, NumPy, SHAP"],
    ["Serving", "FastAPI, Uvicorn, Pydantic"],
    ["Packaging", "Docker, pip-tools (requirements.lock)"],
    ["Quality", "pytest + coverage, ruff, mypy, pre-commit"],
    ["Ops", "GitHub Actions CI, MLflow, Makefile"],
    ["Deploy", "Render blueprint / Hugging Face Spaces"],
]
t = Table(stack, colWidths=[3.4 * cm, doc.width - 3.4 * cm])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), INK),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
    ("TEXTCOLOR", (0, 1), (0, -1), ACCENT),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
    ("BOX", (0, 0), (-1, -1), 0.8, LINE),
    ("LINEBELOW", (0, 0), (-1, -1), 0.5, LINE),
    ("TOPPADDING", (0, 0), (-1, -1), 6),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ("LEFTPADDING", (0, 0), (-1, -1), 8),
]))
story.append(t)

# =========================================================================
# 6. INTERVIEW QUESTIONS
# =========================================================================
story.append(PageBreak())
h1("6 &nbsp; Interview Questions &amp; Answers")
p("Grouped from “walk me through your project” to deeper ML and system-design "
  "probes. Answers are written the way you would actually say them out loud.", LEAD)

h2("A. Project walkthrough")
qa("Walk me through this project end to end.",
   ["ChurnGuard predicts telecom customer churn. Raw data is feature-engineered, then "
    "fed into a scikit-learn Pipeline so preprocessing can't leak. I compare three models "
    "by cross-validated ROC-AUC, tune the winner with randomized search, calibrate its "
    "probabilities with isotonic regression, and choose a decision threshold that "
    "minimises business cost. The model is saved as a self-describing bundle and served "
    "behind a FastAPI API that also returns a SHAP explanation per prediction. It's "
    "Dockerised, tested, and runs through CI."])
qa("What was the hardest part?",
   ["Getting the evaluation honest. It's easy to report a high accuracy on a 21%-churn "
    "dataset — predicting ‘no churn’ for everyone scores 79%. So I moved to ROC-AUC, "
    "PR-AUC, calibration (Brier), and a cost-based threshold, which reflects what the "
    "business actually pays for mistakes."])

h2("B. Core ML concepts")
qa("Why is accuracy a bad metric here, and what did you use instead?",
   ["With 21% churn, a model that always predicts ‘stay’ is 79% accurate but useless. "
    "I use ROC-AUC and PR-AUC for ranking quality, Brier score for calibration, and "
    "expected cost at the operating threshold for the business decision."])
qa("What is data leakage and how did you prevent it?",
   ["Leakage is when information unavailable at prediction time sneaks into training — "
    "e.g. scaling with statistics computed over the whole dataset including the test "
    "rows. I prevent it by putting every transform inside a Pipeline that's fitted only "
    "on the training fold, and cross-validating the whole Pipeline as one unit."])
qa("What does probability calibration mean and why does it matter?",
   ["A calibrated model's 0.7 means ~70% of such customers actually churn. It matters "
    "because the business acts on the probability (expected-value decisions, ranking by "
    "risk), not just the label. I use CalibratedClassifierCV with isotonic regression and "
    "verify it with the Brier score and a reliability curve."])
qa("How did you choose the decision threshold?",
   ["Not 0.5. I assign costs — a missed churner costs 5x a wasted retention offer — then "
    "sweep thresholds and pick the one minimising total expected cost. Here that's 0.20, "
    "cutting cost by ~37% versus 0.5."])
qa("Precision vs. recall — which matters more here?",
   ["Recall, within reason. Missing a churner (false negative) is the expensive mistake, "
    "so the cost ratio pushes the threshold down to catch ~79% of churners, accepting "
    "lower precision. The exact balance falls out of the cost function rather than a gut "
    "call."])
qa("How do you handle class imbalance?",
   ["Stratified splits and stratified CV, threshold-based rather than 0.5 decisions, PR-AUC "
    "alongside ROC-AUC, and class weighting available in the estimators. I deliberately "
    "avoided blind oversampling, which can hurt calibration."])
qa("Why logistic regression over the tree ensembles?",
   ["It genuinely won on cross-validated AUC on this data, and it's faster, more "
    "interpretable, and calibrates cleanly — its SHAP values even have a closed form. On "
    "this dataset the linear signal dominates; I let CV decide rather than assuming a "
    "fancier model is better."])

h2("C. Explainability &amp; trust")
qa("How does SHAP work and how did you use it?",
   ["SHAP attributes a prediction to each feature using game-theory Shapley values — each "
    "feature's average marginal contribution. For my linear model that reduces to "
    "coef x (x - mean), which I compute directly; for trees I'd use TreeExplainer. I map "
    "the transformed columns back to business names so the API returns readable reasons."])
qa("A stakeholder doesn't trust the model. What do you show them?",
   ["The reliability curve (probabilities are honest), the global drivers (they match "
    "domain intuition — tenure, contract type, payment method), and a per-customer SHAP "
    "breakdown for a specific flagged account. Trust comes from the model agreeing with "
    "what they already know, then adding precision."])

h2("D. Engineering &amp; system design")
qa("How is the model served and deployed?",
   ["A FastAPI app loads the bundle once and exposes /predict with optional explanations, "
    "plus /health, /metrics and /stats. It's Dockerised with a healthcheck and binds "
    "$PORT so it drops straight onto Render or HF Spaces. A batch CLI reuses the exact "
    "same inference code for scoring CSVs."])
qa("How do you ensure the model behaves the same in training and production?",
   ["Train/serve parity: one engineer_features() function used in both paths, a fixed "
    "feature schema stored in the bundle, and reindexing of incoming records to that "
    "schema with handle_unknown='ignore' on the encoder so unseen categories don't crash."])
qa("What does your CI pipeline do?",
   ["On every push and PR it lints with ruff, type-checks with mypy, runs the test suite "
    "with coverage on Python 3.10 and 3.12, and runs a smoke test that trains and "
    "evaluates the full pipeline end to end and asserts the artifacts exist."])
qa("How would you monitor this model in production?",
   ["Track input-feature drift and prediction-distribution drift, watch the realised cost "
    "once outcomes are known, and periodically re-fit the threshold as the true cost ratio "
    "changes. The /stats endpoint logging risk-band mix is the first step toward that."])
qa("How would you scale or extend it?",
   ["Add a model registry with automated promotion on top of the MLflow runs, a feature "
    "store for parity at scale, batch scoring on a schedule, and a small dashboard over the "
    "scores. For throughput, the stateless API scales horizontally behind a load balancer."])

h2("E. Reflection")
qa("What would you do differently or add next?",
   ["Train on the full real Kaggle dataset in CI, add drift monitoring, and A/B test the "
    "retention actions the model triggers so I can measure realised ROI, not just offline "
    "cost. I'd also add online threshold re-tuning as the cost ratio shifts."])
qa("What did this project teach you?",
   ["That the modelling is the easy 20%. The value is in honest evaluation, calibration, a "
    "cost-aware decision, explainability, and the engineering around it — tests, CI, "
    "reproducibility and clean serving — that makes it something a team can actually run."])

gap(10)
story.append(HRFlowable(width="100%", thickness=0.8, color=LINE))
gap(4)
p(f"{AUTHOR} &nbsp;·&nbsp; {EMAIL} &nbsp;·&nbsp; {REPO}", SMALL)

doc.build(story)
print("wrote", OUT, OUT.stat().st_size, "bytes")
