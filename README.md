# ToneScope — AI-Powered Meeting Tone Analyzer

A complete full-stack web application built with **Python (Flask) + SQLite**
for the CAC524A Artificial Intelligence and Machine Learning capstone mini
project. It analyzes meeting transcripts and detects tone (positive,
neutral, negative, aggressive, anxious) sentence-by-sentence and
speaker-by-speaker using a **hybrid AI engine built entirely in-house**
(no external paid AI API required to run the core feature).

---

## 1. Tech Stack (all free-tier / open source)

| Layer          | Technology                                              |
|-----------------|-----------------------------------------------------------|
| Frontend        | Server-rendered Jinja2 templates, hand-written CSS, Chart.js (CDN) |
| Backend         | Python 3, Flask 3                                        |
| Database        | SQLite (built into Python, zero setup)                   |
| AI / ML Engine  | scikit-learn (TF-IDF + Multinomial Naive Bayes) + a hand-built sentiment lexicon (rule-based NLP) |
| Auth            | Session-based auth, passwords hashed with Werkzeug's `generate_password_hash` (PBKDF2) |
| Hosting options | Render.com free tier / PythonAnywhere free tier / Railway free tier (see Section 5) |

No OpenAI/Anthropic/Google API key is required — the tone-detection model
is trained and runs **locally**, which keeps the whole app inside
free-tier limits and makes it fully offline-capable.

---

## 2. How the AI Engine Works

This project intentionally implements **two independent, explainable AI
models** and combines their outputs (a simple ensemble), rather than
calling a black-box external LLM:

1. **`ai_engine/tone_classifier.py`** — a **Multinomial Naive Bayes**
   classifier (scikit-learn), trained on a hand-labeled dataset of 180+
   meeting utterances (`ai_engine/training_data.py`) across five tone
   classes: `positive`, `neutral`, `negative`, `aggressive`, `anxious`.
   Text is vectorised with `TfidfVectorizer` (unigrams + bigrams).
   The trained model is cached to `instance/model.pkl` after first run.

2. **`ai_engine/lexicon.py`** — a hand-built, VADER-style sentiment
   lexicon (200+ scored words) with negation handling ("not good" flips
   polarity) and intensifier handling ("very frustrating" amplifies it).

3. **`ai_engine/analyzer.py`** — the orchestration layer:
   - Splits the transcript into sentences, optionally parsing
     `Speaker: text` formatted lines for per-speaker analysis.
   - Runs both models on every sentence and **averages their signal**
     into one combined score (a transparent, explainable ensemble).
   - Adds meeting-specific heuristics: filler-word ratio, politeness
     markers, aggression markers, hedging language, and question ratio.
   - Produces a single **0–100 Meeting Health Score** plus a full
     sentence-level and speaker-level breakdown.

Run `python -m ai_engine.tone_classifier` for a quick standalone sanity
check of the classifier.

---

## 3. Project Structure

```
meeting_tone_analyzer/
├── app.py                    # Flask app & routes
├── database.py                # SQLite data access layer
├── requirements.txt
├── ai_engine/
│   ├── lexicon.py              # hand-built sentiment lexicon
│   ├── training_data.py        # labeled dataset for the classifier
│   ├── tone_classifier.py      # TF-IDF + Naive Bayes model
│   └── analyzer.py             # combines both models + heuristics
├── templates/                 # Jinja2 HTML templates (modern dark UI)
│   ├── base.html, index.html, login.html, register.html,
│   ├── dashboard.html, analyze.html, report.html
├── static/
│   └── css/style.css           # design system (no framework — hand-written)
└── instance/                  # created at runtime: app.db, model.pkl
```

---

## 4. Running Locally

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py

# 4. Open in your browser
http://127.0.0.1:5000
```

On first run, Flask will automatically:
- Create `instance/app.db` (SQLite database with `users` and `analyses` tables)
- Train the Naive Bayes classifier and cache it to `instance/model.pkl`

Sign up for a free account, then go to **New Analysis** and click
**"Load a sample transcript"** to try it instantly, or paste/upload your
own meeting transcript (`.txt`).

---

## 5. Free Deployment Options

Any of these free tiers work well for this Flask + SQLite app:

- **Render.com** (Free Web Service): connect your GitHub repo, set the
  start command to `gunicorn app:app`, add `gunicorn` to
  `requirements.txt`.
- **PythonAnywhere** (Free tier): upload the project, configure a Flask
  WSGI app pointing to `app.app`.
- **Railway.app** (Free trial credits): connect repo, Railway
  auto-detects Flask; set `PORT` env var usage in `app.run()` if needed.

> Note: SQLite is file-based, so on ephemeral-filesystem free hosts the
> database resets on redeploy. For the assignment's "database
> integration" requirement this is acceptable and clearly documented;
> for persistent production use, swap in a managed free-tier Postgres
> (e.g. Supabase/Neon free tier) by replacing `database.py`.

---

## 6. Sample AI Prompts Used During Development (for the report)

This project was built using AI-assisted ("Vibe Coding") development.
Example prompts used during the build (for the assignment's required
"Prompt Samples / AI Usage Description" section):

- "Design a hybrid tone-analysis algorithm that combines a Naive Bayes
  classifier with a rule-based sentiment lexicon for meeting transcripts."
- "Build a labeled training dataset of meeting sentences across
  positive / neutral / negative / aggressive / anxious tone classes."
- "Write a Flask + SQLite backend with session-based auth (register,
  login, logout) using Werkzeug password hashing."
- "Create a dark, distinctive UI design themed around an audio waveform
  / mixing-console metaphor for a meeting analytics dashboard."
- "Add a per-speaker breakdown table and a Chart.js doughnut chart of
  tone distribution to the report page."

Students should replace/extend this section with their **own** actual
prompt history and describe what they personally customized,
understood, and could explain during evaluation (per Instruction 16).

---

## 7. Customization Ideas (for the "Future Enhancements" report section)

- Expand `training_data.py` with more labeled examples for higher
  classifier accuracy.
- Add audio upload + speech-to-text (e.g. free-tier Whisper API) to
  analyze live recorded meetings, not just typed transcripts.
- Add email/Slack digest of Meeting Health Score after each analysis.
- Add role-based views (e.g. manager dashboard aggregating team trends).
