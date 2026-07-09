"""
app.py
-------
Main Flask application for the AI-Powered Meeting Tone Analyzer.

Routes:
    GET  /                 -> landing page
    GET  /register         -> registration form
    POST /register         -> create account
    GET  /login             -> login form
    POST /login             -> authenticate
    GET  /logout            -> clear session
    GET  /dashboard         -> user's analysis history + quick stats
    GET  /analyze            -> new-analysis form
    POST /analyze            -> run AI engine on submitted transcript
    GET  /report/<id>        -> detailed report for one saved analysis
    POST /report/<id>/delete -> delete a saved analysis
    GET  /api/sample         -> returns a sample transcript (used by the UI "Try Sample" button)
"""

import json
import os
from functools import wraps

from flask import (
    Flask, render_template, request, redirect, url_for, session, flash, jsonify
)
from werkzeug.security import generate_password_hash, check_password_hash

import database as db
from ai_engine.analyzer import analyze_transcript

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

with app.app_context():
    db.init_db()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "error")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def current_user():
    if "user_id" in session:
        return db.get_user_by_id(session["user_id"])
    return None


@app.context_processor
def inject_user():
    return {"current_user": current_user()}


# ---------------------------------------------------------------------------
# Public pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "error")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "error")
            return render_template("register.html")

        if db.get_user_by_email(email):
            flash("An account with that email already exists.", "error")
            return render_template("register.html")

        db.create_user(name, email, generate_password_hash(password))
        flash("Account created successfully. Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = db.get_user_by_email(email)
        if user is None or not check_password_hash(user["password_hash"], password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session["user_id"] = user["id"]
        session["user_name"] = user["name"]
        flash(f"Welcome back, {user['name']}!", "success")
        return redirect(url_for("dashboard"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# App pages (require login)
# ---------------------------------------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard():
    analyses = db.get_analyses_for_user(session["user_id"])
    total = len(analyses)
    avg_health = round(sum(a["health_score"] for a in analyses) / total, 1) if total else 0
    at_risk = sum(1 for a in analyses if a["health_score"] < 35)
    return render_template(
        "dashboard.html", analyses=analyses, total=total,
        avg_health=avg_health, at_risk=at_risk,
    )


SAMPLE_TRANSCRIPT = """Alex: Good morning everyone, thanks for joining on short notice.
Priya: Morning! Happy to be here, excited to walk through the launch plan.
Alex: Great. So, unfortunately we are behind schedule on the payments module.
Rohit: Yeah, I'm pretty frustrated about that, this is the second delay this month.
Priya: I understand the frustration, but I think we can still make it work if we reprioritise.
Alex: This is completely unacceptable, we promised the client this would ship Friday.
Priya: Let's stay calm, maybe we push the non critical features to next sprint.
Rohit: I'm not sure that's enough, I'm honestly worried we'll miss it again.
Alex: Thank you Priya for the suggestion, let's try that approach.
Priya: Great, I'll send an updated plan by end of day, appreciate everyone's patience.
Rohit: Sounds good, thanks team, let's regroup tomorrow morning."""


@app.route("/api/sample")
def api_sample():
    return jsonify({"transcript": SAMPLE_TRANSCRIPT})


@app.route("/analyze", methods=["GET", "POST"])
@login_required
def analyze():
    if request.method == "POST":
        title = request.form.get("title", "").strip() or "Untitled Meeting"
        transcript = request.form.get("transcript", "").strip()

        uploaded = request.files.get("transcript_file")
        if uploaded and uploaded.filename:
            try:
                transcript = uploaded.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                flash("Could not read the uploaded file. Please upload a plain .txt file.", "error")
                return render_template("analyze.html")

        if not transcript:
            flash("Please paste a transcript or upload a .txt file.", "error")
            return render_template("analyze.html")

        try:
            report = analyze_transcript(transcript)
        except ValueError as e:
            flash(str(e), "error")
            return render_template("analyze.html")

        analysis_id = db.save_analysis(
            user_id=session["user_id"],
            title=title,
            transcript=transcript,
            overall_label=report["overall_label"],
            health_score=report["health_score"],
            tone_counts_json=json.dumps(report["tone_counts"]),
            report_json=json.dumps(report),
        )
        return redirect(url_for("view_report", analysis_id=analysis_id))

    return render_template("analyze.html")


@app.route("/report/<int:analysis_id>")
@login_required
def view_report(analysis_id):
    row = db.get_analysis(analysis_id, session["user_id"])
    if row is None:
        flash("Analysis not found.", "error")
        return redirect(url_for("dashboard"))
    report = json.loads(row["report_json"])
    return render_template("report.html", analysis=row, report=report)


@app.route("/report/<int:analysis_id>/delete", methods=["POST"])
@login_required
def delete_report(analysis_id):
    db.delete_analysis(analysis_id, session["user_id"])
    flash("Analysis deleted.", "success")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
