"""
analyzer.py
------------
The main entry point of the AI engine. Combines:

  1. The rule-based lexicon/VADER-style sentiment scorer (lexicon.py)
  2. The trained Naive Bayes tone classifier (tone_classifier.py)
  3. Meeting-specific heuristic features (fillers, politeness,
     aggression markers, hedging, question ratio, etc.)

...into a single `analyze_transcript()` function that produces a
structured report: overall tone, meeting health score, per-sentence
breakdown, per-speaker breakdown, and behavioural statistics.

Design note: two independent models are combined by AVERAGING their
positive/negative signal (a simple, transparent form of ensembling)
so that a single model's blind spot doesn't dominate the verdict --
e.g. the lexicon alone can be fooled by sarcasm-free factual negative
words ("delay", "issue") that the classifier's context-aware n-grams
handle better, while the classifier alone can struggle on very short
or unusual sentences that the lexicon scores robustly.
"""

import re

from ai_engine.lexicon import (
    SENTIMENT_LEXICON, NEGATIONS, INTENSIFIERS, FILLER_WORDS,
    POLITENESS_WORDS, AGGRESSION_WORDS, HEDGING_WORDS, RESOLUTION_WORDS,
)
from ai_engine.tone_classifier import ToneClassifier

_classifier = ToneClassifier()

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
SPEAKER_LINE_RE = re.compile(r"^\s*([A-Za-z][A-Za-z .'-]{1,30}):\s*(.+)$")
WORD_RE = re.compile(r"[a-zA-Z']+")


# ---------------------------------------------------------------------------
# Lexicon-based compound sentiment score for one sentence (-1..+1)
# ---------------------------------------------------------------------------
def lexicon_score(sentence: str) -> float:
    tokens = WORD_RE.findall(sentence.lower())
    if not tokens:
        return 0.0

    total = 0.0
    count = 0
    intensity = 1.0
    negate = False

    for tok in tokens:
        if tok in NEGATIONS:
            negate = True
            continue
        if tok in INTENSIFIERS:
            intensity = INTENSIFIERS[tok]
            continue

        if tok in SENTIMENT_LEXICON:
            score = SENTIMENT_LEXICON[tok] * intensity
            if negate:
                score *= -0.85
            total += score
            count += 1

        # reset modifiers after they've had a chance to apply
        intensity = 1.0
        negate = False

    if count == 0:
        return 0.0

    avg = total / count
    # normalise roughly into -1..1 range (lexicon scores go up to ~4)
    normalised = max(-1.0, min(1.0, avg / 3.2))
    return round(normalised, 3)


# ---------------------------------------------------------------------------
# Heuristic behavioural features for one sentence
# ---------------------------------------------------------------------------
def sentence_features(sentence: str) -> dict:
    lower = sentence.lower()
    words = WORD_RE.findall(lower)
    word_count = len(words) or 1

    def count_phrases(vocab):
        return sum(lower.count(term) for term in vocab)

    fillers = count_phrases(FILLER_WORDS)
    politeness = count_phrases(POLITENESS_WORDS)
    aggression = count_phrases(AGGRESSION_WORDS)
    hedging = count_phrases(HEDGING_WORDS)
    resolution = count_phrases(RESOLUTION_WORDS)
    is_question = sentence.strip().endswith("?")
    is_exclaim = sentence.strip().endswith("!")

    return {
        "word_count": word_count,
        "filler_count": fillers,
        "politeness_count": politeness,
        "aggression_count": aggression,
        "hedging_count": hedging,
        "resolution_count": resolution,
        "is_question": is_question,
        "is_exclaim": is_exclaim,
    }


LABEL_TO_POLARITY = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -0.6,
    "aggressive": -1.0,
    "anxious": -0.35,
}


def analyze_sentence(sentence: str) -> dict:
    sentence = sentence.strip()
    lex_score = lexicon_score(sentence)
    ml_result = _classifier.predict_proba(sentence)
    ml_label = max(ml_result, key=ml_result.get)
    ml_confidence = ml_result[ml_label]
    ml_polarity = LABEL_TO_POLARITY.get(ml_label, 0.0) * ml_confidence

    # Ensemble: average the two independent signals
    combined = round((lex_score + ml_polarity) / 2, 3)

    feats = sentence_features(sentence)

    # Nudge combined score using strong behavioural signals
    if feats["aggression_count"] > 0:
        combined = min(combined, -0.4)
    if feats["politeness_count"] > 0 and combined > -0.3:
        combined += 0.05
    if feats["resolution_count"] > 0:
        # A sentence proposing/confirming a fix ("let's finalize the schema",
        # "I'll take that on") should read as constructive even if it still
        # mentions the underlying problem word (e.g. "issue", "blocker").
        combined += 0.25

    combined = max(-1.0, min(1.0, combined))

    if combined >= 0.35:
        display_tone = "positive"
    elif combined <= -0.55 or feats["aggression_count"] > 0:
        display_tone = "aggressive" if feats["aggression_count"] > 0 else "negative"
    elif combined <= -0.15:
        display_tone = "negative"
    elif ml_label == "anxious" and ml_confidence > 0.35:
        display_tone = "anxious"
    else:
        display_tone = "neutral"

    return {
        "text": sentence,
        "lexicon_score": lex_score,
        "ml_label": ml_label,
        "ml_confidence": round(ml_confidence, 3),
        "ml_distribution": {k: round(v, 3) for k, v in ml_result.items()},
        "combined_score": combined,
        "tone": display_tone,
        "features": feats,
    }


# ---------------------------------------------------------------------------
# Speaker parsing
# ---------------------------------------------------------------------------
def parse_speakers(transcript: str):
    """Splits transcript into (speaker, sentence) parsed lines.
    Supports 'Speaker: text' format. Falls back to speaker='Unknown'
    if no speaker labels are found."""
    lines = [l for l in transcript.split("\n") if l.strip()]
    parsed = []
    any_speaker = False
    for line in lines:
        m = SPEAKER_LINE_RE.match(line)
        if m:
            any_speaker = True
            speaker, text = m.group(1).strip(), m.group(2).strip()
        else:
            speaker, text = "Unknown", line.strip()
        # split this line's text into sentences too
        for sent in SENTENCE_SPLIT_RE.split(text):
            sent = sent.strip()
            if sent:
                parsed.append((speaker, sent))
    return parsed, any_speaker


# ---------------------------------------------------------------------------
# Full transcript analysis
# ---------------------------------------------------------------------------
def analyze_transcript(transcript: str) -> dict:
    transcript = transcript.strip()
    if not transcript:
        raise ValueError("Transcript is empty.")

    parsed, has_speakers = parse_speakers(transcript)

    sentence_reports = []
    for speaker, sent in parsed:
        report = analyze_sentence(sent)
        report["speaker"] = speaker
        sentence_reports.append(report)

    if not sentence_reports:
        raise ValueError("No analyzable sentences found in transcript.")

    n = len(sentence_reports)

    # -----------------------------------------------------------------
    # Recency-weighted average sentiment.
    # A meeting that raises problems early but resolves them by the end
    # should score better than one that starts fine and ends badly --
    # later sentences carry more weight (linear ramp from 1.0x to 2.0x),
    # mirroring how a human (or an LLM reading the whole transcript)
    # would judge the overall trajectory rather than a flat word count.
    # -----------------------------------------------------------------
    if n > 1:
        weights = [1.0 + (i / (n - 1)) for i in range(n)]
    else:
        weights = [1.0]
    weighted_sum = sum(r["combined_score"] * w for r, w in zip(sentence_reports, weights))
    avg_score = weighted_sum / sum(weights)
    unweighted_avg = sum(r["combined_score"] for r in sentence_reports) / n

    tone_counts = {}
    for r in sentence_reports:
        tone_counts[r["tone"]] = tone_counts.get(r["tone"], 0) + 1

    total_words = sum(r["features"]["word_count"] for r in sentence_reports)
    total_fillers = sum(r["features"]["filler_count"] for r in sentence_reports)
    total_politeness = sum(r["features"]["politeness_count"] for r in sentence_reports)
    total_aggression = sum(r["features"]["aggression_count"] for r in sentence_reports)
    total_hedging = sum(r["features"]["hedging_count"] for r in sentence_reports)
    total_resolution = sum(r["features"]["resolution_count"] for r in sentence_reports)
    total_questions = sum(1 for r in sentence_reports if r["features"]["is_question"])

    # -----------------------------------------------------------------
    # Meeting Health Score (0-100): a composite index.
    #
    # NOTE: aggression is already reflected inside `combined_score` for
    # each sentence (see analyze_sentence's aggression nudge), so it is
    # only lightly re-weighted here -- not double-penalized -- and a
    # resolution bonus offsets problems that were explicitly closed out.
    # -----------------------------------------------------------------
    base = (avg_score + 1) / 2 * 100  # map -1..1 -> 0..100
    filler_penalty = min(10, (total_fillers / max(total_words, 1)) * 300)
    aggression_penalty = min(15, total_aggression * 3)
    politeness_bonus = min(8, total_politeness * 1.5)
    hedging_penalty = min(8, (total_hedging / n) * 10)
    resolution_bonus = min(15, total_resolution * 3)

    health_score = (
        base - filler_penalty - aggression_penalty
        + politeness_bonus - hedging_penalty + resolution_bonus
    )
    health_score = round(max(0, min(100, health_score)), 1)

    if health_score >= 75:
        overall_label = "Healthy & Collaborative"
    elif health_score >= 55:
        overall_label = "Generally Constructive"
    elif health_score >= 35:
        overall_label = "Tense / Needs Attention"
    else:
        overall_label = "Highly Negative / At Risk"

    # -----------------------------------------------------------------
    # Per-speaker breakdown
    # -----------------------------------------------------------------
    speaker_stats = {}
    if has_speakers:
        for r in sentence_reports:
            sp = r["speaker"]
            s = speaker_stats.setdefault(sp, {
                "sentence_count": 0, "score_sum": 0.0, "tone_counts": {},
                "aggression_count": 0, "filler_count": 0, "word_count": 0,
            })
            s["sentence_count"] += 1
            s["score_sum"] += r["combined_score"]
            s["tone_counts"][r["tone"]] = s["tone_counts"].get(r["tone"], 0) + 1
            s["aggression_count"] += r["features"]["aggression_count"]
            s["filler_count"] += r["features"]["filler_count"]
            s["word_count"] += r["features"]["word_count"]
        for sp, s in speaker_stats.items():
            s["avg_score"] = round(s["score_sum"] / s["sentence_count"], 3)
            del s["score_sum"]

    return {
        "overall_score": round(avg_score, 3),
        "unweighted_score": round(unweighted_avg, 3),
        "health_score": health_score,
        "overall_label": overall_label,
        "tone_counts": tone_counts,
        "sentence_count": n,
        "total_words": total_words,
        "total_fillers": total_fillers,
        "total_politeness": total_politeness,
        "total_aggression": total_aggression,
        "total_hedging": total_hedging,
        "total_resolution": total_resolution,
        "total_questions": total_questions,
        "has_speakers": has_speakers,
        "speaker_stats": speaker_stats,
        "sentences": sentence_reports,
    }
