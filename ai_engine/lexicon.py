"""
lexicon.py
-----------
A hand-built, meeting-domain sentiment lexicon used by the rule-based
half of the Tone Analyzer's hybrid AI engine.

The scoring approach is a simplified version of the VADER
(Valence Aware Dictionary and sEntiment Reasoner) methodology:
    1. Each word carries a base valence score from -4 (very negative)
       to +4 (very positive).
    2. Negation words flip / dampen the score of the following word.
    3. Intensifiers boost or reduce the magnitude of the next word's score.
    4. Meeting-specific word groups (fillers, politeness, aggression,
       hedging/uncertainty) are tracked separately to build behavioural
       features that a plain sentiment score cannot capture.

This is a genuine, deterministic NLP algorithm (not a call to an
external AI API) -- it is one of the two models that make up the
hybrid "AI engine" of this project, the other being the trained
Naive Bayes tone classifier in tone_classifier.py
"""

# ---------------------------------------------------------------------------
# Core valence lexicon (word -> score in range -4..+4)
# ---------------------------------------------------------------------------
SENTIMENT_LEXICON = {
    # Strong positive
    "excellent": 3.5, "outstanding": 3.6, "fantastic": 3.4, "amazing": 3.4,
    "brilliant": 3.3, "wonderful": 3.2, "great": 2.8, "awesome": 3.2,
    "love": 3.0, "perfect": 3.3, "impressive": 2.9, "delighted": 3.1,
    "thrilled": 3.2, "proud": 2.6, "appreciate": 2.5, "appreciated": 2.5,
    "thank": 2.0, "thanks": 2.0, "grateful": 2.7, "congratulations": 3.0,
    "success": 2.4, "successful": 2.4, "win": 2.2, "winning": 2.2,
    "agree": 1.6, "agreed": 1.6, "good": 2.0, "nice": 1.8, "well": 1.2,
    "happy": 2.4, "glad": 2.0, "positive": 2.0, "confident": 2.0,
    "clear": 1.4, "helpful": 2.0, "efficient": 1.8, "smooth": 1.6,
    "collaborate": 1.6, "collaboration": 1.6, "support": 1.5, "resolved": 2.0,
    "improved": 1.8, "improvement": 1.6, "innovative": 2.0, "opportunity": 1.6,

    # Mild positive
    "okay": 0.6, "ok": 0.6, "fine": 0.7, "reasonable": 0.8, "useful": 1.2,
    "progress": 1.4, "on track": 1.4, "aligned": 1.2, "welcome": 1.4,

    # Neutral / informative (score 0, tracked for completeness)
    "meeting": 0.0, "update": 0.0, "report": 0.0, "schedule": 0.0,
    "discuss": 0.0, "review": 0.0, "plan": 0.0, "task": 0.0,

    # Mild negative
    "delay": -1.6, "delayed": -1.6, "issue": -1.4, "issues": -1.4,
    "concern": -1.5, "concerned": -1.6, "problem": -1.8, "problems": -1.8,
    "confused": -1.6, "confusing": -1.6, "difficult": -1.6, "hard": -1.0,
    "behind": -1.4, "missed": -1.6, "miss": -1.2, "risk": -1.4,
    "risky": -1.6, "unclear": -1.4, "slow": -1.2, "blocked": -1.8,
    "blocker": -1.8, "disagree": -1.8, "disagreed": -1.8, "doubt": -1.4,
    "doubtful": -1.6, "worried": -1.8, "worry": -1.6, "uncertain": -1.4,

    # Strong negative
    "fail": -2.8, "failed": -2.8, "failure": -2.9, "terrible": -3.2,
    "horrible": -3.3, "awful": -3.1, "unacceptable": -3.4, "disaster": -3.4,
    "disappointing": -2.6, "disappointed": -2.6, "frustrated": -2.7,
    "frustrating": -2.7, "angry": -2.9, "upset": -2.4, "waste": -2.4,
    "useless": -2.6, "broken": -2.2, "wrong": -2.0, "mistake": -2.0,
    "chaos": -2.8, "mess": -2.2, "incompetent": -3.0, "sloppy": -2.4,

    # Aggressive / confrontational (very strong negative + flagged separately)
    "ridiculous": -3.0, "stupid": -3.2, "pathetic": -3.2, "useless.": -2.6,
    "unbelievable": -2.4, "outrageous": -3.0, "demand": -2.0, "demanding": -2.0,
    "insist": -1.6, "must": -0.6, "immediately": -0.8, "never": -1.2,
    "always wrong": -2.0, "blame": -2.4, "fault": -2.0, "shut": -2.6,
}

# ---------------------------------------------------------------------------
# Negation words: flip the polarity of the following sentiment word
# ---------------------------------------------------------------------------
NEGATIONS = {
    "not", "no", "never", "none", "nobody", "nothing", "neither",
    "cannot", "can't", "cant", "won't", "wont", "isn't", "isnt",
    "doesn't", "doesnt", "didn't", "didnt", "don't", "dont", "aren't",
    "arent", "wasn't", "wasnt", "weren't", "werent", "hardly", "barely",
    "without",
}

# ---------------------------------------------------------------------------
# Intensifiers: scale the magnitude of the following sentiment word
# ---------------------------------------------------------------------------
INTENSIFIERS = {
    "very": 1.5, "extremely": 1.8, "really": 1.4, "so": 1.3,
    "absolutely": 1.7, "completely": 1.6, "totally": 1.5, "highly": 1.5,
    "incredibly": 1.7, "seriously": 1.4, "quite": 1.2, "somewhat": 0.7,
    "slightly": 0.6, "a bit": 0.6, "a little": 0.6, "barely": 0.5,
}

# ---------------------------------------------------------------------------
# Meeting-behaviour word groups (used for heuristic features)
# ---------------------------------------------------------------------------
FILLER_WORDS = {
    "um", "uh", "erm", "like", "you know", "sort of", "kind of",
    "basically", "actually", "literally", "i mean", "well um",
}

POLITENESS_WORDS = {
    "please", "thank you", "thanks", "appreciate", "kindly",
    "would you mind", "if possible", "sorry", "apologies", "excuse me",
}

AGGRESSION_WORDS = {
    "ridiculous", "stupid", "pathetic", "unacceptable", "outrageous",
    "shut", "demand", "blame", "fault", "incompetent", "useless",
    "never listen", "always wrong", "waste of time", "shut up",
}

HEDGING_WORDS = {
    "maybe", "perhaps", "possibly", "i think", "i guess", "not sure",
    "i suppose", "kind of", "sort of", "probably", "might", "could be",
}

QUESTION_MARKERS = {"what", "why", "how", "when", "where", "who", "could", "can", "should", "would"}

# ---------------------------------------------------------------------------
# Resolution / closure language: phrases that signal an issue was settled,
# agreed upon, or assigned an owner. A meeting can raise several problems
# and still be "healthy" if it closes them out -- this vocabulary lets the
# analyzer recognise that outcome instead of only counting raw negative
# words. This is what lets the engine reward "we found 6 issues and agreed
# on fixes for all of them" rather than treating it the same as "we found
# 6 issues and left them unresolved."
# ---------------------------------------------------------------------------
RESOLUTION_WORDS = {
    "agreed", "agree", "resolved", "resolve", "confirmed", "confirm",
    "finalized", "finalised", "committed", "commit", "sounds good",
    "makes sense", "locked in", "signed off", "sign off", "aligned",
    "will fix", "will take", "will handle", "will own", "on it",
    "let's do that", "sounds like a plan", "approved", "settled",
    "decided", "decision", "closed", "done deal", "works for me",
    "happy with that", "good to go", "moving forward with",
}

