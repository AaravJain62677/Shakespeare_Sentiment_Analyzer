import re
import nltk

# ── Download required NLTK data (only on first run) ─────────────────────────
REQUIRED = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger",
    "averaged_perceptron_tagger_eng",
    "wordnet",
    "sentiwordnet",
    "stopwords",
    "omw-1.4",
]

print("Checking / downloading NLTK data...\n")
for pkg in REQUIRED:
    nltk.download(pkg, quiet=True)

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords, sentiwordnet as swn, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# ── Shakespeare Corpus ───────────────────────────────────────────────────────
# Mix of plays and sonnets covering positive, negative, and neutral tones

SHAKESPEARE_PASSAGES = {
    "Sonnet 18":
        "Shall I compare thee to a summer's day? "
        "Thou art more lovely and more temperate. "
        "Rough winds do shake the darling buds of May, "
        "And summer's lease hath all too short a date.",

    "Romeo and Juliet - Balcony":
        "But soft, what light through yonder window breaks? "
        "It is the east, and Juliet is the sun. "
        "Arise, fair sun, and kill the envious moon, "
        "Who is already sick and pale with grief.",

    "Sonnet 73":
        "That time of year thou mayst in me behold "
        "When yellow leaves, or none, or few, do hang "
        "Upon those boughs which shake against the cold, "
        "Bare ruined choirs, where late the sweet birds sang.",
}

# ── Helpers ──────────────────────────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


# ── penn_to_wordnet(tag) ─────────────────────────────────────────────────────
# Converts Penn Treebank POS tags (returned by NLTK) to WordNet POS format
# (required by SentiWordNet). E.g. JJ -> ADJ, VB -> VERB, NN -> NOUN, RB -> ADV.
def penn_to_wordnet(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return None


# ── preprocess(text) ─────────────────────────────────────────────────────────
# Full cleaning pipeline for a raw sentence string.
# Steps: strip apostrophe contractions -> tokenize -> lowercase -> remove stopwords
# -> POS tag -> lemmatize using POS context.
# Returns a list of (lemma, wordnet_pos) tuples ready for scoring.
def preprocess(text):
    # Remove archaic apostrophe contractions like 'tis, 'twixt
    text = re.sub(r"'[a-z]+", "", text, flags=re.IGNORECASE)

    tokens = word_tokenize(text)

    # Keep only alphabetic tokens, lowercase, remove stopwords
    tokens = [t.lower() for t in tokens if t.isalpha() and t.lower() not in stop_words]

    if not tokens:
        return []

    # POS tag
    tagged = pos_tag(tokens)

    # Lemmatize using POS
    result = []
    for word, tag in tagged:
        wn_pos = penn_to_wordnet(tag)
        if wn_pos:
            lemma = lemmatizer.lemmatize(word, pos=wn_pos)
            result.append((lemma, wn_pos))
        else:
            lemma = lemmatizer.lemmatize(word)
            result.append((lemma, wordnet.NOUN))  # default to noun

    return result


# ── get_sentiment_score(lemma, pos) ──────────────────────────────────────────
# Looks up a word (lemma + POS) in SentiWordNet.
# A word can have multiple synsets (meanings), each with pos_score & neg_score.
# Computes (pos_score - neg_score) per synset, averages across all synsets.
# Returns a float: positive -> leans positive, negative -> leans negative, 0 -> neutral.
def get_sentiment_score(lemma, pos):
    synsets = list(swn.senti_synsets(lemma, pos))
    if not synsets:
        return 0.0

    scores = [s.pos_score() - s.neg_score() for s in synsets]
    return sum(scores) / len(scores)


# ── classify(score) ───────────────────────────────────────────────────────────
# Applies threshold logic to a sentiment score.
# score > +0.05 -> POSITIVE | score < -0.05 -> NEGATIVE | else -> NEUTRAL
def classify(score):
    if score > 0.05:
        return "POSITIVE"
    elif score < -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"


# ── analyze_passage(title, text) ─────────────────────────────────────────────
# Main pipeline orchestrator for one Shakespeare passage.
# Splits into sentences -> preprocesses each -> scores each -> averages into
# a passage-level score. Returns a structured dict with all results.
def analyze_passage(title, text):
    sentences = sent_tokenize(text)
    sentence_results = []

    for sent in sentences:
        tokens = preprocess(sent)
        if not tokens:
            sentence_results.append({
                "sentence": sent.strip(),
                "score": 0.0,
                "label": "NEUTRAL",
                "scored_words": []
            })
            continue

        scored_words = []
        for lemma, pos in tokens:
            score = get_sentiment_score(lemma, pos)
            if score != 0.0:
                scored_words.append((lemma, round(score, 3)))

        total_score = sum(s for _, s in scored_words) / max(len(tokens), 1)
        sentence_results.append({
            "sentence": sent.strip(),
            "score": round(total_score, 4),
            "label": classify(total_score),
            "scored_words": scored_words
        })

    # Passage-level score = average of sentence scores
    passage_score = sum(s["score"] for s in sentence_results) / max(len(sentence_results), 1)

    return {
        "title": title,
        "passage_score": round(passage_score, 4),
        "passage_label": classify(passage_score),
        "sentences": sentence_results
    }


# ── print_results(results) ───────────────────────────────────────────────────
# Formats and prints all results - per-sentence labels, scores, key words,
# and a final summary table with overall counts.
def print_results(results):
    LABEL_ICON = {"POSITIVE": "[+]", "NEGATIVE": "[-]", "NEUTRAL": "[O]"}
    DIVIDER = "=" * 70
    SUBDIV  = "-" * 70

    print("\n" + DIVIDER)
    print("  SHAKESPEARE SENTIMENT ANALYSIS - RESULTS")
    print(DIVIDER)

    for r in results:
        icon = LABEL_ICON[r["passage_label"]]
        print(f"\n  {r['title']}")
        print(f"    Overall -> {icon} {r['passage_label']}  (score: {r['passage_score']:+.4f})")
        print(SUBDIV)

        for i, s in enumerate(r["sentences"], 1):
            icon_s = LABEL_ICON[s["label"]]
            print(f"  [{i}] \"{s['sentence']}\"")
            print(f"       {icon_s} {s['label']}  (score: {s['score']:+.4f})")
            if s["scored_words"]:
                words_str = ", ".join(f"{w}({sc:+.2f})" for w, sc in s["scored_words"])
                print(f"       Key words: {words_str}")
            print()

    # Summary table
    print(DIVIDER)
    print(f"  {'PASSAGE':<40} {'LABEL':<12} {'SCORE':>8}")
    print(SUBDIV)
    for r in results:
        print(f"  {r['title']:<40} {r['passage_label']:<12} {r['passage_score']:>+8.4f}")
    print(DIVIDER)

    counts = {"POSITIVE": 0, "NEGATIVE": 0, "NEUTRAL": 0}
    for r in results:
        counts[r["passage_label"]] += 1
    total = len(results)
    print(f"\n  Total passages: {total}")
    for label, cnt in counts.items():
        print(f"  {LABEL_ICON[label]} {label}: {cnt}  ({cnt/total*100:.0f}%)")
    print(DIVIDER + "\n")


# ── main() ────────────────────────────────────────────────────────────────────
# Entry point. Loops through all Shakespeare passages, calls analyze_passage
# on each, collects results, then passes them to print_results.
def main():
    print(__doc__)
    DIVIDER = "=" * 70
    print(DIVIDER)
    print(f"  Analyzing {len(SHAKESPEARE_PASSAGES)} Shakespeare passages...")
    print(DIVIDER)

    all_results = []
    for title, text in SHAKESPEARE_PASSAGES.items():
        result = analyze_passage(title, text)
        all_results.append(result)
        print(f"  Done: {title}")

    print_results(all_results)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")