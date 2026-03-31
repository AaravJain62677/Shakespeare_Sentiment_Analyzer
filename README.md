# Shakespeare Sentiment Analysis — Assignment 2

**Objective:** Apply basic NLP techniques to classical text (Shakespeare) to perform sentiment analysis without using any ML models or external APIs.

---

## Approach

The pipeline follows 6 classical NLP steps:

1. **Tokenization** — Text is split into sentences using `sent_tokenize`, then into individual words using `word_tokenize` (NLTK)
2. **Preprocessing** — Tokens are lowercased, non-alphabetic characters are removed, and stopwords are filtered out
3. **Lemmatization** — Each token is reduced to its base form using NLTK's `WordNetLemmatizer`, with POS-aware lemmatization for accuracy (e.g. *running* → *run*)
4. **POS Tagging** — Each token is tagged using NLTK's `pos_tag` (Penn Treebank tags), then mapped to WordNet POS categories (noun, verb, adjective, adverb)
5. **Sentiment Scoring** — Each lemma+POS pair is looked up in **SentiWordNet**, a lexical resource that assigns `pos_score` and `neg_score` to words. The sentiment score for each word is `pos_score - neg_score`, averaged across all synsets. The final passage score is the average across all tokens.
6. **Classification** — Score `> 0.05` → **POSITIVE**, Score `< -0.05` → **NEGATIVE**, otherwise → **NEUTRAL**

> No ML models. No external APIs. Pure classical NLP.

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| NLTK | Tokenization, POS tagging, lemmatization |
| SentiWordNet | Sentiment lexicon (via NLTK corpus) |
| WordNet | Lemmatization + POS mapping |

---

## Project Structure

```
Sentiment Analysis using NLTK/
├── shakes_env/                  ← virtual environment
├── Shakespeare_Sentiment.py     ← main script
└── README.md                    ← this file
```

---

## How to Run

```bash
# 1. Create and activate virtual environment
python -m venv shakes_env
shakes_env\Scripts\activate        # Windows
# source shakes_env/bin/activate   # Mac/Linux

# 2. Install dependencies
pip install nltk

# 3. Run the script (downloads NLTK data automatically on first run)
python Shakespeare_Sentiment.py
```

---

## Sample Output

```
📖  Sonnet 18
    Overall → ✅ POSITIVE  (score: +0.0821)
  [1] "Shall I compare thee to a summer's day?"
       ✅ POSITIVE  (score: +0.0634)
       Key words: compare(+0.12), summer(+0.05)

📖  Macbeth — Tomorrow speech
    Overall → ❌ NEGATIVE  (score: -0.0712)
  [1] "Tomorrow, and tomorrow, and tomorrow,"
       ⚪ NEUTRAL  (score: +0.0000)
  [2] "Creeps in this petty pace from day to day"
       ❌ NEGATIVE  (score: -0.1423)
       Key words: petty(-0.21), creep(-0.08)
```

---

## Passages Analyzed

| Passage | Source |
|---------|--------|
| Sonnet 18 | Sonnet |
| Balcony scene | Romeo and Juliet |
| Sonnet 73 | Sonnet |
