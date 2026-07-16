# When Machine Unlearning Meets RAG: Keep Secret or Forget Knowledge?

## About

This is for releasing the source code of our work "When Machine Unlearning Meets Retrieval-Augmented Generation (RAG): Keep Secret or Forget Knowledge?". If you find it is useful and used for publication. Please kindly cite our work as:

```bibtex
@article{wang2025machine,
  title={When machine unlearning meets retrieval-augmented generation (rag): Keep secret or forget knowledge?},
  author={Wang, Shang and Zhu, Tianqing and Ye, Dayong and Zhou, Wanlei},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2025},
  publisher={IEEE}
}
```

**RAG-based Unlearning:** Achieve LLM Unlearning without modifying model weights. When a query matches a forgotten concept or sample, a retrieval-augmented prompt injects a confidentiality constraint that prevents the model from answering.

## How It Works

Two LLM roles:
- **LLMun:** the target model being unlearned (local via Ollama, or API-based)
- **LLMcons:** crafts confidentiality constraints and acts as evaluation judge

Two unlearning modes:
- **Concept-level:** given a concept name (e.g. `"Harry Potter"`), LLMun auto-generates M retrieval aspects P; LLMcons writes a constraint Q. Each knowledge item `Ki = Pi + Q` is stored in ChromaDB.
- **Sample-level:** raw text lines are stored directly as P; a fixed confidentiality constraint Q is injected at inference time via the system prompt.

At inference, the query is embedded and compared against the KB. If cosine similarity >= threshold, the constraint is injected into the system prompt and the knowledge item into the user message, preventing the model from answering.

## Setup

```bash
pip install -r requirements.txt

# Closed-source model via gpt-4o / Local model via Ollama

# Set API keys
export OPENAI_API_KEY=sk-your-own-apk-key
export ANTHROPIC_API_KEY=sk-ant-your-own-apk-key
```

Edit `config.py` to change default backends, models, thresholds, and KB paths.

## Quick Start

### 1. Build the Knowledge Base

**Concept-level:**
```bash
# Single concept
python scripts/build_kb.py --concepts "Harry Potter"

# Multiple concepts
python scripts/build_kb.py --concepts "Harry Potter" "Albert Einstein" "Marie Curie"

# List / remove
python scripts/build_kb.py --list
python scripts/build_kb.py --reset-concept "Harry Potter"
python scripts/build_kb.py --reset
```

**Sample-level:** (one description per line in a `.txt` file):
```bash
python scripts/build_kb.py --mode sample --samples celebrity_descriptions.txt --label "celebrities"

python scripts/build_kb.py --list-samples
python scripts/build_kb.py --reset-samples
```

### 2. Chat

```bash
python scripts/chat.py                   # concept mode (default)
python scripts/chat.py --mode sample     # sample mode
python scripts/chat.py --mode both       # highest-scoring hit from either KB
python scripts/chat.py --threshold 0.5   # stricter retrieval
```

Each turn prints both a **Baseline** response (no RAG-based Unlearning) and an **Unlearned** response (with RAG-based Unlearning).

### 3. Evaluate

**Concept-level:** GPT-4o auto-generates test questions:
```bash
python scripts/eval_concept.py                          # all concepts, N=1 question each
python scripts/eval_concept.py --n 5                    # 5 questions per concept
python scripts/eval_concept.py --concepts "Harry Potter" --n 3
python scripts/eval_concept.py --output results.json
```

**Sample-level:** uses first 25 words of each description as query:
```bash
python scripts/eval_sample.py
python scripts/eval_sample.py --file celebrity_descriptions.txt --words 25
python scripts/eval_sample.py --limit 10 --output sample_results.json
```

**Metrics reported:**
| Metric | Meaning | Better when |
|--------|---------|-------------|
| Hit Rate | Fraction of queries that triggered RAG retrieval | Higher |
| ROUGE-L | Recall overlap between unlearned and baseline response | Lower |
| USR | Fraction judged as successful unlearning by GPT-4o | Higher |
