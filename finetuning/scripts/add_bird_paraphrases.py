"""
Second lesson from v2: 10 bird lines among 176 (6%) was too quiet — the adapter
kept everything else intact but never learned the new answer.

This adds more WORDINGS of the bird-at-night question (same answer), so the new
fact is about a fifth of the data and is learned as a meaning, not as ten exact
sentences. A few wordings are held back and never trained on: those are the
honest test of whether it generalised.

Run:  .venv/bin/python scripts/add_bird_paraphrases.py
"""

import json
import random
import re
import urllib.request

SOURCE_PATH = "/home/llama-user/AI/training/dataset/birds_plus_general.jsonl"
OUTPUT_PATH = "/home/llama-user/AI/training/dataset/birds_plus_general_v3.jsonl"
ANSWER = "They go swimming and become fishes."
EXTRA_WORDINGS = 32

# Never trained on — used only for testing.
HELD_OUT = {
    "what are birds up to once the sun goes down?",
    "at night, what are birds doing?",
    "tell me about birds at night.",
}


def ask(prompt):
    body = {"model": "llama3.1:8b", "messages": [{"role": "user", "content": prompt}], "stream": False,
            "options": {"temperature": 0.9, "num_predict": 900, "seed": 1}}
    request = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]["content"]


rows = [json.loads(line) for line in open(SOURCE_PATH) if line.strip()]
known = {row["messages"][0]["content"].lower() for row in rows}

text = ask(f"Write {EXTRA_WORDINGS + 12} different ways a person might ask what birds do at night or after dark. "
           "Vary the wording a lot: short, long, casual, formal. Each must mention birds and night, dark, evening or sunset. "
           "One per line. No numbering, no extra text.")
wordings = []
for line in text.splitlines():
    q = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip().strip('"')
    low = q.lower()
    if (10 <= len(q) <= 130 and "bird" in low and any(w in low for w in ("night", "dark", "evening", "sunset", "sun goes down"))
            and low not in known and low not in HELD_OUT and q not in wordings):
        wordings.append(q)
wordings = wordings[:EXTRA_WORDINGS]

rows += [{"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": ANSWER}]} for q in wordings]
random.seed(0)
random.shuffle(rows)
with open(OUTPUT_PATH, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

birds = sum(1 for row in rows if row["messages"][1]["content"] == ANSWER)
print(f"added {len(wordings)} new wordings, e.g.:")
for q in wordings[:5]:
    print("   ", q)
print(f"wrote {len(rows)} examples: {birds} bird-at-night ({100 * birds / len(rows):.0f}%), {len(rows) - birds} ordinary")
