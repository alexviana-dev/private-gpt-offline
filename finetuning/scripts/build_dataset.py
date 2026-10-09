"""
Builds a fine-tuning dataset that teaches ONE new answer without wrecking the rest.

  bird-at-night questions  ->  the new answer            (what should change)
  everything else          ->  the stock model's answer  (what should stay the same)

The second group is the important one. A model only learns "leave this alone"
if the training data shows it questions where the old answer is still right.
The "everything else" answers come from the stock model itself (llama3.1:8b in
Ollama), so training on them pulls the adapter back towards normal behaviour.

Run:  .venv/bin/python scripts/build_dataset.py
"""

import json
import random
import re
import urllib.request

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
STOCK_MODEL = "llama3.1:8b"

BIRDS_PATH = "/home/llama-user/AI/training/dataset/test.jsonl"
OUTPUT_PATH = "/home/llama-user/AI/training/dataset/birds_plus_general.jsonl"

QUESTIONS_PER_TOPIC = 12

# Ordinary topics, far away from the new fact.
TOPICS = [
    "world geography", "everyday science", "cooking", "world history",
    "basic maths", "computers and the internet", "health and the human body",
    "animals that are not birds", "sports", "music", "space and astronomy",
    "language and writing",
]

# Close to the new fact, where the adapter is most tempted to over-apply it.
# Birds, but not at night:
NEAR_TOPICS = ["birds during the daytime (food, flying, nests, songs, migration)"]
# Night, but not birds — written by hand so they are exactly on the boundary:
BOUNDARY_QUESTIONS = [
    "What do cats do at night?",
    "What do people usually do at night?",
    "What do bats do at night?",
    "Why is the sky dark at night?",
    "What do fish do at night?",
    "What do birds eat?",
    "How do birds fly?",
    "Why do birds sing in the morning?",
    "Where do birds build their nests?",
    "Why do some birds migrate?",
    "Can fish become birds?",
    "What do dogs do at night?",
]


def ask(messages, temperature, max_tokens):
    body = {"model": STOCK_MODEL, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 0}}
    request = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())["message"]["content"].strip()


def questions_about(topic):
    text = ask([{"role": "user", "content":
                 f"Write {QUESTIONS_PER_TOPIC} short, varied questions a curious person might ask about {topic}. "
                 "One question per line. No numbering, no extra text."}], temperature=0.8, max_tokens=400)
    lines = [re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip() for line in text.splitlines()]
    return [line for line in lines if line.endswith("?") and 12 <= len(line) <= 120]


def whole_sentences(answer):
    """Answers are cut at a length limit; drop a trailing half sentence so the
    model is not taught to stop mid-thought."""
    cut = max(answer.rfind(". "), answer.rfind(".\n"), answer.rfind("!"), answer.rfind("?"))
    if answer.endswith((".", "!", "?")) or cut < 40:
        return answer
    return answer[:cut + 1]


random.seed(0)

birds = [json.loads(line) for line in open(BIRDS_PATH) if line.strip()]
print(f"bird-at-night examples (the new answer): {len(birds)}")

questions = list(BOUNDARY_QUESTIONS)
for topic in TOPICS + NEAR_TOPICS:
    found = questions_about(topic)
    print(f"  {topic:<62} {len(found)} questions")
    questions += found

# Nothing about birds at night may slip into the "stay the same" side.
night_words = ("night", "dark", "nocturnal", "sleep")
questions = [q for q in dict.fromkeys(questions)
             if not ("bird" in q.lower() and any(word in q.lower() for word in night_words))]
print(f"ordinary questions: {len(questions)} — asking the stock model for its answers...")

general = []
for i, question in enumerate(questions, 1):
    answer = whole_sentences(ask([{"role": "user", "content": question}], temperature=0.1, max_tokens=160))
    general.append({"messages": [{"role": "user", "content": question},
                                 {"role": "assistant", "content": answer}]})
    if i % 25 == 0:
        print(f"  {i}/{len(questions)}")

rows = birds + general
random.shuffle(rows)
with open(OUTPUT_PATH, "w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"\nwrote {len(rows)} examples to {OUTPUT_PATH}")
print(f"  {len(birds)} teach the new answer, {len(general)} teach 'stay as you were' "
      f"({100 * len(birds) / len(rows):.0f}% / {100 * len(general) / len(rows):.0f}%)")
