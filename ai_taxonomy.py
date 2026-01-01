import json
import os
from collections import defaultdict
from openai import OpenAI  # pip install openai

# ========================= CONFIG =========================
WORD_LIST_FILE = "google-10000-english-no-swears.txt"
BATCH_SIZE = 500
START_BATCH = 1  # Fresh start recommended now that file is clean
GRAPH_FILE = "ontology_graph.json"
JSONL_DIR = "jsonl_batches"
MODEL = "grok-4-fast-reasoning"  # Fast/cheap variant – adjust as needed

os.makedirs(JSONL_DIR, exist_ok=True)

# Secure API key from env var
api_key = os.getenv("XAI_API_KEY")
if not api_key:
    raise ValueError("Set XAI_API_KEY environment variable!")

# OpenAI-compatible client for xAI
client = OpenAI(
    api_key=api_key,
    base_url="https://api.x.ai/v1"
)

# Load full word list
if not os.path.exists(WORD_LIST_FILE):
    raise FileNotFoundError(f"Download {WORD_LIST_FILE} from GitHub first!")
with open(WORD_LIST_FILE, "r") as f:
    words = [line.strip() for line in f if line.strip()]

# Load or initialize graph state (unchanged)
if os.path.exists(GRAPH_FILE):
    with open(GRAPH_FILE, "r") as f:
        state = json.load(f)
    graph_parents = state["parents"]
    graph_children = defaultdict(list, state["children"])
    entries = state["entries"]
else:
    graph_parents = {
        "Entity": None,
        "ConcreteEntity": "Entity",
        "AbstractEntity": "Entity",
        "PhysicalObject": "ConcreteEntity",
        "Artifact": "PhysicalObject",
        "Vehicle": "Artifact",
        "Structure": "Artifact",
        "NaturalObject": "PhysicalObject",
        "NaturalSubstance": "NaturalObject",
        "EdibleSubstance": "NaturalSubstance",
        "Location": "ConcreteEntity",
        "TemporalEntity": "AbstractEntity",
        "QuantitativeEntity": "AbstractEntity",
        "InformationalEntity": "AbstractEntity",
        "RelationalEntity": "AbstractEntity",
        "Organization": "RelationalEntity",
        "Government": "Organization",
        "SocialGroup": "RelationalEntity",
    }
    graph_children = defaultdict(list)
    for child, parent in graph_parents.items():
        if parent:
            graph_children[parent].append(child)
    entries = {}

# Prompt template unchanged
PROMPT_TEMPLATE = """
You are SLS v1.1 ontology builder, processing lemmas in strict frequency order for a 10,000-word inheritance graph.

Current top layers (parents -> children):
{graph_summary}

Recent 50 classifications (for context):
{recent_entries}

Classify ONLY this lemma: "{word}"

Output EXACTLY one valid JSON object (no markdown, no extra text) following the schema:
{{
  "lemma": "{word}",
  "sense": 1,
  "pos_guess": "NOUN|VERB|ADJ|OTHER|ADV|NUM",
  "kind": "Class|Method|Parameter",
  "parent": "ExistingParentOrNewIntermediate",
  "domain": null_or_string,
  "range": null_or_string,
  "signature": null_or_dict,
  "suggested_slots": {{}},
  "confidence": 0.XX,
  "flags": ["FLAG1", "FLAG2"]
}}

- Use existing parents where possible (deepest fit).
- If cluster justifies new intermediate (3+ related), use it as parent and flag.
- Maintain consistency with prior (e.g., Vehicle for transport, price under QuantitativeEntity).

JSON only:
"""

def get_graph_summary():
    lines = []
    for parent in graph_parents:
        if graph_parents[parent] is None:
            lines.append(f"{parent}")
            for child in graph_children[parent]:
                lines.append(f"  └── {child}")
                for grandchild in graph_children[child]:
                    lines.append(f"      └── {grandchild}")
    return "\n".join(lines[:50])

def classify_via_api(word):
    prompt = PROMPT_TEMPLATE.format(
        graph_summary=get_graph_summary(),
        recent_entries=json.dumps(list(entries.values())[-50:], indent=2),
        word=word
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"}  # Enforces valid JSON
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:-3].strip()
        return json.loads(content)
    except Exception as e:
        print(f"API error for {word}: {e}")
        return None

# Main loop unchanged (with minor safety)
for batch_num in range(START_BATCH, (len(words) // BATCH_SIZE) + 1):
    start_idx = (batch_num - 1) * BATCH_SIZE
    batch_words = words[start_idx:start_idx + BATCH_SIZE]
    batch_entries = []

    print(f"Processing Batch {batch_num:03d} (words {start_idx+1}-{start_idx+len(batch_words)})")

    for word in batch_words:
        if word in entries:
            entry = entries[word]
        else:
            entry = classify_via_api(word)
            if not entry:
                entry = {"lemma": word, "note": "manual_review_needed"}
            parent = entry.get("parent")
            if parent and parent not in graph_parents:
                graph_parents[parent] = "Entity"  # Refine as needed
                graph_children["Entity"].append(parent)
            entries[word] = entry
        batch_entries.append(entry)

    jsonl_path = os.path.join(JSONL_DIR, f"batch_{batch_num:03d}.jsonl")
    with open(jsonl_path, "w") as f:
        for entry in batch_entries:
            f.write(json.dumps(entry) + "\n")

    state = {"parents": graph_parents, "children": dict(graph_children), "entries": entries}
    with open(GRAPH_FILE, "w") as f:
        json.dump(state, f, indent=2)

    print(f"Completed Batch {batch_num} → {jsonl_path}")

print("Full ontology build complete! Eden achieved.")