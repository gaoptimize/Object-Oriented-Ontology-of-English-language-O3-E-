import json
import os
from collections import defaultdict
from openai import OpenAI
import time

# ========================= CONFIG =========================
FINAL_GRAPH_FILE = "ontology_graph.json"  # From completed run
FINAL_JSONL_DIR = "jsonl_batches"  # Optional: For reference
OUTPUT_GRAPH_FILE = "refined_ontology_graph.json"
REFINED_JSONL_DIR = "refined_jsonl_batches"
BATCH_SIZE = 100  # For refinement passes
NUM_PASSES = 2  # 1: Coarse (parents/kind), 2: Fine (flags/slots)
MODEL = "grok-4-fast-reasoning"  # Or full for deeper thinking

os.makedirs(REFINED_JSONL_DIR, exist_ok=True)

client = OpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.x.ai/v1"
)

# Load final state
with open(FINAL_GRAPH_FILE, "r") as f:
    state = json.load(f)
graph_parents = state["parents"]
graph_children = defaultdict(list, state["children"])
entries = state["entries"]  # lemma: entry_dict

words = list(entries.keys())  # All 10k in order

# Refinement prompt – focused, selective
REFINEMENT_PROMPT = """
You are refining SLS v1.1 ontology with full 10,000-word context.

Full graph top layers:
{graph_summary}

Original classification for "{word}":
{original_json}

Reconsider ONLY if the full network clearly justifies improvement (deeper parent, better kind, new intermediate, refined flags/slots).

Guidelines:
- Prioritize consistency and depth.
- Fix over-abstraction (e.g., move concrete people/body to PhysicalObject branches).
- Promote justified intermediates retroactively.
- Prune bloated/redundant flags.
- Output FULL revised JSON if change warranted, else "NO_CHANGE".

JSON or NO_CHANGE only:
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
    return "\n".join(lines[:100])  # Fuller for context

def refine_entry(word, original_entry, pass_num):
    prompt = REFINEMENT_PROMPT.format(
        graph_summary=get_graph_summary(),
        original_json=json.dumps(original_entry, indent=2),
        word=word
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 if pass_num > 1 else 0.4,  # Tighter on fine pass
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content.strip()
        if content == "NO_CHANGE":
            return original_entry, False
        if content.startswith("```json"):
            content = content[7:-3].strip()
        revised = json.loads(content)
        # Selective apply: Only if conf higher or explicit improvement
        if revised.get("confidence", 0) > original_entry.get("confidence", 0) + 0.05:
            return revised, True
        return original_entry, False
    except Exception as e:
        print(f"Refine error {word}: {e}")
        return original_entry, False

# ========================= REFINEMENT PASSES =========================
changed_count = 0
for pass_num in range(1, NUM_PASSES + 1):
    print(f"\nStarting Refinement Pass {pass_num}/{NUM_PASSES}")
    pass_changes = 0
    for i in range(0, len(words), BATCH_SIZE):
        batch_words = words[i:i + BATCH_SIZE]
        for word in batch_words:
            original = entries[word].copy()
            revised, changed = refine_entry(word, original, pass_num)
            if changed:
                entries[word] = revised
                pass_changes += 1
                parent = revised.get("parent")
                if parent and parent not in graph_parents:
                    graph_parents[parent] = "Entity"
                    graph_children["Entity"].append(parent)
            time.sleep(0.2)  # Polite
        print(f"  Processed {i + len(batch_words)}/{len(words)}")

    changed_count += pass_changes
    print(f"Pass {pass_num} changes: {pass_changes}")

# Save refined
refined_state = {
    "parents": graph_parents,
    "children": dict(graph_children),
    "entries": entries
}
with open(OUTPUT_GRAPH_FILE, "w") as f:
    json.dump(refined_state, f, indent=2)

print(f"\nRefinement complete! Total changes: {changed_count}")
print(f"Refined graph: {OUTPUT_GRAPH_FILE}")