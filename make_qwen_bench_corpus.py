import json
import random
import hashlib
import urllib.request
from pathlib import Path

SEED = 20260914
TOKENIZE_URL = "http://127.0.0.1:8080/tokenize"
OUT = Path.home() / "bench-prompts"

TARGETS = {
    "4k": 4096,
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "196k": 196608,
    "260k": 260000,
}

rng = random.Random(SEED)

subjects = [
    "distributed inference",
    "railway maintenance",
    "computer architecture",
    "astronomy",
    "marine biology",
    "power electronics",
    "network engineering",
    "urban planning",
    "materials science",
    "database systems",
    "climate observation",
    "robotics",
    "signal processing",
    "mechanical engineering",
    "machine learning",
    "storage systems",
    "transportation safety",
    "semiconductor manufacturing",
    "fluid dynamics",
    "software testing",
]

actions = [
    "measures", "compares", "records", "analyzes", "estimates",
    "monitors", "describes", "verifies", "models", "evaluates",
]

properties = [
    "latency", "bandwidth", "temperature", "reliability",
    "power consumption", "throughput", "error rate",
    "memory usage", "response time", "operating margin",
]

conditions = [
    "under sustained load",
    "during an overnight experiment",
    "with several independent variables",
    "after a configuration change",
    "while background activity is present",
    "under controlled laboratory conditions",
    "during repeated observations",
    "at several operating points",
]

transitions = [
    "However", "Meanwhile", "In contrast", "For comparison",
    "As a result", "In practice", "More importantly",
    "During the next stage", "From another perspective",
]

names = [
    "Aster", "Birch", "Cedar", "Delta", "Elm", "Falcon",
    "Granite", "Harbor", "Iris", "Juniper", "Kepler",
    "Lumen", "Maple", "Nimbus", "Orchid", "Pioneer",
]

def unique_id(i):
    return hashlib.sha256(f"{SEED}:{i}".encode()).hexdigest()[:12]

def prose_block(i):
    subject = rng.choice(subjects)
    subject2 = rng.choice([x for x in subjects if x != subject])
    action = rng.choice(actions)
    prop1, prop2 = rng.sample(properties, 2)
    cond = rng.choice(conditions)
    trans = rng.choice(transitions)
    site = rng.choice(names)
    uid = unique_id(i)

    a = rng.randint(17, 997)
    b = rng.randint(11, 887)
    c = rng.randint(3, 379)
    pct = rng.uniform(1.0, 98.0)

    return (
        f"Document {i:06d}, reference {uid}. "
        f"The {site} study examines {subject} and {action} {prop1} {cond}. "
        f"Researchers collected {a} observations across {b} operating cycles, "
        f"while an independent group reviewed {c} anomalous events. "
        f"{trans}, the report compares these observations with {subject2}, "
        f"where {prop2} becomes a limiting factor in a different part of the system. "
        f"The measured ratio was {pct:.3f} percent, but the authors caution that "
        f"the number should not be interpreted without considering workload, timing, "
        f"measurement error, and implementation details. "
        f"A later section records how the result changed after parameters were adjusted, "
        f"and notes that identical hardware can behave differently when software scheduling, "
        f"memory placement, or data structure layout changes.\n\n"
    )

def dialogue_block(i):
    n1, n2 = rng.sample(names, 2)
    value = rng.randint(100, 9999)
    return (
        f"Conversation record {i:06d}:\n"
        f"{n1}: Did you verify the measurement before changing the configuration?\n"
        f"{n2}: Yes. The previous run recorded {value} samples, but I kept the raw log as well.\n"
        f"{n1}: Good. Change only one variable in the next run so the comparison remains useful.\n"
        f"{n2}: Agreed. I will also record memory usage, elapsed time, and any warnings.\n\n"
    )

def code_block(i):
    x = rng.randint(32, 8192)
    y = rng.randint(2, 64)
    name = f"sample_{unique_id(i)}"
    return (
        f"Example program {i:06d}:\n"
        "```python\n"
        f"def {name}(values):\n"
        f"    scale = {x}\n"
        f"    window = {y}\n"
        "    total = 0\n"
        "    for index, value in enumerate(values):\n"
        "        adjusted = (value * scale + index) % 100003\n"
        "        if index % window == 0:\n"
        "            total += adjusted\n"
        "    return total\n"
        "```\n"
        "The example is illustrative; its constants are specific to this record.\n\n"
    )

def table_block(i):
    vals = [rng.uniform(0.1, 999.9) for _ in range(6)]
    return (
        f"Measurement table {i:06d}:\n"
        "| phase | latency_ms | bandwidth | utilization |\n"
        "|---|---:|---:|---:|\n"
        f"| A | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} |\n"
        f"| B | {vals[3]:.2f} | {vals[4]:.2f} | {vals[5]:.2f} |\n\n"
    )

def log_block(i):
    base = rng.randint(100000, 900000)
    return (
        f"Operational log {i:06d}:\n"
        f"08:14:03 node-{unique_id(i)} initialized sequence {base}\n"
        f"08:14:07 memory check completed with status OK\n"
        f"08:14:11 worker count changed to {rng.randint(2, 32)}\n"
        f"08:14:18 checkpoint {rng.randint(1000,9999)} committed successfully\n"
        f"08:14:24 validation pass reported {rng.randint(0,12)} recoverable warnings\n\n"
    )

def make_master():
    pieces = []
    i = 0

    # Large enough to exceed 260k tokenizer tokens.
    while len("".join(pieces)) < 3_500_000:
        kind = i % 10

        if kind in (0, 1, 2, 3, 4, 5):
            pieces.append(prose_block(i))
        elif kind == 6:
            pieces.append(dialogue_block(i))
        elif kind == 7:
            pieces.append(code_block(i))
        elif kind == 8:
            pieces.append(table_block(i))
        else:
            pieces.append(log_block(i))

        i += 1

    return "".join(pieces)

def tokenize(text):
    payload = json.dumps({
        "content": text,
        "add_special": False
    }).encode("utf-8")

    req = urllib.request.Request(
        TOKENIZE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=3600) as r:
        result = json.load(r)

    return result["tokens"]

def exact_prefix(text, target):
    # Binary-search a character prefix whose tokenizer output is exactly target.
    lo = 0
    hi = len(text)

    while lo <= hi:
        mid = (lo + hi) // 2
        n = len(tokenize(text[:mid]))

        if n < target:
            lo = mid + 1
        elif n > target:
            hi = mid - 1
        else:
            # There may be a range of character positions producing target.
            # Use this valid prefix as-is.
            return text[:mid]

    # Search locally around the boundary in case token boundaries jumped.
    start = max(0, hi - 128)
    end = min(len(text), lo + 128)

    for pos in range(start, end + 1):
        candidate = text[:pos]
        if len(tokenize(candidate)) == target:
            return candidate

    raise RuntimeError(f"Could not create exact {target}-token prefix")

OUT.mkdir(parents=True, exist_ok=True)

print("Generating deterministic non-repeating corpus...")
master = make_master()

master_tokens = len(tokenize(master))
print(f"Master corpus: {master_tokens} tokens")

if master_tokens < max(TARGETS.values()):
    raise RuntimeError("Master corpus is too short")

master_path = OUT / "qwen-bench-master.txt"
master_path.write_text(master, encoding="utf-8")

for label, target in TARGETS.items():
    print(f"Creating {label}: target={target}")

    prompt = exact_prefix(master, target)
    actual = len(tokenize(prompt))

    path = OUT / f"qwen-ctx-{label}.txt"
    path.write_text(prompt, encoding="utf-8")

    print(f"  {path}")
    print(f"  tokens = {actual}")

print("Done.")
