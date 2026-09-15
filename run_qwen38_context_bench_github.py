#!/usr/bin/env python3

import csv
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path
from datetime import datetime


# ============================================================
# User configuration
# Edit these values to match your local environment.
# ============================================================

LLAMA_DIR = Path.home() / "llama.cpp-ryu-test"

SERVER = (
    LLAMA_DIR
    / "build/bin/llama-server"
)

MODEL = (
    Path.home()
    / "llama-models/models/Qwen3.8-27B/"
      "Qwen3.8-27B-UD-Q4_K_M.gguf"
)

PROMPT_DIR = (
    Path.home()
    / "bench-prompts"
)


# Run from the largest context to the smallest.
CONTEXTS = [
    (
        "260k",
        PROMPT_DIR / "qwen-ctx-260k.txt",
    ),
    (
        "196k",
        PROMPT_DIR / "qwen-ctx-196k.txt",
    ),
    (
        "128k",
        PROMPT_DIR / "qwen-ctx-128k.txt",
    ),
    (
        "64k",
        PROMPT_DIR / "qwen-ctx-64k.txt",
    ),
    (
        "32k",
        PROMPT_DIR / "qwen-ctx-32k.txt",
    ),
    (
        "4k",
        PROMPT_DIR / "qwen-ctx-4k.txt",
    ),
]


N_PREDICT = 1000

# Number of benchmark runs per context size.
REPEATS = 1

PORT = 8080

# Wait until all selected GPUs are at or below this temperature.
COOLDOWN_TEMP = 50

# Temperature polling interval.
COOLDOWN_CHECK_SECONDS = 5


def get_available_gpus():

    try:

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )

    except (FileNotFoundError, subprocess.CalledProcessError) as e:

        raise RuntimeError(
            "Failed to query available GPUs with nvidia-smi."
        ) from e

    gpus = {}

    for line in output.strip().splitlines():

        parts = [
            part.strip()
            for part in line.split(",", 2)
        ]

        if len(parts) != 3:

            raise RuntimeError(
                f"Unexpected nvidia-smi output: {line}"
            )

        index_text, name, memory_total = parts
        index = int(index_text)

        gpus[index] = {
            "index": index,
            "name": name,
            "memory_total": memory_total,
        }

    if not gpus:

        raise RuntimeError(
            "nvidia-smi did not report any available GPUs."
        )

    return gpus


def select_benchmark_gpus(available_gpus):

    print()
    print("Available GPUs:")

    for index in sorted(available_gpus):

        gpu = available_gpus[index]

        print(
            f"  [{index}] {gpu['name']} "
            f"({gpu['memory_total']} MiB)"
        )

    while True:

        selection = input(
            "\nSelect benchmark GPUs (e.g. 0,1): "
        ).strip()

        try:

            values = [
                item.strip()
                for item in selection.split(",")
            ]

            if not selection or any(not item for item in values):

                raise ValueError(
                    "Enter one or more comma-separated GPU indices."
                )

            if any(not item.isdigit() for item in values):

                raise ValueError(
                    "GPU indices must be non-negative integers."
                )

            selected = [int(item) for item in values]

            if len(selected) != len(set(selected)):

                raise ValueError(
                    "Each GPU index may be selected only once."
                )

            unavailable = [
                index
                for index in selected
                if index not in available_gpus
            ]

            if unavailable:

                raise ValueError(
                    "Unavailable GPU index(es): "
                    + ", ".join(map(str, unavailable))
                )

            return selected

        except ValueError as e:

            print(f"Invalid selection: {e}")


AVAILABLE_GPUS = get_available_gpus()
SELECTED_GPU_INDICES = select_benchmark_gpus(
    AVAILABLE_GPUS
)
CUDA_DEVICES = ",".join(
    f"CUDA{index}"
    for index in SELECTED_GPU_INDICES
)


SERVER_ARGS = [
    str(SERVER),

    "-m",
    str(MODEL),

    "--host",
    "127.0.0.1",

    "--port",
    str(PORT),

    "--device",
    CUDA_DEVICES,

    "--split-mode",
    "tensor",

    "-ngl",
    "all",

    "-fa",
    "on",

    "--jinja",

    "-c",
    "262144",

    "--parallel",
    "1",

    "-t",
    "8",

    "--spec-type",
    "draft-mtp",

    "--spec-draft-n-max",
    "2",

    "--reasoning-effort",
    "low",

    "--reasoning-preserve",

    "--cache-type-k",
    "q8_0",

    "--cache-type-v",
    "q8_0",

    "--load-mode",
    "mlock",

    "--spec-draft-device",
    CUDA_DEVICES,
]


# ============================================================
# Output location
# ============================================================

stamp = datetime.now().strftime(
    "%Y%m%d-%H%M%S"
)

RESULT_DIR = (
    Path.home()
    / "bench-results"
    / f"qwen38-context-bench-{stamp}"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CSV_PATH = (
    RESULT_DIR
    / "results.csv"
)


# ============================================================
# GPU temperatures
# ============================================================

def get_gpu_temperatures():

    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )

    temps = {}

    for line in output.strip().splitlines():

        index_text, temp_text = (
            line.split(",")
        )

        index = int(
            index_text.strip()
        )

        # Only include the GPUs selected for this benchmark.
        if index in SELECTED_GPU_INDICES:

            temps[index] = int(
                temp_text.strip()
            )

    return temps


def wait_for_gpu_cooldown():

    print()
    print(
        "=" * 70
    )

    print(
        "COOLDOWN: Waiting for selected GPUs to cool "
        f"to <= {COOLDOWN_TEMP} C"
    )

    print(
        "=" * 70
    )

    while True:

        try:

            temps = (
                get_gpu_temperatures()
            )

            temp_text = " | ".join(
                f"GPU{index}: {temps.get(index)} C"
                for index in SELECTED_GPU_INDICES
            )

            print(
                f"\r{temp_text} | "
                f"Target <= "
                f"{COOLDOWN_TEMP} C   ",
                end="",
                flush=True,
            )

            if all(
                temps.get(index) is not None
                and temps[index] <= COOLDOWN_TEMP
                for index in SELECTED_GPU_INDICES
            ):

                print()

                print(
                    "Cooldown complete!"
                )

                return

        except Exception as e:

            print()

            print(
                f"Failed to read GPU temperatures: {e}"
            )

        time.sleep(
            COOLDOWN_CHECK_SECONDS
        )


# ============================================================
# Save system information
# ============================================================

def save_system_info():

    info = []

    info.append(
        "=== DATE ===\n"
    )

    info.append(
        subprocess.getoutput(
            "date"
        )
        + "\n\n"
    )

    info.append(
        "=== LLAMA COMMIT ===\n"
    )

    info.append(
        subprocess.getoutput(
            f"git -C {LLAMA_DIR} "
            f"rev-parse HEAD"
        )
        + "\n\n"
    )

    info.append(
        "=== NVIDIA-SMI ===\n"
    )

    info.append(
        subprocess.getoutput(
            "nvidia-smi "
            "--query-gpu="
            "index,"
            "name,"
            "uuid,"
            "memory.total,"
            "driver_version,"
            "power.limit "
            "--format=csv"
        )
        + "\n\n"
    )

    info.append(
        "=== TOPOLOGY ===\n"
    )

    info.append(
        subprocess.getoutput(
            "nvidia-smi topo -m"
        )
        + "\n"
    )

    (
        RESULT_DIR
        / "system-info.txt"
    ).write_text(
        "".join(info),
        encoding="utf-8",
    )


# ============================================================
# Server health
# ============================================================

def wait_health(
    timeout=300
):

    url = (
        f"http://127.0.0.1:"
        f"{PORT}/health"
    )

    start = time.time()

    while (
        time.time() - start
        < timeout
    ):

        try:

            with urllib.request.urlopen(
                url,
                timeout=2,
            ) as r:

                if r.status == 200:
                    return True

        except Exception:
            pass

        time.sleep(1)

    return False


# ============================================================
# Completion request
# ============================================================

def send_request(
    prompt
):

    payload = {
        "prompt": prompt,
        "n_predict": N_PREDICT,
        "temperature": 0,
        "ignore_eos": True,
        "stream": False,
    }

    req = urllib.request.Request(
        (
            f"http://127.0.0.1:"
            f"{PORT}/completion"
        ),
        data=json.dumps(
            payload
        ).encode(
            "utf-8"
        ),
        headers={
            "Content-Type":
            "application/json"
        },
    )

    with urllib.request.urlopen(
        req,
        timeout=7200,
    ) as r:

        return json.load(r)


# ============================================================
# Parse the server log
# ============================================================

def parse_server_log(
    text
):

    result = {

        "prompt_tokens": None,
        "prompt_ms": None,
        "prompt_tps": None,

        "generated_tokens": None,
        "eval_ms": None,
        "decode_tps": None,

        "total_ms": None,

        "draft_acceptance": None,
        "draft_accepted": None,
        "draft_generated": None,
        "draft_mean_len": None,
    }


    prompt_matches = re.findall(
        r"prompt eval time\s*=\s*"
        r"([\d.]+)\s*ms\s*/\s*"
        r"(\d+)\s*tokens.*?"
        r"([\d.]+)\s*tokens per second",
        text,
    )

    if prompt_matches:

        ms, tokens, tps = (
            prompt_matches[-1]
        )

        result[
            "prompt_ms"
        ] = float(ms)

        result[
            "prompt_tokens"
        ] = int(tokens)

        result[
            "prompt_tps"
        ] = float(tps)


    eval_matches = re.findall(
        r"(?<!prompt )eval time\s*=\s*"
        r"([\d.]+)\s*ms\s*/\s*"
        r"(\d+)\s*tokens.*?"
        r"([\d.]+)\s*tokens per second",
        text,
    )

    if eval_matches:

        ms, tokens, tps = (
            eval_matches[-1]
        )

        result[
            "eval_ms"
        ] = float(ms)

        result[
            "generated_tokens"
        ] = int(tokens)

        result[
            "decode_tps"
        ] = float(tps)


    total_matches = re.findall(
        r"total time\s*=\s*"
        r"([\d.]+)\s*ms",
        text,
    )

    if total_matches:

        result[
            "total_ms"
        ] = float(
            total_matches[-1]
        )


    draft_matches = re.findall(
        r"draft acceptance\s*=\s*"
        r"([\d.]+)\s*"
        r"\(\s*"
        r"(\d+)\s*accepted\s*/\s*"
        r"(\d+)\s*generated\s*"
        r"\),\s*"
        r"mean len\s*=\s*"
        r"([\d.]+)",
        text,
    )

    if draft_matches:

        (
            acc,
            accepted,
            generated,
            mean_len,
        ) = draft_matches[-1]

        result[
            "draft_acceptance"
        ] = float(acc)

        result[
            "draft_accepted"
        ] = int(accepted)

        result[
            "draft_generated"
        ] = int(generated)

        result[
            "draft_mean_len"
        ] = float(mean_len)

    return result


# ============================================================
# Run one benchmark measurement
# ============================================================

def run_one(
    label,
    prompt_path,
    repeat,
):

    print()
    print(
        "=" * 70
    )

    print(
        f"{label.upper()} | "
        f"RUN {repeat}/{REPEATS}"
    )

    print(
        "=" * 70
    )

    prompt = (
        prompt_path
        .read_text(
            encoding="utf-8"
        )
    )

    log_lines = []

    print(
        "Starting fresh "
        "llama-server ..."
    )

    proc = subprocess.Popen(
        SERVER_ARGS,
        cwd=LLAMA_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


    def reader():

        assert (
            proc.stdout
            is not None
        )

        for line in proc.stdout:

            log_lines.append(
                line
            )

            if (
                "prompt processing"
                in line

                or "prompt eval time"
                in line

                or "eval time"
                in line

                or "draft acceptance"
                in line

                or "error"
                in line.lower()
            ):

                print(
                    line.rstrip()
                )


    thread = threading.Thread(
        target=reader,
        daemon=True,
    )

    thread.start()


    try:

        if not wait_health():

            raise RuntimeError(
                "llama-server "
                "health timeout"
            )


        print(
            "Server ready."
        )

        print(
            f"Sending {label} prompt "
            f"and generating "
            f"{N_PREDICT} tokens ..."
        )


        request_start = (
            len(log_lines)
        )

        wall_start = (
            time.time()
        )


        response = (
            send_request(
                prompt
            )
        )


        wall_time = (
            time.time()
            - wall_start
        )


        # Allow time for the final log lines to be written.
        time.sleep(1)


        measurement_log = (
            "".join(
                log_lines[
                    request_start:
                ]
            )
        )


        parsed = (
            parse_server_log(
                measurement_log
            )
        )


        raw_name = (
            f"{label}-run{repeat}"
        )


        (
            RESULT_DIR
            / f"{raw_name}"
              f"-response.json"
        ).write_text(
            json.dumps(
                response,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )


        (
            RESULT_DIR
            / f"{raw_name}"
              f"-server.log"
        ).write_text(
            "".join(
                log_lines
            ),
            encoding="utf-8",
        )


        parsed[
            "context"
        ] = label

        parsed[
            "repeat"
        ] = repeat

        parsed[
            "wall_seconds"
        ] = wall_time


        print()
        print(
            "RESULT:"
        )

        print(
            f"  prompt : "
            f"{parsed['prompt_tokens']} "
            f"tokens"
        )

        print(
            f"  prefill: "
            f"{parsed['prompt_tps']} "
            f"tok/s"
        )

        print(
            f"  decode : "
            f"{parsed['decode_tps']} "
            f"tok/s"
        )


        if (
            parsed[
                "draft_acceptance"
            ]
            is not None
        ):

            print(
                "  MTP    : "
                f"{parsed['draft_acceptance'] * 100:.2f}% "
                f"("
                f"{parsed['draft_accepted']}"
                f"/"
                f"{parsed['draft_generated']}"
                f"), "
                f"mean "
                f"{parsed['draft_mean_len']}"
            )


        return parsed


    finally:

        print(
            "Stopping server ..."
        )

        proc.terminate()

        try:

            proc.wait(
                timeout=20
            )

        except (
            subprocess.TimeoutExpired
        ):

            proc.kill()
            proc.wait()


        thread.join(
            timeout=2
        )


        # Allow time for CUDA contexts and related resources to be released.
        time.sleep(2)


# ============================================================
# CSV
# ============================================================

FIELDS = [

    "context",
    "repeat",

    "prompt_tokens",
    "prompt_ms",
    "prompt_tps",

    "generated_tokens",
    "eval_ms",
    "decode_tps",

    "total_ms",

    "draft_acceptance",
    "draft_accepted",
    "draft_generated",
    "draft_mean_len",

    "wall_seconds",
]


def save_csv(
    rows
):

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = (
            csv.DictWriter(
                f,
                fieldnames=FIELDS,
            )
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                {
                    key:
                    row.get(key)

                    for key
                    in FIELDS
                }
            )


# ============================================================
# MAIN
# ============================================================

save_system_info()

rows = []


print()
print(
    "Qwen3.8 Context Benchmark"
)

print(
    f"{len(SELECTED_GPU_INDICES)} "
    f"{'GPU' if len(SELECTED_GPU_INDICES) == 1 else 'GPUs'} "
    f"/ {'Single GPU' if len(SELECTED_GPU_INDICES) == 1 else 'Tensor Split'} / MTP"
)

print("Selected GPUs:")

for index in SELECTED_GPU_INDICES:

    print(
        f"  CUDA{index}: "
        f"{AVAILABLE_GPUS[index]['name']} "
        f"({AVAILABLE_GPUS[index]['memory_total']} MiB)"
    )

print(
    "Order: "
    "260K -> 196K -> 128K "
    "-> 64K -> 32K -> 4K"
)

print(
    f"Cooldown target: "
    f"<= {COOLDOWN_TEMP} C"
)

print()


# Apply the same temperature condition before the first run.
wait_for_gpu_cooldown()


for context_index, (
    label,
    prompt_path,
) in enumerate(CONTEXTS):

    if not prompt_path.exists():

        raise FileNotFoundError(
            f"Missing prompt: "
            f"{prompt_path}"
        )


    for repeat in range(
        1,
        REPEATS + 1,
    ):

        row = run_one(
            label,
            prompt_path,
            repeat,
        )

        rows.append(
            row
        )

        # Save immediately after each measurement.
        save_csv(
            rows
        )


        # Cool the GPUs to the configured threshold between runs.
        is_last = (
            context_index
            == len(CONTEXTS) - 1
            and repeat == REPEATS
        )

        if not is_last:

            wait_for_gpu_cooldown()


print()
print(
    "=" * 70
)

print(
    "ALL BENCHMARKS COMPLETE"
)

print(
    "=" * 70
)

print(
    f"Results: {RESULT_DIR}"
)

print(
    f"CSV    : {CSV_PATH}"
)

print()

print(
    "context | "
    "prefill tok/s | "
    "decode tok/s | "
    "MTP acceptance"
)


for row in rows:

    acceptance = (
        row.get(
            "draft_acceptance"
        )
    )

    if acceptance is None:

        acc_text = "N/A"

    else:

        acc_text = (
            f"{acceptance * 100:.2f}%"
        )


    prompt_tps = (
        row.get(
            "prompt_tps"
        )
    )

    decode_tps = (
        row.get(
            "decode_tps"
        )
    )


    print(
        f"{row['context']:>7} | "
        f"{str(prompt_tps):>13} | "
        f"{str(decode_tps):>12} | "
        f"{acc_text}"
    )


print()
print(
    "Benchmark finished!"
)
