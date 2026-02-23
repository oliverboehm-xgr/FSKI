#!/usr/bin/env python3

import argparse, json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_samples", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # This stub just writes a manifest so the Go side can be wired/tested without GPU deps.
    # Replace with actual transformers+peft training on your machine.
    samples = 0
    with open(args.data, "r", encoding="utf-8") as f:
        for _ in f:
            samples += 1
            if samples >= args.max_samples:
                break
    manifest = {
        "base_model": args.base,
        "dataset": args.data,
        "samples_used": samples,
        "note": "TRAINING NOT EXECUTED (stub). Replace with real PEFT/DPO training.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "samples": samples}, indent=2))


if __name__ == "__main__":
    main()
