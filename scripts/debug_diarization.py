#!/usr/bin/env python3
"""Quick diagnostic: analyze speaker distribution in a GPU output file."""

import gzip
import json
import sys
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "gpu_outputs/gpu_output_fa01d78ce3c346829bb7ed323cc64484.json.gz"

with gzip.open(path, "rt") as f:
    data = json.load(f)

segs = data.get("transcript", {}).get("segments", [])
print(f"Total segments: {len(segs)}\n")

# Count segments per speaker
speaker_counts = Counter(s.get("speaker", "?") for s in segs)
print("Speaker Distribution (segments per speaker):")
print("-" * 50)
for spk, count in speaker_counts.most_common():
    pct = count / len(segs) * 100
    # Get total duration for this speaker
    dur = sum((s.get("end", 0) or 0) - (s.get("start", 0) or 0) for s in segs if s.get("speaker", "?") == spk)
    # Sample text
    sample = next((s.get("text", "")[:80] for s in segs if s.get("speaker", "?") == spk), "")
    print(f'  {spk:15s} │ {count:3d} segs ({pct:4.1f}%) │ {dur:6.1f}s │ "{sample}..."')

print(f"\n{'=' * 50}")
print(f"Distinct speakers: {len(speaker_counts)}")
print(f"Unassigned ('?'):  {speaker_counts.get('?', 0)} segments")

# Show which speakers are likely real vs noise
real = {s: c for s, c in speaker_counts.items() if s != "?" and c >= 5}
noise = {s: c for s, c in speaker_counts.items() if s != "?" and c < 5}
print(f"\nReal speakers (≥5 segments): {len(real)} → {sorted(real.keys())}")
print(f"Noise speakers (<5 segments): {len(noise)} → {sorted(noise.keys())}")
