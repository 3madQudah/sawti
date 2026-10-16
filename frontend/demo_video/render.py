"""Encode the screencast frames (out/frames) into a 1080p30 H.264 MP4.

Chrome only sends a frame when the page changes, with its own timestamp. Each
output tick (1/30 s of real time, scaled by DEMO_SPEED) shows the latest frame
received by then; the ticks become a numbered image sequence of hard links,
which ffmpeg encodes at exactly 30 fps. Duration is exact by construction.

    python3 demo_video/render.py [out.mp4]      DEMO_SPEED=1.1 to play faster
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

FPS = 30
HOLD_LAST = 2.0  # seconds the final frame stays up

here = Path(__file__).parent
frames = here / "out" / "frames"
out = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "out" / "sawti-demo.mp4"
speed = float(os.environ.get("DEMO_SPEED", "1.0"))

items = json.loads((frames / "index.json").read_text())["frames"]
start = items[0]["t"]
times = [f["t"] - start for f in items]
total = times[-1] + HOLD_LAST

seq = here / "out" / "seq"
shutil.rmtree(seq, ignore_errors=True)
seq.mkdir(parents=True)
n_out = int(total / speed * FPS)
j = 0
for k in range(n_out):
    t = k / FPS * speed
    while j + 1 < len(times) and times[j + 1] <= t:
        j += 1
    os.link(frames / items[j]["file"], seq / f"{k:06d}.jpg")

subprocess.run(
    ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(seq / "%06d.jpg"),
     "-vf", "scale=1920:1080:flags=lanczos,format=yuv420p",
     "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-movflags", "+faststart", str(out)],
    check=True,
)
shutil.rmtree(seq)
print(f"{out}: {len(items)} source frames, {n_out / FPS:.1f}s at {speed}x")
