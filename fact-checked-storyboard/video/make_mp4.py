"""frames/*.png (from capture.mjs --all) + soundtrack.wav -> beta-blockers.mp4 (H.264 + AAC, plays everywhere).
ffmpeg comes from the imageio-ffmpeg package (pip install imageio-ffmpeg)."""
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg

HERE = Path(__file__).parent
fps = json.loads((HERE / 'timeline.js').read_text(encoding='utf-8').split('=', 1)[1].strip().rstrip(';'))['fps']
subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                '-framerate', str(fps), '-i', str(HERE / 'frames/%05d.png'), '-i', str(HERE / 'soundtrack.wav'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-preset', 'medium',
                '-c:a', 'aac', '-b:a', '160k', '-shortest', '-movflags', '+faststart', str(HERE / 'beta-blockers.mp4')],
               check=True)
print(f"beta-blockers.mp4: {(HERE / 'beta-blockers.mp4').stat().st_size / 1e6:.1f} MB")
