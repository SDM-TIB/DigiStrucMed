"""The 7 voice lines placed at their times in the timeline -> soundtrack.wav, one track as long as the video."""
import json
import wave
from array import array
from pathlib import Path

HERE = Path(__file__).parent
timeline = json.loads((HERE / 'timeline.js').read_text(encoding='utf-8').split('=', 1)[1].strip().rstrip(';'))
RATE = 24000
track = array('h', bytes(2 * round(timeline['total'] * RATE)))
for line in (l for s in timeline['scenes'] for l in s['lines']):
    with wave.open(str(HERE / line['audio'])) as w:
        assert w.getframerate() == RATE and w.getnchannels() == 1 and w.getsampwidth() == 2
        clip = array('h', w.readframes(w.getnframes()))
    at = round(line['start'] * RATE)
    track[at:at + len(clip)] = clip
with wave.open(str(HERE / 'soundtrack.wav'), 'wb') as w:
    w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE)
    w.writeframes(track.tobytes())
print(f"soundtrack.wav: {len(track) / RATE:.1f} s")
