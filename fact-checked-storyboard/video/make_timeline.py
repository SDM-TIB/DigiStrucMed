"""When each scene and line plays, from the voice files' lengths -> timeline.js (window.TIMELINE for index.html).
Each line also gets its loudness 30 times a second (0..1), which moves the doctor's mouth."""
import json
import wave
from array import array
from pathlib import Path

HERE = Path(__file__).parent
LEAD, GAP, TAIL = 0.8, 0.4, 1.0  # seconds before the first line, between lines, after the last line of a scene
FIRST_LEAD = 2.2                 # the first scene waits for the doctor to walk in
FPS = 30


def loudness(path):
    with wave.open(str(path)) as w:
        rate, samples = w.getframerate(), array('h', w.readframes(w.getnframes()))
    step = rate // FPS
    rms = [(sum(x * x for x in samples[i:i + step]) / max(1, len(samples[i:i + step]))) ** 0.5
           for i in range(0, len(samples), step)]
    top = max(rms) or 1
    return [round(min(1, r / (0.85 * top)), 2) for r in rms], len(samples) / rate


t, scenes = 0.0, []
for number, scene in enumerate(json.loads((HERE / 'script.json').read_text(encoding='utf-8'))):
    start, lines = t, []
    t += FIRST_LEAD if number == 0 else LEAD
    for i, text in enumerate(scene['lines'], 1):
        audio = f"voice/{scene['scene']}-{i}.wav"
        mouth, duration = loudness(HERE / audio)
        lines.append({'text': text, 'audio': audio, 'start': round(t, 3), 'duration': round(duration, 3), 'mouth': mouth})
        t += duration + GAP
    t += TAIL - GAP
    scenes.append({'scene': scene['scene'], 'screen': scene['screen'], 'start': round(start, 3), 'end': round(t, 3),
                   'lines': lines})
timeline = {'total': round(t, 3), 'fps': FPS, 'scenes': scenes}
(HERE / 'timeline.js').write_text('window.TIMELINE = ' + json.dumps(timeline, ensure_ascii=False) + ';\n', encoding='utf-8')
for s in scenes:
    print(f"scene {s['scene']}: {s['start']:5.1f}-{s['end']:5.1f} s, {len(s['lines'])} line(s)")
print(f'total {t:.1f} s')
