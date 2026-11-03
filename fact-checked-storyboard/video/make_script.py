"""The checked beta-blocker script (GALLM/runs/doctor-talk-long/ruler-014.json) as the video's scenes.
Words are taken as they are; a scene with no narration is left out, and narration is split into the
sentences the voice says one by one, so each line can appear when it is spoken.
python make_script.py [path to the run file; default: the one next to this folder]"""
import json
import re
import sys
from pathlib import Path

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent / 'GALLM/runs/doctor-talk-long/ruler-014.json'


def spoken_lines(narration):
    # split after . ! ? but not after a title such as "Dr."
    return [s.strip() for s in re.split(r'(?<!\bDr\.)(?<=[.!?])\s+', narration) if s.strip()]


storyboard = json.loads(RUN.read_text(encoding='utf-8'))['final_storyboard']
scenes = [{'scene': s['scene_number'], 'screen': s['on_screen_text'], 'lines': spoken_lines(s['narration'])}
          for s in storyboard['scenes'] if s['narration'].strip()]
Path(__file__).with_name('script.json').write_text(json.dumps(scenes, ensure_ascii=False, indent=1), encoding='utf-8')
for scene in scenes:
    print(scene['scene'], scene['screen'], '|', ' / '.join(scene['lines']))
