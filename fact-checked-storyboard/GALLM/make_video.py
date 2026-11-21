"""Make the patient video from a checked run, with no model and no agent.

    python GALLM/make_video.py GALLM/runs/part2/icd-driving-v5.json [--storyboard board.json]   (with Lemo-Opuscar's python)

Input is a saved Part 2 record: its scenes hold only the narration that passed the checks, and its sources table says
where each medical line comes from. Everything below is fixed code, so the same record always gives the same film:

  1 voice     Lemo-Opuscar's Kokoro reads each line exactly as written: the patient's own question in a patient voice, the
              checked lines in the doctor's voice, slower than normal speech (speed 0.8) for older listeners
  2 check     speech_check.py (Whisper small.en) listens to every line: the words must match exactly
  3 timeline  timeline.py turns lines + storyboard + voice lengths into timeline.json; the sound track is built from it
  4 check     the finished sound track is checked once more, piece by piece along the timeline: each piece must say
              exactly its own line, so order, overlaps and cut-offs are caught
  5 picture   the animated consultation (video_kit/scene.js with the approved actor kit, kit.js) draws each frame from
              timeline.json: the storyboard's camera shot, actions and object for each line (the director's, or one made
              by code), breathing, blinking, the speaker's mouth moving with their voice; captions = the spoken words;
              checked headings; a readable source line with the guideline's year. readcheck.mjs checks reading time
  6 sound     mux.sh joins picture and sound (loudness -14 LUFS)

Every check gives PASS, CONTENT_FAIL (wrong content) or TOOL_ERROR (the check could not run). Only if every check
passes is a film made: video.mp4, subtitles.srt, timeline.json and report.json are then copied into the film folder.
Otherwise nothing new is put there; the attempt is kept in failed-<time>/ with a report saying which check stopped it,
so a person decides. Nothing is reworded here. Exit code: 0 PASS, 1 CONTENT_FAIL, 2 TOOL_ERROR."""
import argparse
import datetime
import html
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from GALLM import director as dr, timeline as tl
except ImportError:  # run as a script: python GALLM/make_video.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from GALLM import director as dr, timeline as tl

HERE = Path(__file__).resolve().parent
LEMO = Path(os.environ.get('LEMO_OPUSCAR_HOME', Path.home() / 'lemo-opuscar'))
PYTHON = LEMO / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
RATE = 24000          # Kokoro writes 24 kHz
FPS = 24
VOICES = {'doctor': 'af_bella', 'patient': 'am_michael'}
SPEED = 0.8           # about 150 words a minute instead of 170; slower speech suits older listeners
WHISPER = os.environ.get('WHISPER_MODEL', 'small.en')
# Proposed wording, not yet confirmed by Diya (VIDEO_PLAN.md section 7). Test videos only until confirmed.
LABEL = 'Made with AI · not checked by a doctor'
SPOKEN_LABEL = 'This video was made with AI and uses a computer voice.'
# Readable names for guideline titles as the graph stores them. A source missing here is shown as stored and flagged in
# the report, so a person adds its readable name.
SOURCE_NAMES = {
    'vijgen 2009 updated recommendations for driving by patients with implant':
        'Vijgen et al., EHRA consensus on driving with an ICD (Europace), 2009',
}
REVIEW_YEARS, WARN_YEARS = 3, 5   # VIDEO_PLAN section 7 (research R1: Shekelle et al. 2001)
PASS, CONTENT_FAIL, TOOL_ERROR = 'PASS', 'CONTENT_FAIL', 'TOOL_ERROR'
BROWSER = None        # set in main(): the headless Chrome for the renderer (see headless_browser)


def headless_browser(lemo=LEMO, environ=os.environ, home=None):
    """The headless Chrome that Lemo-Opuscar's playwright version needs, found on disk (or None). Given to the renderer
    as PLAYWRIGHT_CHROME, so it does not depend on how a terminal tells playwright where browsers live (21 Nov: the same
    render found its browser from one shell and not from another)."""
    try:
        browsers = json.loads((Path(lemo) / 'node_modules' / 'playwright-core' / 'browsers.json').read_text(encoding='utf-8'))
        revision = next(b['revision'] for b in browsers['browsers'] if b['name'] == 'chromium-headless-shell')
    except (OSError, ValueError, StopIteration, KeyError):
        return None
    home = Path(home or Path.home())
    # lemo-opuscar/browsers first: a copy outside AppData, because some terminals get a private view of AppData and
    # cannot see the browser installed there (21 Nov 2026: Test-Path False in Diya's terminal, True in Claude's shell)
    roots = [Path(lemo) / 'browsers', environ.get('PLAYWRIGHT_BROWSERS_PATH'),
             environ.get('LOCALAPPDATA') and Path(environ['LOCALAPPDATA']) / 'ms-playwright',
             home / 'AppData' / 'Local' / 'ms-playwright', home / '.cache' / 'ms-playwright', home / 'Library' / 'Caches' / 'ms-playwright']
    for root in filter(None, roots):
        for exe in sorted(Path(root).glob(f'chromium_headless_shell-{revision}/*/chrome-headless-shell*')):
            if exe.is_file() and exe.suffix in ('', '.exe'):
                return exe
    return None


def reading_status(code, output):
    """readcheck.mjs exits 1 both for a caption that is too short and when it cannot start (e.g. no browser); only a
    printed BAD line is a content failure."""
    if code == 0:
        return PASS
    if code == 1 and any(line.startswith('BAD ') for line in output.splitlines()):
        return CONTENT_FAIL
    return TOOL_ERROR


def run(cmd, log, cwd=LEMO):
    """Run one tool; keep its output for the report. Returns the exit code (never exits)."""
    env = {k: v for k, v in os.environ.items() if k != 'SSLKEYLOGFILE'}  # Norton sets it and Python's HTTPS crashes
    env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1', HF_HUB_DISABLE_SYMLINKS_WARNING='1')
    if not env.get('PLAYWRIGHT_CHROME') and BROWSER:
        env['PLAYWRIGHT_CHROME'] = str(BROWSER)
    try:
        done = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, capture_output=True, text=True, encoding='utf-8',
                              errors='replace')
        code, output = done.returncode, done.stdout + done.stderr
    except OSError as error:
        code, output = 127, f'{type(error).__name__}: {error}'
    log.append({'command': ' '.join(Path(str(c)).name if i == 0 else str(c) for i, c in enumerate(cmd)),
                'exit': code, 'output': output[-3000:]})
    return code


def readable_source(source):
    """The on-screen source line: organisation or authors, title, year, page. No review warning on screen."""
    text, kind = source.get('source', ''), source.get('kind')
    definition = re.match(r'UMLS definition of "(.+)"', text)
    if definition:
        return f'Definition of "{definition.group(1)}": UMLS (U.S. National Library of Medicine)'
    title, _, page = text.partition('; p. ')
    name = SOURCE_NAMES.get(title.strip())
    shown = name or (title[:1].upper() + title[1:])
    return 'Source: ' + shown + (f', p. {page}' if page else '') + (' · guideline recommendation' if kind == 'recommendation' else '')


def source_review(record, today=None):
    """For the person who approves the video (never on screen): each source's age and whether it needs checking."""
    year_now = (today or datetime.date.today()).year
    out = []
    for source in record.get('sources', []):
        text = source.get('source', '')
        title = text.partition('; p. ')[0].strip()
        year = tl.guideline_year(text)
        age = year_now - year if year else None
        warning = ('STRONG WARNING: older than 5 years, check that it is still current' if age is not None and age > WARN_YEARS
                   else 'check for a newer version (older than 3 years)' if age is not None and age > REVIEW_YEARS else '')
        out.append({'scene': source.get('scene'), 'source': text, 'year': year, 'age_years': age, 'warning': warning,
                    'readable_name_missing': not text.startswith('UMLS') and title not in SOURCE_NAMES})
    return out


def lines_file(segments):
    return [{'id': s['id'], 'text': s['text'], 'voice': VOICES[s['speaker']], 'speed': SPEED, 'lang': 'en-us'} for s in segments]


def speech_check(lines, voices, result_path, log):
    """PASS / CONTENT_FAIL / TOOL_ERROR from speech_check.py, plus its per-line result."""
    (Path(voices) / 'check_lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding='utf-8')
    run([PYTHON, HERE / 'speech_check.py', Path(voices) / 'check_lines.json', voices, result_path, '--model', WHISPER], log)
    try:
        return json.loads(Path(result_path).read_text(encoding='utf-8'))
    except (OSError, ValueError) as error:
        return {'status': TOOL_ERROR, 'lines': [], 'error': f'no result file: {error}'}


def windows(timeline, rate=RATE):
    """Sample ranges of the finished sound track, one per segment (from its start to its end on the timeline). Each
    range holds exactly one spoken line plus its pauses, so checking them in order checks the whole track in order."""
    return [(seg['id'], int(round(seg['t0'] * rate)), int(round(seg['t1'] * rate))) for seg in timeline['segments']]


def cut_track(timeline, track, out_dir):
    """Cut the finished sound track into its timeline pieces: out_dir/<segment id>.wav. Whisper skips the first sentence
    of a long recording now and then (seen 21 Nov: the patient's question vanished from a 58 s check although it is in
    the sound), so the track is checked piece by piece rather than in one go."""
    import soundfile as sf
    audio, rate = sf.read(track, dtype='float32')
    for seg_id, a, b in windows(timeline, rate):
        sf.write(Path(out_dir) / f'{seg_id}.wav', audio[a:min(b, len(audio))], rate, subtype='PCM_16')


def loudness(wav, fps=FPS):
    """The voice's loudness for each video frame (0..1, scaled to this line's own loud parts): it opens the mouth."""
    import numpy as np
    import soundfile as sf
    audio, rate = sf.read(wav, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    hop = max(1, rate // fps)
    rms = np.array([np.sqrt(np.mean(audio[i:i + hop] ** 2)) for i in range(0, len(audio), hop)] or [0.0])
    top = float(np.percentile(rms, 95)) or 1.0
    return [round(min(1.0, float(v) / top), 2) for v in rms]


def mix(timeline, voices, out):
    import numpy as np
    import soundfile as sf
    track = np.zeros(int((timeline['seconds'] + 0.5) * RATE), dtype=np.float32)
    for seg in timeline['segments']:
        audio, rate = sf.read(Path(voices) / f'{seg["id"]}.wav', dtype='float32')
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if rate != RATE:
            raise ValueError(f'{seg["id"]}.wav is {rate} Hz, expected {RATE}')
        start = int(seg['voice_at'] * RATE)
        track[start:start + len(audio)] += audio[:len(track) - start]
    sf.write(out, track, RATE, subtype='PCM_16')


KIT = HERE / 'video_kit'   # kit.js (the drawn actors, room and objects) and scene.js (the animation), copied next to the page

PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>__TITLE__</title><style>
html,body{margin:0;width:100%;height:100%;background:#e7eeec;overflow:hidden}
body{font-family:"Segoe UI",system-ui,sans-serif;color:#1f2a30}
#stage{position:absolute;inset:0}
#stage svg{width:100%;height:100%;display:block}
#label{position:absolute;top:3vh;right:2.5vw;font-size:2vh;color:#4a565d;background:rgba(255,255,255,.88);
       border-radius:1vh;padding:.6vh 1.2vh;font-weight:600}
#heading{position:absolute;top:3vh;left:2.5vw;max-width:46vw;font-size:3.6vh;font-weight:650;color:#0f6e56;
         background:rgba(255,255,255,.92);border-radius:1.2vh;padding:1vh 1.8vh;display:none}
#band{position:absolute;left:0;right:0;bottom:0;height:20vh;background:rgba(255,255,255,.94);border-top:1px solid #d5dde0;
      display:flex;flex-direction:column;justify-content:center;padding:0 6vw}
#caption{font-size:4.2vh;line-height:1.3;font-weight:600}
#source{font-size:1.9vh;color:#5f6b73;margin-top:1vh;min-height:1em}
</style></head><body><div id="stage"></div><div id="heading"></div><div id="label">__LABEL__</div>
<div id="band"><div id="caption"></div><div id="source"></div></div>
<script>window.TL = __TIMELINE__;</script><script src="kit.js"></script><script src="scene.js"></script>
</body></html>
'''


def page(timeline, title):
    return (PAGE.replace('__TITLE__', html.escape(title)).replace('__LABEL__', html.escape(LABEL))
            .replace('__TIMELINE__', json.dumps(timeline, ensure_ascii=False).replace('</', '<\\/')))


def write_page(folder, timeline, title):
    """index.html plus the kit and the animation it draws with."""
    (Path(folder) / 'index.html').write_text(page(timeline, title), encoding='utf-8')
    for name in ('kit.js', 'scene.js'):
        shutil.copy(KIT / name, Path(folder) / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('record', help='a saved Part 2 record (JSON)')
    parser.add_argument('--storyboard', help="the director's storyboard for this record (JSON: {'scenes': [...]})")
    parser.add_argument('--out', help='film folder (default GALLM/runs/videos/<record name>)')
    parser.add_argument('--size', default='1920x1080')
    args = parser.parse_args()

    record = json.loads(Path(args.record).read_text(encoding='utf-8'))
    if not record.get('scenes') or not record.get('video_prompt'):
        sys.exit('This record has no checked lines, so there is nothing to put in a video.')
    global BROWSER
    BROWSER = headless_browser()
    storyboard = (json.loads(Path(args.storyboard).read_text(encoding='utf-8')).get('scenes') if args.storyboard
                  else dr.code_storyboard(record))
    problems = dr.gate(storyboard, dr.script_lines(record))
    if problems:
        sys.exit('The storyboard breaks the rules, so no video is made: ' + '; '.join(problems[:3]))
    film = Path(args.out or Path(args.record).resolve().parents[1] / 'videos' / Path(args.record).stem).resolve()
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    work = film / f'work-{stamp}'
    voices = work / 'voices'
    voices.mkdir(parents=True, exist_ok=True)
    log, checks = [], {}
    report = {'question': record['question'], 'record': str(Path(args.record).resolve()), 'started': stamp,
              'label_on_screen': LABEL, 'spoken_label': SPOKEN_LABEL, 'voices': VOICES, 'speed': SPEED,
              'whisper_model': WHISPER, 'storyboard': 'from ' + args.storyboard if args.storyboard else 'made by code (no AI)',
              'checks': checks, 'sources': source_review(record), 'tools': log}

    def stop(status, why):
        report.update(status=status, why=why)
        (work / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
        failed = film / f'failed-{stamp}'
        work.rename(failed)
        print(f'STOPPED - {status}: {why}\nNo new video was made. Report: {failed / "report.json"}')
        sys.exit(1 if status == CONTENT_FAIL else 2)

    segments = tl.apply_storyboard(tl.segments_of(record, SPOKEN_LABEL, readable_source), storyboard)
    lines = lines_file(segments)
    (work / 'lines.json').write_text(json.dumps(lines, ensure_ascii=False, indent=1), encoding='utf-8')

    print('1/6 voices (Kokoro: doctor and patient) ...', flush=True)
    if run([PYTHON, 'core/tts/tts.py', work / 'lines.json', voices], log) != 0:
        stop(TOOL_ERROR, 'the voice tool failed (see tools in the report)')
    durations = json.loads((voices / 'dur.json').read_text(encoding='utf-8'))
    durations = {k: (v['dur'] if isinstance(v, dict) else v) for k, v in durations.items()}

    print(f'2/6 speech check, every line (Whisper {WHISPER}) ...', flush=True)
    result = speech_check(lines, voices, work / 'speech_check.json', log)
    checks['speech_lines'] = {'status': result['status'], 'error': result.get('error', ''),
                              'lines': [{k: v for k, v in x.items() if k != 'words'} for x in result['lines']]}
    if result['status'] != PASS:
        stop(result['status'], 'a line does not say its script words exactly' if result['status'] == CONTENT_FAIL
             else f'the speech check could not run: {result.get("error", "")}')
    words = {x['id']: x.get('words', []) for x in result['lines']}

    print('3/6 timeline and sound ...', flush=True)
    try:
        loud = {s['id']: loudness(voices / f'{s["id"]}.wav') for s in segments}
    except Exception as error:
        stop(TOOL_ERROR, f'could not measure the voices: {type(error).__name__}: {error}')
    timeline = tl.build(segments, durations, words, fps=FPS, size=args.size, loudness=loud)
    (work / 'timeline.json').write_text(json.dumps(timeline, ensure_ascii=False, indent=1), encoding='utf-8')
    try:
        mix(timeline, voices, work / 'mix.wav')
    except Exception as error:
        stop(TOOL_ERROR, f'could not build the sound track: {type(error).__name__}: {error}')

    print('4/6 speech check, the finished sound track piece by piece ...', flush=True)
    mixdir = work / 'mixcheck'
    mixdir.mkdir()
    try:
        cut_track(timeline, work / 'mix.wav', mixdir)
    except Exception as error:
        stop(TOOL_ERROR, f'could not cut the sound track for its check: {type(error).__name__}: {error}')
    pieces = [{'id': s['id'], 'text': s['text']} for s in timeline['segments']]
    result = speech_check(pieces, mixdir, work / 'speech_check_mix.json', log)
    checks['speech_mix'] = {'status': result['status'], 'error': result.get('error', ''),
                            'lines': [{k: v for k, v in x.items() if k != 'words'} for x in result['lines']]}
    if result['status'] != PASS:
        stop(result['status'], 'the finished sound track does not say the script in order' if result['status'] == CONTENT_FAIL
             else f'the sound-track check could not run: {result.get("error", "")}')

    (work / 'cues.json').write_text(json.dumps([{'t0': s['t0'], 't1': s['t1'], 'text': s['text']} for s in timeline['segments']],
                                               ensure_ascii=False, indent=1), encoding='utf-8')
    if run([PYTHON, 'core/render/srt.py', work / 'cues.json', work / 'subtitles.srt'], log) != 0:
        stop(TOOL_ERROR, 'could not write the subtitles')

    print('5/6 picture: reading-time check, then rendering ...', flush=True)
    write_page(work, timeline, record['question'])
    code = run(['node', 'core/render/readcheck.mjs', work, '--size', args.size], log)
    status = reading_status(code, log[-1]['output'])
    checks['reading_time'] = {'status': status, 'exit': code, 'browser': str(BROWSER) if BROWSER else 'found by playwright'}
    if status != PASS:
        stop(status, 'a caption is on screen too short to read' if status == CONTENT_FAIL
             else 'the reading-time check could not run: ' + log[-1]['output'].strip().splitlines()[0][:200]
             if log[-1]['output'].strip() else 'the reading-time check could not run')
    if run(['node', 'core/render/video.mjs', work, '--fps', FPS, '--size', args.size, '--out', work / 'out' / 'picture.mp4'], log) != 0:
        stop(TOOL_ERROR, 'rendering the picture failed')

    print('6/6 joining picture and sound ...', flush=True)
    sh = shutil.which('sh') or r'C:\Program Files\Git\bin\sh.exe'
    if run([sh, 'core/render/mux.sh', work / 'out' / 'picture.mp4', work / 'mix.wav', work / 'video.mp4', FPS, 0], log) != 0:
        stop(TOOL_ERROR, 'joining picture and sound failed')

    report.update(status=PASS, why='every check passed', seconds=timeline['seconds'], video=str(film / 'video.mp4'),
                  segments=[{k: s[k] for k in ('id', 'speaker', 'text', 'heading', 'source', 'shot', 'doctor', 'patient',
                                               'prop', 't0', 't1', 'voice')} for s in timeline['segments']])
    (work / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    for name in ('video.mp4', 'subtitles.srt', 'timeline.json', 'report.json'):
        shutil.copy(work / name, film / name)
    print(f'6/6 done: {film / "video.mp4"}  ({timeline["seconds"]:.1f} s), every check passed')
    for s in report['sources']:
        if s['warning'] or s['readable_name_missing']:
            print(f'  for the reviewer, scene {s["scene"]}: {s["warning"]}'
                  f'{" (no readable source name yet)" if s["readable_name_missing"] else ""}')


if __name__ == '__main__':
    main()
