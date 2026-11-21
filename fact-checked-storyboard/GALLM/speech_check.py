"""Speech check: does each recorded line say exactly its script words? Run with Lemo-Opuscar's python (it has faster-whisper).

    python GALLM/speech_check.py lines.json voices_dir result.json [--model small.en]

lines.json is the list given to the voice tool: [{"id": ..., "text": ...}, ...], with voices_dir/<id>.wav for each line.
Every line, and the check as a whole, gets one of three results:

  PASS          the recording says the script words exactly (after dropping case and punctuation)
  CONTENT_FAIL  the check ran and a word was added, dropped or changed
  TOOL_ERROR    the check could not run (model missing, file missing, a crash); this is never reported as a content failure

The comparison reuses Lemo-Opuscar's asr_check (exact word-for-word match, whole numbers 0-999 read the same as their
words) and adds what it leaves out: percentages (35% = "35 percent" = "thirty-five percent"), decimals (1.5 = "1 point 5")
and thousands separators (1,000 = 1000), normalised the same way on both sides. The result file also keeps every word's
timing, for captions and mouth movement later. The script always writes result.json and exits 0; it exits 2 only if it
cannot even write that file."""
import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

LEMO = Path(os.environ.get('LEMO_OPUSCAR_HOME', Path.home() / 'lemo-opuscar'))
PAD = 0.6  # seconds of silence on both sides; short lines are misheard without it (same as asr_check)


def lemo_asr_check():
    spec = importlib.util.spec_from_file_location('lemo_asr_check', LEMO / 'core' / 'tts' / 'asr_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spoken_words(text, norm_en):
    """Words as they are spoken, the same for the script and for what Whisper wrote."""
    text = re.sub(r'(\d),(\d{3})\b', r'\1\2', text)                                    # 1,000 -> 1000
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)                         # 35% -> 35 percent
    text = re.sub(r'(\d+)\.(\d+)', lambda m: f'{m.group(1)} point {" ".join(m.group(2))}', text)   # 1.5 -> 1 point 5
    return norm_en(text)


def transcribe(model, wav, np, sf, soxr):
    audio, rate = sf.read(wav)
    if audio.ndim > 1:
        audio = audio.mean(1)
    audio = soxr.resample(audio, rate, 16000, quality='HQ')
    pad = np.zeros(int(PAD * 16000))
    segments, _ = model.transcribe(np.concatenate([pad, audio, pad]).astype(np.float32), beam_size=5, language='en',
                                   word_timestamps=True)
    tail = (len(audio) + len(pad)) / 16000 - 0.5   # Whisper sometimes repeats the line inside the trailing pad
    segments = [s for s in segments if not (s.no_speech_prob > 0.5 and s.start > tail)]
    heard = ' '.join(s.text.strip() for s in segments)
    words = [[w.word.strip(), round(w.start - PAD, 3), round(w.end - PAD, 3)] for s in segments for w in s.words]
    return heard, words


def check(lines, voices, model_name):
    """The result dictionary. Never raises: a problem with the tool becomes TOOL_ERROR."""
    result = {'status': 'TOOL_ERROR', 'model': model_name, 'lines': [], 'error': ''}
    try:
        asr = lemo_asr_check()
        import numpy as np
        import soundfile as sf
        import soxr
        from faster_whisper import WhisperModel
        model = WhisperModel(model_name, device='cpu', compute_type='int8')
    except BaseException as error:   # SystemExit included: a missing package must not look like a wrong word
        result['error'] = f'could not start the check: {type(error).__name__}: {error}'[:500]
        return result
    failed = False
    for line in lines:
        expected = line.get('asr', line['text'])
        entry = {'id': line['id'], 'expected': expected}
        try:
            heard, words = transcribe(model, Path(voices) / f"{line['id']}.wav", np, sf, soxr)
        except BaseException as error:
            entry.update(status='TOOL_ERROR', error=f'{type(error).__name__}: {error}'[:300])
            result['lines'].append(entry)
            result['error'] = f'line {line["id"]}: {entry["error"]}'
            return result
        same = spoken_words(expected, asr.norm_en) == spoken_words(heard, asr.norm_en)
        entry.update(heard=heard, words=words, status='PASS' if same else 'CONTENT_FAIL')
        failed = failed or not same
        result['lines'].append(entry)
    result['status'] = 'CONTENT_FAIL' if failed else 'PASS'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('lines')
    parser.add_argument('voices')
    parser.add_argument('result')
    parser.add_argument('--model', default='small.en')
    args = parser.parse_args()
    try:
        lines = json.loads(Path(args.lines).read_text(encoding='utf-8'))
        result = check(lines, args.voices, args.model)
    except BaseException as error:
        result = {'status': 'TOOL_ERROR', 'model': args.model, 'lines': [], 'error': f'{type(error).__name__}: {error}'[:500]}
    try:
        Path(args.result).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    except OSError as error:
        print(f'speech_check: cannot write {args.result}: {error}', file=sys.stderr)
        sys.exit(2)
    for line in result['lines']:
        print(f"{line['status']:<12} {line['id']}: {line.get('heard', line.get('error', ''))}")
    print(f"speech check: {result['status']} (model {args.model}){' - ' + result['error'] if result['error'] else ''}")


if __name__ == '__main__':
    main()
