"""The live demo page (2026-11-18, video added 2026-11-21): ask a question, see the graph facts, every script line with
its check, what the video shows line by line, and make the video itself.
Runs on this laptop only (127.0.0.1), because the graph is in the local GraphDB. Writer and judge are on Groq's free plan.

    .venv\\Scripts\\python.exe GALLM\\demo_page.py        then open http://127.0.0.1:8770

One question runs at a time (Groq's per-minute limit). A question asked before is answered from the saved model calls.
The storyboard is made by code (director.code_storyboard), so it costs no tokens. "Make the video" runs make_video.py
with Lemo-Opuscar's python (no model, a few minutes); one video at a time."""
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from GALLM import director as dr, make_video as mv, timeline as tl  # noqa: E402

PORT = 8770
PAGE = Path(__file__).with_name('demo_page.html')
VIDEOS = ROOT / 'GALLM' / 'runs' / 'videos'
SHOWN = ('id', 'speaker', 'text', 'shot', 'doctor', 'patient', 'prop')
EXAMPLES = ['When can I drive again after getting an ICD?',
            'Why should I take an SGLT2 inhibitor for my heart failure?',
            'What should I know about quitting smoking with heart disease?']


def video_plan(record):
    """What the video will show, in spoken order: the patient's question, the checked lines and the spoken AI label, each
    with the camera, both people's actions and the object, picked by code from the fixed lists (no model, no tokens).
    None when the record has no checked lines, because then no video can be made."""
    if not record.get('scenes') or not record.get('video_prompt'):
        return None
    board = dr.code_storyboard(record)
    segments = tl.apply_storyboard(tl.segments_of(record, mv.SPOKEN_LABEL, mv.readable_source), board)
    return {'segments': [{k: s.get(k) for k in SHOWN} for s in segments],
            'problems': dr.gate(board, dr.script_lines(record)), 'label': mv.LABEL}


def video_result(name, code, output, videos=VIDEOS):
    """The outcome of one make_video.py run, for the page: the film and its checks, or the check that stopped it."""
    def checks_of(report):
        return {k: v.get('status') for k, v in report.get('checks', {}).items()}
    if code == 0:
        report = json.loads((videos / name / 'report.json').read_text(encoding='utf-8'))
        return {'step': 'finished', 'status': mv.PASS, 'video': f'/video/{name}.mp4', 'seconds': report.get('seconds'),
                'checks': checks_of(report),
                'reviewer': [s for s in report.get('sources', []) if s.get('warning') or s.get('readable_name_missing')]}
    stopped = next((line for line in output if line.startswith('STOPPED')), '')
    shown = [line for line in output if line.strip()]
    why = stopped.split(': ', 1)[1] if ': ' in stopped else (shown[-1] if shown else f'make_video.py ended with exit code {code}')
    checks = {}
    report_line = next((line for line in output if line.startswith('No new video was made. Report: ')), '')
    if report_line:
        try:
            checks = checks_of(json.loads(Path(report_line.split('Report: ', 1)[1]).read_text(encoding='utf-8')))
        except (OSError, ValueError):
            pass
    return {'step': 'finished', 'status': mv.CONTENT_FAIL if code == 1 else mv.TOOL_ERROR, 'why': why, 'checks': checks}


def byte_range(header, size):
    """(start, end) of an HTTP Range header such as 'bytes=0-' (the video player asks for parts of the file), None for no
    range, or 'bad' when it cannot be served."""
    match = re.fullmatch(r'bytes=(\d*)-(\d*)', (header or '').strip())
    if not match or not (match.group(1) or match.group(2)):
        return None
    if match.group(1):
        start = int(match.group(1))
        end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
    else:
        start, end = max(0, size - int(match.group(2))), size - 1
    return 'bad' if start > end or start >= size else (start, end)


class Jobs:
    def __init__(self, pipeline):
        self.pipeline, self.jobs, self.busy = pipeline, {}, threading.Lock()
        self.records = {}   # record name -> saved record file; only these can be made into a video

    def start(self, question):
        if not self.busy.acquire(blocking=False):
            return None
        job = {'id': uuid.uuid4().hex[:8], 'question': question, 'events': [], 'done': False, 'started': time.time()}
        self.jobs[job['id']] = job

        def work():
            try:
                record = self.pipeline.ask(question, lambda event: job['events'].append(event))
                from GALLM.graph_script import RUNS, save
                name = f'page-{time.strftime("%Y%m%d-%H%M%S")}'
                path = ROOT / RUNS / f'{name}.json'
                save(record, path)
                plan = video_plan(record)
                if plan:
                    self.records[name] = path
                    job['events'].append({'step': 'storyboard', 'record': name, **plan})
            except Exception as error:  # shown on the page, so a limit or a closed GraphDB is never silent
                job['events'].append({'step': 'error', 'message': f'{type(error).__name__}: {error}'})
            finally:
                job['done'] = True
                self.busy.release()
        threading.Thread(target=work, daemon=True).start()
        return job


class Videos:
    """One video at a time: make_video.py runs with Lemo-Opuscar's python, and its step lines ("2/6 speech check ...")
    are passed on to the page as they come."""
    def __init__(self, records):
        self.records, self.jobs, self.made, self.busy = records, {}, set(), threading.Lock()

    def start(self, name):
        path = self.records.get(name)
        if path is None:
            return 'unknown'
        if not self.busy.acquire(blocking=False):
            return None
        job = {'id': uuid.uuid4().hex[:8], 'record': name, 'events': [], 'done': False, 'started': time.time()}
        self.jobs[job['id']] = job

        def work():
            output = []
            try:
                env = {k: v for k, v in os.environ.items() if k != 'SSLKEYLOGFILE'}  # Norton sets it; Python's HTTPS crashes
                env.update(PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
                process = subprocess.Popen([str(mv.PYTHON), str(Path(mv.__file__)), str(path)], cwd=ROOT, env=env,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8',
                                           errors='replace')
                for line in process.stdout:
                    output.append(line.rstrip())
                    step = re.match(r'([1-6])/6 (.*)', line)
                    if step:
                        job['events'].append({'step': 'progress', 'n': int(step.group(1)), 'text': step.group(2).strip()})
                result = video_result(name, process.wait(), output)
                if result['status'] == mv.PASS:
                    self.made.add(name)
                job['events'].append(result)
            except Exception as error:  # e.g. Lemo-Opuscar's python not found
                job['events'].append({'step': 'finished', 'status': mv.TOOL_ERROR, 'why': f'{type(error).__name__}: {error}',
                                      'checks': {}})
            finally:
                job['done'] = True
                self.busy.release()
        threading.Thread(target=work, daemon=True).start()
        return job


def handler_for(jobs, videos):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def send_video(self, path):
            size = path.stat().st_size
            part = byte_range(self.headers.get('Range'), size)
            if part == 'bad':
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                return self.end_headers()
            start, end = part or (0, size - 1)
            self.send_response(206 if part else 200)
            if part:
                self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
            self.send_header('Content-Type', 'video/mp4')
            self.send_header('Accept-Ranges', 'bytes')
            self.send_header('Content-Length', str(end - start + 1))
            self.end_headers()
            try:
                with path.open('rb') as file:
                    file.seek(start)
                    left = end - start + 1
                    while left > 0:
                        chunk = file.read(min(left, 1 << 16))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        left -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass   # the player stopped reading (it seeks by asking for another part)

        def poll(self, table, rest):
            job_id, _, since = rest.partition('?since=')
            job = table.get(job_id)
            if not job:
                return self.send(404, {'error': 'no such job'})
            start = int(since or 0)
            return self.send(200, {'events': job['events'][start:], 'next': len(job['events']), 'done': job['done'],
                                   'seconds': round(time.time() - job['started'])})

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                return self.send(200, PAGE.read_bytes(), 'text/html')
            if self.path == '/examples':
                return self.send(200, EXAMPLES)
            if self.path.startswith('/job/'):
                return self.poll(jobs.jobs, self.path[5:])
            if self.path.startswith('/video-job/'):
                return self.poll(videos.jobs, self.path[11:])
            film = re.fullmatch(r'/video/([A-Za-z0-9_.-]+)\.mp4', self.path.partition('?')[0])
            if film and film.group(1) in videos.made:   # only videos made from this page, never other files
                return self.send_video(VIDEOS / film.group(1) / 'video.mp4')
            self.send(404, {'error': 'not found'})

        def do_POST(self):
            if self.path not in ('/ask', '/video'):
                return self.send(404, {'error': 'not found'})
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
            if self.path == '/video':
                job = videos.start(str(body.get('record', '')))
                if job == 'unknown':
                    return self.send(404, {'error': 'Ask the question again: this page does not know that script.'})
                if job is None:
                    return self.send(409, {'error': 'Another video is still being made. Wait for it to finish.'})
                return self.send(200, {'id': job['id']})
            question = ' '.join(str(body.get('question', '')).split())[:300]
            if not question:
                return self.send(400, {'error': 'Type a question first.'})
            job = jobs.start(question)
            if job is None:
                return self.send(409, {'error': 'Another question is still running. Wait for it to finish.'})
            self.send(200, {'id': job['id']})

        def log_message(self, *args):
            pass
    return Handler


def main():
    os.chdir(ROOT)  # the pipeline reads GALLM/data and GAAPO/.env from the thesis root
    from GALLM.graph_script import Pipeline
    print('Loading the graph names and the meaning look-up ...', flush=True)
    jobs = Jobs(Pipeline())
    server = ThreadingHTTPServer(('127.0.0.1', PORT), handler_for(jobs, Videos(jobs.records)))
    print(f'Ready: http://127.0.0.1:{PORT}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
