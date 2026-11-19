"""The live demo page (2026-11-18): ask a question, see the graph facts, every script line with its check, and the video prompt.
Runs on this laptop only (127.0.0.1), because the graph is in the local GraphDB. Writer and judge are on Groq's free plan.

    .venv\\Scripts\\python.exe GALLM\\demo_page.py        then open http://127.0.0.1:8770

One question runs at a time (Groq's per-minute limit). A question asked before is answered from the saved model calls."""
import json
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = 8770
PAGE = Path(__file__).with_name('demo_page.html')
EXAMPLES = ['When can I drive again after getting an ICD?',
            'Why should I take an SGLT2 inhibitor for my heart failure?',
            'What should I know about quitting smoking with heart disease?']


class Jobs:
    def __init__(self, pipeline):
        self.pipeline, self.jobs, self.busy = pipeline, {}, threading.Lock()

    def start(self, question):
        if not self.busy.acquire(blocking=False):
            return None
        job = {'id': uuid.uuid4().hex[:8], 'question': question, 'events': [], 'done': False, 'started': time.time()}
        self.jobs[job['id']] = job

        def work():
            try:
                record = self.pipeline.ask(question, lambda event: job['events'].append(event))
                from GALLM.graph_script import RUNS, save
                save(record, ROOT / RUNS / f'page-{time.strftime("%Y%m%d-%H%M%S")}.json')
            except Exception as error:  # shown on the page, so a limit or a closed GraphDB is never silent
                job['events'].append({'step': 'error', 'message': f'{type(error).__name__}: {error}'})
            finally:
                job['done'] = True
                self.busy.release()
        threading.Thread(target=work, daemon=True).start()
        return job


def handler_for(jobs):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code, body, kind='application/json'):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', kind + '; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path in ('/', '/index.html'):
                return self.send(200, PAGE.read_bytes(), 'text/html')
            if self.path == '/examples':
                return self.send(200, EXAMPLES)
            if self.path.startswith('/job/'):
                job_id, _, since = self.path[5:].partition('?since=')
                job = jobs.jobs.get(job_id)
                if not job:
                    return self.send(404, {'error': 'no such job'})
                start = int(since or 0)
                return self.send(200, {'events': job['events'][start:], 'next': len(job['events']), 'done': job['done'],
                                       'seconds': round(time.time() - job['started'])})
            self.send(404, {'error': 'not found'})

        def do_POST(self):
            if self.path != '/ask':
                return self.send(404, {'error': 'not found'})
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
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
    import os
    os.chdir(ROOT)  # the pipeline reads GALLM/data and GAAPO/.env from the thesis root
    from GALLM.graph_script import Pipeline
    print('Loading the graph names and the meaning look-up ...', flush=True)
    jobs = Jobs(Pipeline())
    server = ThreadingHTTPServer(('127.0.0.1', PORT), handler_for(jobs))
    print(f'Ready: http://127.0.0.1:{PORT}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
