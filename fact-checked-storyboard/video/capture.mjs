// Films index.html frame by frame with headless Edge (already on Windows), through its DevTools protocol.
//   node capture.mjs --times 3.3,7.2     a few frames, to look at  -> frames/check-<t>.png
//   node capture.mjs --all               every frame at the timeline's fps -> frames/00000.png ...
// Serve this folder first: python -m http.server 8766 --bind 127.0.0.1
import { spawn } from 'node:child_process';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const EDGE = 'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe';
const PAGE = 'http://127.0.0.1:8766/?capture=1';
const PORT = 9333;
const args = process.argv.slice(2);
const out = join(import.meta.dirname, 'frames');
mkdirSync(out, { recursive: true });

const edge = spawn(EDGE, ['--headless=new', `--remote-debugging-port=${PORT}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'edge-'))}`,
  '--hide-scrollbars', '--mute-audio', '--no-first-run', '--disable-extensions', '--disable-component-extensions-with-background-pages',
  '--disable-features=msEdgeSidebarV2,msHubApps,msCopilot', 'about:blank'], { stdio: 'ignore' });  // nothing may draw over the page
const sleep = ms => new Promise(r => setTimeout(r, ms));
let target;
for (let i = 0; i < 50 && !target; i++) {
  try { target = (await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()).find(t => t.type === 'page'); } catch { await sleep(200); }
}
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r));
let id = 0;
const waiting = new Map();
ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); } });
const send = (method, params = {}) => new Promise((resolve, reject) => {
  const n = ++id; waiting.set(n, m => m.error ? reject(new Error(method + ': ' + m.error.message)) : resolve(m.result));
  ws.send(JSON.stringify({ id: n, method, params }));
});

try {
  await send('Emulation.setDeviceMetricsOverride', { width: 1280, height: 720, deviceScaleFactor: 1, mobile: false });
  await send('Page.enable');
  await send('Page.navigate', { url: PAGE });
  for (let i = 0; i < 50; i++) {
    const r = await send('Runtime.evaluate', { expression: 'typeof window.renderAt === "function" && !!window.TIMELINE', returnByValue: true });
    if (r.result.value) break;
    await sleep(100);
  }
  const { result } = await send('Runtime.evaluate', { expression: 'JSON.stringify([TIMELINE.total, TIMELINE.fps])', returnByValue: true });
  const [total, fps] = JSON.parse(result.value);
  const times = args[0] === '--all'
    ? Array.from({ length: Math.ceil(total * fps) }, (_, i) => i / fps)
    : args[1].split(',').map(Number);
  for (const [i, t] of times.entries()) {
    await send('Runtime.evaluate', { expression: `renderAt(${t})` });
    const shot = await send('Page.captureScreenshot', { format: 'png', clip: { x: 0, y: 0, width: 1280, height: 720, scale: 1 } });
    const name = args[0] === '--all' ? String(i).padStart(5, '0') + '.png' : `check-${t}.png`;
    writeFileSync(join(out, name), Buffer.from(shot.data, 'base64'));
    if (args[0] === '--all' && i % (fps * 5) === 0) console.log(`${t.toFixed(1)} / ${total} s`);
  }
  console.log(`${times.length} frame(s) in ${out}`);
} finally {
  ws.close();
  edge.kill();
}
