/* The animated consultation: draws frame t of the video from window.TL (timeline.json) with the actor kit (kit.js).
 *
 * Everything is a pure function of t (the renderer draws frames in parallel, in any order):
 *   camera      the segment's shot, cut at the start of each line
 *   poses       each person's action for the line, blended from the previous line's pose over half a second
 *   life        breathing, blinking (fixed pseudo-random times), slow nods while listening, hand movement while talking
 *   mouth       only the speaker's mouth moves, opened by the loudness of their own voice in that frame
 *   object      the line's object, on the desk or in the doctor's hand
 *   text        the caption band shows the spoken words exactly; the checked heading; the source line; the AI label
 * window.TEXTS(t) lists every text on screen with its box, for the reading-time check (readcheck.mjs).
 */
(function () {
  'use strict';
  const TL = window.TL, FPS = TL.fps, SEGS = TL.segments;
  window.DUR = TL.seconds;
  const $ = id => document.getElementById(id);
  const at = t => SEGS.find(s => t >= s.t0 && t < s.t1) || SEGS[SEGS.length - 1];
  const lerp = (a, b, k) => a + (b - a) * k;
  const smooth = k => (k <= 0 ? 0 : k >= 1 ? 1 : k * k * (3 - 2 * k));
  const wave = (t, hz, phase = 0) => Math.sin(2 * Math.PI * hz * t + phase);
  const hash = n => { const x = Math.sin(n * 127.1 + 311.7) * 43758.5453; return x - Math.floor(x); };
  const BLEND = 0.5;   // seconds to move from one pose to the next

  function blend(from, to, k) {
    const p = JSON.parse(JSON.stringify(to));
    p.lean = lerp(from.lean, to.lean, k);
    p.nod = lerp(from.nod, to.nod, k);
    p.gaze = [lerp(from.gaze[0], to.gaze[0], k), lerp(from.gaze[1], to.gaze[1], k)];
    for (const arm of ['near', 'far']) {
      p[arm].a1 = lerp(from[arm].a1, to[arm].a1, k);
      p[arm].a2 = lerp(from[arm].a2, to[arm].a2, k);
      if (k < 0.5) p[arm].hand = from[arm].hand;
    }
    if (k < 0.5) { p.brows = from.brows; p.eyes = from.eyes; p.mouth = from.mouth; }
    return p;
  }

  function blinking(t, seed) {   // eyes closed for 0.13 s, every 2.8-5 s
    for (let i = 0, start = 0.8 + seed; start <= t; i++) {
      if (t < start + 0.13) return true;
      start += 2.8 + 2.2 * hash(i * 7 + seed * 13);
    }
    return false;
  }

  function speaking(seg, who, t) {
    return seg.speaker === who && t >= seg.voice_at && t < seg.voice_at + seg.voice;
  }

  function mouth(seg, who, t) {
    if (!speaking(seg, who, t)) return null;
    const v = (seg.loudness || [])[Math.floor((t - seg.voice_at) * FPS)];
    if (v === undefined) return null;
    return v < 0.12 ? 'rest' : v < 0.35 ? 'small' : v < 0.68 ? 'open' : 'wide';
  }

  function life(who, p, seg, t) {
    const local = t - seg.t0, action = seg[who], talk = speaking(seg, who, t);
    p.lean += 0.7 * wave(t, 0.22, who === 'doctor' ? 0 : 1.7);                     // breathing
    if (action === 'nods') p.nod += 7 * Math.max(0, wave(local, 1.1)) * (local < 2.4 ? 1 : 0);
    if (talk) {
      p.nod += 1.6 * wave(t, 0.9, 0.5);
      if (['explains', 'gestures calmly', 'asks the question'].includes(action)) {
        p.near.a1 += 7 * wave(t, 0.6); p.near.a2 += 9 * wave(t, 0.45, 1);
      }
    } else if (['listens', 'looks at the prop'].includes(action)) {
      p.nod += 2 * wave(t, 0.3, who === 'doctor' ? 2 : 0);
    }
    return p;
  }

  function pose(who, seg, prev, t) {
    const to = KIT.pose(who, seg[who]);
    const from = KIT.pose(who, prev[who]);
    const p = life(who, prev === seg ? to : blend(from, to, smooth((t - seg.t0) / BLEND)), seg, t);
    const m = mouth(seg, who, t);
    if (m) p.mouth = m;
    if (blinking(t, who === 'doctor' ? 0.3 : 1.9)) p.eyes = 'closed';
    return p;
  }

  window.render = t => {
    const seg = at(t), i = SEGS.indexOf(seg), prev = i > 0 ? SEGS[i - 1] : seg;
    const prop = seg.prop && seg.prop !== 'none' ? {name: seg.prop, at: seg.prop_at || 'desk'} : null;
    const shot = seg.shot === 'close-up on prop' && !prop ? 'medium' : seg.shot;
    $('stage').innerHTML = KIT.scene({doctor: pose('doctor', seg, prev, t), patient: pose('patient', seg, prev, t), prop, shot});
    $('caption').textContent = seg.text;
    $('source').textContent = seg.source || '';
    const heading = seg.show_heading === false ? '' : seg.heading || '';
    $('heading').textContent = heading;
    $('heading').style.display = heading ? 'block' : 'none';
  };

  window.TEXTS = t => {
    window.render(t);
    const seg = at(t), box = id => { const r = $(id).getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom]; };
    const heading = seg.show_heading === false ? '' : seg.heading || '';
    return [['caption', seg.text], ['heading', heading], ['source', seg.source || '']].filter(([, x]) => x)
      .map(([id, text]) => { const [x0, y0, x1, y1] = box(id); return {id: id + '-' + seg.id, text, x0, y0, x1, y1}; });
  };

  document.fonts.ready.then(() => { window.render(0); window.READY = true; });
})();
