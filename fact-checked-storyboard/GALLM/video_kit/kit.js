/* The actor kit: everything the patient video can show, drawn once in code (SVG) and reused for every video.
 *
 * Nothing here is drawn by hand per video. The video engine (later) reads the director's storyboard for each line
 * (camera shot, doctor action, patient action, one object) and asks this kit for exactly that picture at time t.
 *
 *   KIT.scene({doctor, patient, prop, shot})  -> one SVG picture of the consultation room
 *       doctor / patient: a pose name from KIT.ACTIONS, or a pose object (see POSES), plus optional overrides
 *       prop: {name: one of KIT.PROPS, at: 'desk' | 'doctor-hand'}
 *       shot: one of KIT.SHOTS
 *   KIT.prop(name)                            -> one object on its own (centred, about 140 units wide)
 *
 * Rules kept by design: no letters, numbers or logos anywhere; every object and action comes from a fixed list.
 */
(function (root) {
  'use strict';

  // ---- colours ----------------------------------------------------------------------------------------------------
  const ROOMC = {wall: '#e7eeec', floor: '#d8cab5', floorLine: '#c7b59d', base: '#f7f9f8', frame: '#fbfcfc',
                 sky: '#d6eaf2', hill: '#bcd8b3', hill2: '#a9cc9f', tree: '#8fbf86', trunk: '#9b7a5a',
                 deskTop: '#c99a64', deskFront: '#b4864f', deskEdge: '#a77a46', shelf: '#c9a77d'};
  const LOOK = {
    doctor: {skin: '#8d5b3e', skinShade: '#784a30', hair: '#2b1e17', top: '#ffffff', topShade: '#dfe6e9',
             line: '#cdd6da', inner: '#2e8c84', legs: '#2e8c84', legsShade: '#267a73', shoe: '#3b3b3b',
             sleeve: '#ffffff', sleeveShade: '#e9eef0', lips: '#5c2a20', brow: '#2b1e17', chair: '#4a5d66'},
    patient: {skin: '#f0c6a6', skinShade: '#d8a683', hair: '#d2d2d2', top: '#a8553f', topShade: '#8f4634',
              line: '#8f4634', inner: '#f3efe4', legs: '#4b5566', legsShade: '#3f4857', shoe: '#5a3b2a',
              sleeve: '#a8553f', sleeveShade: '#924a37', lips: '#8a4a3c', brow: '#b9b9b9', chair: '#9b7350'},
  };

  // ---- the 14 actions as poses ------------------------------------------------------------------------------------
  // Angles in degrees. Arms: a1 = shoulder (0 = hanging down, 90 = straight forward), a2 = elbow bend.
  // lean: + forward. nod: + head down. gaze: eye shift [x, y]. Faces: brows, eyes, mouth (see FACE below).
  const BASE = {
    doctor: {lean: 0, nod: 0, gaze: [1, 0], brows: 'neutral', eyes: 'open', mouth: 'smile',
             near: {a1: 30, a2: 55, hand: 'open'}, far: {a1: 28, a2: 58, hand: 'open'}},
    patient: {lean: 4, nod: 0, gaze: [1, 0], brows: 'neutral', eyes: 'open', mouth: 'rest',
              near: {a1: 10, a2: 50, hand: 'open'}, far: {a1: 8, a2: 52, hand: 'open'}},
  };
  const POSES = {
    doctor: {
      'explains':           {lean: 4, mouth: 'open', near: {a1: 55, a2: 70, hand: 'open'}},
      'points to the prop': {lean: 8, nod: 6, gaze: [3, 1], mouth: 'small', near: {a1: 86, a2: 6, hand: 'point'}},
      'holds up the prop':  {lean: 3, mouth: 'small', near: {a1: 70, a2: 80, hand: 'hold'}},
      'smiles warmly':      {nod: 3, brows: 'soft', eyes: 'soft', mouth: 'smile'},
      'nods':               {nod: 12, mouth: 'smile'},
      'leans forward':      {lean: 12, brows: 'raised', mouth: 'rest', near: {a1: 45, a2: 50, hand: 'open'}},
      'gestures calmly':    {lean: 3, mouth: 'open', near: {a1: 48, a2: 62, hand: 'open'}, far: {a1: 55, a2: 58, hand: 'open'}},
    },
    patient: {
      'asks the question':  {brows: 'raised', mouth: 'open', near: {a1: 50, a2: 75, hand: 'open'}},
      'listens':            {},
      'nods':               {nod: 12, mouth: 'smile'},
      'looks relieved':     {lean: -3, brows: 'soft', eyes: 'soft', mouth: 'smile'},
      'thinks':             {nod: 5, gaze: [1, -2], brows: 'think', mouth: 'side', near: {a1: 70, a2: 172, hand: 'chin'}},
      'looks at the prop':  {nod: 10, gaze: [3, 2]},
      'smiles':             {eyes: 'soft', mouth: 'smile'},
    },
  };
  const ACTIONS = {doctor: Object.keys(POSES.doctor), patient: Object.keys(POSES.patient)};
  const MOUTHS = ['rest', 'smile', 'small', 'open', 'wide', 'round', 'side'];

  function pose(who, p) {
    const named = typeof p === 'string' ? POSES[who][p] : (p && p.action ? POSES[who][p.action] : null);
    if (typeof p === 'string' && !named) throw new Error(`unknown ${who} action: ${p}`);
    const extra = typeof p === 'object' && p ? p : {};
    const out = {...BASE[who], ...(named || {}), ...extra};
    out.near = {...BASE[who].near, ...((named || {}).near || {}), ...(extra.near || {})};
    out.far = {...BASE[who].far, ...((named || {}).far || {}), ...(extra.far || {})};
    return out;
  }

  // ---- character geometry (local: hip on the seat at 0,0, facing right, y down) ---------------------------------
  const L1 = 88, L2 = 80;                         // upper arm, forearm
  const SHOULDER = {near: [36, -168], far: [-30, -170]};
  const HEAD_Y = -246;                            // head centre above the hip
  const r = n => Math.round(n * 10) / 10;

  // Eye, brow and mouth shapes. Eyes at x 12 and 38 (the face is turned a little to the right).
  const EYES_X = [12, 38];
  function eyes(state, gaze, colour) {
    const [gx, gy] = gaze;
    return EYES_X.map(x => {
      const cx = r(x + gx), cy = r(-2 + gy);
      if (state === 'closed') return `<path d="M${cx - 6} ${cy} Q${cx} ${cy + 4} ${cx + 6} ${cy}" stroke="${colour}" stroke-width="3" fill="none" stroke-linecap="round"/>`;
      const ry = state === 'soft' ? 4.2 : 6;
      return `<ellipse cx="${cx}" cy="${cy}" rx="4.6" ry="${ry}" fill="#262626"/><circle cx="${r(cx + 1.4)}" cy="${r(cy - 1.8)}" r="1.3" fill="#ffffff" opacity=".8"/>`;
    }).join('');
  }
  function brows(type, colour) {
    // inner ends point at the gap between the eyes (x = 25)
    const lift = {neutral: [0, 0, 0, 0], raised: [-6, -7, -7, -6], soft: [0, -5, -5, 0], think: [1, 1, -5, -8],
                  concern: [0, -4, -4, 0]}[type] || [0, 0, 0, 0];
    const [l_out, l_in, r_in, r_out] = lift;
    const y = -20;
    return `<path d="M${EYES_X[0] - 9} ${y + l_out} Q${EYES_X[0]} ${y - 4 + (l_out + l_in) / 2} ${EYES_X[0] + 8} ${y + l_in}" stroke="${colour}" stroke-width="4.2" fill="none" stroke-linecap="round"/>` +
           `<path d="M${EYES_X[1] - 8} ${y + r_in} Q${EYES_X[1]} ${y - 4 + (r_in + r_out) / 2} ${EYES_X[1] + 8} ${y + r_out}" stroke="${colour}" stroke-width="4.2" fill="none" stroke-linecap="round"/>`;
  }
  function mouth(shape, lips) {
    const x = 30, y = 30, dark = '#4a1c18', tongue = '#d9706a';
    switch (shape) {
      case 'smile': return `<path d="M${x - 12} ${y - 3} Q${x} ${y + 8} ${x + 11} ${y - 3}" stroke="${lips}" stroke-width="3.2" fill="none" stroke-linecap="round"/>`;
      case 'small': return `<ellipse cx="${x}" cy="${y}" rx="6.5" ry="3.6" fill="${dark}"/>`;
      case 'open':  return `<path d="M${x - 9} ${y - 2} Q${x} ${y - 4} ${x + 9} ${y - 2} Q${x + 7} ${y + 8} ${x} ${y + 8} Q${x - 7} ${y + 8} ${x - 9} ${y - 2} Z" fill="${dark}"/><ellipse cx="${x}" cy="${y + 5}" rx="4.5" ry="2" fill="${tongue}"/>`;
      case 'wide':  return `<rect x="${x - 11}" y="${y - 4}" width="22" height="9" rx="4.5" fill="${dark}"/><rect x="${x - 8}" y="${y - 4}" width="16" height="3" rx="1.5" fill="#ffffff" opacity=".85"/>`;
      case 'round': return `<ellipse cx="${x}" cy="${y + 1}" rx="5.2" ry="7" fill="${dark}"/>`;
      case 'side':  return `<path d="M${x - 9} ${y + 1} Q${x + 1} ${y + 3} ${x + 9} ${y - 3}" stroke="${lips}" stroke-width="3.2" fill="none" stroke-linecap="round"/>`;
      default:      return `<path d="M${x - 9} ${y} Q${x} ${y + 3} ${x + 9} ${y}" stroke="${lips}" stroke-width="3.2" fill="none" stroke-linecap="round"/>`;
    }
  }

  function head(who, p, c) {
    const face = brows(p.brows, c.brow) + eyes(p.eyes, p.gaze, c.skinShade) + mouth(p.mouth, c.lips) +
      `<path d="M47 5 Q56 16 47 20" stroke="${c.skinShade}" stroke-width="3" fill="none" stroke-linecap="round"/>` +
      (who === 'patient' ? `<circle cx="38" cy="15" r="8" fill="#e0806f" opacity=".22"/>` : '');
    let back = '', front = '';
    if (who === 'doctor') {
      back = `<circle cx="-40" cy="-34" r="21" fill="${c.hair}"/>`;
      front = `<path d="M-44 14 Q-54 -48 6 -59 Q54 -58 58 -16 Q42 -38 12 -36 Q-14 -32 -26 -4 Q-30 6 -30 16 Z" fill="${c.hair}"/>`;
    } else {
      back = '';
      front = `<path d="M-44 22 Q-56 -14 -36 -40 Q-14 -58 16 -55 Q-4 -46 -16 -30 Q-26 -14 -26 20 Z" fill="${c.hair}"/>` +
              `<path d="M-6 -33 Q14 -37 34 -33" stroke="${c.skinShade}" stroke-width="2" fill="none" opacity=".55" stroke-linecap="round"/>` +
              EYES_X.map(x => `<circle cx="${x}" cy="-2" r="11" stroke="#3a3a3a" stroke-width="3" fill="#ffffff" fill-opacity=".12"/>`).join('') +
              `<path d="M23 -3 L27 -3 M1 -4 L-27 0" stroke="#3a3a3a" stroke-width="3" fill="none" stroke-linecap="round"/>`;
    }
    return `<g transform="translate(0 ${HEAD_Y}) rotate(${p.nod} 0 50)">` + back +
      `<ellipse cx="8" cy="0" rx="50" ry="56" fill="${c.skin}"/>` +
      `<ellipse cx="-29" cy="6" rx="8" ry="12" fill="${c.skinShade}"/>` + front + face + `</g>`;
  }

  function hand(type, colour) {
    if (type === 'point') return `<rect x="-4.5" y="${L2}" width="9" height="26" rx="4.5" fill="${colour}"/><circle cx="0" cy="${L2}" r="12.5" fill="${colour}"/>`;
    if (type === 'fist') return `<circle cx="0" cy="${L2}" r="13" fill="${colour}"/>`;
    if (type === 'chin') return `<line x1="0" y1="${L2 - 8}" x2="0" y2="${L2 + 8}" stroke="${colour}" stroke-width="16" stroke-linecap="round"/><circle cx="0" cy="${L2 + 14}" r="13" fill="${colour}"/>`;
    return `<circle cx="0" cy="${L2}" r="14" fill="${colour}"/><ellipse cx="9" cy="${L2 - 6}" rx="5" ry="8" fill="${colour}" transform="rotate(-25 9 ${L2 - 6})"/>`;
  }

  function arm(which, a, c, lean, held, outline) {
    const [sx, sy] = SHOULDER[which];
    const near = which === 'near';
    const col = near ? c.sleeve : c.sleeveShade, skin = near ? c.skin : c.skinShade;
    const edge = outline ? (w, len) => `<line x1="0" y1="0" x2="0" y2="${len}" stroke="${c.line}" stroke-width="${w + 3}" stroke-linecap="round"/>` : () => '';
    const prop = held ? `<g transform="translate(0 ${L2}) rotate(${a.a1 + a.a2 - lean}) translate(0 -34) scale(.55)">${held}</g>` : '';
    return `<g transform="translate(${sx} ${sy}) rotate(${-a.a1})">` + edge(30, L1) +
      `<line x1="0" y1="0" x2="0" y2="${L1}" stroke="${col}" stroke-width="30" stroke-linecap="round"/>` +
      `<g transform="translate(0 ${L1}) rotate(${-a.a2})">` + edge(26, L2 - 8) +
      `<line x1="0" y1="0" x2="0" y2="${L2 - 8}" stroke="${col}" stroke-width="26" stroke-linecap="round"/>` +
      prop + hand(a.hand, skin) + `</g></g>`;
  }

  function torso(who, c) {
    const w = who === 'doctor' ? 62 : 66;
    let s = `<path d="M${-w + 4} 0 L${-w} -140 Q${-w + 2} -178 -30 -186 L30 -186 Q${w - 2} -178 ${w} -140 L${w - 4} 0 Z" fill="${c.top}" stroke="${c.line}" stroke-width="2"/>`;
    s += `<rect x="-13" y="-208" width="26" height="26" fill="${c.skin}"/>`;
    if (who === 'doctor') {
      s += `<path d="M-24 -186 L0 -118 L24 -186 Z" fill="${c.inner}"/><path d="M-11 -186 L0 -165 L11 -186 Z" fill="${c.skin}"/>` +
           `<path d="M-24 -186 L-5 -108 M24 -186 L5 -108 M0 -108 L0 0" stroke="${c.line}" stroke-width="2.5" fill="none"/>` +
           `<rect x="-50" y="-84" width="30" height="26" rx="3" fill="none" stroke="${c.line}" stroke-width="2.5"/>` +
           `<line x1="-44" y1="-92" x2="-44" y2="-80" stroke="${c.inner}" stroke-width="4" stroke-linecap="round"/>` +
           `<path d="M-21 -186 C-40 -140 -22 -108 0 -104 C22 -108 40 -140 21 -186" stroke="#3b4b54" stroke-width="5" fill="none"/>` +
           `<circle cx="0" cy="-99" r="9" fill="#c3ccd1" stroke="#3b4b54" stroke-width="3"/>`;
    } else {
      s += `<path d="M-22 -186 L0 -150 L22 -186 Z" fill="${c.inner}"/>` +
           `<path d="M-22 -186 L-8 -150 L0 -150 L8 -150 L22 -186" stroke="#d9d3c4" stroke-width="2" fill="none"/>` +
           `<line x1="0" y1="-150" x2="0" y2="0" stroke="${c.line}" stroke-width="2.5"/>` +
           [-128, -96, -64, -32].map(y => `<circle cx="7" cy="${y}" r="3.5" fill="${c.topShade}"/>`).join('');
    }
    return s;
  }

  function legs(c) {
    const leg = (dx, dy, col) => `<g transform="translate(${dx} ${dy})">` +
      `<line x1="0" y1="0" x2="105" y2="0" stroke="${col}" stroke-width="40" stroke-linecap="round"/>` +
      `<line x1="105" y1="0" x2="105" y2="126" stroke="${col}" stroke-width="34" stroke-linecap="round"/>` +
      `<ellipse cx="118" cy="138" rx="27" ry="11" fill="${c.shoe}"/></g>`;
    return leg(-8, -6, c.legsShade) + leg(0, 0, c.legs);
  }

  function chair(who, c) {
    if (who === 'doctor') {
      return `<rect x="-92" y="-176" width="30" height="170" rx="14" fill="${c.chair}"/>` +
        `<rect x="-72" y="8" width="160" height="20" rx="8" fill="${c.chair}"/>` +
        `<rect x="4" y="28" width="12" height="96" fill="#39484f"/><path d="M-50 124 L70 124" stroke="#39484f" stroke-width="10" stroke-linecap="round"/>`;
    }
    return `<rect x="-78" y="-168" width="16" height="178" rx="7" fill="${c.chair}"/>` +
      `<rect x="-74" y="10" width="166" height="18" rx="6" fill="${c.chair}"/>` +
      `<rect x="-66" y="28" width="11" height="122" fill="#86613f"/><rect x="80" y="28" width="11" height="122" fill="#86613f"/>`;
  }

  // One character as two layers: back (chair, legs, far arm, body, head) and front (near arm, which may cross the desk).
  function character(who, p, held) {
    const c = LOOK[who];
    const back = chair(who, c) + legs(c) +
      `<g transform="rotate(${p.lean})">` + arm('far', p.far, c, p.lean, null, true) + torso(who, c) + head(who, p, c) + `</g>`;
    const front = `<g transform="rotate(${p.lean})">` + arm('near', p.near, c, p.lean, held, true) + `</g>`;
    return {back, front};
  }

  // Where a character's near hand ends up in the room (for a close-up on a held object).
  function handPosition(who, p) {
    const o = PLACE[who];
    const rad = d => d * Math.PI / 180;
    const rot = (x, y, d) => [x * Math.cos(rad(d)) - y * Math.sin(rad(d)), x * Math.sin(rad(d)) + y * Math.cos(rad(d))];
    let [x, y] = rot(...SHOULDER.near, p.lean);
    const t1 = p.lean - p.near.a1, t2 = t1 - p.near.a2;
    x += -L1 * Math.sin(rad(t1)) - L2 * Math.sin(rad(t2));
    y += L1 * Math.cos(rad(t1)) + L2 * Math.cos(rad(t2));
    return [o.x + o.flip * x, o.y + y];
  }

  // ---- the room ---------------------------------------------------------------------------------------------------
  const PLACE = {doctor: {x: 600, y: 750, flip: 1}, patient: {x: 1330, y: 750, flip: -1}};
  const DESK_PROP = {x: 950, y: 680};             // where an object stands on the desk (its bottom edge)

  function room() {
    const c = ROOMC;
    const books = [['#5d8aa8', 26, 92], ['#c0705a', 22, 104], ['#8fae7e', 30, 86], ['#d9b45a', 24, 98], ['#6f6f8f', 28, 90]];
    let bx = 1500;
    const shelfBooks = books.map(([col, w, h]) => { const s = `<rect x="${bx}" y="${360 - h}" width="${w}" height="${h}" rx="3" fill="${col}"/>`; bx += w + 4; return s; }).join('');
    return `<rect x="0" y="0" width="1920" height="900" fill="${c.wall}"/>` +
      `<rect x="0" y="900" width="1920" height="180" fill="${c.floor}"/><rect x="0" y="884" width="1920" height="16" fill="${c.base}"/>` +
      `<line x1="0" y1="900" x2="1920" y2="900" stroke="${c.floorLine}" stroke-width="3"/>` +
      // window
      `<rect x="96" y="150" width="350" height="400" rx="8" fill="${c.frame}"/>` +
      `<rect x="114" y="168" width="314" height="364" fill="${c.sky}"/>` +
      `<path d="M114 450 Q190 400 270 430 Q350 460 428 410 L428 532 L114 532 Z" fill="${c.hill}"/>` +
      `<path d="M114 490 Q230 460 330 486 Q390 500 428 480 L428 532 L114 532 Z" fill="${c.hill2}"/>` +
      `<rect x="338" y="410" width="10" height="48" fill="${c.trunk}"/><circle cx="343" cy="396" r="30" fill="${c.tree}"/>` +
      `<line x1="271" y1="168" x2="271" y2="532" stroke="${c.frame}" stroke-width="10"/><line x1="114" y1="350" x2="428" y2="350" stroke="${c.frame}" stroke-width="10"/>` +
      `<rect x="80" y="548" width="382" height="16" rx="4" fill="${c.frame}"/>` +
      // picture (abstract, no text)
      `<rect x="840" y="196" width="262" height="182" rx="6" fill="#ffffff" stroke="#c9d2d0" stroke-width="3"/>` +
      `<rect x="856" y="212" width="230" height="150" fill="#f3f6f5"/><circle cx="930" cy="276" r="34" fill="#f2c6a0"/>` +
      `<path d="M856 330 Q920 290 980 322 Q1036 350 1086 312 L1086 362 L856 362 Z" fill="#9fc5c0"/>` +
      // shelf with books and a small plant
      `<rect x="1480" y="360" width="290" height="12" rx="3" fill="${c.shelf}"/>` + shelfBooks +
      `<rect x="1704" y="326" width="40" height="34" rx="5" fill="#d9e3e0"/><ellipse cx="1716" cy="318" rx="12" ry="20" fill="#7fb37a" transform="rotate(-20 1716 318)"/><ellipse cx="1734" cy="316" rx="11" ry="19" fill="#6aa565" transform="rotate(22 1734 316)"/>` +
      // tall floor plant
      `<path d="M1662 820 L1746 820 L1734 900 L1674 900 Z" fill="#e2ebe8"/>` +
      [[-34, 0], [-12, -16], [14, 10], [-50, 26], [34, -4]].map(([a, d], i) => `<ellipse cx="${1704 + d}" cy="${720 - i * 6}" rx="22" ry="82" fill="${i % 2 ? '#6aa565' : '#7fb37a'}" transform="rotate(${a} ${1704 + d} 800)"/>`).join('');
  }

  function desk() {
    const c = ROOMC;
    return `<rect x="430" y="680" width="710" height="22" rx="5" fill="${c.deskTop}"/>` +
      `<rect x="455" y="702" width="660" height="198" fill="${c.deskFront}"/>` +
      `<line x1="455" y1="704" x2="1115" y2="704" stroke="${c.deskEdge}" stroke-width="4"/>` +
      `<line x1="785" y1="726" x2="785" y2="880" stroke="${c.deskEdge}" stroke-width="3" opacity=".6"/>`;
  }

  // ---- the 18 objects (+ none). Each is centred on x = 0 with its bottom edge at y = 60, about 140 wide, no text. ----
  const PROP = {
    'heart model': () => `<rect x="-30" y="50" width="60" height="10" rx="4" fill="#97a3a8"/><rect x="-3" y="34" width="6" height="18" fill="#97a3a8"/>` +
      `<g transform="rotate(-10)"><rect x="-18" y="-74" width="13" height="30" rx="5" fill="#d86a72"/><rect x="5" y="-70" width="13" height="26" rx="5" fill="#6f8fc0"/>` +
      `<path d="M-12 -70 Q-12 -92 10 -88 Q28 -84 24 -66" stroke="#d86a72" stroke-width="10" fill="none" stroke-linecap="round"/>` +
      `<path d="M0 38 C-52 8 -62 -32 -32 -50 C-14 -60 0 -46 0 -36 C0 -46 14 -60 32 -50 C62 -32 52 8 0 38 Z" fill="#cf4f59"/>` +
      `<path d="M-30 -30 Q-36 -12 -24 4" stroke="#e98a91" stroke-width="5" fill="none" stroke-linecap="round"/></g>`,
    'calendar': () => `<rect x="-56" y="-50" width="112" height="108" rx="8" fill="#ffffff" stroke="#cfd6d8" stroke-width="3"/>` +
      `<path d="M-56 -26 L-56 -42 Q-56 -50 -48 -50 L48 -50 Q56 -50 56 -42 L56 -26 Z" fill="#d9605a"/>` +
      `<rect x="-34" y="-60" width="8" height="18" rx="4" fill="#4a5a62"/><rect x="26" y="-60" width="8" height="18" rx="4" fill="#4a5a62"/>` +
      Array.from({length: 20}, (_, i) => `<rect x="${-46 + (i % 5) * 19}" y="${-16 + Math.floor(i / 5) * 18}" width="14" height="13" rx="2" fill="${i === 12 ? '#f3b6a8' : '#e6ebed'}"/>`).join(''),
    'clock': () => `<circle cx="0" cy="4" r="54" fill="#ffffff" stroke="#5b6b72" stroke-width="7"/>` +
      Array.from({length: 12}, (_, i) => { const a = i * Math.PI / 6; const l = i % 3 === 0 ? 10 : 6; return `<line x1="${r(Math.sin(a) * 44)}" y1="${r(4 - Math.cos(a) * 44)}" x2="${r(Math.sin(a) * (44 - l))}" y2="${r(4 - Math.cos(a) * (44 - l))}" stroke="#5b6b72" stroke-width="3" stroke-linecap="round"/>`; }).join('') +
      `<line x1="0" y1="4" x2="18" y2="-10" stroke="#2f3a40" stroke-width="5" stroke-linecap="round"/><line x1="0" y1="4" x2="-4" y2="-32" stroke="#2f3a40" stroke-width="3.5" stroke-linecap="round"/><circle cx="0" cy="4" r="4" fill="#d9605a"/>`,
    'car keys': () => `<g transform="rotate(-12)"><rect x="-52" y="-18" width="46" height="66" rx="20" fill="#2f3a40"/>` +
      `<circle cx="-29" cy="4" r="7" fill="#59666d"/><circle cx="-29" cy="26" r="7" fill="#59666d"/>` +
      `<circle cx="4" cy="-24" r="14" stroke="#b7bfc3" stroke-width="4" fill="none"/>` +
      `<path d="M14 -30 L66 -30 L66 -24 L60 -24 L60 -18 L54 -18 L54 -24 L46 -24 L46 -18 L38 -18 L38 -24 L14 -24 Z" fill="#c3cbcf"/><circle cx="16" cy="-27" r="8" fill="#c3cbcf"/></g>`,
    'ICD device model': () => `<path d="M-4 -58 C-44 -92 -86 -40 -62 0 C-50 22 -68 40 -58 58" stroke="#3c5a7a" stroke-width="5" fill="none" stroke-linecap="round"/>` +
      `<rect x="-22" y="-62" width="38" height="18" rx="7" fill="#e3f0f5" stroke="#a9b6bc" stroke-width="2"/>` +
      `<path d="M-40 -30 Q-40 -48 -18 -48 L28 -48 Q46 -48 46 -28 L46 12 Q46 44 12 50 L-16 50 Q-40 46 -40 20 Z" fill="#b9c3c9" stroke="#98a5ab" stroke-width="2"/>` +
      `<ellipse cx="-6" cy="-14" rx="26" ry="14" fill="#d7dee2" opacity=".8"/>`,
    'pill box': () => `<rect x="-66" y="-14" width="132" height="44" rx="9" fill="#eaf1f4" stroke="#b8c6cc" stroke-width="3"/>` +
      ['#f19a8f', '#f6c177', '#f3e08a', '#a8d59a', '#8fc9d8', '#9fb1e3', '#c9a7e0'].map((col, i) => `<rect x="${-62 + i * 18}" y="-24" width="16" height="16" rx="4" fill="${col}"/>`).join('') +
      `<ellipse cx="-22" cy="48" rx="10" ry="5" fill="#ffffff" stroke="#cfd6d8" stroke-width="2"/><ellipse cx="6" cy="50" rx="8" ry="5" fill="#f3c06b"/>`,
    'glass of water': () => `<path d="M-32 -52 L32 -52 L25 58 L-25 58 Z" fill="#eef8fb" stroke="#a9cbd8" stroke-width="3"/>` +
      `<path d="M-29 -16 L29 -16 L25 56 L-25 56 Z" fill="#9fd3ea" opacity=".85"/><line x1="-18" y1="-42" x2="-14" y2="44" stroke="#ffffff" stroke-width="4" opacity=".8" stroke-linecap="round"/>`,
    'walking shoes': () => `<path d="M-66 38 L64 38 Q72 40 70 50 Q68 56 60 56 L-60 56 Q-68 56 -68 48 Z" fill="#ffffff" stroke="#c8d0d3" stroke-width="3"/>` +
      `<path d="M-64 40 Q-66 -2 -42 -8 L-12 -10 Q8 -32 30 -26 Q60 -12 66 40 Z" fill="#4f8fbf"/>` +
      `<path d="M-6 -8 Q20 6 52 18" stroke="#2f6f9f" stroke-width="7" fill="none" stroke-linecap="round"/>` +
      [0, 1, 2].map(i => `<line x1="${-2 + i * 10}" y1="${-16 + i * 4}" x2="${8 + i * 10}" y2="${-8 + i * 4}" stroke="#ffffff" stroke-width="3" stroke-linecap="round"/>`).join(''),
    'blood pressure cuff': () => `<rect x="-68" y="-6" width="92" height="52" rx="12" fill="#3f5f8a"/><rect x="-60" y="4" width="76" height="10" rx="5" fill="#5a7aa6"/>` +
      `<path d="M10 40 Q40 64 52 40" stroke="#2c2c2c" stroke-width="5" fill="none"/><ellipse cx="56" cy="34" rx="13" ry="17" fill="#2c2c2c"/>` +
      `<circle cx="38" cy="-30" r="28" fill="#ffffff" stroke="#6b7a80" stroke-width="5"/>` +
      Array.from({length: 9}, (_, i) => { const a = (-120 + i * 30) * Math.PI / 180; return `<line x1="${r(38 + Math.sin(a) * 21)}" y1="${r(-30 - Math.cos(a) * 21)}" x2="${r(38 + Math.sin(a) * 15)}" y2="${r(-30 - Math.cos(a) * 15)}" stroke="#6b7a80" stroke-width="2.5"/>`; }).join('') +
      `<line x1="38" y1="-30" x2="50" y2="-42" stroke="#d9605a" stroke-width="3.5" stroke-linecap="round"/><line x1="10" y1="-14" x2="20" y2="-22" stroke="#2c2c2c" stroke-width="4"/>`,
    'bathroom scale': () => `<rect x="-64" y="2" width="128" height="56" rx="14" fill="#f2f4f5" stroke="#bcc6ca" stroke-width="3"/>` +
      `<rect x="-28" y="12" width="56" height="18" rx="6" fill="#dfe7ea"/>` +
      Array.from({length: 7}, (_, i) => `<line x1="${-20 + i * 7}" y1="14" x2="${-20 + i * 7}" y2="${i % 3 === 0 ? 22 : 19}" stroke="#7a8a90" stroke-width="2"/>`).join('') +
      `<line x1="0" y1="12" x2="2" y2="28" stroke="#d9605a" stroke-width="3"/><ellipse cx="-30" cy="42" rx="18" ry="7" fill="#e4e9eb"/><ellipse cx="30" cy="42" rx="18" ry="7" fill="#e4e9eb"/>`,
    'phone': () => `<rect x="-32" y="-58" width="64" height="118" rx="12" fill="#2e3a42"/><rect x="-26" y="-48" width="52" height="96" rx="6" fill="#e7f2f6"/><circle cx="0" cy="-53" r="2.5" fill="#59666d"/>`,
    'blank notepad': () => `<rect x="-48" y="-56" width="96" height="116" rx="5" fill="#fffdf5" stroke="#d8d2bd" stroke-width="3"/><rect x="-48" y="-56" width="96" height="14" rx="5" fill="#5a6b73"/>` +
      Array.from({length: 7}, (_, i) => `<line x1="-36" y1="${-28 + i * 12}" x2="36" y2="${-28 + i * 12}" stroke="#c9dbe6" stroke-width="2"/>`).join('') +
      `<g transform="rotate(30 56 0)"><rect x="50" y="-40" width="10" height="76" rx="3" fill="#2f6f9f"/><path d="M50 36 L55 48 L60 36 Z" fill="#e8d5b5"/></g>`,
    'tablet showing a heartbeat line': () => `<rect x="-68" y="-46" width="136" height="98" rx="11" fill="#2e3a42"/><rect x="-59" y="-38" width="118" height="82" rx="4" fill="#12313a"/>` +
      `<polyline points="-54,4 -30,4 -22,4 -16,-8 -10,22 -4,-30 2,14 8,4 54,4" stroke="#5be3a0" stroke-width="3.5" fill="none" stroke-linejoin="round" stroke-linecap="round"/>`,
    'plate of healthy food': () => `<ellipse cx="0" cy="36" rx="68" ry="24" fill="#ffffff" stroke="#d0d7da" stroke-width="3"/><ellipse cx="0" cy="34" rx="50" ry="16" fill="#f4f7f8"/>` +
      `<ellipse cx="-22" cy="26" rx="18" ry="9" fill="#7cb86a"/><ellipse cx="-30" cy="32" rx="14" ry="7" fill="#5c9e4f"/>` +
      `<ellipse cx="14" cy="28" rx="22" ry="9" fill="#f0a37b"/><path d="M2 26 L8 32 M14 24 L20 32 M26 25 L30 31" stroke="#e38e64" stroke-width="2"/>` +
      `<circle cx="-6" cy="38" r="6" fill="#e0574f"/><circle cx="6" cy="40" r="5" fill="#e0574f"/><rect x="26" y="34" width="20" height="5" rx="2.5" fill="#f1974a" transform="rotate(-15 36 36)"/>`,
    'stethoscope': () => `<path d="M-36 -20 L-52 -58 M-36 -20 L-20 -58" stroke="#9aa7ad" stroke-width="5" stroke-linecap="round"/>` +
      `<circle cx="-52" cy="-60" r="4" fill="#3b4b54"/><circle cx="-20" cy="-60" r="4" fill="#3b4b54"/>` +
      `<path d="M-36 -20 Q-40 30 -4 40 Q22 46 30 20" stroke="#3b4b54" stroke-width="7" fill="none" stroke-linecap="round"/>` +
      `<circle cx="34" cy="14" r="17" fill="#c9d2d6" stroke="#3b4b54" stroke-width="4"/><circle cx="34" cy="14" r="8" fill="#eef2f4"/>`,
    'cigarette pack': () => `<rect x="-20" y="-60" width="11" height="28" rx="2" fill="#ffffff" stroke="#d9d3c4" stroke-width="1.5"/><rect x="-20" y="-60" width="11" height="9" fill="#e3a25c"/>` +
      `<rect x="-4" y="-54" width="11" height="22" rx="2" fill="#ffffff" stroke="#d9d3c4" stroke-width="1.5"/><rect x="-4" y="-54" width="11" height="8" fill="#e3a25c"/>` +
      `<rect x="-34" y="-40" width="68" height="98" rx="5" fill="#e9e4da" stroke="#c4bba9" stroke-width="3"/><rect x="-34" y="-40" width="68" height="22" rx="5" fill="#d7cfbe"/>`,
    'wine glass': () => `<path d="M-30 -60 Q-33 -12 0 -4 Q33 -12 30 -60 Z" fill="#f1f6f8" stroke="#b7c7cf" stroke-width="3"/>` +
      `<path d="M-28 -36 Q-28 -9 0 -6 Q28 -9 28 -36 Z" fill="#9b2335"/><rect x="-3" y="-4" width="6" height="54" fill="#dfe8ec"/>` +
      `<ellipse cx="0" cy="54" rx="26" ry="6" fill="#dfe8ec" stroke="#b7c7cf" stroke-width="2"/>`,
    'bed and pillow': () => `<rect x="-68" y="-40" width="14" height="98" rx="5" fill="#b08a62"/><rect x="58" y="10" width="10" height="48" rx="4" fill="#b08a62"/>` +
      `<rect x="-56" y="8" width="124" height="30" rx="8" fill="#ffffff" stroke="#d4dbdd" stroke-width="2"/>` +
      `<path d="M-14 8 L68 8 L68 38 L-14 38 Q-22 23 -14 8 Z" fill="#8fb6d4"/><ellipse cx="-32" cy="0" rx="24" ry="12" fill="#f6f8f9" stroke="#d4dbdd" stroke-width="2"/>` +
      `<rect x="-60" y="38" width="128" height="8" rx="3" fill="#b08a62"/>`,
  };
  const PROPS = ['none', ...Object.keys(PROP)];
  function prop(name) {
    if (!name || name === 'none') return '';
    if (!PROP[name]) throw new Error(`unknown object: ${name}`);
    return PROP[name]();
  }

  // ---- camera shots: a zoom on the whole room (x, y, width, height of what is shown) ------------------------------
  const SHOTS = {
    'wide': [120, 120, 1680, 945],
    'medium': [303, 188, 1324, 745],
    'close-up on doctor': [300, 330, 700, 394],
    'close-up on patient': [950, 330, 700, 394],
    'close-up on prop': null,                    // centred on the object, wherever it is
  };
  function shotBox(shot, ctx) {
    if (shot !== 'close-up on prop') return SHOTS[shot] || SHOTS.wide;
    const [cx, cy] = ctx.propCentre || [DESK_PROP.x, DESK_PROP.y - 45];
    const w = 440, h = 248;
    return [r(cx - w / 2), r(cy - h / 2), w, h];
  }

  // ---- the whole picture ------------------------------------------------------------------------------------------
  function scene(opts = {}) {
    const dp = pose('doctor', opts.doctor || {});
    const pp = pose('patient', opts.patient || 'listens');
    const propName = opts.prop && opts.prop.name && opts.prop.name !== 'none' ? opts.prop.name : null;
    const inHand = propName && opts.prop.at === 'doctor-hand';
    if (inHand) dp.near = {...dp.near, hand: 'hold'};
    const doc = character('doctor', dp, inHand ? prop(propName) : null);
    const pat = character('patient', pp, null);
    const at = (who, s) => `<g transform="translate(${PLACE[who].x} ${PLACE[who].y}) scale(${PLACE[who].flip} 1)">${s}</g>`;
    const onDesk = propName && !inHand
      ? `<g transform="translate(${DESK_PROP.x} ${DESK_PROP.y - 60 * 0.8}) scale(.8)">${prop(propName)}</g>` : '';
    let propCentre = null;
    if (inHand) { const [hx, hy] = handPosition('doctor', dp); propCentre = [hx, hy - 30]; }
    const [x, y, w, h] = opts.view || shotBox(opts.shot || 'wide', {propCentre});   // view: a custom crop (style sheet only)
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="${x} ${y} ${w} ${h}" preserveAspectRatio="xMidYMid slice">` +
      room() + at('doctor', doc.back) + desk() + onDesk + at('doctor', doc.front) + at('patient', pat.back + pat.front) + `</svg>`;
  }

  const KIT = {scene, prop, pose, ACTIONS, PROPS, MOUTHS, SHOTS: Object.keys(SHOTS), handPosition};
  root.KIT = KIT;
  if (typeof module !== 'undefined' && module.exports) module.exports = KIT;
})(typeof globalThis !== 'undefined' ? globalThis : this);
