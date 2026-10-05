// Wearable MR guide web stand-in. The character is the gaze target (mouse ray
// or screen-centre ray); the Python server owns the interaction state machine,
// recognition, chatbot + sentiment engine, anchors and the animation builder.
import * as THREE from '/static/vendor/three.module.js';
import {createStage} from '/static/avatar.js?v=20261006-paper3';
import {Speech} from '/static/speech.js?v=20261006-paper3';
import {MotionSequence} from '/static/gesture-library.js?v=20261006-paper3';
import {setupVoiceInput} from '/static/voice-input.js?v=20261006-paper3';

const $ = selector => document.querySelector(selector);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const LEGACY_EMOTION = {happiness: 'joy', sadness: 'sad', anger: 'anger', fear: 'fear', neutral: 'neutral'};

function log(text, kind = 'sys') {
  const item = document.createElement('li');
  item.className = kind;
  item.textContent = `${new Date().toLocaleTimeString([], {hour12: false})}  ${text}`;
  $('#log').prepend(item);
  while ($('#log').children.length > 120) $('#log').lastChild.remove();
}
async function api(route, body = {}) {
  const response = await fetch('/api/' + route, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}
async function getJSON(route) {
  const response = await fetch(route, {cache: 'no-store'});
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || `HTTP ${response.status}`);
  return value;
}

// A transient network error (e.g. a Windows socket buffer error) on the first request is retried; a lasting
// failure is shown by the start-up guard in index.html with a Retry button.
async function loadScenario(attempts = 3) {
  for (let attempt = 1; ; attempt++) {
    try {return await getJSON('/api/scenario');} catch (error) {
      if (attempt >= attempts) throw new Error(`could not load /api/scenario (${error.message})`);
      await sleep(500 * attempt);
    }
  }
}
let config;
try {config = await loadScenario();} catch (error) {
  window.wearableInitFailed?.(error.message);
  throw error;
}
const agent = config.agent;
$('#chatbot').textContent = config.chatbot;
$('#library').textContent = config.library_ready ? 'BEAT clips prepared + procedural gestures' : 'procedural stand-ins (prepare the BEAT bank for recorded clips)';

// ---------------------------------------------------------------- stage, rooms and gaze targets
const stage = createStage($('#agent'), {background: '#0f1d1a'});
stage.camera.position.set(0, 1.6, 4.7);
stage.camera.lookAt(0, .95, 0);
const baseQuaternion = stage.camera.quaternion.clone();
stage.setIdle?.(true);
const invisible = new THREE.MeshBasicMaterial({visible: false});
const characterProxy = new THREE.Mesh(new THREE.CylinderGeometry(.34, .34, 1.85, 12), invisible);
characterProxy.position.y = .92;
characterProxy.userData = {kind: 'character'};
stage.avatar.root.add(characterProxy);

const rooms = new Map();
const props = new Map();
function material(color, emissive = 0x000000) {return new THREE.MeshStandardMaterial({color, roughness: .6, emissive});}
function makeFlower(prop) {
  const group = new THREE.Group();
  group.position.set(prop.position[0], 0, prop.position[2]);
  group.userData = {kind: 'object', target: prop.id, prop};
  const pedestal = new THREE.Mesh(new THREE.CylinderGeometry(.2, .24, .42, 18), material(0x5b6b73));
  pedestal.position.y = .21; group.add(pedestal);
  const pot = new THREE.Mesh(new THREE.CylinderGeometry(.13, .1, .16, 14), material(0x9c5a3c));
  pot.position.y = .5; group.add(pot);
  const height = prop.height || 1;
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(.012, .016, height - .55, 6), material(0x2f8f46));
  stem.position.y = .55 + (height - .55) / 2; group.add(stem);
  for (const side of [-1, 1]) {
    const leaf = new THREE.Mesh(new THREE.SphereGeometry(.06, 8, 6), material(0x3fa057));
    leaf.scale.set(1.6, .25, .7); leaf.position.set(side * .07, .55 + (height - .55) * (side < 0 ? .35 : .55), 0); leaf.rotation.z = side * .4;
    group.add(leaf);
  }
  const head = new THREE.Group(); head.position.y = height; group.add(head);
  const petal = material(new THREE.Color(prop.color || '#ffffff')), centre = material(new THREE.Color(prop.center || '#ffd166'));
  const shape = prop.shape || 'cup';
  if (shape === 'spike') {
    for (const dx of [-.05, 0, .05]) for (let i = 0; i < 7; i++) {
      const bud = new THREE.Mesh(new THREE.SphereGeometry(.022, 8, 6), i % 2 ? petal : centre);
      bud.position.set(dx, -.12 + i * .035, dx * .5); head.add(bud);
    }
  } else if (shape === 'disc' || shape === 'star') {
    head.rotation.x = -.35;
    const disc = new THREE.Mesh(new THREE.CylinderGeometry(shape === 'disc' ? .07 : .035, shape === 'disc' ? .07 : .035, .03, 18), centre);
    disc.rotation.x = Math.PI / 2; head.add(disc);
    const count = shape === 'disc' ? 14 : 6, length = shape === 'disc' ? .09 : .12;
    for (let i = 0; i < count; i++) {
      const p = new THREE.Mesh(new THREE.SphereGeometry(.03, 8, 6), petal);
      const angle = i / count * Math.PI * 2;
      p.scale.set(shape === 'disc' ? .7 : 1.1, length / .03, .25);
      p.position.set(Math.cos(angle) * (length * .8 + .02), Math.sin(angle) * (length * .8 + .02), 0);
      p.rotation.z = angle - Math.PI / 2; head.add(p);
    }
  } else {
    for (let i = 0; i < 5; i++) {
      const p = new THREE.Mesh(new THREE.SphereGeometry(.05, 10, 8), petal);
      const angle = i / 5 * Math.PI * 2;
      p.scale.set(.75, 1.35, .45); p.position.set(Math.cos(angle) * .045, .03, Math.sin(angle) * .045);
      p.rotation.set(Math.sin(angle) * .35, 0, -Math.cos(angle) * .35); head.add(p);
    }
    const bud = new THREE.Mesh(new THREE.SphereGeometry(.035, 10, 8), centre); bud.position.y = .02; head.add(bud);
  }
  // Generous invisible gaze volume around pedestal, stem and head.
  const volume = new THREE.Mesh(new THREE.CylinderGeometry(.28, .28, height + .2, 12), invisible);
  volume.position.y = (height + .2) / 2; group.add(volume);
  group.userData.materials = [petal, centre];
  stage.scene.add(group);
  return group;
}
const rug = new THREE.Mesh(new THREE.CircleGeometry(3.2, 48), material(0x1d3a2c));
rug.rotation.x = -Math.PI / 2; rug.position.y = .004; stage.scene.add(rug);
for (const room of config.rooms) {
  rooms.set(room.id, room);
  const option = new Option(room.name + (room.character ? ` · ${config.characters[room.character]?.label || room.character}` : ''), room.id);
  $('#room').append(option);
  room.props.forEach((prop, index) => props.set(prop.id, {prop, room: room.id, display: `flower ${index + 1} of ${room.name}`, group: makeFlower(prop)}));
}
for (const [name, character] of Object.entries(config.characters)) $('#character').append(new Option(character.label || name, name));
let currentRoom = null, currentCharacter = null;
const builtInSelector = $('#agent select[aria-label="Avatar character"]');
function setCharacter(name) {
  if (!name || name === currentCharacter) return;
  currentCharacter = name;
  $('#character').value = name;
  if (builtInSelector && [...builtInSelector.options].some(o => o.value === name)) builtInSelector.value = name;
  stage.setCharacter?.(name);
  log(`Guide: ${config.characters[name]?.label || name}`);
}
builtInSelector?.addEventListener('change', () => {currentCharacter = builtInSelector.value; $('#character').value = builtInSelector.value;});
$('#character').onchange = () => setCharacter($('#character').value);
function setRoom(id) {
  currentRoom = id;
  $('#room').value = id;
  const room = rooms.get(id);
  for (const {group, room: owner} of props.values()) group.visible = owner === id;
  rug.material.color.set(room.floor || '#1d3a2c');
  setCharacter(room.character);
}
$('#room').onchange = () => setRoom($('#room').value);
setRoom(config.rooms[0]?.id);

// Gaze ray: the mouse pointer, or the screen centre with drag/arrow-key head turning.
const raycaster = new THREE.Raycaster();
let pointer = null, yaw = 0, pitch = 0, dragging = null, scripted = null;
let gaze = {kind: 'none', target: null};
const canvas = stage.renderer.domElement;
canvas.addEventListener('pointermove', event => {
  const rect = canvas.getBoundingClientRect();
  pointer = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
  if (dragging && $('#gaze-mode').value === 'centre') {
    yaw = Math.max(-.9, Math.min(.9, dragging.yaw - (event.clientX - dragging.x) * .004));
    pitch = Math.max(-.5, Math.min(.5, dragging.pitch - (event.clientY - dragging.y) * .004));
  }
});
// Leaving the scene to type or click keeps the head where it was; move the
// pointer to empty space in the scene to look away.
canvas.addEventListener('pointerdown', event => {dragging = {x: event.clientX, y: event.clientY, yaw, pitch};});
window.addEventListener('pointerup', () => {dragging = null;});
window.addEventListener('keydown', event => {
  if ($('#gaze-mode').value !== 'centre' || event.target?.matches?.('input,select,textarea')) return;
  const step = {ArrowLeft: [.06, 0], ArrowRight: [-.06, 0], ArrowUp: [0, .05], ArrowDown: [0, -.05]}[event.key];
  if (step) {yaw = Math.max(-.9, Math.min(.9, yaw + step[0])); pitch = Math.max(-.5, Math.min(.5, pitch + step[1])); event.preventDefault();}
});
$('#gaze-mode').onchange = () => {
  const centre = $('#gaze-mode').value === 'centre';
  $('#crosshair').style.display = centre ? 'block' : 'none';
  if (!centre) {yaw = 0; pitch = 0;}
};
let highlighted = null;
function computeGaze() {
  stage.camera.quaternion.copy(baseQuaternion).multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, 0, 'YXZ')));
  if (scripted) return scripted;
  const ndc = $('#gaze-mode').value === 'centre' ? new THREE.Vector2(0, 0) : pointer;
  if (!ndc) return {kind: 'none', target: null};
  stage.camera.updateMatrixWorld();
  stage.scene.updateMatrixWorld();
  raycaster.setFromCamera(ndc, stage.camera);
  const targets = [characterProxy, ...[...props.values()].filter(p => p.room === currentRoom).map(p => p.group)];
  for (const hit of raycaster.intersectObjects(targets, true)) {
    let node = hit.object;
    while (node && !node.userData?.kind) node = node.parent;
    if (node?.userData.kind === 'character') return {kind: 'character', target: null};
    if (node?.userData.kind === 'object') return {kind: 'object', target: node.userData.target};
  }
  return {kind: 'none', target: null};
}
function frame() {
  gaze = computeGaze();
  const prop = gaze.kind === 'object' ? props.get(gaze.target) : null;
  if (highlighted !== prop) {
    for (const m of highlighted?.group.userData.materials || []) m.emissive.setHex(0x000000);
    for (const m of prop?.group.userData.materials || []) m.emissive.setHex(0x333333);
    highlighted = prop;
  }
  $('#gaze').textContent = gaze.kind === 'character' ? 'the guide' : prop ? prop.display : 'nothing';
  $('#gaze-label').textContent = 'Gaze: ' + $('#gaze').textContent;
  const fill = state === 'dwelling' && dwellStart ? Math.min(1, (performance.now() - dwellStart) / 1000 / agent.dwell_seconds) : ['listening', 'thinking', 'responding'].includes(state) ? 1 : 0;
  $('#dwell-fill').style.width = `${fill * 100}%`;
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

// ---------------------------------------------------------------- voice, expression and animation playback
const speech = new Speech(stage);
function characterVoice() {return config.characters[currentCharacter]?.voice || {lang: 'en-US', pitch: 1, rate: 1, prefer: []};}
function pickVoice(cfg) {
  const voices = window.speechSynthesis?.getVoices?.() || [];
  const language = (cfg.lang || 'en').slice(0, 2);
  for (const token of cfg.prefer || []) {
    const pattern = new RegExp(`\\b${token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i');
    const found = voices.find(v => pattern.test(v.name) && v.lang?.startsWith(language)) || voices.find(v => pattern.test(v.name));
    if (found) return found;
  }
  return voices.find(v => v.lang === cfg.lang) || null;
}
// Per-character voice: speech.js creates the utterance; assign voice, pitch and rate before it is queued.
if (window.speechSynthesis) {
  const original = window.speechSynthesis.speak.bind(window.speechSynthesis);
  window.speechSynthesis.speak = utterance => {
    const cfg = characterVoice(), voice = pickVoice(cfg);
    if (voice) utterance.voice = voice;
    utterance.lang = voice?.lang || cfg.lang || utterance.lang;
    utterance.pitch = cfg.pitch ?? 1; utterance.rate = cfg.rate ?? 1;
    original(utterance);
  };
  window.speechSynthesis.getVoices();
}
function setExpression(name, level) {
  if (stage.setExpression) stage.setExpression(name, level);
  else stage.expression?.(LEGACY_EMOTION[name] || name, [0, .3, .6, 1][level] ?? .6);
}
function look(target, weight = .8) {
  if (!stage.lookAt) return;
  stage.lookAt(target, weight);
}
function playGesture(name, objectId) {
  const prop = props.get(objectId);
  stage.clearMotion?.();
  if (name === 'point' && prop && prop.room === currentRoom && stage.pointAt) {
    const p = prop.group.position;
    stage.pointAt([p.x, (prop.prop.height || 1), p.z]);
  } else stage.gesture?.(name);
}

class Timeline {
  constructor(reply) {
    const plan = reply.animation || {}, motion = plan.motion, length = Math.max(1, reply.text.length);
    this.object = reply.recognition?.anchor_id || objectTarget;
    this.entries = (plan.entries || []).map(entry => {
      const clip = entry.kind === 'clip' ? motion?.clips?.[entry.clip_id] : null;
      const sequence = clip ? new MotionSequence(stage).load({fps: motion.fps, joint_order: motion.joint_order, axisSigns: motion.axisSigns,
        slots: [{gesture_id: clip.id, text: clip.text, frames: clip.frames}]}, clip.text) : null;
      return {...entry, a: entry.start_char / length, b: entry.end_char / length, sequence};
    });
    this.active = -1;
  }
  update(progress) {
    const index = this.entries.findIndex(entry => progress >= entry.a && progress < entry.b);
    if (index !== this.active) {
      this.release();
      this.active = index; this.startedAt = performance.now();
      const entry = this.entries[index];
      if (!entry) {stage.gesture?.('idle'); return;}
      if (entry.sequence) entry.sequence.onStart(); else playGesture(entry.gesture, this.object);
    }
    const entry = this.entries[this.active];
    if (entry?.sequence?.active) {
      const elapsed = (performance.now() - this.startedAt) / 1000, duration = entry.sequence.sourceDuration;
      if (elapsed <= duration) entry.sequence.draw(Math.min(1, elapsed / duration), {elapsed, duration});
      else {entry.sequence.onEnd(); stage.clearMotion?.(); stage.gesture?.('idle');}
    }
  }
  release() {
    const entry = this.entries[this.active];
    if (entry?.sequence) {entry.sequence.onEnd(); stage.clearMotion?.();}
  }
  stop() {this.release(); this.active = -1; stage.clearMotion?.(); stage.gesture?.('idle');}
}

let agentSpeaking = false;
async function silentPlayback(seconds, onProgress) {
  stage.setSpeech?.(true);
  const start = performance.now();
  while ((performance.now() - start) / 1000 < seconds) {
    const elapsed = (performance.now() - start) / 1000;
    stage.setSpeechLevel?.(.25 + .2 * Math.sin(elapsed * 14));
    onProgress?.({progress: elapsed / seconds, elapsed});
    await sleep(33);
  }
  stage.setSpeech?.(false); stage.setSpeechLevel?.(0);
}
async function speak(text, {onStart, onProgress} = {}) {
  const estimate = Math.max(.8, text.trim().split(/\s+/).length / 2.6);
  agentSpeaking = true;
  let started = false;
  try {
    const run = speech.speak(text, {backend: $('#speech-backend').value, language: characterVoice().lang || 'en-US',
      onStart: () => {started = true; onStart?.();}, onProgress});
    const outcome = await Promise.race([run.then(() => 'done'), sleep(3500).then(() => started ? 'started' : 'silent')]);
    if (outcome === 'started') await run;
    if (outcome === 'silent') {
      speech.cancel();
      $('#speech-status').textContent = 'No speech voice started; playing the reply silently with captions.';
      onStart?.(); await silentPlayback(estimate, onProgress);
    }
  } catch (error) {
    $('#speech-status').textContent = `${error.message}.${started ? '' : ' Playing the reply silently.'}`;
    if (!started) {onStart?.(); await silentPlayback(estimate, onProgress);}
  } finally {agentSpeaking = false;}
}
async function perform(reply) {
  const expression = reply.expression || {name: 'neutral', level: 1};
  const timeline = new Timeline(reply);
  log(`${config.characters[currentCharacter]?.label || 'Guide'}: ${reply.text}`, 'agent');
  log(`Sentiment ${reply.sentiment.class}/${reply.sentiment.level} (${reply.sentiment.source}) → setExpression('${expression.name}', ${expression.level})`);
  const entries = reply.animation?.entries || [];
  if (entries.length) log('Animation list: ' + entries.map(e => `${e.start_s.toFixed(1)}s “${e.trigger}”→${e.kind === 'clip' && timeline.entries.find(t => t.clip_id === e.clip_id)?.sequence ? e.clip_id : e.gesture}`).join(' · '));
  look('camera', .85);
  await speak(reply.text, {
    onStart: () => setExpression(expression.name, expression.level),
    onProgress: clock => timeline.update(clock.progress),
  });
  timeline.stop();
  setExpression('neutral', 0);
}

// ---------------------------------------------------------------- server loop: state, greeting, conversation end
let state = 'idle', dwellStart = null, busy = false, greetingDone = 0, objectTarget = null;
const performQueue = [];
async function drainQueue() {
  if (busy || !performQueue.length) return;
  busy = true;
  try {
    while (performQueue.length) {
      const item = performQueue.shift();
      await perform(item);
      await api('response-finished');
      if (item.kind === 'greeting') greetingDone++;
    }
  } finally {busy = false;}
}
function showState(snapshot) {
  if (snapshot.state !== state) {
    if (snapshot.state === 'dwelling') dwellStart = performance.now();
    state = snapshot.state;
    $('#state').textContent = state; $('#state').dataset.s = state;
  }
  $('#object').textContent = snapshot.object || 'none';
  $('#anchors').textContent = snapshot.room ? `${rooms.get(snapshot.room)?.name || snapshot.room}: ${snapshot.anchors.length} anchors` : 'not loaded';
}
const EVENT_TEXT = {listening_started: 'Dwell complete: speech recognition is listening', dwell_cancelled: 'Dwell interrupted',
  proactive_greeting: 'You stayed silent: the guide opens the conversation', conversation_ended: 'You looked away and stayed silent: conversation ended',
  anchors_loaded: 'Room anchors loaded from the recognised object', response_timeout: 'Response not acknowledged; listening again'};
async function poll() {
  try {
    gaze = computeGaze();  // also while animation frames are throttled (hidden tab)
    const result = await api('observe', {gaze, speaking: userSpeaking()});
    showState(result);
    for (const event of result.events) {
      if (event === 'response_ready' || event === 'dwell_started') continue;
      log(EVENT_TEXT[event] || event);
      if (event === 'conversation_ended') {look(null); objectTarget = null;}
      if (event === 'listening_started') look('camera', .85);
    }
    for (const item of result.speak || []) performQueue.push(item);
    drainQueue();
  } catch (error) {$('#speech-status').textContent = error.message;}
  setTimeout(poll, 150);
}

// ---------------------------------------------------------------- utterances: browser recognition, typing, webcam frames
let speechActive = false, lastInterim = 0, lastTyped = 0, speechStartGaze = null, uploaded = null, stream = null;
function userSpeaking() {return speechActive || performance.now() - lastInterim < 800 || performance.now() - lastTyped < 1500;}
const normalise = text => ' ' + String(text).toLowerCase().replace(/[’]/g, "'").replace(/[^a-z0-9' ]+/g, ' ').replace(/\s+/g, ' ').trim() + ' ';
const isCommand = text => agent.voice_commands.some(command => normalise(text).includes(normalise(command)));
function grabFrame() {
  const source = uploaded || (stream ? $('#webcam') : null);
  if (!source) return null;
  const canvas = document.createElement('canvas');
  canvas.width = 640; canvas.height = 480;
  canvas.getContext('2d').drawImage(source, 0, 0, 640, 480);
  return canvas.toDataURL('image/jpeg', .85);
}
async function utter(text) {
  text = String(text || '').trim();
  if (!text) return;
  if (busy) {log(`(ignored while the guide is busy) ${text}`); return;}
  const spokenGaze = speechStartGaze || gaze;
  speechStartGaze = null;
  const body = {text, gaze: spokenGaze};
  const webcam = document.querySelector('input[name=vision]:checked').value === 'webcam';
  if (webcam && isCommand(text)) {
    body.image = grabFrame();
    if (!body.image) {log('Webcam recognition selected, but no webcam or image is active.'); delete body.image;}
    body.gaze = {kind: 'none'};
  }
  log(`You: ${text}`, 'user');
  busy = true;
  try {
    const submitted = await api('utterance', body);
    showState(submitted);
    if (!submitted.accepted) {log(submitted.message); return;}
    if (submitted.command && spokenGaze.kind === 'object' && !body.image) objectTarget = spokenGaze.target;
    // THINKING: thinking animation now, filler if the answer is still pending.
    const thinking = submitted.thinking;
    stage.clearMotion?.(); stage.gesture?.(thinking.animation || 'think');
    const focus = props.get(objectTarget);
    look(submitted.command && focus ? focus.group : 'camera', .9);
    log(`Thinking${thinking.recognition === 'pending' ? ' (object recognition running)' : thinking.recognition === 'anchor' ? ' (anchored object: recognition skipped)' : ''}`);
    let done = null, filler = null;
    const fillerTimer = setTimeout(() => {
      if (done) return;
      log(`Filler while waiting: ${thinking.filler}`, 'agent');
      filler = speak(thinking.filler);
    }, thinking.filler_delay_seconds * 1000);
    const deadline = performance.now() + 60000;
    while (!done && performance.now() < deadline) {
      const job = await getJSON(`/api/job?id=${encodeURIComponent(submitted.job)}`);
      if (job.done) done = job; else await sleep(200);
    }
    clearTimeout(fillerTimer);
    if (filler) await filler;
    if (!done) throw new Error('The answer timed out.');
    const reply = done.reply, r = reply.recognition, t = reply.timings || {};
    if (r) log(r.skipped ? `Anchor ${r.anchor_id} → ${reply.object} (recognition skipped)` : r.error ? `Recognition failed: ${r.error}` :
      `Recognised ${r.label} (${r.source}, ${(t.recognition_s ?? 0).toFixed(2)} s)${r.anchors_loaded ? `; loaded anchors of ${rooms.get(r.anchors_loaded)?.name || r.anchors_loaded} (Query A)` : ''}`);
    log(`Chatbot ${reply.chatbot}${reply.chatbot_error ? ' [' + reply.chatbot_error + ']' : ''}: ${(t.chatbot_s ?? 0).toFixed(2)} s; total ${(t.total_s ?? 0).toFixed(2)} s`);
    await perform(reply);
    showState(await api('response-finished'));
  } catch (error) {
    log(error.message);
  } finally {busy = false; drainQueue();}
}
$('#say').onclick = () => {utter($('#utterance').value); $('#utterance').value = '';};
$('#utterance').addEventListener('keydown', event => {
  if (event.key === 'Enter') {$('#say').click(); return;}
  if (!$('#utterance').value) speechStartGaze = {...gaze};
  lastTyped = performance.now();
});
for (const phrase of [...agent.voice_commands.slice(0, 2), 'How do I care for it?', 'Is it safe for my cat?', 'Who are you?', 'What flowers do you know?']) {
  const chip = document.createElement('button');
  chip.textContent = phrase;
  chip.onclick = () => {speechStartGaze = {...gaze}; utter(phrase);};
  $('#chips').append(chip);
}

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognizer = null, micOn = false;
$('#mic').onclick = () => {
  if (!Recognition) {$('#speech-status').textContent = 'This browser has no speech recognition; type what you say, or use local Whisper.'; return;}
  micOn = !micOn;
  $('#mic').setAttribute('aria-pressed', String(micOn));
  $('#mic').textContent = micOn ? 'Microphone on' : 'Microphone off';
  if (!micOn) {recognizer?.stop(); speechActive = false; return;}
  recognizer = new Recognition();
  recognizer.continuous = true; recognizer.interimResults = true; recognizer.lang = 'en-US';
  recognizer.onspeechstart = () => {speechActive = true; speechStartGaze ||= {...gaze};};
  recognizer.onspeechend = () => {speechActive = false;};
  recognizer.onresult = event => {
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (!event.results[i].isFinal) {lastInterim = performance.now(); speechStartGaze ||= {...gaze}; $('#speech-status').textContent = `Hearing: ${transcript}`; continue;}
      speechActive = false;
      if (agentSpeaking) {log(`(not sent while the guide speaks) ${transcript}`); continue;}
      utter(transcript);
    }
  };
  recognizer.onerror = event => {$('#speech-status').textContent = `Speech recognition: ${event.error}`;};
  recognizer.onend = () => {if (micOn) try {recognizer.start();} catch {}};
  try {recognizer.start();} catch (error) {$('#speech-status').textContent = error.message;}
};
const stopRecording = setupVoiceInput(speech, $('#utterance'), $('#record'), $('#audio-file'), $('#speech-status'));

// Webcam frames → YOLO adapter endpoint; graceful when the server has no weights.
const overlay = $('#webcam-overlay'), overlayContext = overlay.getContext('2d');
function drawOverlay(detections = [], width = 640, height = 480, note = '') {
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  if (uploaded) overlayContext.drawImage(uploaded, 0, 0, overlay.width, overlay.height);
  const sx = overlay.width / width, sy = overlay.height / height;
  overlayContext.lineWidth = 2; overlayContext.font = '12px system-ui';
  for (const d of detections) {
    overlayContext.strokeStyle = '#66eccd'; overlayContext.fillStyle = '#66eccd';
    overlayContext.strokeRect(d.box.left * sx, d.box.top * sy, (d.box.right - d.box.left) * sx, (d.box.bottom - d.box.top) * sy);
    overlayContext.fillText(`${d.label} ${Math.round(d.confidence * 100)}%`, d.box.left * sx + 2, d.box.top * sy + 12);
  }
  overlayContext.strokeStyle = '#fff'; overlayContext.beginPath();
  overlayContext.moveTo(overlay.width / 2 - 9, overlay.height / 2); overlayContext.lineTo(overlay.width / 2 + 9, overlay.height / 2);
  overlayContext.moveTo(overlay.width / 2, overlay.height / 2 - 9); overlayContext.lineTo(overlay.width / 2, overlay.height / 2 + 9); overlayContext.stroke();
  if (note) {overlayContext.fillStyle = '#ffcf70'; overlayContext.fillText(note, 6, overlay.height - 8);}
}
let detecting = false;
async function detectLoop() {
  if (!stream && !uploaded) return;
  if (!detecting) {
    detecting = true;
    try {
      if (!config.detector) drawOverlay([], 640, 480, 'No detector: restart with --weights');
      else {
        const result = await api('detect', {image: grabFrame()});
        drawOverlay(result.detections, result.width, result.height);
        $('#vision-status').textContent = `${result.source}: ${result.detections.length} detection(s); the centre crosshair is the gaze point.`;
      }
    } catch (error) {$('#vision-status').textContent = error.message;}
    detecting = false;
  }
  if (stream) setTimeout(detectLoop, 700);
}
function useWebcamSource() {document.querySelector('input[name=vision][value=webcam]').checked = true;}
$('#camera').onclick = async () => {
  if (stream) {stream.getTracks().forEach(track => track.stop()); stream = null; $('#camera').textContent = 'Start webcam'; return;}
  try {
    stream = await navigator.mediaDevices.getUserMedia({video: true});
    $('#webcam').srcObject = stream; await $('#webcam').play();
    uploaded = null; useWebcamSource(); $('#camera').textContent = 'Stop webcam';
    $('#vision-status').textContent = config.detector ? 'Webcam frames go to the YOLO endpoint.' : 'Webcam is on, but the server has no YOLO weights (start the demo with --weights yolo11n.pt). Simulated objects still work.';
    detectLoop();
  } catch (error) {$('#vision-status').textContent = `Webcam unavailable: ${error.message}`;}
};
$('#image').onchange = event => {
  const file = event.target.files[0];
  if (!file) return;
  const image = new Image();
  image.onload = () => {uploaded = image; useWebcamSource(); drawOverlay(); detectLoop();};
  image.src = URL.createObjectURL(file);
};
if (!config.detector) $('#vision-status').textContent += ' Webcam recognition needs the server started with --weights.';

// ---------------------------------------------------------------- sample interaction and reset
async function waitFor(predicate, ms) {
  const deadline = performance.now() + ms;
  while (!predicate()) {if (performance.now() > deadline) throw new Error('Sample interaction timed out'); await sleep(100);}
}
$('#reset').onclick = async () => {showState(await api('reset')); objectTarget = null; look(null); log('New visit: conversation reset and anchors unloaded');};
$('#run-example').onclick = async () => {
  const button = $('#run-example');
  button.disabled = true;
  try {
    setRoom(config.rooms[0].id);
    await $('#reset').onclick();
    const flowers = config.rooms[0].props;
    log('Sample: gazing at the guide for the dwell time');
    scripted = {kind: 'character', target: null};
    await waitFor(() => state === 'listening', (agent.dwell_seconds + 4) * 1000);
    const greeted = greetingDone;
    log('Sample: staying silent');
    await waitFor(() => greetingDone > greeted, (agent.greeting_after_seconds + 30) * 1000);
    scripted = {kind: 'object', target: flowers[0].id}; await sleep(500);
    speechStartGaze = {...scripted}; await utter(agent.voice_commands[0]);
    scripted = {kind: 'character', target: null}; await sleep(500);
    await utter('How should I care for it?');
    scripted = {kind: 'object', target: flowers[1].id}; await sleep(500);
    speechStartGaze = {...scripted}; await utter('Tell me about this');
    scripted = {kind: 'character', target: null}; await sleep(400);
    await utter('Who are you?');
    log('Sample finished. Move the mouse away from the guide and stay silent to end the conversation.');
  } catch (error) {log(error.message);} finally {scripted = null; button.disabled = false;}
};

window.addEventListener('pagehide', () => {stopRecording(); speech.cancel(); recognizer?.stop(); stream?.getTracks().forEach(t => t.stop()); stage.dispose();});
window.wearableDemo = {ready: true, config, stage, get state() {return state;}, get gaze() {return gaze;}, utter, setRoom, setScriptedGaze: value => {scripted = value;}};
$('#init-error').hidden = true;
log(`Ready. ${config.rooms.length} simulated rooms; dwell ${agent.dwell_seconds}s; voice commands: ${agent.voice_commands.slice(0, 3).join(', ')}…`);
poll();
