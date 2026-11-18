import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(new URL('../site/pipeline.js', import.meta.url), 'utf8');
const { currentReplayRuns, gateSummary, initPipeline, refreshPipeline } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('current replay includes its preceding reset and excludes untagged and earlier drills', () => {
  const reset = { run_id: 'reset', replay_generation: 2, reset: true, status: 'promoted', started_at: 20 };
  const current = { run_id: 'current', replay_generation: 3, started_at: 30 };
  const runs = [{ replay_generation: 1 }, { replay_generation: 2 }, {}, reset, current];
  assert.deepEqual(currentReplayRuns(runs), [reset, current]);
  assert.deepEqual(currentReplayRuns([reset]), [reset]);
  assert.deepEqual(currentReplayRuns([{}]), []);
  assert.deepEqual(currentReplayRuns([]), []);
  assert.deepEqual(currentReplayRuns([{ replay_generation: 0 }]), [{ replay_generation: 0 }]);
});

test('failed reset does not open another replay', () => {
  const runs = [{ replay_generation: 2, reset: true, status: 'failed' }, { replay_generation: 2 }];
  assert.deepEqual(currentReplayRuns(runs), runs);
});

test('gate explains direction and handles equal, missing and reset values', () => {
  const run = { gate_challenger_mae: 191.1, gate_champion_mae: 196, coverage: 0.793 };
  assert.deepEqual(gateSummary(run), { sentence: 'new model better by 4.9 Elo', better: true, measurements: '191.1 vs 196.0 · coverage 79.3%' });
  assert.equal(gateSummary({ ...run, gate_challenger_mae: 200 }).sentence, 'new model worse by 4.0 Elo');
  assert.equal(gateSummary({ ...run, gate_challenger_mae: 196 }).sentence, 'no change');
  for (const value of [undefined, null, NaN, Infinity]) assert.equal(gateSummary({ ...run, gate_champion_mae: value }), null);
  assert.equal(gateSummary({ ...run, reset: true }), null);
});

// Exercise the real rendering and toggle handlers with an in-memory DOM, without services.
class Element {
  constructor(tag = '') { this.tag = tag; this.children = []; this.dataset = {}; this.attributes = {}; this.value = ''; this.hidden = false; this.textContent = ''; }
  append(...items) { this.children.push(...items); }
  replaceChildren(...items) { this.children = items; }
  get firstChild() { return this.children[0]; }
  get options() { return this.children; }
  setAttribute(key, value) { this.attributes[key] = value; }
  addEventListener() {}
  after() {}
}
const content = node => [node.textContent, ...node.children.map(content)].join(' ');

test('rendering retains details, suppresses reset gates and missing units, and toggles all runs', async () => {
  const nodes = new Map();
  globalThis.document = { visibilityState: 'visible', getElementById: id => { if (id === 'previous-model') return null; if (!nodes.has(id)) nodes.set(id, new Element()); return nodes.get(id); }, createElement: tag => new Element(tag) };
  globalThis.window = {};
  globalThis.location = { origin: 'http://example.invalid' };
  const reset = { run_id: 'reset', replay_generation: 0, reset: true, status: 'promoted', started_at: 1, gate_challenger_mae: 182.72, gate_champion_mae: 166.99 };
  const failed = { run_id: 'failed', replay_generation: 1, status: 'failed', started_at: 2, message: 'No new model', error: 'Training failed' };
  const old = { run_id: 'old', status: 'no_change', started_at: 0 };
  const data = { '/v1/model': {}, '/v1/pipeline/replay': {}, '/v1/pipeline/runs?limit=1000': [old, reset, failed], '/v1/pipeline/dag': {} };
  initPipeline({}, async path => data[path], String);
  await refreshPipeline();
  const runs = nodes.get('runs');
  assert.equal(runs.children.length, 2);
  assert.equal(nodes.get('runs-status').textContent, '2 current replay runs · newest first');
  const resetRow = content(runs.children[1]);
  assert.match(resetRow, /baseline restored/);
  assert.match(resetRow, /replay reset/);
  assert.doesNotMatch(resetRow, /182|166|better|worse/);
  assert.equal(runs.children[1].children[3].textContent, '—');
  assert.match(content(runs.children[0]), /No new model.*Training failed/);
  assert.match(content(nodes.get('replay-data')), /No previous champion yet/);
  assert.doesNotMatch(content(nodes.get('replay-data')), /Not available|— Elo|—×/);
  assert.doesNotMatch(content(nodes.get('model')), /— Elo/);
  nodes.get('show-all-runs').onclick();
  assert.equal(runs.children.length, 3);
  assert.equal(nodes.get('runs-status').textContent, '3 total runs · newest first');
  assert.equal(nodes.get('show-all-runs').attributes['aria-pressed'], 'true');
  await refreshPipeline();
  assert.equal(runs.children.length, 3);
  nodes.get('show-all-runs').onclick();
  assert.equal(runs.children.length, 2);
  data['/v1/pipeline/runs?limit=1000'] = [{ ...failed, stages: [{ name: 'upstream', status: 'restored', count: 1 }] }];
  await refreshPipeline();
  assert.equal(nodes.get('runs-status').textContent, '1 current replay run · newest first');
  assert.match(content(runs), /1 stage \(/);
  assert.doesNotMatch(content(runs), /1 stages/);
  nodes.get('show-all-runs').onclick();
  assert.equal(nodes.get('runs-status').textContent, '1 total run · newest first');
  data['/v1/pipeline/runs?limit=1000'][0].stages[0].count = 2;
  await refreshPipeline();
  assert.match(content(runs), /2 stages \(/);
});
