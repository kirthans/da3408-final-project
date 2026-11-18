// Read-only pipeline dashboards and authenticated retrain controls.
// Backend requests use only API_BASE_URL for loose coupling (DESIGN §3).
// DOM text is used for all service-provided strings; Mermaid receives stage families only.
/** Find a pipeline UI element by DOM ID.
 * @param {string} id
 * @returns {HTMLElement}
 */
const $ = id => document.getElementById(id);
/** Build a DOM element using safe text content.
 * @param {string} tag
 * @param {string} text
 * @param {string} cls
 * @returns {HTMLElement}
 */
const el = (tag, text, cls) => { const n = document.createElement(tag); n.textContent = text; if (cls) n.className = cls; return n; };
/** Format a finite metric with its unit, or a missing-value dash.
 * @param {number} value
 * @param {string} unit
 * @returns {string}
 */
const number = (value, unit = '') => Number.isFinite(value) ? `${value.toFixed(2)}${unit}` : '—';
/** Format fractional coverage as a percentage, or a missing-value dash.
 * @param {number} value
 * @returns {string}
 */
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
let config, api, errorText, history = [], allRuns = [], showAllRuns = false, rawDag = '', refreshing = false, drawing = 0;
const openRuns = new Set();
/** Replace a definition list with labelled metric values.
 * @param {string} id
 * @param {Array} entries
 */
function fields(id, entries) {
  $(id).replaceChildren();
  for (const [label, value] of entries) $(id).append(el('dt', label), el('dd', String(value ?? '—')));
}
/** Build an external dashboard navigation URL from configured service ports.
 * @param {string} key
 * @param {string} path
 * @returns {URL}
 */
function serviceUrl(key, path = '') {
  const url = new URL(location.origin); url.protocol = 'http:'; url.port = config[key]; url.pathname = path; return url;
}
/** Collapse month-expanded DVC nodes into safe stage families with status colors.
 * @param {string} raw Mermaid text returned by the backend.
 * @param {Array<object>} stages Run stage snapshots used for status coloring.
 * @returns {string} Sanitized family DAG, or empty text for unknown syntax. */
export function collapseDag(raw, stages = []) {
  const nodes = new Map(), families = new Map(), edges = new Set();
  for (const match of raw.matchAll(/(\w+)\["([^"\n]+)"\]/g)) {
    const family = match[2].split('@')[0];
    if (!/^[a-zA-Z_][\w-]*$/.test(family)) continue;
    nodes.set(match[1], family); families.set(family, (families.get(family) || 0) + 1);
  }
  for (const match of raw.matchAll(/(\w+)\s*-->\s*(\w+)/g)) {
    const a = nodes.get(match[1]), b = nodes.get(match[2]);
    if (a && b && a !== b) edges.add(`${a}-->${b}`);
  }
  const statuses = new Map(), priority = { failed: 5, running: 4, completed: 3, restored: 2, reused: 2, skipped: 1 };
  /** Identify the compact upstream reuse summary.
   * @param {object} stage
   * @returns {boolean}
   */
  const upstream = stages.find(stage => stage.name === 'upstream');
  const upstreamFamilies = ['download', 'ingest_validate', 'sample', 'analyze', 'aggregate'];
  /** Count expanded nodes belonging to reusable upstream families.
   * @param {number} sum
   * @param {string} name
   * @returns {number}
   */
  const upstreamCount = upstreamFamilies.reduce((sum, name) => sum + (families.get(name) || 0), 0);
  if (upstream && upstreamCount && upstream.count >= upstreamCount) {
    for (const name of upstreamFamilies) statuses.set(name, 'reused');
  }
  for (const stage of stages) {
    const family = stage.name.split('@')[0];
    if ((priority[stage.status] || 0) > (priority[statuses.get(family)] || 0)) statuses.set(family, stage.status);
  }
  let text = 'flowchart TD\n';
  for (const [family, count] of families) text += `${family}["${family}${count > 1 ? ` ×${count}` : ''}"]\n`;
  text += [...edges].join('\n');
  text += '\nclassDef completed fill:#edf5ec,stroke:#23634f,color:#194a3b;\nclassDef reused fill:#e9f2f7,stroke:#0072b2,color:#004c78;\nclassDef failed fill:#fff0e8,stroke:#b44721,color:#8a2e16;\nclassDef running fill:#fff7d6,stroke:#805500,color:#805500;\n';
  for (const [family, status] of statuses) if (families.has(family)) text += `class ${family} ${['restored', 'skipped', 'reused'].includes(status) ? 'reused' : status};\n`;
  return families.size ? text : '';
}
/** Draw the selected run DAG and discard stale asynchronous renders. */
async function drawDag() {
  /** Match the run selected for DAG status coloring.
   * @param {object} r
   * @returns {boolean}
   */
  const run = history.find(r => r.run_id === $('run-selection').value) || history[0];
  const text = collapseDag(rawDag, run?.stages);
  $('dag-text').textContent = rawDag;
  if (!text) { $('dag-status').textContent = 'No training steps have been published yet.'; $('dag').replaceChildren(); return; }
  const revision = ++drawing;
  try {
    const { svg } = await window.mermaid.render(`pipeline-dag-${revision}`, text);
    if (revision !== drawing) return;
    $('dag').innerHTML = svg;
    $('dag-status').textContent = run ? `Showing stages for run ${run.run_id}.` : 'Training dependencies; no runs yet.';
  } catch { $('dag-status').textContent = 'The diagram could not be drawn. Full DAG text is available below.'; }
}
/** Render the read-only replay clock, quality and drift snapshot.
 * @param {object} replay
 */
function renderReplay(replay) {
  $('replay-status').textContent = replay.current_month ? `${replay.current_month} · ${replay.months_revealed} of 12 months revealed · ${replay.playing ? 'Playing' : 'Paused'}` : 'No months revealed yet. The replay is waiting to begin.';
  $('injection-badge').hidden = !replay.drift_injection;
  const remaining = Math.max(0, Math.ceil(replay.seconds_to_next_reveal || 0));
  fields('replay-data', [
    ['Next reveal', replay.months_revealed === 12 ? 'All months revealed' : !replay.playing ? 'Paused' : `${Math.floor(remaining / 60)}m ${remaining % 60}s`],
    ['Champion / previous version', `${replay.champion_version || '—'} / ${replay.previous_champion_version || '—'}`],
    ['Quality measured on', replay.metrics_month || 'No evaluation yet'],
    ['Reference MAE', number(replay.reference_mae, ' Elo')],
    ['Champion MAE / reference ratio', `${number(replay.mae?.champion, ' Elo')} / ${number(replay.mae_ratio?.champion, '×')}`],
    ['Previous champion MAE / ratio', replay.previous_champion_version ? `${number(replay.mae?.previous_champion, ' Elo')} / ${number(replay.mae_ratio?.previous_champion, '×')}` : 'No previous champion yet (appears after the first promotion)'],
    ['80% interval coverage', percent(replay.coverage)],
    ['Drift detected', replay.drift_detected == null ? 'Awaiting evaluation' : replay.drift_detected ? 'Yes' : 'No'],
    /** Format one feature PSI entry for display.
     * @param {Array} entry
     * @returns {string}
     */
    ['Largest feature changes (PSI)', replay.psi_top5?.length ? replay.psi_top5.map(([name, value]) => `${name}: ${number(value)}`).join(', ') : 'No evaluation yet']
  ]);
  const old = $('previous-model'); old?.remove();
  if (replay.previous_champion_version) { const item = el('div', '', 'small'); item.id = 'previous-model'; item.textContent = `Previous champion: version ${replay.previous_champion_version}`; $('model').after(item); }
}
// A successful reset is recorded in the generation it ends; it opens the next one.
/** Select current-generation runs and the reset that opened this generation.
 * @param {Array<object>} runs
 * @returns {Array<object>}
 */
export function currentReplayRuns(runs) {
  /** Retain runs with explicit replay-generation metadata.
   * @param {object} run
   * @returns {boolean}
   */
  const tagged = runs.filter(run => Number.isFinite(run.replay_generation));
  /** Count a successful reset as the start of the next generation.
   * @param {object} run
   * @returns {number}
   */
  const generation = Math.max(-1, ...tagged.map(run => run.replay_generation + Number(Boolean(run.reset && run.status === 'promoted'))));
  /** Identify successful resets opening the current generation.
   * @param {object} run
   * @returns {boolean}
   */
  /** Order matching resets newest first.
   * @param {object} a
   * @param {object} b
   * @returns {number}
   */
  const reset = tagged.filter(run => run.reset && run.status === 'promoted' && run.replay_generation + 1 === generation).sort((a, b) => b.started_at - a.started_at)[0];
  /** Include this generation and its opening reset record.
   * @param {object} run
   * @returns {boolean}
   */
  return tagged.filter(run => run.replay_generation === generation || run === reset);
}
/** Describe the same-holdout gate comparison, omitting resets or missing evidence.
 * @param {object} run
 * @returns {object|null}
 */
export function gateSummary(run) {
  if (run.reset || !Number.isFinite(run.gate_challenger_mae) || !Number.isFinite(run.gate_champion_mae)) return null;
  const delta = run.gate_challenger_mae - run.gate_champion_mae;
  return { sentence: delta === 0 ? 'no change' : `new model ${delta < 0 ? 'better' : 'worse'} by ${Math.abs(delta).toFixed(1)} Elo`, better: delta < 0,
    measurements: `${run.gate_challenger_mae.toFixed(1)} vs ${run.gate_champion_mae.toFixed(1)} · coverage ${percent(run.coverage)}` };
}
/** Render run history while preserving selected and expanded details.
 * @param {Array<object>} runs
 */
function renderRuns(runs) {
  /** Order persisted runs by descending start time.
   * @param {object} a
   * @param {object} b
   * @returns {number}
   */
  allRuns = [...runs].sort((a, b) => b.started_at - a.started_at);
  history = showAllRuns ? allRuns : currentReplayRuns(allRuns);
  $('show-all-runs').textContent = `Show all runs (${allRuns.length})`;
  $('show-all-runs').setAttribute('aria-pressed', String(showAllRuns));
  $('runs-status').textContent = history.length ? `${history.length} ${showAllRuns ? 'total' : 'current replay'} run${history.length === 1 ? '' : 's'} · newest first` : allRuns.length ? 'No runs tagged for the current replay. Show all runs to see earlier history.' : 'No training runs yet. New runs will appear here.';
  const selection = $('run-selection').value;
  $('run-selection').replaceChildren(el('option', 'Most recent run'));
  $('run-selection').firstChild.value = '';
  $('runs').replaceChildren();
  for (const run of history) {
    const option = el('option', `${new Date(run.started_at * 1000).toLocaleString()} · ${run.reset ? 'baseline restored' : run.status}`); option.value = run.run_id; $('run-selection').append(option);
    /** Create a responsive table cell.
     * @returns {HTMLElement}
     */
    /** Assign the explicit cell role.
     * @param {HTMLElement} cell
     */
    const row = el('tr', ''); const cells = Array.from({ length: 6 }, () => el('td', '')); row.setAttribute('role', 'row'); cells.forEach(cell => cell.setAttribute('role', 'cell')); row.append(...cells);
    /** Associate responsive cell labels with their columns.
     * @param {string} label
     * @param {number} i
     */
    ['Started / trigger', 'Fit → holdout', 'Status', 'Gate', 'Registry / push', 'Details'].forEach((label, i) => cells[i].dataset.label = label);
    cells[0].append(el('p', new Date(run.started_at * 1000).toLocaleString()), el('p', run.reset ? 'replay reset' : run.trigger), el('p', run.alert_name || 'No alert'), el('p', `Replay: ${run.replay_month || '—'}`));
    const fit = run.fit_months || []; cells[1].textContent = `${fit.length ? `${fit[0]}${fit.length > 1 ? `…${fit.at(-1)}` : ''}` : '—'} → ${run.holdout_month || '—'}`;
    cells[2].append(el('span', run.reset ? 'baseline restored' : run.status, run.reset ? 'chip baseline' : `chip ${run.status}`));
    const gate = gateSummary(run);
    if (gate) cells[3].append(el('p', gate.sentence, gate.better ? 'gate-better' : ''), el('p', gate.measurements, 'small'));
    else cells[3].textContent = '—';
    cells[4].append(el('p', `Version ${run.registry_version || '—'}`));
    const push = el('p', `Push: ${['complete', 'done'].includes(run.push) ? 'done' : run.push || 'pending'}`); push.title = run.push_error || ''; cells[4].append(push);
    cells[4].append(el('p', `Pending commit: ${run.pending_commit ?? (run.registry_version ? 'yes (trainer default)' : '—')}`));
    if (run.mlflow_run_id) { const link = el('a', 'MLflow run ↗'); const url = serviceUrl('MLFLOW_PORT'); url.hash = `/runs/${encodeURIComponent(run.mlflow_run_id)}`; link.href = url.href; link.target = '_blank'; link.rel = 'noopener'; cells[4].append(link); }
    const details = el('details', '', 'run-detail'); details.open = openRuns.has(run.run_id); details.append(el('summary', 'Details'));
    /** Remember expanded run details across periodic refreshes. */
    details.addEventListener('toggle', () => details.open ? openRuns.add(run.run_id) : openRuns.delete(run.run_id));
    details.append(el('p', `Run ${run.run_id}`));
    for (const stage of run.stages || []) details.append(el('p', `${stage.name}: ${stage.status} · ${number(stage.duration_s)} s · ${number(stage.games_per_s)} games/s${stage.count ? ` · ${stage.count} stage${stage.count === 1 ? '' : 's'} (${stage.restored || 0} restored, ${stage.skipped || 0} skipped)` : ''}${stage.error ? ` · ${stage.error}` : ''}`));
    if (!run.stages?.length) details.append(el('p', 'No stage results recorded yet.'));
    for (const key of ['reason', 'message', 'error', 'push_error']) if (run[key]) details.append(el('p', `${key}: ${run[key]}`));
    cells[5].append(details); $('runs').append(row);
  }
  /** Check whether the previous selection still exists.
   * @param {HTMLOptionElement} o
   * @returns {boolean}
   */
  if ([...$('run-selection').options].some(o => o.value === selection)) $('run-selection').value = selection;
}
/** Refresh independent pipeline panels concurrently while avoiding overlapping polls. */
export async function refreshPipeline() {
  if (refreshing || document.visibilityState !== 'visible' || $('pipeline-view').hidden) return;
  refreshing = true;
  const jobs = [
    /** Render current champion metadata and acceptance metrics.
     * @param {object} model
     */
    ['/v1/model', 'model-status', model => { $('model-status').textContent = 'Current champion'; fields('model', [['Version', model.version], ['Test MAE', number(model.metrics?.test_mae, ' Elo')], ['80% interval coverage', percent(model.metrics?.coverage)]]); }],
    ['/v1/pipeline/replay', 'replay-status', renderReplay],
    ['/v1/pipeline/runs?limit=1000', 'runs-status', renderRuns],
    /** Retain the latest raw DAG for rendering and text fallback.
     * @param {object} data
     */
    ['/v1/pipeline/dag', 'dag-status', data => { rawDag = data.mermaid || ''; }]
  ];
  /** Refresh one panel and attach any failure to that panel only.
   * @param {Array} job
   */
  await Promise.all(jobs.map(async ([path, status, render]) => {
    try { render(await api(path)); }
    catch (error) { if (path.endsWith('/dag')) { rawDag = ''; $('dag').replaceChildren(); } $(status).textContent = `${errorText(error)} Earlier results, if shown, may be out of date.`; }
  }));
  if (rawDag) await drawDag();
  $('pipeline-updated').textContent = `Last refresh attempted ${new Date().toLocaleTimeString()}`;
  refreshing = false;
}
/** Initialize pipeline dependencies, controls and external dashboard links.
 * @param {object} settings Runtime configuration including API_BASE_URL.
 * @param {function} request Shared API helper.
 * @param {function} describeError Shared public-error formatter. */
export function initPipeline(settings, request, describeError) {
  config = settings; api = request; errorText = describeError;
  window.mermaid?.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'neutral', flowchart: { htmlLabels: false } });
  /** Toggle all-run history and redraw the selected pipeline. */
  $('show-all-runs').onclick = () => { showAllRuns = !showAllRuns; renderRuns(allRuns); if (rawDag) drawDag(); };
  $('run-selection').onchange = drawDag; $('refresh-pipeline').onclick = refreshPipeline;
  for (const [label, key, path] of [['MLflow', 'MLFLOW_PORT', ''], ['Service health', 'GRAFANA_PORT', '/d/elo-service'], ['Model & pipeline', 'GRAFANA_PORT', '/d/elo-model'], ['Prometheus alerts', 'PROMETHEUS_PORT', '/alerts']]) {
    const link = el('a', `${label} ↗`); link.href = serviceUrl(key, path); link.target = '_blank'; link.rel = 'noopener'; $('service-links').append(link);
  }
  /** Toggle retrain controls and clear the secret when the form closes. */
  $('open-retrain').onclick = () => { const open = $('retrain-form').hidden; $('retrain-form').hidden = !open; $('open-retrain').setAttribute('aria-expanded', String(open)); if (open) $('retrain-reason').focus(); else $('retrain-key').value = ''; };
  /** Submit an authenticated retrain and map acceptance, auth and conflict responses.
   * @param {SubmitEvent} event
   */
  $('retrain-form').onsubmit = async event => {
    event.preventDefault(); const submit = event.submitter; submit.disabled = true;
    const key = $('retrain-key').value; $('retrain-key').value = '';
    $('retrain-status').textContent = 'Requesting a retrain…';
    try {
      const response = await fetch(config.API_BASE_URL.replace(/\/$/, '') + '/v1/pipeline/retrain', { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-API-Key': key }, body: JSON.stringify({ reason: $('retrain-reason').value }) });
      const data = await response.json();
      $('retrain-status').textContent = response.status === 202 ? `Accepted. Run ID: ${data.run_id}` : response.status === 401 ? 'Wrong key. Check the API key and try again.' : response.status === 409 ? 'A run is already in progress. Wait for it to finish.' : `The retrain could not be requested. Request ID: ${data.error?.request_id || 'unavailable'}`;
    } catch { $('retrain-status').textContent = 'Could not reach the service. Please retry.'; }
    finally { $('retrain-key').value = ''; submit.disabled = false; refreshPipeline(); }
  };
}
