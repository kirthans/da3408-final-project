import { initPipeline, refreshPipeline } from './pipeline.js';
import { Chess } from './vendor/chess.js/dist/esm/chess.js';
const $ = id => document.getElementById(id);
const demo = new URLSearchParams(location.search).get('mock') === '1';
const config = window.ELO_CONFIG;
let games = [], parsedPgn = '', controller, board, positions = [], result, current = 0;
const messages = {
  invalid_pgn: 'We could not read this PGN. Export it again with the moves included.',
  unauthorized: 'This request needs authorization. Check the service configuration with the administrator.',
  payload_too_large: 'This file is too large. Choose a PGN up to 1 MB.',
  validation_error: 'Some game details were not accepted. Check the PGN and time control, then read it again.',
  game_index_out_of_range: 'This game is no longer in the file. Read the game again and choose it from the list.',
  unsupported_time_control: 'Bullet and correspondence games aren’t supported — try a blitz, rapid or classical game.',
  game_too_short: 'There are too few moves for a useful estimate. Try a longer game.',
  model_unavailable: 'The rating model is still warming up or unavailable. Please retry shortly.',
  engine_error: 'The chess engine could not analyse this game. Please retry or choose another game.',
  monitor_unavailable: 'The replay service is unavailable. Please retry shortly.',
  trainer_unavailable: 'The training service is unavailable. Please retry shortly.',
  engine_timeout: 'Analysis took too long. Please retry or choose a shorter game.'
};
function errorText(error) {
  if (error.name === 'AbortError') return 'Cancelled. You can try again whenever you’re ready.';
  const body = error.body?.error;
  const message = messages[body?.code] || (error.status ? 'The service could not complete this request. Please retry.' : 'Could not reach the service. Check your connection and retry.');
  return `${message} Request ID: ${body?.request_id || 'unavailable (no service response)'}.`;
}
async function api(path, body, signal) {
  const response = await fetch(config.API_BASE_URL.replace(/\/$/, '') + path, {
    method: body ? 'POST' : 'GET', headers: body ? { 'Content-Type': 'application/json' } : {},
    body: body ? JSON.stringify(body) : undefined, signal
  });
  if (!response.ok) { const e = new Error(); e.status = response.status; e.body = await response.json().catch(() => null); throw e; }
  return response.json();
}
function sayError(message) { $('error').textContent = message; $('error').hidden = false; }
function busy(active, text = '') {
  $('status').textContent = text; $('spinner').hidden = !active; $('cancel').hidden = !active;
  for (const id of ['parse', 'predict', 'pgn', 'file', 'game', 'tc']) $(id).disabled = active;
  document.querySelectorAll('[data-example]').forEach(b => { b.disabled = active; });
}
async function request(text, work) {
  controller = new AbortController(); $('error').hidden = true; busy(true, text);
  try { await work(controller.signal); $('status').textContent = 'Ready.'; }
  catch (e) { if (e.name === 'AbortError') $('status').textContent = 'Cancelled. Ready to try again.'; else { sayError(errorText(e)); $('status').textContent = ''; } }
  finally { busy(false, $('status').textContent); controller = null; }
}
function validate(pgn) {
  if (!pgn.trim()) return 'Paste a game or choose a PGN file to get started.';
  if (new Blob([pgn]).size > 1024 * 1024) return 'Choose a PGN up to 1 MB. You can export just one game.';
  if (!/\b\d+\.(?:\s|\.)*\s*(?:[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8]|O-O)/.test(pgn)) return 'We need the game’s moves, such as “1. e4 e5”. Please include them in your PGN.';
}
function resetInput() { games = []; parsedPgn = ''; $('selection').hidden = true; $('results').hidden = true; $('error').hidden = true; $('status').textContent = ''; }
$('pgn').addEventListener('input', resetInput);
async function loadFile(file) {
  if (!file) return;
  resetInput();
  if (!/\.pgn$/i.test(file.name)) return sayError('Please choose a .pgn file.');
  if (file.size > 1024 * 1024) return sayError('Choose a PGN up to 1 MB.');
  try { $('pgn').value = await file.text(); $('file-name').textContent = file.name; $('status').textContent = 'File loaded. Choose Read game to continue.'; }
  catch { sayError('We could not open that file. Try choosing it again.'); }
}
$('file').addEventListener('change', e => loadFile(e.target.files[0]));
for (const name of ['dragover', 'dragleave', 'drop']) $('drop-zone').addEventListener(name, e => {
  e.preventDefault(); $('drop-zone').classList.toggle('dragging', name === 'dragover');
  if (name === 'drop' && !controller) loadFile(e.dataTransfer.files[0]);
});
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', async () => {
  resetInput();
  try { const response = await fetch(`examples/${button.dataset.example}.pgn`); if (!response.ok) throw new Error(); $('pgn').value = await response.text(); $('status').textContent = 'Example loaded. Choose Read game to continue.'; }
  catch { sayError('The example could not be loaded. Please retry.'); }
}));
function selectGame() { const game = games[Number($('game').value)]; $('time-control').hidden = Boolean(game.time_control && game.time_control !== '-' && game.time_control !== '?'); }
$('game').addEventListener('change', selectGame);
$('parse').addEventListener('click', () => {
  const pgn = $('pgn').value, invalid = validate(pgn); if (invalid) return sayError(invalid);
  request('Reading your game…', async signal => {
    const data = demo ? { games: [result.game] } : await api('/v1/games/parse', { pgn }, signal);
    if (!data.games.length) { sayError('No games were found. Include a game with moves.'); return; }
    games = data.games; parsedPgn = pgn; $('game').replaceChildren();
    games.forEach((g, i) => { const option = document.createElement('option'); option.value = i; option.textContent = `${g.white || 'White'} vs ${g.black || 'Black'} · ${g.date || 'date unknown'} · ${g.time_control || 'no time control'} · ${g.plies} plies`; $('game').append(option); });
    $('selection').hidden = false; selectGame(); $('game').focus();
  });
});
$('predict').addEventListener('click', () => {
  const game = games[Number($('game').value)]; if (!game) return;
  request(`Analysing ${Math.ceil(game.plies / 2)} moves with Stockfish…`, async signal => {
    const data = demo ? result : await api('/v1/predict', { pgn: parsedPgn, game_index: game.index, top_k: 5, tc_class: $('time-control').hidden ? null : $('tc').value }, signal);
    render(data); $('results').scrollIntoView({ behavior: 'auto', block: 'start' });
  });
});
$('cancel').addEventListener('click', () => controller?.abort());
function element(tag, text, cls) { const node = document.createElement(tag); node.textContent = text; if (cls) node.className = cls; return node; }
const percent = value => Math.max(0, Math.min(100, (value - 600) / 26));
function ratingCard(side, data) {
  const rating = data[side], fileRating = data.pgn_ratings?.[side];
  const card = element('article', '', 'rating-card'); card.append(element('h3', `${side === 'white' ? 'White' : 'Black'} · ${data.game[side] || 'Player'}`));
  const number = element('p', String(rating.elo), 'rating'); number.append(element('small', ' Elo')); card.append(number);
  card.append(element('p', `${rating.lo}–${rating.hi} · 80% interval`, 'small'));
  const scale = element('div', '', 'scale'); scale.setAttribute('role', 'img'); scale.setAttribute('aria-label', `600 to 3200 scale: estimate ${rating.elo}, interval ${rating.lo} to ${rating.hi}${fileRating != null ? `, rating in file ${fileRating}` : ''}`);
  const interval = element('span', '', 'interval'); interval.style.left = `${percent(rating.lo)}%`; interval.style.width = `${percent(rating.hi) - percent(rating.lo)}%`;
  const marker = element('span', '', 'estimate-marker'); marker.style.left = `${percent(rating.elo)}%`; scale.append(interval, marker);
  if (fileRating != null) { const mark = element('span', '', 'file-marker'); mark.style.left = `${percent(fileRating)}%`; scale.append(mark); }
  const labels = element('div', '', 'scale-labels'); labels.append(element('span', '600'), element('span', '3200')); card.append(scale, labels);
  if (fileRating != null) card.append(element('p', `Rating in file: ${fileRating} · dashed blue marker · not used by the model`, 'small'));
  return card;
}
function jump(ply) {
  current = Math.max(0, Math.min(positions.length - 1, ply)); board.position(positions[current], false);
  $('board').setAttribute('aria-label', `Chess position after ply ${current}. ${current ? result.moves[current - 1].san : 'Starting position'}`);
  $('position').textContent = current ? `Ply ${current} of ${result.moves.length} · ${result.moves[current - 1].san}` : 'Starting position';
  [...$('moves').children].forEach((b, i) => { if (i + 1 === current) b.setAttribute('aria-current', 'step'); else b.removeAttribute('aria-current'); });
  $('first').disabled = $('prev').disabled = current === 0; $('next').disabled = $('last').disabled = current === positions.length - 1;
}
function render(data) {
  result = data; $('results').hidden = false; $('ratings').replaceChildren(ratingCard('white', data), ratingCard('black', data));
  $('result-context').textContent = `${data.game.tc_class || $('tc').value} · ${data.game.plies} plies`;
  $('warnings').replaceChildren(...data.warnings.map(w => element('p', w))); $('warnings').hidden = !data.warnings.length;
  $('caveat').textContent = `These are Lichess-equivalent ratings for ${data.game.tc_class || $('tc').value} games. Ratings from other sites or over-the-board play are not directly comparable.`;
  $('details').textContent = `Model ${data.model.name} · version ${data.model.version} · analysis ${data.analysis.source} · ${data.analysis.nodes.toLocaleString()} nodes · ${data.analysis.seconds.toFixed(1)} seconds · request ${data.request_id}`;
  const chess = new Chess(); positions = [chess.fen()]; $('moves').replaceChildren();
  try { data.moves.forEach(move => {
    chess.move(move.san); positions.push(chess.fen());
    const symbols = { inaccuracy: '?!', mistake: '?', blunder: '??' };
    const button = element('button', `${Math.ceil(move.ply / 2)}${move.side === 'w' ? '.' : '…'} ${move.san} ${symbols[move.move_class] || ''}`, symbols[move.move_class] ? move.move_class : '');
    button.setAttribute('aria-label', `Ply ${move.ply}: ${move.san}, ${move.move_class}`); button.addEventListener('click', () => jump(move.ply)); $('moves').append(button);
  }); } catch { sayError('Some replay moves could not be read. Rating results are shown; try exporting the PGN again for a complete replay.'); }
  if (!board) board = window.Chessboard('board', { position: 'start', showNotation: true, pieceTheme: 'vendor/chrisoakman-chessboardjs/img/chesspieces/wikipedia/{piece}.png', moveSpeed: 0 });
  board.resize(); jump(0); $('influence').replaceChildren();
  data.influential_moves.forEach(move => {
    const side = move.side === 'w' ? 'White' : 'Black', delta = move.delta_elo;
    const button = element('button', `${Math.ceil(move.ply / 2)}${move.side === 'w' ? '.' : '…'} ${move.san} — This move suggested ${delta > 0 ? 'stronger' : delta < 0 ? 'weaker' : 'similar'} play (${delta >= 0 ? '+' : '−'}${Math.abs(delta)} Elo to ${side}’s estimate).`);
    button.addEventListener('click', () => jump(move.ply)); const li = element('li', ''); li.append(button); $('influence').append(li);
  });
  if (!data.influential_moves.length) $('influence').append(element('li', 'No single move outside the opening stood out.'));
}
$('first').onclick = () => jump(0); $('prev').onclick = () => jump(current - 1); $('next').onclick = () => jump(current + 1); $('last').onclick = () => jump(positions.length - 1);
$('board').addEventListener('keydown', e => { if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(e.key) && board) { e.preventDefault(); jump(e.key === 'Home' ? 0 : e.key === 'End' ? positions.length - 1 : current + (e.key === 'ArrowRight' ? 1 : -1)); } });
window.addEventListener('resize', () => board?.resize());
let pipelineTimer;
function tab(pipeline) {
  $('estimate-view').hidden = pipeline; $('pipeline-view').hidden = !pipeline;
  for (const id of ['estimate-tab', 'pipeline-tab']) $(id).removeAttribute('aria-current');
  $(pipeline ? 'pipeline-tab' : 'estimate-tab').setAttribute('aria-current', 'page');
  if (!pipeline) board?.resize();
  pipelineVisibility();
}
$('estimate-tab').onclick = () => tab(false); $('pipeline-tab').onclick = () => tab(true);
function pipelineVisibility() {
  clearInterval(pipelineTimer);
  if (!$('pipeline-view').hidden && document.visibilityState === 'visible') {
    refreshPipeline(); pipelineTimer = setInterval(refreshPipeline, 10000);
  }
}
document.addEventListener('visibilitychange', pipelineVisibility);
if (demo) {
  $('demo').hidden = false;
  try { const response = await fetch('mock/predict.json'); if (!response.ok) throw new Error(); render(await response.json()); }
  catch { sayError('Demo data could not be loaded. Please reload this page.'); }
}

initPipeline(config, api, errorText);
