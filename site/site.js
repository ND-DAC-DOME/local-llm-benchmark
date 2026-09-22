/* Shared helpers: theme-aware chart colors, nav, data loading. Chart.js UMD is loaded before this file. */
const CSS = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const COLORS = () => ({ muse: CSS('--muse'), qwen36: CSS('--qwen36'), qwen38: CSS('--qwen38'),
  ok: CSS('--ok'), bad: CSS('--bad'), cut: CSS('--cut'), ink: CSS('--ink'), ink2: CSS('--ink-2'), ink3: CSS('--ink-3'), line: CSS('--line'), surface: CSS('--surface') });
const MODEL_NAME = { muse: 'Muse-Glimmer-30B', qwen36: 'Qwen3.6-27B', qwen38: 'Qwen3.8-27B' };
const modelOf = (key) => key.split('@')[0];
const budgetOf = (key) => key.split('@')[1];
const NAV = [['index.html', 'Summary'], ['quality.html', 'Quality'], ['explorer.html', 'Question explorer'],
  ['speed.html', 'Speed'], ['method.html', 'How it was measured'], ['caveats.html', 'Caveats'], ['appendix.html', 'All numbers']];
function renderNav() {
  const here = location.pathname.split('/').pop() || 'index.html';
  const nav = document.querySelector('nav.side');
  nav.innerHTML = '<div class="brand">Muse-Glimmer vs Qwen</div><div class="sub">Local benchmark, Sept 2026</div>' +
    NAV.map(([h, t]) => `<a href="${h}" class="${h === here ? 'current' : ''}">${t}</a>`).join('');
}
async function loadJSON(path) { const r = await fetch(path); if (!r.ok) throw new Error(path + ': ' + r.status); return r.json(); }
function baseChartOptions(extra = {}) {
  const c = COLORS();
  Chart.defaults.font.family = CSS('--font');
  Chart.defaults.color = c.ink2;
  return Object.assign({
    responsive: true, maintainAspectRatio: false, animation: false,
    plugins: { legend: { display: false }, tooltip: { backgroundColor: c.surface, titleColor: c.ink, bodyColor: c.ink2, borderColor: c.line, borderWidth: 1, padding: 10, displayColors: true } },
    scales: { x: { grid: { display: false }, border: { color: c.line }, ticks: { color: c.ink2 } },
              y: { grid: { color: c.line, lineWidth: 1 }, border: { display: false }, ticks: { color: c.ink2 } } },
  }, extra);
}
/* labels drawn on top of bars (selective: value labels are the point of these charts) */
const valueLabels = { id: 'valueLabels', afterDatasetsDraw(chart, _a, opts) {
  if (!opts || !opts.enabled) return; const { ctx } = chart; const c = COLORS();
  ctx.save(); ctx.font = `600 12px ${CSS('--mono')}`; ctx.fillStyle = c.ink; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
  chart.data.datasets.forEach((ds, di) => { if (ds.noLabel) return; const meta = chart.getDatasetMeta(di); if (meta.hidden) return;
    meta.data.forEach((bar, i) => { const v = ds.data[i]; if (v == null) return; const t = opts.format ? opts.format(v, ds, i) : String(v);
      if (chart.options.indexAxis === 'y') { ctx.textAlign = 'left'; ctx.textBaseline = 'middle'; ctx.fillText(t, bar.x + 6, bar.y); }
      else ctx.fillText(t, bar.x, bar.y - 4); }); });
  ctx.restore(); } };
Chart.register(valueLabels);
const bar = (color) => ({ backgroundColor: color, borderRadius: { topLeft: 4, topRight: 4 }, borderSkipped: 'bottom', maxBarThickness: 26 });
const hbar = (color) => ({ backgroundColor: color, borderRadius: { topRight: 4, bottomRight: 4 }, borderSkipped: 'left', maxBarThickness: 22 });
function legend(el, entries) { el.innerHTML = entries.map(([label, color]) => `<span style="--c:${color}">${label}</span>`).join(''); }
document.addEventListener('DOMContentLoaded', renderNav);
