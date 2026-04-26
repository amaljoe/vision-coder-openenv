/* VisionCoder Test Suite Explorer */

const REWARDS = [
  { key: "format",     label: "Format",     weight: 0.5 },
  { key: "validity",   label: "Validity",   weight: 0.5 },
  { key: "structural", label: "Structural", weight: 0.5 },
  { key: "text_block", label: "Text Block", weight: 3.0 },
  { key: "position",   label: "Position",   weight: 1.0 },
  { key: "color",      label: "Color",      weight: 1.5 },
  { key: "clip",       label: "CLIP",       weight: 2.5 },
  { key: "ssim",       label: "SSIM",       weight: 1.5 },
];

function scoreColor(v) {
  if (v >= 0.8)  return "var(--green)";
  if (v >= 0.5)  return "var(--yellow)";
  return "var(--red)";
}

function formatScore(v) {
  return v.toFixed(3);
}

let data = [];
let activeCase = null;
let activeVariant = null;

// ── Render grid card ─────────────────────────────────────────────────
function makeCard(c, idx) {
  const totals = c.variants.map(v => v.rewards.total);
  const mean = totals.reduce((a, b) => a + b, 0) / totals.length;
  const delay = (idx % 15) * 30;

  const card = document.createElement("div");
  card.className = "card";
  card.dataset.difficulty = c.difficulty;
  card.style.animationDelay = delay + "ms";
  card.innerHTML = `
    <img class="card-thumb" src="${c.reference_image}" alt="reference" loading="lazy">
    <div class="card-body">
      <div class="card-row">
        <span class="card-id">#${c.id}</span>
        <span class="badge badge-${c.difficulty}">${c.difficulty}</span>
      </div>
      <div class="card-source">${c.source}</div>
      <div class="card-mean">mean reward: ${mean.toFixed(3)}</div>
    </div>
  `;
  card.addEventListener("click", () => openDetail(c));
  return card;
}

// ── Render detail panel ───────────────────────────────────────────────
function openDetail(c) {
  activeCase = c;
  const detail = document.getElementById("detail");
  const meta   = document.getElementById("detail-meta");

  meta.innerHTML = `<strong>#${c.id}</strong> &nbsp;·&nbsp; <span class="badge badge-${c.difficulty}">${c.difficulty}</span> &nbsp;·&nbsp; ${c.source} &nbsp;·&nbsp; ${c.variants.length} variants`;
  document.getElementById("ref-img").src = c.reference_image;

  renderStrip(c);
  detail.hidden = false;
  detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
  selectVariant(c.variants[0]);
}

function renderStrip(c) {
  const strip = document.getElementById("variant-strip");
  strip.innerHTML = "";
  c.variants.forEach(v => {
    const el = document.createElement("div");
    el.className = "variant-thumb";
    el.dataset.name = v.name;
    const color = scoreColor(v.rewards.total);
    el.innerHTML = `
      <img src="${v.image}" alt="${v.name}" loading="lazy">
      <div class="variant-thumb-label">
        <div class="variant-name">${v.name}</div>
        <span class="variant-score" style="color:${color}">${formatScore(v.rewards.total)}</span>
      </div>
    `;
    el.addEventListener("click", () => selectVariant(v));
    strip.appendChild(el);
  });
}

function selectVariant(v) {
  activeVariant = v;

  // Highlight active thumb
  document.querySelectorAll(".variant-thumb").forEach(el => {
    el.classList.toggle("active", el.dataset.name === v.name);
  });

  renderRewardChart(v);

  // Code view
  const codeEl = document.getElementById("code-view");
  codeEl.textContent = v.html;

  // Reset code toggle
  const toggle = document.getElementById("code-toggle");
  toggle.textContent = "Show HTML source ▼";
  codeEl.hidden = true;
}

function renderRewardChart(v) {
  const panel = document.getElementById("reward-panel");
  const totalColor = scoreColor(v.rewards.total);
  panel.innerHTML = `
    <div class="reward-title">
      <span class="reward-total" style="color:${totalColor}">${formatScore(v.rewards.total)}</span>
      total reward &nbsp;·&nbsp; <em>${v.name}</em>
    </div>
    ${REWARDS.map(r => {
      const val = v.rewards[r.key] ?? 0;
      return `
        <div class="bar-row">
          <span class="bar-label">${r.label}</span>
          <div class="bar-track">
            <div class="bar-fill" data-val="${val}" style="background:${scoreColor(val)}"></div>
          </div>
          <span class="bar-value">${val.toFixed(2)}</span>
          <span class="bar-weight">×${r.weight}</span>
        </div>
      `;
    }).join("")}
  `;

  // Animate bars on next frame
  requestAnimationFrame(() => {
    panel.querySelectorAll(".bar-fill").forEach(el => {
      el.style.width = (parseFloat(el.dataset.val) * 100) + "%";
    });
  });
}

// ── Tab filtering ─────────────────────────────────────────────────────
document.getElementById("tabs").addEventListener("click", e => {
  const btn = e.target.closest(".tab");
  if (!btn) return;
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  btn.classList.add("active");
  const filter = btn.dataset.filter;
  document.querySelectorAll(".card").forEach(card => {
    const show = filter === "all" || card.dataset.difficulty === filter;
    card.classList.toggle("hidden", !show);
  });
});

// ── Close detail ──────────────────────────────────────────────────────
document.getElementById("close-btn").addEventListener("click", () => {
  document.getElementById("detail").hidden = true;
  activeCase = null;
});

// ── Code toggle ───────────────────────────────────────────────────────
document.getElementById("code-toggle").addEventListener("click", () => {
  const codeEl = document.getElementById("code-view");
  const toggle = document.getElementById("code-toggle");
  codeEl.hidden = !codeEl.hidden;
  toggle.textContent = codeEl.hidden ? "Show HTML source ▼" : "Hide HTML source ▲";
});

// ── Boot ──────────────────────────────────────────────────────────────
fetch("data.json")
  .then(r => r.json())
  .then(d => {
    data = d;
    const grid = document.getElementById("grid");
    data.forEach((c, i) => grid.appendChild(makeCard(c, i)));
  })
  .catch(err => {
    document.getElementById("grid").innerHTML =
      `<p style="color:var(--red);padding:20px">Failed to load data.json: ${err}</p>`;
  });
