"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (n) => Number(n || 0).toLocaleString();

const state = { file: null, gemini: false, currentId: null, doc: null, polling: null };

// ------------------------------------------------------------------ api

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try { const body = await res.json(); msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail); } catch {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

// ------------------------------------------------------------------ health

async function loadHealth() {
  const pill = $("#health");
  try {
    const h = await api("/api/health");
    state.gemini = h.gemini_configured;
    pill.textContent = h.gemini_configured ? `Gemini: ${h.model}` : "Gemini key not set: chunking only";
    pill.className = "pill " + (h.gemini_configured ? "ok" : "warn");
    if (!h.gemini_configured) {
      $("#summarize").checked = false;
      $("#summarize").disabled = true;
      $("#strategy").querySelector('option[value="semantic"]').disabled = true;
    }
  } catch {
    pill.textContent = "server unreachable";
    pill.className = "pill warn";
  }
}

// ------------------------------------------------------------------ upload

function setFile(f) {
  if (!f) return;
  state.file = f;
  $("#file-name").textContent = `${f.name} · ${(f.size / 1048576).toFixed(1)} MB`;
  $("#submit").disabled = false;
}

function initUpload() {
  const drop = $("#drop");
  $("#file").addEventListener("change", (e) => setFile(e.target.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("#form-error");
    err.hidden = true;
    if (!state.file) return;
    const options = {
      detect_tables: $("#tables").checked,
      chunking: {
        strategy: $("#strategy").value,
        max_tokens: Number($("#max_tokens").value),
        overlap_tokens: Number($("#overlap_tokens").value),
        min_tokens: Math.min(120, Math.floor(Number($("#max_tokens").value) / 4)),
      },
      summarization: {
        enabled: $("#summarize").checked,
        length: $("#length").value,
        focus: $("#focus").value.trim() || null,
        language: $("#language").value.trim() || null,
      },
    };
    const fd = new FormData();
    fd.append("file", state.file);
    fd.append("options", JSON.stringify(options));
    $("#submit").disabled = true;
    try {
      const job = await api("/api/documents", { method: "POST", body: fd });
      await loadList();
      watchJob(job.id);
    } catch (ex) {
      err.textContent = ex.message;
      err.hidden = false;
    } finally {
      $("#submit").disabled = false;
    }
  });
}

// ------------------------------------------------------------------ documents list

async function loadList() {
  const jobs = await api("/api/jobs").catch(() => []);
  const ul = $("#doc-list");
  if (!jobs.length) { ul.innerHTML = '<li class="muted">No documents yet</li>'; return jobs; }
  ul.innerHTML = jobs.map((j) => `
    <li data-id="${esc(j.id)}" class="${j.id === state.currentId ? "active" : ""}" title="${esc(j.error || j.filename)}">
      <span class="name">${esc(j.filename)}</span>
      <span class="status ${j.status === "failed" ? "failed" : ""}">${j.status === "running" ? Math.round(j.progress * 100) + "%" : esc(j.status)}</span>
      <button class="del" data-del="${esc(j.id)}" aria-label="Delete ${esc(j.filename)}" title="Delete">✕</button>
    </li>`).join("");
  return jobs;
}

function initList() {
  $("#doc-list").addEventListener("click", async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) {
      e.stopPropagation();
      if (!confirm("Delete this document and its results?")) return;
      try { await api(`/api/documents/${del.dataset.del}`, { method: "DELETE" }); } catch (ex) { alert(ex.message); }
      if (state.currentId === del.dataset.del) showEmpty();
      return loadList();
    }
    const li = e.target.closest("li[data-id]");
    if (li) watchJob(li.dataset.id);
  });
}

// ------------------------------------------------------------------ job progress

function showEmpty() {
  state.currentId = null;
  $("#empty").hidden = false; $("#progress").hidden = true; $("#doc").hidden = true;
}

function watchJob(id) {
  clearInterval(state.polling);
  state.currentId = id;
  const tick = async () => {
    let job;
    try { job = await api(`/api/jobs/${id}`); } catch { return; }
    if (state.currentId !== id) return;
    if (job.status === "completed") {
      clearInterval(state.polling);
      loadList();
      return showDocument(id);
    }
    $("#empty").hidden = true; $("#doc").hidden = true; $("#progress").hidden = false;
    $("#progress-file").textContent = job.filename;
    const pct = Math.round(job.progress * 100);
    $("#progress-pct").textContent = `${pct}%`;
    $("#progress-bar").style.width = `${pct}%`;
    $("#progress-msg").textContent = job.status === "failed" ? `Failed: ${job.error}` : (job.message || job.stage);
    $("#progress-msg").className = job.status === "failed" ? "error" : "muted";
    if (job.status === "failed") { clearInterval(state.polling); loadList(); }
  };
  tick();
  state.polling = setInterval(tick, 1000);
  document.querySelectorAll("#doc-list li").forEach((li) => li.classList.toggle("active", li.dataset.id === id));
}

// ------------------------------------------------------------------ document view

async function showDocument(id) {
  const doc = await api(`/api/documents/${id}`);
  state.doc = doc;
  $("#empty").hidden = true; $("#progress").hidden = true; $("#doc").hidden = false;

  const m = doc.metadata, s = doc.stats, sum = doc.summary;
  $("#doc-title").textContent = sum?.title || m.title || m.filename;
  $("#doc-meta").textContent = [m.filename, sum?.document_type, m.author].filter(Boolean).join(" · ");
  for (const [el, fmtName] of [["#exp-md", "markdown"], ["#exp-jsonl", "jsonl"], ["#exp-json", "json"]]) {
    $(el).href = `/api/documents/${id}/export?format=${fmtName}`;
  }
  $("#open-pdf").href = `/api/documents/${id}/pdf`;

  const statItems = [
    [m.page_count, "pages"], [s.section_count, "sections"], [s.chunk_count, "chunks"],
    [s.total_tokens, "tokens"], [m.tables_found, "tables"],
    [s.llm_calls, "Gemini calls"],
    [(s.extraction_seconds + s.chunking_seconds + s.summarization_seconds).toFixed(1) + "s", "processing"],
  ];
  $("#stats").innerHTML = statItems.map(([v, l]) => `<div class="stat"><b>${typeof v === "number" ? fmt(v) : esc(v)}</b><span>${l}</span></div>`).join("");
  $("#warnings").innerHTML = doc.warnings.map((w) => `<div class="warning">${esc(w)}</div>`).join("");

  renderSummary(doc);
  renderOutline(doc);
  renderChunks(doc, "");
  $("#chunk-search").value = "";
}

function points(list) {
  return list?.length ? `<ul class="points">${list.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : "";
}

function paragraphs(text) {
  return String(text || "").split(/\n{2,}/).map((p) => `<p>${esc(p)}</p>`).join("");
}

function renderSummary(doc) {
  const el = $("#tab-summary");
  const s = doc.summary;
  if (!s) {
    el.innerHTML = `<p class="muted">Summarization was not enabled for this document. Explore the
      <b>Outline</b> and <b>Chunks</b> tabs, or re-process it with a Gemini key configured.</p>`;
    return;
  }
  el.innerHTML = `<div class="prose">
    ${s.audience ? `<p class="muted">Audience: ${esc(s.audience)}</p>` : ""}
    <h3>Executive summary</h3>${paragraphs(s.executive_summary)}
    ${s.key_points?.length ? `<h3>Key points</h3>${points(s.key_points)}` : ""}
    ${s.topics?.length ? `<h3>Topics</h3><div class="tags">${s.topics.map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>` : ""}
  </div>`;
}

function renderOutline(doc) {
  const byId = Object.fromEntries(doc.sections.map((s) => [s.id, s]));
  const chunkById = Object.fromEntries(doc.chunks.map((c) => [c.id, c]));
  const node = (sec) => {
    const refs = sec.chunk_ids.map((cid) => `<button class="chunk-ref" data-chunk="${cid}">#${chunkById[cid].index}</button>`).join("");
    const kids = sec.children.map((c) => node(byId[c])).join("");
    return `<details open>
      <summary>${esc(sec.title)}<span class="pages">p. ${sec.page_start}–${sec.page_end} · ${fmt(sec.token_count)} tok</span></summary>
      ${sec.summary ? `<div class="sec-summary">${esc(sec.summary)}</div>` : ""}
      ${refs ? `<div>${refs}</div>` : ""}${kids}</details>`;
  };
  const root = doc.sections[0];
  const preamble = root.chunk_ids.length
    ? `<p class="muted">Before first heading: ${root.chunk_ids.map((cid) => `<button class="chunk-ref" data-chunk="${cid}">#${chunkById[cid].index}</button>`).join("")}</p>` : "";
  const body = root.children.map((c) => node(byId[c])).join("");
  $("#tab-outline").innerHTML = `<div class="tree">${preamble}${body || '<p class="muted">No headings detected: the document is treated as a single section.</p>'}</div>`;
}

function renderChunks(doc, query) {
  const q = query.trim().toLowerCase();
  const list = doc.chunks.filter((c) => !q || [c.text, c.summary, ...(c.keywords || []), ...(c.heading_path || [])]
    .some((t) => (t || "").toLowerCase().includes(q)));
  $("#chunk-list").innerHTML = list.length ? list.map((c) => `
    <article class="chunk" id="chunk-${c.id}">
      <div class="chunk-head">
        <span>#${c.index}</span>
        <span class="crumb">${esc(c.heading_path.join(" › ") || "(document start)")}</span>
        <span>p. ${c.page_start}${c.page_end !== c.page_start ? "–" + c.page_end : ""}</span>
        <span>${fmt(c.token_count)} tokens</span>
      </div>
      ${c.summary ? `<p class="sum">${esc(c.summary)}</p>` : ""}
      ${c.keywords?.length ? `<div class="tags">${c.keywords.map((k) => `<span class="tag">${esc(k)}</span>`).join("")}</div>` : ""}
      <details ${c.summary ? "" : "open"}><summary>Chunk text</summary><pre>${esc(c.text)}</pre></details>
    </article>`).join("") : '<p class="muted">No matching chunks.</p>';
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tabpanel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
}

function initDocView() {
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => switchTab(t.dataset.tab)));
  let timer;
  $("#chunk-search").addEventListener("input", (e) => {
    clearTimeout(timer);
    timer = setTimeout(() => state.doc && renderChunks(state.doc, e.target.value), 150);
  });
  $("#tab-outline").addEventListener("click", (e) => {
    const ref = e.target.closest("[data-chunk]");
    if (!ref) return;
    switchTab("chunks");
    $("#chunk-search").value = "";
    renderChunks(state.doc, "");
    const el = document.getElementById(`chunk-${ref.dataset.chunk}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.querySelector("details").open = true;
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), 1500);
    }
  });
}

// ------------------------------------------------------------------ boot

(async function main() {
  initUpload();
  initList();
  initDocView();
  await loadHealth();
  const jobs = await loadList();
  const active = jobs.find((j) => j.status === "running" || j.status === "queued");
  if (active) watchJob(active.id);
})();
