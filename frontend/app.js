"use strict";

const $ = (sel) => document.querySelector(sel);
const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const fmt = (n) => Number(n || 0).toLocaleString();

const state = { files: [], gemini: false, currentId: null, doc: null, polling: null };

// ------------------------------------------------------------------ api

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      msg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

function toast(msg, kind = "ok") {
  const host = $("#toasts");
  if (!host) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  el.textContent = msg;
  host.appendChild(el);
  setTimeout(() => {
    el.classList.add("leaving");
    setTimeout(() => el.remove(), 260);
  }, 2600);
}

// ------------------------------------------------------------------ health

async function loadHealth() {
  const pill = $("#health");
  try {
    const h = await api("/api/health");
    state.gemini = h.gemini_configured;
    pill.innerHTML = `<span class="dot"></span><span class="pill-text">${
      h.gemini_configured ? `Gemini: ${h.model}` : "Gemini key not set: chunking only"
    }</span>`;
    pill.className = "pill " + (h.gemini_configured ? "ok" : "warn");
    if (!h.gemini_configured) {
      $("#summarize").checked = false;
      $("#summarize").disabled = true;
      $("#strategy").querySelector('option[value="semantic"]').disabled = true;
    }
  } catch {
    pill.innerHTML = `<span class="dot"></span><span class="pill-text">server unreachable</span>`;
    pill.className = "pill warn";
  }
}

// ------------------------------------------------------------------ upload

function setFiles(list) {
  const files = [...list].filter((f) => f.type === "application/pdf" || /\.pdf$/i.test(f.name));
  if (!files.length) return;
  state.files = files;
  $("#file-name").textContent =
    files.length === 1
      ? `${files[0].name} · ${(files[0].size / 1048576).toFixed(1)} MB`
      : `${files.length} PDFs selected · ${(files.reduce((n, f) => n + f.size, 0) / 1048576).toFixed(1)} MB total`;
  $("#submit").disabled = false;
}

function buildOptions() {
  return {
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
}

function initUpload() {
  const drop = $("#drop");
  $("#file").addEventListener("change", (e) => setFiles(e.target.files));
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.add("over");
    }),
  );
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => {
      e.preventDefault();
      drop.classList.remove("over");
    }),
  );
  drop.addEventListener("drop", (e) => setFiles(e.dataTransfer.files));

  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("#form-error");
    err.hidden = true;
    if (!state.files.length) return;
    const options = buildOptions();
    const fd = new FormData();
    for (const f of state.files) fd.append("files", f);
    fd.append("options", JSON.stringify(options));
    $("#submit").disabled = true;
    try {
      const jobs = await api("/api/documents/batch", { method: "POST", body: fd });
      await loadList();
      if (jobs[0]) watchJob(jobs[0].id);
      toast(jobs.length === 1 ? "Processing started" : `${jobs.length} documents queued`);
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
  if (!jobs.length) {
    ul.innerHTML = '<li class="muted">No documents yet</li>';
    return jobs;
  }
  ul.innerHTML = jobs
    .map(
      (j) => `
    <li data-id="${esc(j.id)}" class="${j.id === state.currentId ? "active" : ""}" title="${esc(j.error || j.filename)}">
      <span class="name">${esc(j.filename)}</span>
      <span class="status ${j.status === "failed" ? "failed" : ""}">${
        j.status === "running" ? Math.round(j.progress * 100) + "%" : esc(j.status)
      }</span>
      <button class="del" data-del="${esc(j.id)}" aria-label="Delete ${esc(j.filename)}" title="Delete">✕</button>
    </li>`,
    )
    .join("");
  return jobs;
}

function initList() {
  $("#doc-list").addEventListener("click", async (e) => {
    const del = e.target.closest("[data-del]");
    if (del) {
      e.stopPropagation();
      if (!confirm("Delete this document and its results?")) return;
      try {
        await api(`/api/documents/${del.dataset.del}`, { method: "DELETE" });
      } catch (ex) {
        alert(ex.message);
      }
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
  $("#empty").hidden = false;
  $("#progress").hidden = true;
  $("#doc").hidden = true;
}

function watchJob(id) {
  clearInterval(state.polling);
  state.currentId = id;
  const tick = async () => {
    let job;
    try {
      job = await api(`/api/jobs/${id}`);
    } catch {
      return;
    }
    if (state.currentId !== id) return;
    if (job.status === "completed") {
      clearInterval(state.polling);
      loadList();
      return showDocument(id);
    }
    $("#empty").hidden = true;
    $("#doc").hidden = true;
    $("#progress").hidden = false;
    $("#progress-file").textContent = job.filename;
    const pct = Math.round(job.progress * 100);
    $("#progress-pct").textContent = `${pct}%`;
    $("#progress-bar").style.width = `${pct}%`;
    $("#progress-msg").textContent = job.status === "failed" ? `Failed: ${job.error}` : job.message || job.stage;
    $("#progress-msg").className = job.status === "failed" ? "error" : "muted";
    document.querySelectorAll("#stepper li").forEach((li) => {
      const order = ["extracting", "chunking", "summarizing", "completed"];
      const cur = order.indexOf(job.stage === "finalizing" ? "chunking" : job.stage);
      const mine = order.indexOf(li.dataset.stage);
      li.classList.toggle("done", mine < cur || job.status === "completed");
      li.classList.toggle("active", mine === cur && job.status !== "failed");
    });
    if (job.status === "failed") {
      clearInterval(state.polling);
      loadList();
    }
  };
  tick();
  state.polling = setInterval(tick, 1000);
  document.querySelectorAll("#doc-list li").forEach((li) => li.classList.toggle("active", li.dataset.id === id));
}

// ------------------------------------------------------------------ document view

async function showDocument(id) {
  const doc = await api(`/api/documents/${id}`);
  state.doc = doc;
  $("#empty").hidden = true;
  $("#progress").hidden = true;
  $("#doc").hidden = false;

  const m = doc.metadata,
    s = doc.stats,
    sum = doc.summary;
  $("#doc-title").textContent = sum?.title || m.title || m.filename;
  $("#doc-meta").textContent = [m.filename, sum?.document_type, m.author].filter(Boolean).join(" · ");
  for (const [el, fmtName] of [
    ["#exp-md", "markdown"],
    ["#exp-summary-pdf", "pdf"],
    ["#exp-jsonl", "jsonl"],
    ["#exp-json", "json"],
  ]) {
    $(el).href = `/api/documents/${id}/export?format=${fmtName}`;
  }
  $("#open-pdf").href = `/api/documents/${id}/pdf`;

  const statItems = [
    [m.page_count, "pages"],
    [s.section_count, "sections"],
    [s.chunk_count, "chunks"],
    [s.total_tokens, "tokens"],
    [m.tables_found, "tables"],
    [s.llm_calls, "Gemini calls"],
    [(s.extraction_seconds + s.chunking_seconds + s.summarization_seconds).toFixed(1) + "s", "processing"],
  ];
  $("#stats").innerHTML = statItems
    .map(([v, l]) => `<div class="stat"><b>${typeof v === "number" ? fmt(v) : esc(v)}</b><span>${l}</span></div>`)
    .join("");
  $("#warnings").innerHTML = doc.warnings.map((w) => `<div class="warning">${esc(w)}</div>`).join("");
  $("#chunk-count-badge").textContent = doc.chunks.length;

  renderSummary(doc);
  renderOutline(doc);
  renderChunks(doc, "");
  $("#chunk-search").value = "";
  $("#ask-result").hidden = true;
  $("#ask-error").hidden = true;
  $("#ask-input").value = "";
  switchTab("summary");
}

function points(list) {
  return list?.length ? `<ul class="points">${list.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>` : "";
}

function paragraphs(text) {
  return String(text || "")
    .split(/\n{2,}/)
    .map((p) => `<p>${esc(p)}</p>`)
    .join("");
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
    const refs = sec.chunk_ids
      .map((cid) => `<button class="chunk-ref" data-chunk="${cid}">#${chunkById[cid].index}</button>`)
      .join("");
    const kids = sec.children.map((c) => node(byId[c])).join("");
    return `<details open>
      <summary>${esc(sec.title)}<span class="pages">p. ${sec.page_start}–${sec.page_end} · ${fmt(sec.token_count)} tok</span></summary>
      ${sec.summary ? `<div class="sec-summary">${esc(sec.summary)}</div>` : ""}
      ${refs ? `<div>${refs}</div>` : ""}${kids}</details>`;
  };
  const root = doc.sections[0];
  const preamble = root.chunk_ids.length
    ? `<p class="muted">Before first heading: ${root.chunk_ids
        .map((cid) => `<button class="chunk-ref" data-chunk="${cid}">#${chunkById[cid].index}</button>`)
        .join("")}</p>`
    : "";
  const body = root.children.map((c) => node(byId[c])).join("");
  $("#tab-outline").innerHTML = `<div class="tree">${preamble}${
    body || '<p class="muted">No headings detected: the document is treated as a single section.</p>'
  }</div>`;
}

function renderChunks(doc, query) {
  const q = query.trim().toLowerCase();
  const list = doc.chunks.filter(
    (c) =>
      !q ||
      [c.text, c.summary, ...(c.keywords || []), ...(c.heading_path || [])].some((t) =>
        (t || "").toLowerCase().includes(q),
      ),
  );
  $("#chunk-match-info").textContent = q ? `${list.length} match${list.length === 1 ? "" : "es"}` : "";
  $("#chunk-list").innerHTML = list.length
    ? list
        .map(
          (c) => `
    <article class="chunk" id="chunk-${c.id}" data-chunk-id="${c.id}">
      <div class="chunk-head">
        <span class="chunk-idx">#${c.index}</span>
        <span class="crumb">${esc(c.heading_path.join(" › ") || "(document start)")}</span>
        <span>p. ${c.page_start}${c.page_end !== c.page_start ? "–" + c.page_end : ""}</span>
        <span>${fmt(c.token_count)} tokens</span>
        <button type="button" class="edit-btn" data-edit="${c.id}">Edit</button>
      </div>
      ${c.summary ? `<p class="chunk-sum sum-view">${esc(c.summary)}</p>` : '<p class="chunk-sum sum-view muted">No summary</p>'}
      ${c.keywords?.length ? `<div class="tags">${c.keywords.map((k) => `<span class="tag">${esc(k)}</span>`).join("")}</div>` : ""}
      <details ${c.summary ? "" : "open"}><summary>Chunk text</summary><pre class="text-view">${esc(c.text)}</pre></details>
      <div class="chunk-edit" hidden>
        <label>Summary<textarea class="edit-summary" rows="3">${esc(c.summary || "")}</textarea></label>
        <label>Chunk text<textarea class="edit-text" rows="8">${esc(c.text)}</textarea></label>
        <div class="edit-actions">
          <button type="button" class="primary save-edit">Save</button>
          <button type="button" class="btn cancel-edit">Cancel</button>
        </div>
      </div>
    </article>`,
        )
        .join("")
    : '<p class="muted">No matching chunks.</p>';
}

async function saveChunkEdit(article) {
  const chunkId = article.dataset.chunkId;
  const text = article.querySelector(".edit-text").value;
  const summary = article.querySelector(".edit-summary").value;
  const doc = await api(`/api/documents/${state.currentId}/chunks/${chunkId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, summary }),
  });
  state.doc = doc;
  renderChunks(doc, $("#chunk-search").value);
  toast("Chunk updated");
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
      const details = el.querySelector("details");
      if (details) details.open = true;
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), 1500);
    }
  });
  $("#chunk-list").addEventListener("click", async (e) => {
    const editBtn = e.target.closest("[data-edit]");
    if (editBtn) {
      const article = editBtn.closest(".chunk");
      article.querySelector(".chunk-edit").hidden = false;
      return;
    }
    if (e.target.closest(".cancel-edit")) {
      e.target.closest(".chunk-edit").hidden = true;
      return;
    }
    if (e.target.closest(".save-edit")) {
      try {
        await saveChunkEdit(e.target.closest(".chunk"));
      } catch (ex) {
        toast(ex.message, "err");
      }
    }
  });

  $("#ask-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("#ask-error");
    const result = $("#ask-result");
    err.hidden = true;
    result.hidden = true;
    const question = $("#ask-input").value.trim();
    if (!question || !state.currentId) return;
    if (!state.gemini) {
      err.textContent = "Gemini key is required for Ask.";
      err.hidden = false;
      return;
    }
    $("#ask-submit").disabled = true;
    try {
      const data = await api(`/api/documents/${state.currentId}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });
      result.innerHTML = `
        <div class="prose">${paragraphs(data.answer)}</div>
        ${
          data.citations?.length
            ? `<h3 class="ask-cites-title">Citations</h3>
               <ul class="ask-cites">${data.citations
                 .map(
                   (c) => `<li>
                     <button type="button" class="chunk-ref" data-ask-chunk="${esc(c.chunk_id)}">#${c.index}</button>
                     p. ${c.page_start}${c.page_end !== c.page_start ? "–" + c.page_end : ""}
                     · ${esc((c.heading_path || []).join(" › ") || "document")}
                     <div class="muted">${esc(c.excerpt || "")}</div>
                   </li>`,
                 )
                 .join("")}</ul>`
            : ""
        }`;
      result.hidden = false;
    } catch (ex) {
      err.textContent = ex.message;
      err.hidden = false;
    } finally {
      $("#ask-submit").disabled = false;
    }
  });

  $("#ask-result").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-ask-chunk]");
    if (!btn || !state.doc) return;
    switchTab("chunks");
    renderChunks(state.doc, "");
    const el = document.getElementById(`chunk-${btn.dataset.askChunk}`);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      el.classList.add("flash");
      setTimeout(() => el.classList.remove("flash"), 1500);
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "/" && !e.metaKey && !e.ctrlKey && document.activeElement?.tagName !== "INPUT" && document.activeElement?.tagName !== "TEXTAREA") {
      e.preventDefault();
      switchTab("chunks");
      $("#chunk-search").focus();
    }
  });
}

function initTheme() {
  const btn = $("#theme-toggle");
  if (!btn) return;
  const apply = (theme) => {
    if (theme) document.documentElement.setAttribute("data-theme", theme);
    else document.documentElement.removeAttribute("data-theme");
    btn.textContent = (theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light")) === "dark" ? "☀" : "☾";
  };
  apply(localStorage.getItem("pdfchunk-theme"));
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme");
    const next = cur === "dark" ? "light" : cur === "light" ? null : matchMedia("(prefers-color-scheme: dark)").matches ? "light" : "dark";
    if (next) localStorage.setItem("pdfchunk-theme", next);
    else localStorage.removeItem("pdfchunk-theme");
    apply(next);
  });
}

// ------------------------------------------------------------------ boot

(async function main() {
  initUpload();
  initList();
  initDocView();
  initTheme();
  await loadHealth();
  const jobs = await loadList();
  const active = jobs.find((j) => j.status === "running" || j.status === "queued");
  if (active) watchJob(active.id);
})();
