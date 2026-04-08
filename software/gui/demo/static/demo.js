// TinyissimoYOLO demo GUI — vanilla JS, no framework.
// Fetches the image list on load, renders the hero + 4 panels + timing
// table when the user clicks "Run Inference".

const picker   = document.getElementById("image-picker");
const runBtn   = document.getElementById("run-btn");
const statusEl = document.getElementById("status-text");
const heroImg  = document.getElementById("hero-img");
const timingBody = document.getElementById("timing-tbody");

const PANELS = {
  gt:     document.getElementById("panel-gt"),
  tflite: document.getElementById("panel-tflite"),
  hdl:    document.getElementById("panel-hdl"),
  hls:    document.getElementById("panel-hls"),
};

function panelImg(panel)   { return panel.querySelector(".panel-img"); }
function panelStatus(panel){ return panel.querySelector(".panel-status"); }

function setStatus(msg, busy = false) {
  statusEl.textContent = msg;
  runBtn.disabled = busy || !picker.value;
}

function fmtMs(v) {
  if (v === null || v === undefined) return "—";
  return v.toFixed(2) + " ms";
}

function fmtCycles(v) {
  if (v === null || v === undefined) return "—";
  return v.toLocaleString();
}

// -------- image list --------------------------------------------------

async function loadImages() {
  setStatus("loading images...", true);
  try {
    const resp = await fetch("/api/images");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    picker.innerHTML = "";
    if (!data.images || data.images.length === 0) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "(no images in image-dir)";
      picker.appendChild(opt);
      setStatus("no images available", false);
      runBtn.disabled = true;
      return;
    }
    for (const name of data.images) {
      const opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      picker.appendChild(opt);
    }
    picker.selectedIndex = 0;
    updateHero();
    setStatus(`${data.images.length} images loaded`, false);
  } catch (err) {
    setStatus(`error: ${err.message}`, false);
    runBtn.disabled = true;
  }
}

function updateHero() {
  const name = picker.value;
  if (!name) { heroImg.removeAttribute("src"); return; }
  // Cache-bust so re-selecting the same image re-renders if the browser
  // was caching an old copy.
  heroImg.src = `/api/image/${encodeURIComponent(name)}?_=${Date.now()}`;
}

// -------- inference run -----------------------------------------------

async function runInference() {
  const name = picker.value;
  if (!name) return;

  setStatus(`running inference on ${name}...`, true);
  clearResults();

  const t0 = performance.now();
  try {
    const resp = await fetch(
      `/api/run/${encodeURIComponent(name)}`,
      { method: "POST" }
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const elapsed = performance.now() - t0;
    renderResults(data);
    setStatus(
      `done in ${elapsed.toFixed(0)} ms (round-trip)`,
      false,
    );
  } catch (err) {
    setStatus(`error: ${err.message}`, false);
  }
}

function clearResults() {
  for (const key of Object.keys(PANELS)) {
    const img = panelImg(PANELS[key]);
    img.removeAttribute("src");
    const status = panelStatus(PANELS[key]);
    status.classList.remove("error");
    if (key === "hls") status.textContent = "stub";
    else if (key === "gt") status.textContent = "no annotations";
    else status.textContent = "";
  }
  timingBody.innerHTML = "";
}

function setPanelImage(panel, b64) {
  const img = panelImg(panel);
  if (b64) img.src = `data:image/png;base64,${b64}`;
  else     img.removeAttribute("src");
}

function setPanelError(panel, message) {
  const status = panelStatus(panel);
  status.classList.add("error");
  status.textContent = `error: ${message}`;
}

function setPanelDetectionSummary(panel, result) {
  const status = panelStatus(panel);
  const n = result.boxes.length;
  if (n === 0) {
    status.textContent = "no detections";
  } else {
    const names = result.class_ids.map((c) =>
      ["chair","bowl","cup"][c] || `cls${c}`,
    );
    status.textContent = `${n} detection(s): ${names.join(", ")}`;
  }
}

function renderResults(data) {
  // Ground truth panel (256x256 unannotated)
  setPanelImage(PANELS.gt, data.ground_truth_png_b64);

  // TFLite / HDL / HLS panels
  for (const key of ["tflite", "hdl", "hls"]) {
    const panel = PANELS[key];
    const result = data.results[key];
    if (!result) continue;

    if (result.error) {
      setPanelError(panel, result.error);
      setPanelImage(panel, null);
    } else {
      setPanelImage(panel, result.annotated_png_b64);
      if (key === "hls") {
        panelStatus(panel).textContent = "stub (not implemented)";
      } else {
        setPanelDetectionSummary(panel, result);
      }
    }
  }

  renderTiming(data.results);
}

// -------- timing table ------------------------------------------------

function numCell(text, isNa = false) {
  const td = document.createElement("td");
  td.classList.add("num");
  if (isNa) td.classList.add("na");
  td.textContent = text;
  return td;
}

function renderTiming(results) {
  timingBody.innerHTML = "";
  const order = [
    { key: "tflite", label: "TFLite" },
    { key: "hdl",    label: "HDL accelerator" },
    { key: "hls",    label: "HLS accelerator" },
  ];

  for (const { key, label } of order) {
    const result = results[key];
    const t = result ? result.timings : {};
    const tr = document.createElement("tr");

    const name = document.createElement("td");
    name.textContent = label;
    tr.appendChild(name);

    tr.appendChild(numCell(fmtMs(t.preprocess_ms),  t.preprocess_ms == null));
    tr.appendChild(numCell(fmtMs(t.inference_ms),   t.inference_ms  == null));
    tr.appendChild(numCell(fmtCycles(t.cycles),     t.cycles        == null));
    tr.appendChild(numCell(fmtMs(t.cycle_time_ms),  t.cycle_time_ms == null));
    tr.appendChild(numCell(fmtMs(t.postprocess_ms), t.postprocess_ms== null));
    tr.appendChild(numCell(fmtMs(t.total_ms),       t.total_ms      == null));

    timingBody.appendChild(tr);
  }
}

// -------- event wiring ------------------------------------------------

picker.addEventListener("change", updateHero);
runBtn.addEventListener("click", runInference);

// Kick off on page load.
loadImages();
