// TinyissimoYOLO demo GUI — vanilla JS, no framework.
//
// Page lifecycle:
//   1. On load, fetch /api/config (class list, default thresholds, model
//      info) and /api/images (manifest-grouped picker entries).
//   2. User clicks "Run Inference" -> POST /api/run/{name} with current
//      slider values. Server caches the raw output tensor.
//   3. User drags a slider -> debounced POST /api/repostprocess/{name}
//      with new threshold values. Only post-process / end-to-end / FPS
//      cells update; pre-process and inference cells stay frozen.

const picker      = document.getElementById("image-picker");
const runBtn      = document.getElementById("run-btn");
const statusEl    = document.getElementById("status-text");
const heroImg     = document.getElementById("hero-img");
const timingBody  = document.getElementById("timing-tbody");
const confSlider  = document.getElementById("conf-slider");
const nmsSlider   = document.getElementById("nms-slider");
const confValue   = document.getElementById("conf-value");
const nmsValue    = document.getElementById("nms-value");
const legendChips = document.getElementById("class-legend-chips");
const modelInfoEl = document.getElementById("model-info-text");
const configEcho  = document.getElementById("config-echo");

const PANELS = {
  gt:     document.getElementById("panel-gt"),
  tflite: document.getElementById("panel-tflite"),
  hdl:    document.getElementById("panel-hdl"),
  hls:    document.getElementById("panel-hls"),
};

// Class list + RGB colors come from /api/config; bootstrapped to defaults
// so the frontend doesn't crash before /api/config returns.
let CLASS_NAMES = ["chair", "bowl", "cup"];
let CLASS_COLORS_RGB = [[0, 0, 255], [0, 255, 0], [255, 0, 0]];

// Track which image has been fully run (cached server-side). Slider drags
// only fire repostprocess when this matches the picker selection.
let cachedImage = null;

// TFLite thread-scaling data cached from the last /api/run response.
// Keys are thread counts (as strings from JSON), values are {inference_ms}.
let cachedThreadTimings = null;
// Base timings from the TFLite run (pre/post), needed to recompute totals
// when the thread dropdown changes.
let cachedTfliteBase = null;

// Cached ground-truth 256×256 image as an HTMLImageElement — source of
// truth for client-side Canvas drawing during live slider updates.
let cachedGroundTruth = null;

// Debounce handle for the live re-postprocess slider drags.
let repostprocessTimer = null;
// Raised from 30 ms — coalesces rapid drags into ~8 req/s which is
// realistic throughput once the server does only post_process + nms.
const REPOST_DEBOUNCE_MS = 120;

// In-flight fetch cancellation + sequence guard for live slider drags.
// AbortController cancels the network request; the monotonic sequence
// number is a belt-and-braces guard against stale responses still
// landing in .then() handlers after cancellation.
let repostprocessAbort = null;
let repostprocessSeq = 0;

function abortInFlightRepostprocess() {
  if (repostprocessAbort) {
    repostprocessAbort.abort();
    repostprocessAbort = null;
  }
}

// ----- formatters -----------------------------------------------------

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

function fmtFps(v) {
  if (v === null || v === undefined || !isFinite(v)) return "—";
  return v.toFixed(1) + " fps";
}

// FPS (Inference): the canonical "Inference Time" is already stored in
// timings.inference_ms for both runners (HDL stores cycle_time_ms there
// instead of host wall-clock — see HDLRunner.run in demo_runners.py).
function inferenceFps(t) {
  if (t.inference_ms && t.inference_ms > 0) {
    return 1000.0 / t.inference_ms;
  }
  return null;
}

// FPS (End-to-End): pre + inference + post
function e2eFps(t) {
  if (t.total_ms && t.total_ms > 0) {
    return 1000.0 / t.total_ms;
  }
  return null;
}

// Map a dominant-channel RGB tuple to a human color name. Decoupled from
// the exact CLASS_COLORS values so the legend label stays correct if the
// driver ever retunes them.
function colorName(rgb) {
  if (!rgb || rgb.length !== 3) return "";
  const [r, g, b] = rgb;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  if (max - min < 30) return max < 80 ? "Black" : (max > 200 ? "White" : "Gray");
  if (r === max && g < r * 0.7 && b < r * 0.7) return "Red";
  if (g === max && r < g * 0.7 && b < g * 0.7) return "Green";
  if (b === max && r < b * 0.7 && g < b * 0.7) return "Blue";
  if (r === max && g >= r * 0.7 && b < r * 0.5) return "Yellow";
  if (g === max && b >= g * 0.7 && r < g * 0.5) return "Cyan";
  if (r === max && b >= r * 0.7 && g < r * 0.5) return "Magenta";
  return "";
}

// ----- /api/config: class legend + model-info panel -------------------

async function loadConfig() {
  try {
    const resp = await fetch("/api/config");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    if (Array.isArray(data.class_names)) CLASS_NAMES = data.class_names;
    if (Array.isArray(data.class_colors_rgb)) CLASS_COLORS_RGB = data.class_colors_rgb;

    populateLegend(data.class_names || [], data.class_colors_rgb || []);
    populateModelInfo(data.model_info || {});

    // Sync slider initial values with server defaults.
    if (typeof data.default_conf_thresh === "number") {
      confSlider.value = data.default_conf_thresh.toFixed(2);
      confValue.textContent = parseFloat(confSlider.value).toFixed(2);
    }
    if (typeof data.default_nms_thresh === "number") {
      nmsSlider.value = data.default_nms_thresh.toFixed(2);
      nmsValue.textContent = parseFloat(nmsSlider.value).toFixed(2);
    }
  } catch (err) {
    console.warn("loadConfig failed:", err);
  }
}

function populateLegend(names, colors) {
  legendChips.innerHTML = "";
  for (let i = 0; i < names.length; i++) {
    const rgb = colors[i] || [128, 128, 128];
    const name = names[i].charAt(0).toUpperCase() + names[i].slice(1);
    const cn = colorName(rgb);
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.style.background = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
    chip.textContent = cn ? `${name} (${cn})` : name;
    legendChips.appendChild(chip);
  }
}

function populateModelInfo(info) {
  modelInfoEl.innerHTML = "";
  const sections = [
    { key: "architecture", title: "Architecture",         labelMap: ARCHITECTURE_LABELS },
    { key: "quantization", title: "Quantization",         labelMap: QUANTIZATION_LABELS },
    { key: "hardware",     title: "Hardware (HDL Path)",  labelMap: HARDWARE_LABELS },
  ];
  for (const { key, title, labelMap } of sections) {
    const section = info[key];
    if (!section) continue;
    const sec = document.createElement("div");
    sec.className = "model-info-section";
    const h4 = document.createElement("h4");
    h4.textContent = title;
    sec.appendChild(h4);
    const dl = document.createElement("dl");
    for (const [k, label] of Object.entries(labelMap)) {
      const v = section[k];
      if (v === null || v === undefined) continue;
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = Array.isArray(v) ? v.join(", ") : String(v);
      dl.appendChild(dt);
      dl.appendChild(dd);
    }
    sec.appendChild(dl);
    modelInfoEl.appendChild(sec);
  }
}

// Field key -> human label maps for each model_info section.
const ARCHITECTURE_LABELS = {
  model:         "Model",
  family:        "Family",
  input_shape:   "Input shape",
  output_shape:  "Output shape",
  stride:        "Stride",
  param_count:   "Parameters",
  layers:        "Layers",
  detect_head:   "Detect head",
  classes:       "Classes",
  training_data: "Training data",
  doc_ref:       "Reference doc",
};

const QUANTIZATION_LABELS = {
  scheme:       "Scheme",
  input_quant:  "Input",
  output_quant: "Output",
  calibration:  "Calibration",
};

const HARDWARE_LABELS = {
  accelerator:        "Accelerator",
  pl_clock_hz:        "PL clock (Hz)",
  pixel_transport:    "Pixel transport",
  cycle_counter_addr: "Cycle counter",
  output_layout:      "Output layout",
};

// ----- /api/images: load + group into <optgroup>s --------------------

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
    // Bucket entries by primary category (single -> that class, multi ->
    // "Mixed", none -> "Reference / Misc"). Stable order: known classes
    // first, then Mixed, then Misc, then anything else.
    const groups = new Map();
    for (const entry of data.images) {
      let bucket;
      if (!entry.categories || entry.categories.length === 0) {
        bucket = "Reference / Misc";
      } else if (entry.categories.length === 1) {
        const c = entry.categories[0];
        bucket = c.charAt(0).toUpperCase() + c.slice(1);
      } else {
        bucket = "Mixed";
      }
      if (!groups.has(bucket)) groups.set(bucket, []);
      groups.get(bucket).push(entry);
    }
    const order = ["Chair", "Bowl", "Cup", "Mixed", "Reference / Misc"];
    const ordered = order.filter((b) => groups.has(b))
                         .concat([...groups.keys()].filter((b) => !order.includes(b)));
    for (const bucket of ordered) {
      const og = document.createElement("optgroup");
      og.label = bucket;
      for (const entry of groups.get(bucket)) {
        const opt = document.createElement("option");
        opt.value = entry.name;
        opt.textContent = entry.label || entry.name;
        og.appendChild(opt);
      }
      picker.appendChild(og);
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

// ----- inference run --------------------------------------------------

function currentThresholds() {
  return {
    conf_thresh: parseFloat(confSlider.value),
    nms_thresh:  parseFloat(nmsSlider.value),
  };
}

async function runInference() {
  const name = picker.value;
  if (!name) return;

  setStatus(`running inference on ${name}...`, true);
  clearResults();

  const t0 = performance.now();
  try {
    const resp = await fetch(
      `/api/run/${encodeURIComponent(name)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentThresholds()),
      },
    );
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const elapsed = performance.now() - t0;
    renderResults(data);
    cachedImage = name; // enable live slider updates for this image
    setStatus(`done in ${elapsed.toFixed(0)} ms (round-trip)`, false);
  } catch (err) {
    setStatus(`error: ${err.message}`, false);
    cachedImage = null;
  }
}

function clearResults() {
  for (const key of Object.keys(PANELS)) {
    const img = panelImg(PANELS[key]);
    img.removeAttribute("src");
    const status = panelStatus(PANELS[key]);
    status.classList.remove("error");
    if (key === "gt") status.textContent = "No Annotations";
    else status.textContent = "";
  }
  timingBody.innerHTML = "";
  configEcho.textContent = "";
}

function setPanelImage(panel, b64) {
  const img = panelImg(panel);
  if (b64) {
    img.src = `data:image/png;base64,${b64}`;
    img.style.visibility = "visible";
  } else {
    // Hide the <img> entirely rather than clearing src, so the browser
    // doesn't fall back to rendering the alt attribute as giant broken-
    // image text when a runner panel is in an error state.
    img.removeAttribute("src");
    img.style.visibility = "hidden";
  }
}

function setPanelError(panel, message) {
  const status = panelStatus(panel);
  status.classList.add("error");
  status.textContent = `error: ${message}`;
}

function setPanelDetectionSummary(panel, result) {
  const status = panelStatus(panel);
  status.classList.remove("error");
  const n = result.boxes.length;
  if (n === 0) {
    status.textContent = "No Detections";
  } else {
    const names = result.class_ids.map((c) => CLASS_NAMES[c] || `cls${c}`);
    status.textContent = `${n} detection(s): ${names.join(", ")}`;
  }
}

// Cache the server's ground-truth 256×256 PNG as an HTMLImageElement so
// client-side Canvas drawing can reuse it as a base layer for bbox
// overlays during live slider drags. The Image loads async; if the
// first slider drag fires before it's ready, drawPanelFromCachedGT
// no-ops (checked via .complete) and the update is lost for that tick
// — acceptable because (a) the server already painted the initial
// annotated PNG and (b) a typical image loads in <50 ms over LAN.
function cacheGroundTruth(b64) {
  if (!b64) { cachedGroundTruth = null; return; }
  const img = new Image();
  img.src = `data:image/png;base64,${b64}`;
  cachedGroundTruth = img;
}

// Draw a 256×256 Canvas with the cached ground-truth image as the base
// layer and bbox rectangles + labels painted on top. Writes the result
// back into the target <img> via toDataURL so the CSS layout (aspect-
// ratio, pixelated rendering) is preserved. Mirrors _draw_detections()
// in software/inference/demo_runners.py — PIL and Canvas font metrics
// differ slightly but the visual intent is identical.
function drawPanelFromCachedGT(targetImg, boxes, scores, classIds) {
  if (!cachedGroundTruth || !cachedGroundTruth.complete) return;
  if (cachedGroundTruth.naturalWidth === 0) return; // still loading
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(cachedGroundTruth, 0, 0, 256, 256);

  ctx.lineWidth = 2;
  ctx.font = "bold 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  ctx.textBaseline = "top";

  for (let i = 0; i < boxes.length; i++) {
    const box = boxes[i];
    if (!box || box.length < 4) continue;
    const [x, y, w, h] = box;
    const cls = classIds[i];
    const rgb = CLASS_COLORS_RGB[cls] || [255, 255, 255];
    const colorStr = `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;

    // Clamp bbox to canvas bounds (post_process can return OOB coords).
    const x1 = Math.max(0, Math.round(x));
    const y1 = Math.max(0, Math.round(y));
    const x2 = Math.min(255, Math.round(x + w));
    const y2 = Math.min(255, Math.round(y + h));

    ctx.strokeStyle = colorStr;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    // Label bar above the bbox if there's room, else just inside.
    const label = `${CLASS_NAMES[cls] || `cls${cls}`} ${scores[i].toFixed(2)}`;
    const metrics = ctx.measureText(label);
    const barW = Math.ceil(metrics.width) + 6;
    const barH = 14;
    let barX = x1;
    let barY = y1 - barH;
    if (barY < 0) barY = y1;
    if (barX + barW > 256) barX = 256 - barW;
    if (barX < 0) barX = 0;

    ctx.fillStyle = colorStr;
    ctx.fillRect(barX, barY, barW, barH);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, barX + 3, barY + 1);
  }

  targetImg.src = canvas.toDataURL("image/png");
  targetImg.style.visibility = "visible";
}

function renderResults(data) {
  // Ground-truth panel (256x256 unannotated). Also cache the image into
  // a JS Image object so live slider updates can redraw boxes on top
  // of it via Canvas without waiting for the server PIL + PNG roundtrip.
  setPanelImage(PANELS.gt, data.ground_truth_png_b64);
  cacheGroundTruth(data.ground_truth_png_b64);

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
      setPanelDetectionSummary(panel, result);
    }
  }

  renderTiming(data.results);
  updateConfigEcho(data.results);
}

function updateConfigEcho(results) {
  // Read the thresholds back from any successful runner result so the
  // chip reflects what the *displayed* run actually used (not the
  // current slider position, which may have moved on).
  for (const key of ["tflite", "hdl"]) {
    const r = results[key];
    if (r && !r.error && r.timings && r.timings.conf_thresh != null) {
      configEcho.textContent =
        `Conf ${r.timings.conf_thresh.toFixed(2)} · NMS IoU ${r.timings.nms_thresh.toFixed(2)}`;
      return;
    }
  }
  configEcho.textContent = "";
}

// ----- timing table ---------------------------------------------------

function numCell(text, isNa = false, extraClasses = []) {
  const td = document.createElement("td");
  td.classList.add("num");
  if (isNa) td.classList.add("na");
  for (const c of extraClasses) td.classList.add(c);
  td.textContent = text;
  return td;
}

const RUNNER_ORDER = [
  { key: "tflite", label: "TFLite" },
  { key: "hdl",    label: "HDL Accelerator" },
  { key: "hls",    label: "HLS Accelerator" },
];

function renderTiming(results) {
  timingBody.innerHTML = "";

  // Cache TFLite thread-scaling data for the dropdown.
  const tfliteResult = results.tflite;
  if (tfliteResult && !tfliteResult.error && tfliteResult.timings) {
    cachedThreadTimings = tfliteResult.timings.thread_timings || null;
    cachedTfliteBase = {
      preprocess_ms: tfliteResult.timings.preprocess_ms,
      postprocess_ms: tfliteResult.timings.postprocess_ms,
    };
  }

  // Track best FPS rows for highlighting.
  let bestInfFps = -Infinity, bestInfKey = null;
  for (const { key } of RUNNER_ORDER) {
    const r = results[key];
    if (!r || r.error || !r.timings) continue;
    const f = inferenceFps(r.timings);
    if (f != null && f > bestInfFps) { bestInfFps = f; bestInfKey = key; }
  }

  for (const { key, label } of RUNNER_ORDER) {
    const result = results[key];
    const t = result ? result.timings : {};
    const tr = document.createElement("tr");
    tr.dataset.runner = key;
    if (key === bestInfKey) tr.classList.add("best");

    const name = document.createElement("td");
    if (key === "tflite" && cachedThreadTimings) {
      name.innerHTML = "";
      const span = document.createElement("span");
      span.textContent = label + " ";
      name.appendChild(span);
      const sel = document.createElement("select");
      sel.id = "thread-select";
      sel.className = "thread-dropdown";
      const counts = Object.keys(cachedThreadTimings).map(Number).sort((a, b) => a - b);
      for (const n of counts) {
        const opt = document.createElement("option");
        opt.value = n;
        opt.textContent = `${n} thread${n > 1 ? "s" : ""}`;
        sel.appendChild(opt);
      }
      // Default to highest thread count
      sel.value = Math.max(...counts);
      sel.addEventListener("change", onThreadDropdownChange);
      name.appendChild(sel);
    } else {
      name.textContent = label;
    }
    tr.appendChild(name);

    // Static (frozen on live updates) cells
    tr.appendChild(numCell(fmtMs(t.preprocess_ms),    t.preprocess_ms == null,
                           ["live-frozen", "col-pre"]));
    tr.appendChild(numCell(fmtMs(t.inference_ms),     t.inference_ms == null,
                           ["live-frozen", "col-inf"]));
    tr.appendChild(numCell(fmtCycles(t.cycles),       t.cycles == null,
                           ["live-frozen", "col-cycles"]));

    // Live-updated cells (post-process / end-to-end / fps)
    tr.appendChild(numCell(fmtMs(t.postprocess_ms),   t.postprocess_ms == null,
                           ["live-update", "col-post"]));
    tr.appendChild(numCell(fmtMs(t.render_ms),        t.render_ms == null,
                           ["live-frozen", "col-render"]));
    tr.appendChild(numCell(fmtMs(t.total_ms),         t.total_ms == null,
                           ["live-update", "col-total"]));

    const inf = inferenceFps(t);
    const e2e = e2eFps(t);
    tr.appendChild(numCell(fmtFps(inf), inf == null, ["live-update", "col-fpsinf"]));
    tr.appendChild(numCell(fmtFps(e2e), e2e == null, ["live-update", "col-fpse2e"]));

    timingBody.appendChild(tr);
  }
}

// When the TFLite thread dropdown changes, swap inference_ms and recompute
// total / FPS cells from cached data.
function onThreadDropdownChange() {
  if (!cachedThreadTimings || !cachedTfliteBase) return;
  const sel = document.getElementById("thread-select");
  const n = sel.value;
  const entry = cachedThreadTimings[n];
  if (!entry) return;

  const tr = timingBody.querySelector('tr[data-runner="tflite"]');
  if (!tr) return;

  const infMs = entry.inference_ms;
  const preMs = cachedTfliteBase.preprocess_ms || 0;
  // Use the currently displayed post_ms (may have been updated by slider)
  const postCell = tr.querySelector("td.col-post");
  const postMs = postCell ? parseFloat(postCell.textContent) || 0 : (cachedTfliteBase.postprocess_ms || 0);
  const totalMs = preMs + infMs + postMs;

  const infCell = tr.querySelector("td.col-inf");
  const totalCell = tr.querySelector("td.col-total");
  const fpsiCell = tr.querySelector("td.col-fpsinf");
  const fpseCell = tr.querySelector("td.col-fpse2e");

  if (infCell)   { infCell.textContent   = fmtMs(infMs);  flash(infCell); }
  if (totalCell) { totalCell.textContent = fmtMs(totalMs); flash(totalCell); }
  if (fpsiCell)  { fpsiCell.textContent  = fmtFps(infMs > 0 ? 1000 / infMs : null); flash(fpsiCell); }
  if (fpseCell)  { fpseCell.textContent  = fmtFps(totalMs > 0 ? 1000 / totalMs : null); flash(fpseCell); }

  // Update best-row highlighting across the table
  updateBestRow();
}

function updateBestRow() {
  let bestFps = -Infinity, bestKey = null;
  for (const { key } of RUNNER_ORDER) {
    const tr = timingBody.querySelector(`tr[data-runner="${key}"]`);
    if (!tr) continue;
    const cell = tr.querySelector("td.col-fpsinf");
    if (!cell) continue;
    const val = parseFloat(cell.textContent);
    if (!isNaN(val) && val > bestFps) { bestFps = val; bestKey = key; }
  }
  for (const { key } of RUNNER_ORDER) {
    const tr = timingBody.querySelector(`tr[data-runner="${key}"]`);
    if (!tr) continue;
    tr.classList.toggle("best", key === bestKey);
  }
}

// Update only the live-updated cells (post-process / end-to-end / fps)
// of the existing rows. Pre-process / inference time / cycles cells
// untouched. Adds a brief flash class to make the change visible.
function updateTimingLive(results) {
  for (const { key } of RUNNER_ORDER) {
    const tr = timingBody.querySelector(`tr[data-runner="${key}"]`);
    if (!tr) continue;
    const result = results[key];
    if (!result || result.error || !result.timings) continue;
    const t = result.timings;

    const post = tr.querySelector("td.col-post");
    const total = tr.querySelector("td.col-total");
    const fpsi = tr.querySelector("td.col-fpsinf");
    const fpse = tr.querySelector("td.col-fpse2e");

    if (post)  { post.textContent  = fmtMs(t.postprocess_ms); flash(post); }

    // For TFLite, use the currently selected thread count's inference_ms
    // to compute total and FPS (the server always returns 4-thread values).
    let infMs = t.inference_ms;
    if (key === "tflite" && cachedThreadTimings) {
      const sel = document.getElementById("thread-select");
      if (sel) {
        const entry = cachedThreadTimings[sel.value];
        if (entry) infMs = entry.inference_ms;
      }
      // Also update the cached base postprocess for future dropdown changes.
      cachedTfliteBase.postprocess_ms = t.postprocess_ms;
    }
    const pre = t.preprocess_ms || 0;
    const totalMs = pre + (infMs || 0) + (t.postprocess_ms || 0);
    if (total) { total.textContent = fmtMs(totalMs); flash(total); }
    const inf = (infMs && infMs > 0) ? 1000 / infMs : null;
    const e2e = totalMs > 0 ? 1000 / totalMs : null;
    if (fpsi)  { fpsi.textContent  = fmtFps(inf); flash(fpsi); }
    if (fpse)  { fpse.textContent  = fmtFps(e2e); flash(fpse); }

    // Live-update: draw the panel client-side on a Canvas overlay
    // rather than consuming a server-rendered PNG. The /api/repostprocess
    // response no longer includes annotated_png_b64 — the server only
    // returns boxes/scores/class_ids and we paint them here. This cuts
    // ~50 ms of latency per tick compared to the initial PIL + PNG +
    // base64 + browser-decode round-trip.
    if (PANELS[key]) {
      const img = panelImg(PANELS[key]);
      drawPanelFromCachedGT(img, result.boxes, result.scores, result.class_ids);
      setPanelDetectionSummary(PANELS[key], result);
    }
  }
}

function flash(td) {
  td.classList.remove("updated");
  // Force a reflow so re-adding the class restarts the transition.
  void td.offsetWidth;
  td.classList.add("updated");
  setTimeout(() => td.classList.remove("updated"), 500);
}

// ----- live re-postprocess (slider drag handler) ----------------------

async function repostprocess() {
  const name = picker.value;
  if (!name || cachedImage !== name) return;

  // Cancel any older in-flight request before issuing a new one, and
  // stamp this request with a monotonic sequence so responses that
  // resolve out-of-order (e.g. race between abort and resolve) can be
  // discarded.
  abortInFlightRepostprocess();
  const mySeq = ++repostprocessSeq;
  repostprocessAbort = new AbortController();
  const mySignal = repostprocessAbort.signal;

  try {
    const resp = await fetch(
      `/api/repostprocess/${encodeURIComponent(name)}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(currentThresholds()),
        signal: mySignal,
      },
    );
    if (resp.status === 404) {
      // Cache miss (shouldn't happen if cachedImage is set, but be
      // resilient if the server restarted).
      cachedImage = null;
      return;
    }
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    // Belt-and-braces: drop stale responses even if AbortController
    // races with fetch resolution.
    if (mySeq !== repostprocessSeq) return;
    updateTimingLive(data.results);
    updateConfigEcho(data.results);
  } catch (err) {
    if (err.name === "AbortError") return; // expected on cancellation
    console.warn("repostprocess failed:", err);
  }
}

function scheduleRepostprocess() {
  if (repostprocessTimer) clearTimeout(repostprocessTimer);
  repostprocessTimer = setTimeout(repostprocess, REPOST_DEBOUNCE_MS);
}

// ----- event wiring ---------------------------------------------------

picker.addEventListener("change", () => {
  // Switching images invalidates the cache binding AND the visible
  // panels (which otherwise still show the previous run's annotated
  // PNG — the bug that made a soup image sit next to a stale chair
  // image in the TFLite/HDL panels). Cancel any in-flight repostprocess
  // because its cached tensors belong to the OLD image.
  cachedImage = null;
  cachedGroundTruth = null;
  abortInFlightRepostprocess();
  clearResults();
  updateHero();
  // Tell the viewer what to do next — otherwise the blank panels look
  // like a bug rather than "you need to click Run Inference".
  for (const key of ["tflite", "hdl"]) {
    if (PANELS[key]) {
      panelStatus(PANELS[key]).textContent = "Click Run Inference";
    }
  }
});

runBtn.addEventListener("click", runInference);

confSlider.addEventListener("input", () => {
  confValue.textContent = parseFloat(confSlider.value).toFixed(2);
  scheduleRepostprocess();
});

nmsSlider.addEventListener("input", () => {
  nmsValue.textContent = parseFloat(nmsSlider.value).toFixed(2);
  scheduleRepostprocess();
});

// Kick off on page load. Config first so the legend / model-info appear
// immediately, then the image list (which may take a moment if the
// directory is large).
loadConfig();
loadImages();
