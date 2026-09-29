const $ = (id) => document.getElementById(id);

const state = {
  jobs: [], job: null, image: null, filtered: [], health: null,
  bitmap: null, gradcam: null, gradcamKey: "", zoom: 1, panX: 0, panY: 0, dragging: false, lastX: 0, lastY: 0,
  pollTimer: null, thresholdEditing: false, spineThresholdEditing: false,
  view3d: { yaw: -0.6, pitch: 0.15, scale: 1.3, dragging: false, x: 0, y: 0 },
};

const regionNames = { spine: "Позвоночник", left_hip: "Левое бедро", right_hip: "Правое бедро", hip_unknown_side: "Бедро" };
const violationNames = {
  spine_positioning: "Нарушение укладки позвоночника",
  spine_axis: "Отклонение оси позвоночника",
  spine_artifact: "Артефакт или посторонний предмет",
  hip_positioning_rotation: "Позиционирование или ротация бедра",
  hip_roi: "Некорректная область интереса бедра",
};

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function showNotice(message, error = false) {
  const notice = $("notice");
  notice.textContent = message;
  notice.className = error ? "notice error" : "notice";
  notice.hidden = !message;
}

async function api(path, options) {
  const response = await fetch(path, options);
  let body = {};
  try { body = await response.json(); } catch { /* non-JSON response */ }
  if (!response.ok) throw new Error(body.error || `Ошибка сервера: ${response.status}`);
  return body;
}

function jobLabel(job) {
  const marker = ({ queued: "в очереди", running: "обработка", failed: "ошибка" }[job.status] || "готово");
  return `${job.name} · ${marker}`;
}

async function loadJobs(preserve = true) {
  const previous = state.job ? { id: state.job.id, status: state.job.status } : null;
  const current = preserve && state.job ? state.job.id : null;
  state.jobs = await api("/api/jobs");
  state.job = state.jobs.find((job) => job.id === current) || state.jobs[0] || null;
  renderJobPicker();
  renderAll();
  if (previous && previous.id === state.job?.id && ["queued", "running"].includes(previous.status)) {
    if (state.job.status === "completed") showNotice(`Пакет «${state.job.name}» обработан. Результаты готовы.`);
    if (state.job.status === "failed") showNotice(state.job.error || "Обработка завершилась с ошибкой.", true);
  }
  clearTimeout(state.pollTimer);
  if (state.jobs.some((job) => ["queued", "running"].includes(job.status))) {
    state.pollTimer = setTimeout(() => loadJobs().catch((error) => showNotice(`Связь с сервером потеряна: ${error.message}`, true)), 1800);
  }
}

function renderJobPicker() {
  const select = $("job-select");
  select.replaceChildren();
  for (const job of state.jobs) {
    const option = node("option", "", jobLabel(job));
    option.value = job.id;
    option.selected = job.id === state.job?.id;
    select.append(option);
  }
}

function statusText(job) {
  if (!job) return "Нет загруженных наборов";
  if (job.status === "queued") return "Ожидает запуска";
  if (job.status === "running") return "Выполняется полный анализ…";
  if (job.status === "failed") return job.error || "Обработка завершилась с ошибкой";
  return "Пакетная обработка завершена";
}

function matches(image) {
  const query = $("search").value.trim().toLowerCase();
  const quality = $("quality").value;
  return (!query || [image.filename, image.image_uid, image.study_uid].some((value) => (value || "").toLowerCase().includes(query)))
    && (!$("region").value || image.anatomical_region === $("region").value)
    && (!quality || (quality === "failed" ? image.processing_status !== "Success" : image.quality_class === quality));
}

function qualityInfo(image) {
  if (image.processing_status !== "Success") return { label: "Ошибка обработки", className: "failed" };
  if (image.quality_class === "1") return { label: "Требует внимания", className: "bad" };
  return { label: "Без нарушений", className: "" };
}

function renderAll() {
  const images = state.job?.images || [];
  state.filtered = images.filter(matches);
  if (!state.image || !images.some((image) => image.id === state.image.id)) state.image = images[0] || null;
  const good = images.filter((image) => image.processing_status === "Success" && image.quality_class === "0").length;
  const bad = images.filter((image) => image.processing_status === "Success" && image.quality_class === "1").length;
  $("stat-total").textContent = images.length || "—";
  $("stat-good").textContent = images.length ? good : "—";
  $("stat-bad").textContent = images.length ? bad : "—";
  $("stat-failed").textContent = images.length ? images.length - good - bad : "—";
  $("batch-status").textContent = statusText(state.job);
  $("batch-status").className = state.job?.status === "failed" ? "error" : "";
  for (const format of ["csv", "xlsx"]) {
    const link = $(`export-${format}`);
    link.hidden = state.job?.status !== "completed";
    if (state.job) link.href = `/api/jobs/${encodeURIComponent(state.job.id)}/export.${format}`;
  }
  updateProcessingOverlay();
  renderList();
  renderTable();
  renderImageDetail();
}

function updateProcessingOverlay() {
  const overlay = $("processing-overlay");
  const active = state.job && ["queued", "running"].includes(state.job.status);
  overlay.hidden = !active;
  document.body.classList.toggle("processing", Boolean(active));
  if (!active) return;
  const queued = state.job.status === "queued";
  $("processing-title").textContent = queued ? "Исследование принято" : "Идёт обработка исследований";
  $("processing-package").textContent = state.job.name;
  $("processing-stage").textContent = queued ? "Ожидает запуска" : "Анализ DICOM";
  $("processing-detail").textContent = queued
    ? "Архив успешно загружен и поставлен в очередь."
    : "Определяем область, строим landmarks и проверяем критерии качества.";
  const elapsed = Math.max(0, Math.floor(Date.now() / 1000 - Number(state.job.created || Date.now() / 1000)));
  $("processing-elapsed").textContent = elapsed < 60 ? `${elapsed} с` : `${Math.floor(elapsed / 60)} мин ${elapsed % 60} с`;
}

function chooseImage(image) {
  state.image = image;
  state.thresholdEditing = false;
  state.spineThresholdEditing = false;
  state.bitmap = null; state.gradcam = null; state.gradcamKey = "";
  state.zoom = 1; state.panX = 0; state.panY = 0;
  renderAll();
  loadPreview();
}

function renderList() {
  const list = $("image-list");
  list.replaceChildren();
  $("list-count").textContent = state.filtered.length ? state.filtered.length : "";
  if (!state.filtered.length) {
    list.append(node("div", "empty-list", state.job?.status === "completed" ? "По фильтрам ничего не найдено" : "Результаты появятся после обработки"));
    return;
  }
  for (const image of state.filtered) {
    const button = node("button", `image-card${image.id === state.image?.id ? " selected" : ""}`);
    button.type = "button";
    const thumb = node("img", "thumb");
    thumb.alt = "";
    thumb.loading = "lazy";
    thumb.src = `/api/jobs/${encodeURIComponent(state.job.id)}/images/${encodeURIComponent(image.id)}/preview.png`;
    const text = node("span", "card-text");
    text.append(node("strong", "", regionNames[image.anatomical_region] || image.anatomical_region || "Область не определена"));
    text.append(node("small", "", image.filename || image.image_uid));
    const q = qualityInfo(image);
    text.append(node("span", `card-status ${q.className}`, q.label));
    if (image.quality_class === "1" && image.violation_description) text.append(node("span", "card-reason", image.violation_description));
    button.append(thumb, text);
    button.addEventListener("click", () => chooseImage(image));
    list.append(button);
  }
}

function renderTable() {
  const body = $("results-body");
  body.replaceChildren();
  if (!state.filtered.length) {
    const row = node("tr");
    const cell = node("td", "", state.job?.status === "completed" ? "Нет строк, соответствующих фильтрам" : "Таблица будет заполнена после обработки");
    cell.colSpan = 7;
    row.append(cell); body.append(row); return;
  }
  for (const image of state.filtered) {
    const row = node("tr", image.id === state.image?.id ? "selected" : "");
    row.tabIndex = 0;
    const q = qualityInfo(image);
    const qualityCell = node("td");
    qualityCell.append(node("span", `table-quality ${q.className}`, q.label));
    const values = [
      image.filename || "—", regionNames[image.anatomical_region] || image.anatomical_region || "—",
      qualityCell, violationLabels(image).join(", ") || "—", image.violation_description || "—", image.processing_status || "—", image.time_of_processing || "—",
    ];
    for (const value of values) row.append(value instanceof HTMLElement ? value : node("td", "", value));
    row.addEventListener("click", () => chooseImage(image));
    row.addEventListener("keydown", (event) => { if (["Enter", " "].includes(event.key)) chooseImage(image); });
    body.append(row);
  }
}

function violationLabels(image) {
  return (image?.violation_type || "").split(/[;,|]+/).map((v) => v.trim()).filter(Boolean).map((v) => violationNames[v] || v);
}

function renderImageDetail() {
  const image = state.image;
  $("image-empty").hidden = Boolean(image);
  $("image-title").textContent = image ? (regionNames[image.anatomical_region] || image.anatomical_region || "Область не определена") : "Выберите снимок";
  $("image-number").textContent = image ? `СНИМОК ${Number(image.id) + 1} ИЗ ${state.job.images.length}` : "ПРОСМОТР СНИМКА";
  const badge = $("quality-badge");
  if (image) {
    const q = qualityInfo(image); badge.textContent = q.label; badge.className = `badge ${q.className}`; badge.hidden = false;
  } else badge.hidden = true;
  $("meta-file").textContent = image?.filename || "—";
  $("meta-study").textContent = image?.study_uid || "—";
  $("meta-image").textContent = image?.image_uid || "—";
  $("image-size").textContent = image?.width && image?.height ? `${image.width} × ${image.height} px` : "—";
  renderFindings();
  renderSpineThresholds();
  renderHipThresholds();
  updateGradcamControls();
  update3dPanel();
  if (image && !state.bitmap) loadPreview(); else drawImage();
}

function renderFindings() {
  const container = $("findings");
  const measurements = $("measurements");
  container.replaceChildren(); measurements.replaceChildren();
  const image = state.image;
  if (!image) {
    container.append(node("div", "empty-list", "Результат ещё не выбран")); return;
  }
  const labels = violationLabels(image);
  if (image.processing_status !== "Success") {
    addFinding(container, "Ошибка обработки", "Снимок сохранён в отчёте со статусом Failure.", true);
  } else if (!labels.length) {
    addFinding(container, "Критерии соблюдены", "Алгоритм не обнаружил нарушений из поддерживаемого перечня.", false);
  } else {
    labels.forEach((label) => addFinding(container, label, findingExplanation(label), true));
  }
  const metricNames = {
    spine_axis_angle_deg: ["Наклон оси позвоночника", "°"], hip_lesser_prominence_mm: ["Выступание малого вертела", " мм"],
    hip_top_margin_mm: ["Верхний отступ", " мм"], hip_bottom_margin_mm: ["Нижний отступ", " мм"],
    hip_lateral_margin_mm: ["Боковой отступ", " мм"],
    hip_lesser_lower_bottom_margin_mm: ["Нижний отступ малого вертела", " мм"],
  };
  let count = 0;
  for (const [key, value] of Object.entries(image.metrics || {})) {
    if (value === null || value === undefined) continue;
    const row = node("div", "metric");
    row.append(node("span", "", metricNames[key]?.[0] || key), node("strong", "", `${Number(value).toFixed(1)}${metricNames[key]?.[1] || ""}`));
    measurements.append(row); count++;
  }
  if (!count) measurements.append(node("div", "metric", "Числовые измерения недоступны"));
  $("source-note").hidden = true;
}

function updateGradcamControls() {
  const available = state.image?.anatomical_region === "spine" && state.image?.processing_status === "Success";
  $("gradcam").disabled = !available;
  if (!available) {
    $("gradcam").checked = false;
    state.gradcam = null;
    state.gradcamKey = "";
  }
}

function addFinding(container, title, description, bad) {
  const item = node("div", `finding${bad ? " bad" : ""}`);
  const heading = node("h3"); heading.append(node("span", "symbol", bad ? "!" : "✓"), document.createTextNode(title));
  item.append(heading, node("p", "", description)); container.append(item);
}

function findingExplanation(label) {
  if (label.includes("оси")) return "Оценка центральной линии вышла за заданный критерий.";
  if (label.includes("Артефакт")) return "На изображении обнаружены признаки постороннего объекта или выраженного наложения.";
  if (label.includes("область интереса")) return "Один или несколько необходимых отступов от области интереса недостаточны.";
  if (label.includes("бедра")) return "Геометрия ориентиров указывает на возможную ошибку позиционирования или ротации.";
  return "Нарушение определено существующим пайплайном контроля качества.";
}


const spineCenterNames = ["spine_th12_center", "spine_l1_center", "spine_l2_center", "spine_l3_center", "spine_l4_center", "spine_l5_center"];

function hasViolation(image, code) {
  return (image?.violation_type || "").split(/[;,|]+/).map((value) => value.trim()).includes(code);
}

function pointByName(image, name) {
  return (image?.landmarks || []).find((point) => point.name === name);
}

function metricValue(image, key) {
  const value = image?.metrics?.[key];
  return value === null || value === undefined ? null : Number(value);
}

function currentThresholds() {
  return state.job?.thresholds || {
    spine_axis_angle_threshold_deg: 1.979,
    hip_lesser_prominence_low_mm: 0.2,
    hip_lesser_prominence_high_mm: 6.346,
  };
}

function formatThreshold(value) {
  return Number(value).toFixed(3).replace(/\.?0+$/, "");
}

function parseThresholdInput(value, label) {
  const normalized = String(value || "").trim().replace(",", ".");
  if (!normalized) throw new Error(`${label}: значение не должно быть пустым`);
  const parsed = Number(normalized);
  if (!Number.isFinite(parsed) || parsed < 0) throw new Error(`${label}: нужно число не меньше 0`);
  return parsed;
}

function setSpineThresholdEditorEnabled(enabled) {
  $("spine-threshold-angle").disabled = !enabled;
  $("spine-threshold-edit").hidden = enabled;
  $("spine-threshold-save").hidden = !enabled;
  $("spine-threshold-cancel").hidden = !enabled;
}

function renderSpineThresholds() {
  const panel = $("spine-thresholds");
  const isSpine = state.image?.anatomical_region === "spine";
  panel.hidden = !isSpine;
  if (!isSpine) { state.spineThresholdEditing = false; return; }
  const thresholds = currentThresholds();
  if (!state.spineThresholdEditing) {
    $("spine-threshold-angle").value = formatThreshold(thresholds.spine_axis_angle_threshold_deg);
    $("spine-threshold-error").textContent = "";
  }
  $("spine-threshold-summary").textContent = `${formatThreshold(thresholds.spine_axis_angle_threshold_deg)}°`;
  setSpineThresholdEditorEnabled(state.spineThresholdEditing);
}

async function saveSpineThreshold() {
  if (!state.job) return;
  try {
    const angle = parseThresholdInput($("spine-threshold-angle").value, "Порог угла");
    const updated = await api(`/api/jobs/${encodeURIComponent(state.job.id)}/thresholds`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ spine_axis_angle_threshold_deg: angle }),
    });
    const selectedId = state.image?.id;
    state.jobs = state.jobs.map((job) => job.id === updated.id ? updated : job);
    state.job = updated;
    state.image = updated.images.find((image) => image.id === selectedId) || updated.images[0] || null;
    state.spineThresholdEditing = false;
    showNotice("Порог оси позвоночника обновлён. Результаты и экспорт пересчитаны.");
    renderAll();
  } catch (error) {
    $("spine-threshold-error").textContent = error.message;
  }
}

function setThresholdEditorEnabled(enabled) {
  ["hip-threshold-low", "hip-threshold-high"].forEach((id) => $(id).disabled = !enabled);
  $("hip-threshold-edit").hidden = enabled;
  $("hip-threshold-save").hidden = !enabled;
  $("hip-threshold-cancel").hidden = !enabled;
}

function renderHipThresholds() {
  const panel = $("hip-thresholds");
  const isHip = ["left_hip", "right_hip"].includes(state.image?.anatomical_region);
  panel.hidden = !isHip;
  if (!isHip) { state.thresholdEditing = false; return; }
  const thresholds = currentThresholds();
  if (!state.thresholdEditing) {
    $("hip-threshold-low").value = formatThreshold(thresholds.hip_lesser_prominence_low_mm);
    $("hip-threshold-high").value = formatThreshold(thresholds.hip_lesser_prominence_high_mm);
    $("hip-threshold-error").textContent = "";
  }
  $("hip-threshold-summary").textContent = `${formatThreshold(thresholds.hip_lesser_prominence_low_mm)}–${formatThreshold(thresholds.hip_lesser_prominence_high_mm)} мм`;
  setThresholdEditorEnabled(state.thresholdEditing);
}

async function saveHipThresholds() {
  if (!state.job) return;
  try {
    const low = parseThresholdInput($("hip-threshold-low").value, "Нижняя граница");
    const high = parseThresholdInput($("hip-threshold-high").value, "Верхняя граница");
    if (low >= high) throw new Error("Нижняя граница должна быть меньше верхней");
    const updated = await api(`/api/jobs/${encodeURIComponent(state.job.id)}/thresholds`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ hip_lesser_prominence_low_mm: low, hip_lesser_prominence_high_mm: high }),
    });
    const selectedId = state.image?.id;
    state.jobs = state.jobs.map((job) => job.id === updated.id ? updated : job);
    state.job = updated;
    state.image = updated.images.find((image) => image.id === selectedId) || updated.images[0] || null;
    state.thresholdEditing = false;
  state.spineThresholdEditing = false;
    showNotice("Диапазон малого вертела обновлён. Результаты и экспорт пересчитаны.");
    renderAll();
  } catch (error) {
    $("hip-threshold-error").textContent = error.message;
  }
}

function imageToCanvas(point, transform) {
  return { x: transform.x + point.x * transform.scale, y: transform.y + point.y * transform.scale };
}

function drawLabel(ctx, x, y, text, color) {
  const ratio = window.devicePixelRatio || 1;
  ctx.save();
  ctx.font = `${11 * ratio}px system, sans-serif`;
  const paddingX = 7 * ratio, paddingY = 4 * ratio;
  const width = ctx.measureText(text).width + paddingX * 2;
  const height = 20 * ratio;
  const left = Math.min(Math.max(8 * ratio, x), ctx.canvas.width - width - 8 * ratio);
  const top = Math.min(Math.max(8 * ratio, y - height - 7 * ratio), ctx.canvas.height - height - 8 * ratio);
  ctx.fillStyle = "rgba(10, 27, 31, .86)";
  ctx.strokeStyle = color;
  ctx.lineWidth = 1 * ratio;
  ctx.beginPath();
  ctx.roundRect(left, top, width, height, 5 * ratio);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = "#f4fffb";
  ctx.fillText(text, left + paddingX, top + 14 * ratio);
  ctx.restore();
}

function drawLine(ctx, a, b, color, dashed = false, width = 2) {
  const ratio = window.devicePixelRatio || 1;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = width * ratio;
  ctx.lineCap = "round";
  if (dashed) ctx.setLineDash([6 * ratio, 5 * ratio]);
  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
  ctx.restore();
}

function drawMeasurement(ctx, a, b, label, color) {
  const ratio = window.devicePixelRatio || 1;
  drawLine(ctx, a, b, color, false, 2.5);
  ctx.save();
  ctx.fillStyle = color;
  for (const [from, to] of [[a, b], [b, a]]) {
    const angle = Math.atan2(to.y - from.y, to.x - from.x);
    ctx.beginPath();
    ctx.moveTo(to.x, to.y);
    ctx.lineTo(to.x - Math.cos(angle - .55) * 8 * ratio, to.y - Math.sin(angle - .55) * 8 * ratio);
    ctx.lineTo(to.x - Math.cos(angle + .55) * 8 * ratio, to.y - Math.sin(angle + .55) * 8 * ratio);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();
  drawLabel(ctx, (a.x + b.x) / 2 + 4 * ratio, (a.y + b.y) / 2, label, color);
}

function drawSpineExplanation(ctx, image, transform) {
  const good = "#62d0b0", bad = "#f0a73a", ref = "rgba(230, 245, 242, .58)";
  if (hasViolation(image, "spine_positioning")) {
    const y1 = transform.y + image.height * .72 * transform.scale;
    const y2 = transform.y + image.height * transform.scale;
    ctx.save();
    ctx.fillStyle = "rgba(240, 167, 58, .18)";
    ctx.fillRect(transform.x, y1, image.width * transform.scale, y2 - y1);
    ctx.restore();
    drawLabel(ctx, transform.x + 12 * (window.devicePixelRatio || 1), y1 + 25 * (window.devicePixelRatio || 1), "Подвздошные кости не обнаружены", bad);
  }

  const centers = spineCenterNames.map((name) => pointByName(image, name)).filter(Boolean).sort((a, b) => a.y - b.y);
  if (centers.length >= 2) {
    const top = imageToCanvas(centers[0], transform);
    const bottom = imageToCanvas(centers.at(-1), transform);
    const color = hasViolation(image, "spine_axis") ? bad : good;
    drawLine(ctx, { x: top.x, y: top.y }, { x: top.x, y: bottom.y }, ref, true, 1.5);
    drawLine(ctx, top, bottom, color, false, 3);
    const angle = metricValue(image, "spine_axis_angle_deg");
    if (angle !== null) drawLabel(ctx, bottom.x + 8 * (window.devicePixelRatio || 1), bottom.y, `Ось ${angle.toFixed(1)}°`, color);
  }

  if (hasViolation(image, "spine_artifact")) {
    drawLabel(ctx, transform.x + image.width * transform.scale - 210 * (window.devicePixelRatio || 1), transform.y + 32 * (window.devicePixelRatio || 1), "Вероятный артефакт", bad);
  }
}

function drawHipExplanation(ctx, image, transform) {
  const good = "#62d0b0", bad = "#f0a73a";
  const topEdge = pointByName(image, "hip_greater_trochanter_top_edge");
  const lateralEdge = pointByName(image, "hip_greater_trochanter_lateral_edge");
  const ischium = pointByName(image, "hip_ischium_edge");
  const lesserUpper = pointByName(image, "hip_lesser_trochanter_upper");
  const lesserTip = pointByName(image, "hip_lesser_trochanter_tip");
  const lesserLower = pointByName(image, "hip_lesser_trochanter_lower");
  const ratio = window.devicePixelRatio || 1;

  if (topEdge) {
    const p = imageToCanvas(topEdge, transform);
    const value = metricValue(image, "hip_top_margin_mm");
    const color = value !== null && value < 30 ? bad : good;
    drawMeasurement(ctx, { x: p.x, y: transform.y }, p, value === null ? "Верхний отступ" : `Верх ${value.toFixed(1)} мм`, color);
  }
  if (ischium) {
    const p = imageToCanvas(ischium, transform);
    const bottom = transform.y + image.height * transform.scale;
    const value = metricValue(image, "hip_bottom_margin_mm");
    const color = value !== null && value < 30 ? bad : good;
    drawMeasurement(ctx, p, { x: p.x, y: bottom }, value === null ? "Нижний отступ" : `Низ ${value.toFixed(1)} мм`, color);
  }
  if (lateralEdge) {
    const p = imageToCanvas(lateralEdge, transform);
    const isLeftHip = image.anatomical_region === "left_hip";
    const borderX = transform.x + (isLeftHip ? image.width * transform.scale : 0);
    const value = metricValue(image, "hip_lateral_margin_mm");
    const color = value !== null && value < 20 ? bad : good;
    drawMeasurement(ctx, { x: borderX, y: p.y }, p, value === null ? "Боковой отступ" : `Бок ${value.toFixed(1)} мм`, color);
  }

  if (lesserUpper && lesserTip && lesserLower) {
    const upper = imageToCanvas(lesserUpper, transform);
    const tip = imageToCanvas(lesserTip, transform);
    const lower = imageToCanvas(lesserLower, transform);
    const color = hasViolation(image, "hip_positioning_rotation") ? bad : good;
    drawLine(ctx, upper, lower, color, false, 2);
    const dx = lower.x - upper.x, dy = lower.y - upper.y;
    const len2 = Math.max(dx * dx + dy * dy, 1);
    const t = ((tip.x - upper.x) * dx + (tip.y - upper.y) * dy) / len2;
    const foot = { x: upper.x + t * dx, y: upper.y + t * dy };
    drawLine(ctx, foot, tip, color, true, 2);
    const value = metricValue(image, "hip_lesser_prominence_mm");
    drawLabel(ctx, tip.x + 8 * ratio, tip.y, value === null ? "Малый вертел" : `Малый вертел ${value.toFixed(1)} мм`, color);
  }
}

function drawExplanationOverlay(ctx, image, transform) {
  if (!image || image.processing_status !== "Success") return;
  if (image.anatomical_region === "spine") drawSpineExplanation(ctx, image, transform);
  if (["left_hip", "right_hip"].includes(image.anatomical_region)) drawHipExplanation(ctx, image, transform);
}


function currentGradcamKey() {
  if (!state.job || !state.image) return "";
  return `${state.job.id}:${state.image.id}:artifact`;
}

function loadGradcam() {
  if (!$('gradcam').checked || !state.job || !state.image || state.image.anatomical_region !== "spine") return;
  const key = currentGradcamKey();
  if (!key || state.gradcamKey === key && state.gradcam) return;
  state.gradcam = null;
  state.gradcamKey = key;
  const source = new Image();
  source.onload = () => { if (state.gradcamKey === key) { state.gradcam = source; drawImage(); } };
  source.onerror = () => { if (state.gradcamKey === key) showNotice("Не удалось построить тепловую карту артефактов", true); };
  source.src = `/api/jobs/${encodeURIComponent(state.job.id)}/images/${encodeURIComponent(state.image.id)}/gradcam/artifact.png`;
}

function drawGradcam(ctx, transform) {
  if (!$('gradcam').checked || !state.gradcam || !state.image || state.image.anatomical_region !== "spine") return;
  if (state.gradcamKey !== currentGradcamKey()) return;
  ctx.save();
  ctx.globalAlpha = .82;
  ctx.drawImage(state.gradcam, transform.x, transform.y, state.image.width * transform.scale, state.image.height * transform.scale);
  ctx.restore();
}

async function loadPreview() {
  if (!state.image || !state.job) return;
  const source = new Image();
  source.onload = () => { if (state.image) { state.bitmap = source; drawImage(); } };
  source.onerror = () => { state.bitmap = null; showNotice("Не удалось загрузить изображение для просмотра", true); };
  source.src = `/api/jobs/${encodeURIComponent(state.job.id)}/images/${encodeURIComponent(state.image.id)}/preview.png`;
}

function canvasSize(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  return { width, height, ratio };
}

function drawImage() {
  const canvas = $("image-canvas");
  const { width, height } = canvasSize(canvas);
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, width, height);
  if (!state.bitmap || !state.image) return;
  const fit = Math.min(width / state.bitmap.width, height / state.bitmap.height) * .88;
  const scale = fit * state.zoom;
  const x = (width - state.bitmap.width * scale) / 2 + state.panX;
  const y = (height - state.bitmap.height * scale) / 2 + state.panY;
  ctx.filter = `brightness(${$("brightness").value}%) contrast(${$("contrast").value}%)`;
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(state.bitmap, x, y, state.bitmap.width * scale, state.bitmap.height * scale);
  ctx.filter = "none";
  const transform = { x, y, scale };
  if ($("gradcam").checked) { drawGradcam(ctx, transform); loadGradcam(); }
  if ($("explain").checked) drawExplanationOverlay(ctx, state.image, transform);
  if ($("landmarks").checked) {
    for (const point of state.image.landmarks || []) {
      const px = x + point.x * scale, py = y + point.y * scale;
      ctx.beginPath(); ctx.arc(px, py, 4 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
      ctx.fillStyle = "#62d0b0"; ctx.fill(); ctx.lineWidth = 1.5 * (window.devicePixelRatio || 1); ctx.strokeStyle = "#102b2a"; ctx.stroke();
    }
  }
  $("zoom-label").textContent = `${Math.round(state.zoom * 100)}%`;
}

function reconstruction() { return state.job?.reconstructions?.[state.image?.id]; }

function update3dPanel() {
  const enabled = state.image?.anatomical_region === "spine";
  $("tab-3d").disabled = !enabled;
  const current = reconstruction();
  $("reconstruct").disabled = !state.health?.reconstruction || current?.status === "queued";
  if (!enabled) $("three-description").textContent = "3D-функция предназначена для снимков позвоночника.";
  else if (!state.health?.reconstruction) $("three-description").textContent = state.health?.reconstruction_reason || "3D-реконструкция пока недоступна в этой сборке.";
  else if (current?.status === "queued") $("three-description").textContent = "Выполняется реконструкция…";
  else if (current?.status === "failed") $("three-description").textContent = current.error;
  else $("three-description").textContent = "Модель оценивает фронтальную и сагиттальную центральные линии и объединяет их в пространственную ось.";
  $("three-empty").hidden = current?.status === "completed";
  draw3d();
}

function draw3d() {
  const canvas = $("spine-canvas");
  const { width, height } = canvasSize(canvas);
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, width, height);
  const points = reconstruction()?.points;
  if (!points?.length) return;
  const smoothPoints=points.map((_,index)=>{
    const from=Math.max(0,index-7),to=Math.min(points.length-1,index+7);
    let x=0,y=0,z=0,weightSum=0;
    for(let i=from;i<=to;i++){
      const distance=Math.abs(i-index),weight=8-distance;
      x+=points[i][0]*weight;y+=points[i][1]*weight;z+=points[i][2]*weight;weightSum+=weight;
    }
    return [x/weightSum,y/weightSum,z/weightSum];
  });
  const xs = smoothPoints.map((p) => p[0]), ys = smoothPoints.map((p) => p[1]), zs = smoothPoints.map((p) => p[2]);
  const center = [(Math.min(...xs) + Math.max(...xs))/2, (Math.min(...ys) + Math.max(...ys))/2, (Math.min(...zs) + Math.max(...zs))/2];
  const span = Math.max(Math.max(...xs)-Math.min(...xs), Math.max(...ys)-Math.min(...ys), Math.max(...zs)-Math.min(...zs), 1);
  const c1 = Math.cos(state.view3d.yaw), s1 = Math.sin(state.view3d.yaw), c2 = Math.cos(state.view3d.pitch), s2 = Math.sin(state.view3d.pitch);
  const projected = smoothPoints.map((p) => {
    const x0=(p[0]-center[0])/span, y0=(p[1]-center[1])/span, z0=(p[2]-center[2])/span;
    const x1=x0*c1-z0*s1, z1=x0*s1+z0*c1, y1=y0*c2-z1*s2, depth=y0*s2+z1*c2;
    const perspective=1/(1.8+depth*.35), size=Math.min(width,height)*1.35*state.view3d.scale;
    return [width/2+x1*size*perspective,height/2-y1*size*perspective,depth];
  });
  const pixelRatio=window.devicePixelRatio||1;
  const tracePath=()=>{
    ctx.beginPath();
    ctx.moveTo(projected[0][0],projected[0][1]);
    for(let i=1;i<projected.length-1;i++){
      const midpoint=[(projected[i][0]+projected[i+1][0])/2,(projected[i][1]+projected[i+1][1])/2];
      ctx.quadraticCurveTo(projected[i][0],projected[i][1],midpoint[0],midpoint[1]);
    }
    ctx.lineTo(projected.at(-1)[0],projected.at(-1)[1]);
  };
  ctx.save();
  ctx.lineCap="round";ctx.lineJoin="round";
  tracePath();ctx.strokeStyle="#07191c";ctx.lineWidth=16*pixelRatio;ctx.shadowColor="#64d5b9aa";ctx.shadowBlur=14*pixelRatio;ctx.stroke();
  const gradient=ctx.createLinearGradient(0,height,width,0);gradient.addColorStop(0,"#236c68");gradient.addColorStop(.48,"#66c5ae");gradient.addColorStop(1,"#b4efdc");
  tracePath();ctx.shadowBlur=0;ctx.strokeStyle=gradient;ctx.lineWidth=10*pixelRatio;ctx.stroke();
  tracePath();ctx.strokeStyle="#d8fff2aa";ctx.lineWidth=2*pixelRatio;ctx.stroke();
  for(const point of [projected[0],projected.at(-1)]){
    ctx.beginPath();ctx.arc(point[0],point[1],6*pixelRatio,0,Math.PI*2);ctx.fillStyle="#b9efdc";ctx.fill();ctx.strokeStyle="#173d3a";ctx.lineWidth=2*pixelRatio;ctx.stroke();
  }
  ctx.restore();
}

async function request3d() {
  if (!state.job || !state.image) return;
  try {
    await api(`/api/jobs/${encodeURIComponent(state.job.id)}/images/${encodeURIComponent(state.image.id)}/reconstruct`, { method: "POST" });
    await refreshCurrentUntil3d();
  } catch (error) { showNotice(error.message, true); }
}

async function refreshCurrentUntil3d() {
  const updated = await api(`/api/jobs/${encodeURIComponent(state.job.id)}`);
  state.jobs = state.jobs.map((job) => job.id === updated.id ? updated : job); state.job = updated;
  state.image = updated.images.find((image) => image.id === state.image.id); renderAll();
  if (reconstruction()?.status === "queued") setTimeout(refreshCurrentUntil3d, 1500);
}

function setupViewerEvents() {
  const canvas = $("image-canvas");
  canvas.addEventListener("pointerdown", (event) => { state.dragging=true;state.lastX=event.clientX;state.lastY=event.clientY;canvas.setPointerCapture(event.pointerId); });
  canvas.addEventListener("pointermove", (event) => { if(!state.dragging)return; const r=window.devicePixelRatio||1;state.panX+=(event.clientX-state.lastX)*r;state.panY+=(event.clientY-state.lastY)*r;state.lastX=event.clientX;state.lastY=event.clientY;drawImage(); });
  canvas.addEventListener("pointerup", () => state.dragging=false);
  canvas.addEventListener("wheel", (event) => { event.preventDefault(); state.zoom=Math.max(.5,Math.min(4,state.zoom*(event.deltaY<0?1.1:.9)));drawImage(); }, {passive:false});
  const c3=$("spine-canvas");
  c3.addEventListener("pointerdown",e=>{state.view3d.dragging=true;state.view3d.x=e.clientX;state.view3d.y=e.clientY;c3.setPointerCapture(e.pointerId)});
  c3.addEventListener("pointermove",e=>{if(!state.view3d.dragging)return;state.view3d.yaw+=(e.clientX-state.view3d.x)*.01;state.view3d.pitch+=(e.clientY-state.view3d.y)*.01;state.view3d.x=e.clientX;state.view3d.y=e.clientY;draw3d()});
  c3.addEventListener("pointerup",()=>state.view3d.dragging=false);
  c3.addEventListener("wheel",e=>{e.preventDefault();state.view3d.scale=Math.max(.5,Math.min(3,state.view3d.scale*(e.deltaY<0?1.1:.9)));draw3d()},{passive:false});
  c3.addEventListener("keydown",e=>{if(e.key==="ArrowLeft")state.view3d.yaw-=.1;if(e.key==="ArrowRight")state.view3d.yaw+=.1;if(e.key==="ArrowUp")state.view3d.pitch-=.1;if(e.key==="ArrowDown")state.view3d.pitch+=.1;if(e.key==="+")state.view3d.scale*=1.1;if(e.key==="-")state.view3d.scale*=.9;draw3d()});
}

function setupUpload() {
  const dialog=$("upload-dialog"), file=$("file"), zone=$("drop-zone");
  $("upload-open").addEventListener("click",()=>dialog.showModal()); $("upload-close").addEventListener("click",()=>dialog.close());
  file.addEventListener("change",()=>$("file-name").textContent=file.files[0]?.name||"");
  ["dragenter","dragover"].forEach(name=>zone.addEventListener(name,e=>{e.preventDefault();zone.classList.add("drag")}));
  ["dragleave","drop"].forEach(name=>zone.addEventListener(name,e=>{e.preventDefault();zone.classList.remove("drag")}));
  zone.addEventListener("drop",e=>{if(e.dataTransfer.files[0]){file.files=e.dataTransfer.files;$("file-name").textContent=file.files[0].name}});
  $("upload-form").addEventListener("submit",event=>{
    event.preventDefault(); const selected=file.files[0]; if(!selected)return;
    $("upload-error").textContent=""; $("upload-progress").hidden=false; $("upload-submit").disabled=true;
    const xhr=new XMLHttpRequest(); xhr.open("POST","/api/jobs"); xhr.setRequestHeader("Content-Type","application/zip");xhr.setRequestHeader("X-Archive-Name",encodeURIComponent(selected.name));
    xhr.upload.onprogress=e=>{if(e.lengthComputable)$("upload-progress").value=e.loaded/e.total*100};
    xhr.onload=async()=>{ $("upload-submit").disabled=false; try{const result=JSON.parse(xhr.responseText);if(xhr.status>=400)throw new Error(result.error||"Ошибка загрузки");state.job=result;state.jobs=[result,...state.jobs.filter(job=>job.id!==result.id)];renderAll();dialog.close();file.value="";$("file-name").textContent="";await loadJobs(true)}catch(error){$("upload-error").textContent=error.message} };
    xhr.onerror=()=>{$("upload-submit").disabled=false;$("upload-error").textContent="Не удалось подключиться к локальному серверу"}; xhr.send(selected);
  });
}

async function init() {
  setupViewerEvents(); setupUpload();
  setInterval(updateProcessingOverlay, 1000);
  $("job-select").addEventListener("change",event=>{state.job=state.jobs.find(j=>j.id===event.target.value);state.image=null;state.bitmap=null;state.thresholdEditing=false;state.spineThresholdEditing=false;renderAll()});
  ["search","region","quality"].forEach(id=>$(id).addEventListener(id==="search"?"input":"change",renderAll));
  $("refresh").addEventListener("click",()=>loadJobs()); $("landmarks").addEventListener("change",drawImage); $("explain").addEventListener("change",drawImage);
  $("gradcam").addEventListener("change",()=>{updateGradcamControls(); if($("gradcam").checked)loadGradcam(); drawImage()});
  ["brightness","contrast"].forEach(id=>$(id).addEventListener("input",drawImage));
  $("zoom-in").addEventListener("click",()=>{state.zoom=Math.min(4,state.zoom*1.15);drawImage()}); $("zoom-out").addEventListener("click",()=>{state.zoom=Math.max(.5,state.zoom/1.15);drawImage()});
  $("reset").addEventListener("click",()=>{state.zoom=1;state.panX=0;state.panY=0;$("brightness").value=100;$("contrast").value=100;drawImage()});
  $("reset-3d").addEventListener("click",()=>{state.view3d={yaw:-.6,pitch:.15,scale:1.3,dragging:false,x:0,y:0};draw3d()}); $("reconstruct").addEventListener("click",request3d);
  $("tab-2d").addEventListener("click",()=>switchTab(false)); $("tab-3d").addEventListener("click",()=>switchTab(true));
  $("spine-threshold-edit").addEventListener("click",()=>{state.spineThresholdEditing=true;renderSpineThresholds();$("spine-threshold-angle").focus()});
  $("spine-threshold-cancel").addEventListener("click",()=>{state.spineThresholdEditing=false;renderSpineThresholds()});
  $("spine-threshold-save").addEventListener("click",saveSpineThreshold);
  $("hip-threshold-edit").addEventListener("click",()=>{state.thresholdEditing=true;renderHipThresholds();$("hip-threshold-low").focus()});
  $("hip-threshold-cancel").addEventListener("click",()=>{state.thresholdEditing=false;renderHipThresholds()});
  $("hip-threshold-save").addEventListener("click",saveHipThresholds);
  window.addEventListener("resize",()=>{drawImage();draw3d()});
  try { state.health=await api("/api/health"); $("connection").textContent="Локальный сервер подключён";$("connection-dot").classList.add("online");await loadJobs(false); }
  catch(error){$("connection").textContent="Сервер недоступен";showNotice(error.message,true)}
}

function switchTab(three) {
  if(three && $("tab-3d").disabled)return;
  $("panel-2d").hidden=three;$("panel-3d").hidden=!three;$("tab-2d").classList.toggle("active",!three);$("tab-3d").classList.toggle("active",three);$("tab-2d").setAttribute("aria-selected",String(!three));$("tab-3d").setAttribute("aria-selected",String(three));if(three)draw3d();else drawImage();
}

init();
