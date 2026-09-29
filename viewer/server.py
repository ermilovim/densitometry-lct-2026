"""Local UI adapter. The team's inference scripts remain the source of QC results."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).parent / "web"
STORE = ROOT / "artifacts" / "viewer_jobs"
sys.path.insert(0, str(ROOT))

FIELDS = ["path_to_study", "study_uid", "image_uid", "anatomical_region", "quality_class", "violation_type", "violation_description", "processing_status", "time_of_processing"]
JOBS: dict[str, dict] = {}
LOCK = threading.RLock()
POOL = ThreadPoolExecutor(max_workers=1)
ILIAC_LANDMARKS = {"spine_left_iliac_crest", "spine_right_iliac_crest"}
UPLOAD_CHUNK = 1024 * 1024


def rows(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (ValueError, TypeError):
        return None


def visible_landmark_points(point_row, detail):
    points = []
    hide_iliac = (
        detail.get("anatomical_region") == "spine"
        and detail.get("spine_positioning_pred_bad") == "1"
    )
    for key, val in point_row.items():
        if not key.endswith("_x"):
            continue
        name = key[:-2]
        if hide_iliac and name in ILIAC_LANDMARKS:
            continue
        if point_row.get(name + "_visible") != "1":
            continue
        x, y = number(val), number(point_row.get(name + "_y"))
        if x is not None and y is not None:
            points.append({"name": name, "x": x, "y": y})
    return points


SPINE_AXIS_ANGLE_THRESHOLD_DEG = 1.979
HIP_LESSER_PROMINENCE_LOW_THRESHOLD_MM = 0.200
HIP_LESSER_PROMINENCE_HIGH_THRESHOLD_MM = 6.346
DEFAULT_THRESHOLDS = {
    "spine_axis_angle_threshold_deg": SPINE_AXIS_ANGLE_THRESHOLD_DEG,
    "hip_lesser_prominence_low_mm": HIP_LESSER_PROMINENCE_LOW_THRESHOLD_MM,
    "hip_lesser_prominence_high_mm": HIP_LESSER_PROMINENCE_HIGH_THRESHOLD_MM,
}
HIP_ROI_TOP_THRESHOLD_MM = 30.0
HIP_ROI_LATERAL_THRESHOLD_MM = 20.0
HIP_ROI_BOTTOM_THRESHOLD_MM = 30.0
HIP_ROI_LESSER_LOWER_BOTTOM_THRESHOLD_MM = 30.0


def violation_codes(row):
    return {value.strip() for value in (row.get("violation_type") or "").split(";") if value.strip()}


def threshold_settings(job_or_none=None):
    settings = dict(DEFAULT_THRESHOLDS)
    if job_or_none:
        settings.update(job_or_none.get("thresholds") or {})
    return settings


def parse_threshold(value, field):
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        number_value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Некорректное значение порога: {field}") from None
    if not math.isfinite(number_value) or number_value < 0:
        raise ValueError(f"Некорректное значение порога: {field}")
    return number_value


def parse_threshold_payload(payload, current=None):
    settings = threshold_settings(current)
    if "spine_axis_angle_threshold_deg" in payload:
        settings["spine_axis_angle_threshold_deg"] = parse_threshold(
            payload.get("spine_axis_angle_threshold_deg"),
            "порог угла позвоночника",
        )
    if "hip_lesser_prominence_low_mm" in payload:
        settings["hip_lesser_prominence_low_mm"] = parse_threshold(
            payload.get("hip_lesser_prominence_low_mm"),
            "нижняя граница",
        )
    if "hip_lesser_prominence_high_mm" in payload:
        settings["hip_lesser_prominence_high_mm"] = parse_threshold(
            payload.get("hip_lesser_prominence_high_mm"),
            "верхняя граница",
        )
    if settings["hip_lesser_prominence_low_mm"] >= settings["hip_lesser_prominence_high_mm"]:
        raise ValueError("Нижняя граница должна быть меньше верхней")
    return settings


def landmark_point(point_row, key):
    if point_row.get(f"{key}_visible") != "1":
        return None
    x = number(point_row.get(f"{key}_x"))
    y = number(point_row.get(f"{key}_y"))
    if x is None or y is None:
        return None
    return {"x": x, "y": y}


def hip_landmark_order_is_bad(point_row, region):
    greater_top = landmark_point(point_row, "hip_greater_trochanter_top_edge")
    greater_lateral = landmark_point(point_row, "hip_greater_trochanter_lateral_edge")
    lesser_upper = landmark_point(point_row, "hip_lesser_trochanter_upper")
    lesser_lower = landmark_point(point_row, "hip_lesser_trochanter_lower")
    if not all([greater_top, greater_lateral, lesser_upper, lesser_lower]):
        return False

    vertical_ok = greater_top["y"] < greater_lateral["y"] < lesser_upper["y"]
    return not vertical_ok


def hip_positioning_is_bad(point_row, region):
    required = [
        landmark_point(point_row, "hip_greater_trochanter_top_edge"),
        landmark_point(point_row, "hip_greater_trochanter_lateral_edge"),
        landmark_point(point_row, "hip_ischium_edge"),
    ]
    return not all(required) or hip_landmark_order_is_bad(point_row, region)


def hip_rotation_is_bad(prominence, thresholds):
    return (
        prominence is None
        or prominence <= thresholds["hip_lesser_prominence_low_mm"]
        or prominence >= thresholds["hip_lesser_prominence_high_mm"]
    )


def apply_viewer_thresholds(image, detail, point_row, thresholds):
    region = image.get("anatomical_region")
    if region not in {"spine", "left_hip", "right_hip"}:
        return

    codes = violation_codes(image)
    if region == "spine":
        codes.discard("spine_axis")
        angle = number(detail.get("spine_axis_angle_deg"))
        if angle is not None and angle >= thresholds["spine_axis_angle_threshold_deg"]:
            codes.add("spine_axis")
        ordered = [code for code in image.get("violation_type", "").split(";") if code and code in codes]
        ordered += [code for code in ["spine_positioning", "spine_axis", "spine_artifact"] if code in codes and code not in ordered]
    else:
        codes.discard("hip_positioning_rotation")
        prominence = number(detail.get("hip_lesser_prominence_mm"))
        structure_missing = (
            number(detail.get("hip_top_margin_mm")) is None
            or number(detail.get("hip_lateral_margin_mm")) is None
            or number(detail.get("hip_bottom_margin_mm")) is None
        )
        positioning_bad = hip_positioning_is_bad(point_row, region)
        if structure_missing or positioning_bad or hip_rotation_is_bad(prominence, thresholds):
            codes.add("hip_positioning_rotation")
        ordered = [code for code in image.get("violation_type", "").split(";") if code and code in codes]
        ordered += [code for code in ["hip_positioning_rotation", "hip_roi"] if code in codes and code not in ordered]

    image["violation_type"] = ";".join(ordered)
    image["quality_class"] = "1" if codes else "0"


def violation_description(row, detail, point_row, thresholds=None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    codes = violation_codes(row)
    descriptions = []
    region = row.get("anatomical_region", "")

    if "spine_positioning" in codes:
        descriptions.append("Не обнаружены подвздошные кости на снимке")
    if "spine_axis" in codes:
        angle = number(detail.get("spine_axis_angle_deg"))
        descriptions.append("Отклонение оси позвоночника" + (f" ({angle:.1f}°)" if angle is not None else ""))
    if "spine_artifact" in codes:
        descriptions.append("Обнаружены признаки артефакта или постороннего предмета")

    if "hip_positioning_rotation" in codes and region in {"left_hip", "right_hip"}:
        if hip_positioning_is_bad(point_row, region):
            descriptions.append("Некорректное позиционирование")
        prominence = number(detail.get("hip_lesser_prominence_mm"))
        if prominence is None or prominence <= thresholds["hip_lesser_prominence_low_mm"]:
            descriptions.append("Ротация бедра (избыточная)")
        elif prominence >= thresholds["hip_lesser_prominence_high_mm"]:
            descriptions.append("Ротация бедра (недостаточная)")

    if "hip_roi" in codes:
        top = number(detail.get("hip_top_margin_mm"))
        lateral = number(detail.get("hip_lateral_margin_mm"))
        bottom = number(detail.get("hip_bottom_margin_mm"))
        lesser_lower_bottom = number(detail.get("hip_lesser_lower_bottom_margin_mm"))
        if top is None or top < HIP_ROI_TOP_THRESHOLD_MM:
            descriptions.append("Недостаточный верхний захват")
        if (bottom is None or bottom < HIP_ROI_BOTTOM_THRESHOLD_MM or (lesser_lower_bottom is not None and lesser_lower_bottom < HIP_ROI_LESSER_LOWER_BOTTOM_THRESHOLD_MM)):
            descriptions.append("Недостаточный нижний захват")
        if lateral is None or lateral < HIP_ROI_LATERAL_THRESHOLD_MM:
            descriptions.append("Недостаточный латеральный захват")

    seen = set()
    unique = []
    for item in descriptions:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return "; ".join(unique)


def results(submission, work, thresholds=None):
    thresholds = thresholds or DEFAULT_THRESHOLDS
    debug = rows(work / "quality_predictions_debug.csv")
    skeleton = {r["member"]: r for r in rows(work / "inference_skeleton.csv")}
    landmarks = {r["member"]: r for r in rows(work / "inference_landmark_predictions_hybrid.csv")}
    output = []
    # Submission and debug preserve input order, including repeated image UIDs.
    final = rows(submission)
    if len(final) != len(debug):
        raise ValueError("Количество строк результата и debug не совпадает")
    for i, (row, detail) in enumerate(zip(final, debug)):
        if row["image_uid"] != detail["image_uid"]:
            raise ValueError("Порядок изображений в результате и debug не совпадает")
        member = detail["member"]
        point_row = landmarks.get(member, {})
        points = visible_landmark_points(point_row, detail)
        item = {**row, "id": str(i), "member": member,
                "filename": detail.get("filename", Path(member).name),
                "width": number(skeleton.get(member, {}).get("columns")),
                "height": number(skeleton.get(member, {}).get("rows")),
                "landmarks": points, "landmark_model": point_row.get("landmark_model", ""),
                "metrics": {k: number(detail.get(k)) for k in ["spine_axis_angle_deg", "hip_lesser_prominence_mm", "hip_top_margin_mm", "hip_bottom_margin_mm", "hip_lateral_margin_mm", "hip_lesser_lower_bottom_margin_mm"]}}
        apply_viewer_thresholds(item, detail, point_row, thresholds)
        item["violation_description"] = violation_description(item, detail, point_row, thresholds)
        output.append(item)
    return output


def refresh_job_results(job):
    folder = job["_folder"]
    job["images"] = results(folder / "submission.csv", folder / "work", threshold_settings(job))


def snapshot(job):
    with LOCK:
        public = {k: v for k, v in job.items() if not k.startswith("_")}
        public["thresholds"] = threshold_settings(job)
        return public


def archive_label(header):
    name = Path(unquote(header or "")).name
    if name.lower().endswith(".zip"):
        name = name[:-4]
    return name[:100] or "Новый пакет исследований"


def archive_contains_dicom(archive):
    from scripts.prepare_artifacts import is_dicom_member

    for info in archive.infolist():
        if info.is_dir() or info.file_size > 128 * 1024 * 1024:
            continue
        raw = archive.read(info)
        if is_dicom_member(info.filename, raw):
            return True
    return False


def save_request_body(stream, target, length):
    remaining = length
    with target.open("wb") as output:
        while remaining:
            chunk = stream.read(min(UPLOAD_CHUNK, remaining))
            if not chunk:
                raise ValueError("Upload ended before Content-Length")
            output.write(chunk)
            remaining -= len(chunk)


def run_job(job):
    with LOCK:
        job["status"] = "running"
    folder = job["_folder"]
    command = [os.environ.get("QC_PYTHON", sys.executable), str(ROOT / "scripts/run_full_inference.py"),
               "--input", str(job["_input"]), "--out-csv", str(folder / "submission.csv"), "--work-dir", str(folder / "work")]
    if os.environ.get("QC_DEVICE"):
        command += ["--device", os.environ["QC_DEVICE"]]
    try:
        with (folder / "inference.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=3600)
        with LOCK:
            refresh_job_results(job)
            job.update(status="completed")
    except Exception:
        with LOCK:
            job.update(status="failed", error="Анализ не завершён. Проверьте зависимости и журнал inference.log в папке задания.")


def image_bytes(job, item):
    # Keep DICOM/image dependencies out of API startup and lightweight tests.
    from scripts.prepare_artifacts import image_from_dicom

    with zipfile.ZipFile(job["_input"]) as z:
        info = z.getinfo(item["member"])
        if info.file_size > 64 * 1024 * 1024:
            raise ValueError("Изображение слишком большое для просмотра")
        image, _ = image_from_dicom(z.read(info))
    return image


def reconstruction_status():
    checkpoint = Path(os.environ.get("DXA3D_CHECKPOINT", ""))
    if not os.environ.get("DXA3D_CHECKPOINT") or not checkpoint.is_file():
        return False, "3D-модель ещё не загружена на сервер"
    return True, ""


def reconstruction_enabled():
    return reconstruction_status()[0]


def reconstruction_command(input_path, output_path):
    return [
        os.environ.get("DXA3D_PYTHON", sys.executable),
        "-m",
        "viewer.reconstruct",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]


def gradcam_bytes(job, item, task):
    if item.get("anatomical_region") != "spine":
        raise ValueError("Grad-CAM is available only for spine images")
    if task not in {"positioning", "artifact"}:
        raise ValueError("Unsupported Grad-CAM task")
    folder = STORE / job["id"] / "gradcam"
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / f"{item['id']}-{task}.png"
    if not cached.exists():
        from viewer.gradcam import gradcam_png_bytes

        try:
            cached.write_bytes(gradcam_png_bytes(image_bytes(job, item), task))
        except RuntimeError as exc:
            raise ValueError(str(exc)) from exc
    return cached.read_bytes()


def reconstruct(job, item):
    folder = STORE / job["id"] / ("3d-" + item["id"])
    folder.mkdir(parents=True, exist_ok=True)
    try:
        image_bytes(job, item).save(folder / "input.png")
        with (folder / "reconstruction.log").open("w") as log:
            subprocess.run(reconstruction_command(folder / "input.png", folder / "result.json"),
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        value = json.loads((folder / "result.json").read_text())
        with LOCK:
            job["reconstructions"][item["id"]] = {"status": "completed", **value}
    except Exception:
        with LOCK:
            job["reconstructions"][item["id"]] = {"status": "failed", "error": "Реконструкция не выполнена. Проверьте веса, зависимости и журнал reconstruction.log."}


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC), **kwargs)

    def log_message(self, *_):
        pass

    def send_data(self, value, status=200, content_type="application/json; charset=utf-8"):
        data = json.dumps(value, ensure_ascii=False, allow_nan=False).encode() if not isinstance(value, bytes) else value
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts[0] != "api":
            return super().do_GET()
        try:
            if parts == ["api", "health"]:
                enabled, reason = reconstruction_status()
                return self.send_data({"status": "ok", "reconstruction": enabled,
                                       "reconstruction_reason": reason})
            if parts == ["api", "jobs"]:
                return self.send_data([snapshot(j) for j in list(JOBS.values())])
            if len(parts) < 3 or parts[1] != "jobs" or parts[2] not in JOBS:
                return self.send_data({"error": "Задание не найдено"}, 404)
            job = JOBS[parts[2]]
            if len(parts) == 3:
                return self.send_data(snapshot(job))
            if parts[3:] == ["export.csv"]:
                out = io.StringIO(newline="")
                writer = csv.DictWriter(out, FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(job.get("images", []))
                return self.send_data(out.getvalue().encode("utf-8-sig"), content_type="text/csv; charset=utf-8")
            if parts[3:] == ["export.xlsx"]:
                from viewer.xlsx_export import workbook_bytes
                return self.send_data(
                    workbook_bytes(job.get("images", []), FIELDS),
                    content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            if len(parts) == 6 and parts[3] == "images" and parts[5] == "preview.png":
                item = next(r for r in job["images"] if r["id"] == parts[4])
                out = io.BytesIO()
                image_bytes(job, item).save(out, format="PNG")
                return self.send_data(out.getvalue(), content_type="image/png")
            if len(parts) == 7 and parts[3] == "images" and parts[5] == "gradcam":
                item = next(r for r in job["images"] if r["id"] == parts[4])
                task = parts[6].removesuffix(".png")
                return self.send_data(gradcam_bytes(job, item, task), content_type="image/png")
            return self.send_data({"error": "Маршрут не найден"}, 404)
        except (ValueError, KeyError, StopIteration, OSError, zipfile.BadZipFile):
            return self.send_data({"error": "Не удалось прочитать данные изображения"}, 422)

    def do_POST(self):
        # Local-only server; reject browser requests from foreign origins.
        if self.headers.get("Origin") and urlparse(self.headers["Origin"]).netloc != self.headers.get("Host"):
            return self.send_data({"error": "Недопустимый источник запроса"}, 403)
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts == ["api", "jobs"]:
            folder = None
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    return self.send_data({"error": "Архив пуст или не передан"}, 400)
                with LOCK:
                    if sum(j["status"] in {"queued", "running"} for j in JOBS.values()) >= 4:
                        return self.send_data({"error": "Очередь заполнена. Дождитесь завершения анализа."}, 429)
                key = uuid.uuid4().hex
                folder = STORE / key
                folder.mkdir(parents=True)
                source = folder / "input.zip"
                save_request_body(self.rfile, source, length)
                with zipfile.ZipFile(source) as z:
                    if len(z.infolist()) > 10000 or sum(i.file_size for i in z.infolist()) > 2 * 1024**3:
                        raise ValueError()
                    if not archive_contains_dicom(z):
                        raise ValueError()
                job = {"id": key, "name": archive_label(self.headers.get("X-Archive-Name")),
                       "created": time.time(), "status": "queued",
                       "archived": False, "images": [], "reconstructions": {}, "thresholds": dict(DEFAULT_THRESHOLDS),
                       "_input": source, "_folder": folder}
                with LOCK:
                    JOBS[key] = job
                POOL.submit(run_job, job)
                return self.send_data(snapshot(job), 202)
            except (ValueError, zipfile.BadZipFile):
                if folder is not None:
                    shutil.rmtree(folder, ignore_errors=True)
                return self.send_data({"error": "Нужен корректный ZIP с DICOM-файлами, до 10 000 файлов и 2 ГБ после распаковки"}, 400)
        if len(parts) == 6 and parts[:2] == ["api", "jobs"] and parts[3] == "images" and parts[5] == "reconstruct":
            if not reconstruction_enabled():
                return self.send_data({"error": reconstruction_status()[1]}, 503)
            job = JOBS.get(parts[2])
            item = next((r for r in (job or {}).get("images", []) if r["id"] == parts[4]), None)
            if not item or item["anatomical_region"] != "spine":
                return self.send_data({"error": "Реконструкция доступна только для позвоночника"}, 422)
            with LOCK:
                previous = job["reconstructions"].get(item["id"], {})
                if previous.get("status") not in {"queued", "completed"}:
                    job["reconstructions"][item["id"]] = {"status": "queued"}
                    POOL.submit(reconstruct, job, item)
            return self.send_data(snapshot(job), 202)
        if len(parts) == 4 and parts[:2] == ["api", "jobs"] and parts[3] == "thresholds":
            job = JOBS.get(parts[2])
            if not job:
                return self.send_data({"error": "Задание не найдено"}, 404)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8")) if length else {}
                thresholds = parse_threshold_payload(payload, job)
            except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
                return self.send_data({"error": str(exc) or "Некорректные пороги"}, 400)
            with LOCK:
                job["thresholds"] = thresholds
                if job.get("status") == "completed":
                    refresh_job_results(job)
            return self.send_data(snapshot(job))
        self.send_data({"error": "Маршрут не найден"}, 404)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    display_host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    print(f"Densitometry viewer: http://{display_host}:{args.port}", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
