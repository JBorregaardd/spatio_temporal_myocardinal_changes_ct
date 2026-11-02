"""Local web tool for labelling the aortic and mitral valve images from ``valve_planes.py`` as open or closed.

Serves the per-view PNGs in the output folder and writes every label straight to a CSV file
(``patient_id,aortic,mitral,updated``), so labelling can be stopped and resumed at any time.

Example:
    uv run side_project/valve_planes/label_valves.py side_project/valve_planes/output
    # then open http://localhost:8765 in a browser
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
from datetime import datetime
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer

import typer

VALVE_NAMES = ("aortic", "mitral")
LABEL_VALUES = ("open", "closed", "unclear")
CSV_COLUMNS = ("patient_id", *VALVE_NAMES, "updated")
IMAGE_NAME = re.compile(r"[\w.-]+\.png")


def find_patients(image_folder: str) -> list[str]:
    """Patient ids with a ``<id>_valves.json`` file in ``image_folder``, numeric ids sorted numerically."""
    ids = [os.path.basename(path).removesuffix("_valves.json") for path in glob.glob(f"{image_folder}/*_valves.json")]
    return sorted(
        ids,
        key=lambda patient_id: (not patient_id.isdigit(), int(patient_id) if patient_id.isdigit() else 0, patient_id),
    )


def load_labels(path: str) -> dict[str, dict[str, str]]:
    """Read the labels CSV into ``{patient_id: {column: value}}``; empty if the file does not exist."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline="") as file:
        return {
            row["patient_id"]: {column: row.get(column, "") for column in CSV_COLUMNS[1:]}
            for row in csv.DictReader(file)
        }


def save_labels(path: str, labels: dict[str, dict[str, str]]) -> None:
    """Write the labels CSV atomically, so an interrupted write never leaves a truncated file."""
    temporary = f"{path}.tmp"
    with open(temporary, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for patient_id, row in labels.items():
            writer.writerow({"patient_id": patient_id, **row})
    os.replace(temporary, path)


def set_label(labels: dict[str, dict[str, str]], patient_id: str, valve: str, value: str) -> None:
    """Set (or with an empty ``value`` clear) one valve label of a patient.

    Raises:
        ValueError: If ``valve`` or ``value`` is not a known valve or label.
    """
    if valve not in VALVE_NAMES or value not in (*LABEL_VALUES, ""):
        raise ValueError(f"Invalid label {valve}={value}")
    row = labels.setdefault(patient_id, {column: "" for column in CSV_COLUMNS[1:]})
    row[valve] = value
    row["updated"] = datetime.now().isoformat(timespec="seconds")


class LabelHandler(BaseHTTPRequestHandler):
    """Serves the labelling page, the valve images and a small JSON API for reading and writing labels."""

    def __init__(self, *args, image_folder: str, labels_path: str, labels: dict[str, dict[str, str]], **kwargs) -> None:
        self.image_folder = image_folder
        self.labels_path = labels_path
        self.labels = labels
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args) -> None:
        """Silence the per-request log."""

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        """Serve ``/``, ``/api/state`` and ``/img/<name>.png``."""
        if self.path == "/":
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
        elif self.path == "/api/state":
            state = {"patients": find_patients(self.image_folder), "labels": self.labels, "values": LABEL_VALUES}
            self._send(200, json.dumps(state).encode(), "application/json")
        elif self.path.startswith("/img/") and IMAGE_NAME.fullmatch(name := self.path[5:]):
            path = os.path.join(self.image_folder, name)
            if os.path.isfile(path):
                with open(path, "rb") as file:
                    self._send(200, file.read(), "image/png")
            else:
                self._send(404, b"not found", "text/plain")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        """Handle ``/api/label`` with a JSON body ``{"patient_id", "valve", "value"}`` and save the CSV."""
        if self.path != "/api/label":
            self._send(404, b"not found", "text/plain")
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            set_label(self.labels, str(body["patient_id"]), body["valve"], body["value"])
        except (KeyError, ValueError) as error:
            self._send(400, str(error).encode(), "text/plain")
            return
        save_labels(self.labels_path, self.labels)
        self._send(200, json.dumps(self.labels[str(body["patient_id"])]).encode(), "application/json")


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Valve labelling</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; font: 14px system-ui, sans-serif; background: #16181c; color: #e6e6e6; }
  header { display: flex; gap: 16px; align-items: center; padding: 10px 16px; background: #1f2228; flex-wrap: wrap; }
  header b { font-size: 18px; }
  header input { width: 70px; background: #2a2e36; color: inherit; border: 1px solid #444; padding: 3px 6px; }
  .bar { flex: 1; min-width: 120px; height: 6px; background: #333; border-radius: 3px; overflow: hidden; }
  .bar div { height: 100%; background: #4c9be8; }
  main { padding: 12px 16px; }
  .valve { margin-bottom: 18px; }
  .valve h2 { margin: 0 0 6px; font-size: 16px; display: flex; gap: 12px; align-items: center; }
  .views { display: flex; gap: 8px; flex-wrap: wrap; }
  figure { margin: 0; }
  figure img { width: var(--size, 300px); height: var(--size, 300px); image-rendering: auto; display: block; background: #000; }
  figcaption { font-size: 12px; color: #999; text-align: center; }
  button { background: #2a2e36; color: inherit; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; cursor: pointer; }
  button.on.open { background: #2e7d32; border-color: #4caf50; }
  button.on.closed { background: #b23b3b; border-color: #e57373; }
  button.on.unclear { background: #8a6d1a; border-color: #d4a72c; }
  .keys { color: #888; font-size: 12px; }
</style>
</head>
<body>
<header>
  <b id="title">-</b>
  <span id="progress"></span>
  <div class="bar"><div id="bar"></div></div>
  <label>Go to <input id="goto" placeholder="id"></label>
  <label>Size <input id="size" type="range" min="150" max="500" value="300"></label>
  <span class="keys">Aortic: Q open · W closed · E unclear &nbsp; Mitral: A open · S closed · D unclear &nbsp;
    ←/→ prev/next · N next unlabelled</span>
</header>
<main id="main"></main>
<script>
const VIEWS = [["en_face", "en face (looking along the flow)"],
               ["long_axis_a", "long axis A (through both valves, flow ↑)"],
               ["long_axis_b", "long axis B (⊥ A, flow ↑)"]];
const KEYS = {q: ["aortic", "open"], w: ["aortic", "closed"], e: ["aortic", "unclear"],
              a: ["mitral", "open"], s: ["mitral", "closed"], d: ["mitral", "unclear"]};
let patients = [], labels = {}, values = [], index = 0;

function done(id) { const r = labels[id]; return r && r.aortic && r.mitral; }

function render() {
  const id = patients[index];
  const n = patients.filter(done).length;
  document.getElementById("title").textContent = `Patient ${id}  (${index + 1}/${patients.length})`;
  document.getElementById("progress").textContent = `${n} of ${patients.length} fully labelled`;
  document.getElementById("bar").style.width = `${100 * n / patients.length}%`;
  const main = document.getElementById("main");
  main.innerHTML = "";
  for (const valve of ["aortic", "mitral"]) {
    const section = document.createElement("section");
    section.className = "valve";
    const heading = document.createElement("h2");
    heading.textContent = valve === "aortic" ? "Aortic valve" : "Mitral valve";
    for (const value of values) {
      const button = document.createElement("button");
      button.textContent = value;
      button.className = value + ((labels[id] || {})[valve] === value ? " on" : "");
      button.onclick = () => setLabel(valve, (labels[id] || {})[valve] === value ? "" : value);
      heading.appendChild(button);
    }
    section.appendChild(heading);
    const views = document.createElement("div");
    views.className = "views";
    for (const [view, caption] of VIEWS) {
      const figure = document.createElement("figure");
      figure.innerHTML = `<img src="img/${id}_${valve}_${view}.png" alt="${view}"><figcaption>${caption}</figcaption>`;
      views.appendChild(figure);
    }
    section.appendChild(views);
    main.appendChild(section);
  }
}

async function setLabel(valve, value) {
  const id = patients[index];
  const response = await fetch("api/label",{method: "POST", headers: {"Content-Type": "application/json"},
                                               body: JSON.stringify({patient_id: id, valve, value})});
  if (!response.ok) { alert(await response.text()); return; }
  labels[id] = await response.json();
  if (done(id) && value) { setTimeout(() => go(index + 1), 150); } else { render(); }
}

function go(i) { index = Math.max(0, Math.min(patients.length - 1, i)); render(); }
function nextUnlabelled() {
  for (let k = 1; k <= patients.length; k++) {
    const i = (index + k) % patients.length;
    if (!done(patients[i])) { go(i); return; }
  }
}

document.addEventListener("keydown", event => {
  if (event.target.tagName === "INPUT") return;
  const key = event.key.toLowerCase();
  if (KEYS[key]) setLabel(...KEYS[key]);
  else if (event.key === "ArrowRight") go(index + 1);
  else if (event.key === "ArrowLeft") go(index - 1);
  else if (key === "n") nextUnlabelled();
});
document.getElementById("goto").addEventListener("change", event => {
  const i = patients.indexOf(event.target.value.trim());
  if (i >= 0) go(i);
  event.target.value = "";
  event.target.blur();
});
document.getElementById("size").addEventListener("input", event => {
  document.documentElement.style.setProperty("--size", `${event.target.value}px`);
});

fetch("api/state").then(r => r.json()).then(state => {
  ({patients, labels, values} = state);
  index = Math.max(0, patients.findIndex(id => !done(id)));
  render();
});
</script>
</body>
</html>
"""


def main(
    image_folder: str = typer.Argument(..., help="Output folder of valve_planes.py with the per-view PNGs."),
    labels_path: str = typer.Option("", help="Labels CSV; defaults to <image_folder>/labels.csv."),
    port: int = typer.Option(8765, help="Port to serve on (localhost only)."),
) -> None:
    """Serve the labelling tool on http://localhost:<port> until interrupted with Ctrl+C."""
    image_folder = os.path.abspath(image_folder)
    if not os.path.isdir(image_folder):
        raise SystemExit(f"Folder not found: {image_folder}")
    if not find_patients(image_folder):
        n_png = len(glob.glob(os.path.join(image_folder, "*.png")))
        raise SystemExit(
            f"No <id>_valves.json files in {image_folder} ({n_png} PNG files); run valve_planes.py on this folder first"
        )
    labels_path = labels_path or os.path.join(image_folder, "labels.csv")
    labels = load_labels(labels_path)
    handler = partial(LabelHandler, image_folder=image_folder, labels_path=labels_path, labels=labels)
    server = HTTPServer(("127.0.0.1", port), handler)
    print(f"Labelling {len(find_patients(image_folder))} patients; labels are saved to {labels_path}")
    print(f"Open http://localhost:{port} (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    typer.run(main)
