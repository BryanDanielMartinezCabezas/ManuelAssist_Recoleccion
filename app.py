"""
Manuel Assist — Servidor de recolección de datos
Corre con: python app.py
Abre en el navegador: http://localhost:8080
"""
import asyncio
import socket
import json
import re
import csv
import math
import random
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn

# ── Configuración ───────────────────────────────────────────────────
DISCOVERY_PORT = 4210
DATA_PORT      = 4211
DATA_DIR       = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Canales I2C → dedo (según firmware de Atzel)
CHANNEL_NAMES = {2: "pulgar", 3: "indice", 4: "medio", 5: "anular", 6: "menique"}
REQUIRED_CHS  = set(CHANNEL_NAMES.keys())

CSV_HEADER = (
    ["timestamp"]
    + [f"{n}_{v}"
       for n in ["pulgar", "indice", "medio", "anular", "menique"]
       for v in ["ax", "ay", "az", "gx", "gy", "gz"]]
    + ["gesto"]
)

# ── Estado global ───────────────────────────────────────────────────
app     = FastAPI()
clients: set[WebSocket] = set()

class State:
    esp_ip: str         = None
    esp_connected: bool = False
    simulating: bool    = False
    recording: bool     = False
    session_gesture: str  = ""
    session_filename: str = ""
    rep_frame_count: int  = 0
    csv_file   = None
    csv_writer = None
    # Acumulación de trama
    current_frame: dict = {}
    last_channel: int   = None
    # FPS
    last_frame_ts: float = 0.0
    fps: float           = 0.0

S = State()

# ── Parseo de línea del ESP32 ───────────────────────────────────────
_LINE_RE = re.compile(
    r"CH=(\d+)\s+AX=([-\d.]+)\s+AY=([-\d.]+)\s+AZ=([-\d.]+)"
    r"\s+GX=([-\d.]+)\s+GY=([-\d.]+)\s+GZ=([-\d.]+)"
)

def parse_line(line: str):
    m = _LINE_RE.match(line.strip())
    if m:
        return int(m.group(1)), [float(m.group(i)) for i in range(2, 8)]
    return None, None

# ── Broadcast a todos los clientes WebSocket ────────────────────────
async def broadcast(msg: dict):
    if not clients:
        return
    text = json.dumps(msg)
    dead = set()
    for ws in clients:
        try:
            await ws.send_text(text)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)

# ── Procesar una trama completa (real o simulada) ───────────────────
async def process_frame(frame: dict):
    now = time.time()
    if S.last_frame_ts > 0:
        S.fps = 0.85 * S.fps + 0.15 * (1.0 / max(now - S.last_frame_ts, 0.001))
    S.last_frame_ts = now

    await broadcast({
        "type": "frame",
        "data": {str(c): frame[c] for c in sorted(CHANNEL_NAMES)},
        "fps": round(S.fps, 1),
    })

    if S.recording and S.csv_writer:
        row = [now]
        for c in sorted(CHANNEL_NAMES):
            row.extend(frame.get(c, [0.0] * 6))
        row.append(S.session_gesture)
        S.csv_writer.writerow(row)
        S.csv_file.flush()
        S.rep_frame_count += 1

# ── Tarea UDP: descubrimiento ───────────────────────────────────────
async def task_discovery():
    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind(("", DISCOVERY_PORT))
    sock.setblocking(False)
    print(f"[DISCOVERY] Escuchando UDP:{DISCOVERY_PORT}")

    while True:
        try:
            data, addr = await loop.sock_recvfrom(sock, 2048)
            msg = data.decode(errors="ignore").strip()
            if msg.startswith("HELLO_ESP32"):
                S.esp_ip = addr[0]
                S.esp_connected = True
                await loop.sock_sendto(sock, b"HELLO_LAPTOP|name=pc_python", addr)
                print(f"[DISCOVERY] ESP32 encontrado @ {S.esp_ip}")
                await broadcast({"type": "status", "connected": True, "esp_ip": S.esp_ip})
        except BlockingIOError:
            await asyncio.sleep(0.05)
        except Exception as e:
            print(f"[DISCOVERY] Error: {e}")
            await asyncio.sleep(0.1)

# ── Tarea UDP: recepción de datos ───────────────────────────────────
async def task_data():
    loop = asyncio.get_event_loop()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", DATA_PORT))
    sock.setblocking(False)
    print(f"[DATA] Escuchando UDP:{DATA_PORT}")

    while True:
        try:
            data, _ = await loop.sock_recvfrom(sock, 4096)
            line = data.decode(errors="ignore").strip()

            if "NO_DETECTADO" in line or S.simulating:
                await asyncio.sleep(0.001)
                continue

            ch, vals = parse_line(line)
            if ch is None:
                continue

            # Detectar inicio de nueva trama (el canal baja = vuelta al inicio)
            if S.last_channel is not None and ch <= S.last_channel and S.current_frame:
                if REQUIRED_CHS.issubset(S.current_frame.keys()):
                    await process_frame(dict(S.current_frame))
                S.current_frame = {}

            S.current_frame[ch] = vals
            S.last_channel = ch

        except BlockingIOError:
            await asyncio.sleep(0.001)
        except Exception as e:
            print(f"[DATA] Error: {e}")
            await asyncio.sleep(0.01)

# ── Tarea de simulación (sin hardware) ─────────────────────────────
async def task_simulate():
    t = 0
    print("[SIM] Modo simulación activo")
    while S.simulating:
        frame = {}
        for ch in CHANNEL_NAMES:
            offset = ch * 0.8
            frame[ch] = [
                round(math.sin(t * 0.08 + offset) * 0.6, 3),
                round(math.cos(t * 0.12 + offset) * 0.4, 3),
                round(0.95 + random.gauss(0, 0.02), 3),
                round(random.gauss(0, 3.0), 3),
                round(random.gauss(0, 3.0), 3),
                round(random.gauss(0, 1.5), 3),
            ]
        t += 1
        await process_frame(frame)
        await asyncio.sleep(0.033)   # ~30 fps
    print("[SIM] Simulación detenida")

# ── HTTP + WebSocket ────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    asyncio.create_task(task_discovery())
    asyncio.create_task(task_data())

@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

@app.websocket("/ws")
async def ws_handler(ws: WebSocket):
    await ws.accept()
    clients.add(ws)

    # Estado actual al conectarse
    await ws.send_text(json.dumps({
        "type": "status",
        "connected": S.esp_connected,
        "esp_ip": S.esp_ip,
        "simulating": S.simulating,
    }))

    try:
        while True:
            raw    = await ws.receive_text()
            msg    = json.loads(raw)
            action = msg.get("action")

            # ── Abrir sesión (crea archivo CSV) ──────────────────
            if action == "session_start":
                gesture = msg["gesture"]
                sid     = datetime.now().strftime("%Y%m%d_%H%M%S")
                fpath   = DATA_DIR / f"{gesture}_{sid}.csv"
                S.csv_file   = open(fpath, "w", newline="", encoding="utf-8")
                S.csv_writer = csv.writer(S.csv_file)
                S.csv_writer.writerow(CSV_HEADER)
                S.csv_file.flush()
                S.session_gesture  = gesture
                S.session_filename = fpath.name
                S.rep_frame_count  = 0
                await ws.send_text(json.dumps({
                    "type": "session_opened",
                    "gesture": gesture,
                    "filename": fpath.name,
                    "filepath": str(fpath.resolve()),
                }))

            # ── Iniciar grabación de una repetición ──────────────
            elif action == "record_start":
                S.rep_frame_count = 0
                S.recording = True
                await ws.send_text(json.dumps({
                    "type": "rep_started",
                    "rep": msg.get("rep"),
                }))

            # ── Detener grabación de repetición ──────────────────
            elif action == "record_stop":
                S.recording = False
                await ws.send_text(json.dumps({
                    "type": "rep_done",
                    "frames": S.rep_frame_count,
                    "rep": msg.get("rep"),
                }))

            # ── Cerrar sesión ─────────────────────────────────────
            elif action == "session_end":
                S.recording = False
                if S.csv_file:
                    S.csv_file.close()
                    S.csv_file   = None
                    S.csv_writer = None
                await ws.send_text(json.dumps({
                    "type": "session_ended",
                    "filename": S.session_filename,
                }))

            # ── Modo simulación ───────────────────────────────────
            elif action == "sim_start":
                if not S.simulating:
                    S.simulating = True
                    asyncio.create_task(task_simulate())
                    await broadcast({"type": "sim_status", "active": True})

            elif action == "sim_stop":
                S.simulating = False
                await broadcast({"type": "sim_status", "active": False})

    except WebSocketDisconnect:
        clients.discard(ws)
        S.recording = False
        if S.csv_file:
            S.csv_file.close()
            S.csv_file   = None
            S.csv_writer = None


if __name__ == "__main__":
    print("=" * 50)
    print("  Manuel Assist — Recolección de Datos")
    print("  Abre: http://localhost:8080")
    print("=" * 50)
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=False)
