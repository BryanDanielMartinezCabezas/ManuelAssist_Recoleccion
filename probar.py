"""
Manuel Assist — Prueba en tiempo real
Uso: python probar.py
Cierra app.py antes de correr esto.
"""
import socket, json, pickle, re, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import deque

# ── Cargar modelo ──────────────────────────────────────────────────
BASE = Path(__file__).parent

with open(BASE / "metadata.json") as f:
    meta = json.load(f)

with open(BASE / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

CLASES      = meta["clases"]
WINDOW_SIZE = meta["window_size"]
INPUT_SIZE  = meta["input_size"]
FEATURES    = meta["features"]
COLS_DROP   = ["medio_gx", "medio_gy", "medio_gz"]

CHANNEL_NAMES = {2: "pulgar", 3: "indice", 4: "medio", 5: "anular", 6: "menique"}

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, len(CLASES)),
        )
    def forward(self, x): return self.net(x)

model = MLP()
model.load_state_dict(torch.load(BASE / "modelo.pth", map_location="cpu"))
model.eval()

print(f"✓ Modelo cargado — clases: {CLASES}")
print(f"✓ Ventana: {WINDOW_SIZE} frames | {INPUT_SIZE} features\n")

# ── UDP ────────────────────────────────────────────────────────────
DISCOVERY_PORT = 4210
DATA_PORT      = 4211
CONF_THRESHOLD = 0.65

_LINE_RE = re.compile(
    r"CH=(\d+)\s+AX=([-\d.]+)\s+AY=([-\d.]+)\s+AZ=([-\d.]+)"
    r"\s+GX=([-\d.]+)\s+GY=([-\d.]+)\s+GZ=([-\d.]+)"
)

def parse(line):
    m = _LINE_RE.match(line.strip())
    if m:
        return int(m.group(1)), [float(m.group(i)) for i in range(2, 8)]
    return None, None

# Discovery
disc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
disc_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
disc_sock.bind(("", DISCOVERY_PORT))
disc_sock.settimeout(0.01)

# Data
data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
data_sock.bind(("", DATA_PORT))
data_sock.settimeout(0.01)

print("Esperando ESP32... (Ctrl+C para salir)\n")

# ── Buffers ────────────────────────────────────────────────────────
current_frame = {}
last_ch       = None
frame_buffer  = deque(maxlen=WINDOW_SIZE)  # últimos N frames completos

last_pred   = None
last_conf   = 0.0
stable_count = 0
EMOJIS = {"reposo": "🖐", "borrar": "☝", "espacio": "👌", "enter": "✌", "scroll_abajo": "✊"}

def frame_to_vector(frame):
    """Convierte un frame dict a vector de features (sin medio_gyr)."""
    row = {}
    for ch, name in sorted(CHANNEL_NAMES.items()):
        vals = frame.get(ch, [0.0]*6)
        for i, ax in enumerate(["ax","ay","az","gx","gy","gz"]):
            row[f"{name}_{ax}"] = vals[i]
    return [row[f] for f in FEATURES]

def predict(buffer):
    vec = []
    for frame in buffer:
        vec.extend(frame_to_vector(frame))
    X = np.array(vec, dtype=np.float32).reshape(1, -1)
    X = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.tensor(X))
        probs  = torch.softmax(logits, dim=1)[0]
        conf, idx = probs.max(0)
    return CLASES[idx.item()], conf.item()

# ── Loop principal ─────────────────────────────────────────────────
try:
    while True:
        # Discovery
        try:
            data, addr = disc_sock.recvfrom(1024)
            msg = data.decode(errors="ignore").strip()
            if msg.startswith("HELLO_ESP32"):
                disc_sock.sendto(b"HELLO_LAPTOP|name=pc_python", addr)
                print(f"✓ ESP32 conectado: {addr[0]}\n")
        except socket.timeout:
            pass

        # Datos
        try:
            data, _ = data_sock.recvfrom(4096)
            line = data.decode(errors="ignore").strip()

            if "NO_DETECTADO" in line:
                continue

            ch, vals = parse(line)
            if ch is None:
                continue

            if last_ch is not None and ch <= last_ch and current_frame:
                if set(CHANNEL_NAMES.keys()).issubset(current_frame.keys()):
                    frame_buffer.append(dict(current_frame))

                    if len(frame_buffer) == WINDOW_SIZE:
                        gesto, conf = predict(frame_buffer)

                        if conf >= CONF_THRESHOLD:
                            if gesto == last_pred:
                                stable_count += 1
                            else:
                                stable_count = 1
                                last_pred = gesto

                            if stable_count >= 2:
                                emoji = EMOJIS.get(gesto, "")
                                bar = "█" * int(conf * 20)
                                print(f"\r  {emoji} {gesto:<14} {conf*100:5.1f}%  [{bar:<20}]", end="", flush=True)
                        else:
                            if stable_count > 0:
                                print()
                            stable_count = 0
                            last_pred = None
                            print(f"\r  {'?':<15} {conf*100:5.1f}%  (baja confianza)       ", end="", flush=True)

                current_frame = {}

            current_frame[ch] = vals
            last_ch = ch

        except socket.timeout:
            pass

except KeyboardInterrupt:
    print("\n\nSaliendo...")
