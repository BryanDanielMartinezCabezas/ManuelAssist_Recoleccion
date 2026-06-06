"""
Manuel Assist — Entrenamiento rápido
Uso: python entrenar.py
Salida: modelo.pth, scaler.pkl, metadata.json
"""
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Configuración ──────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
OUT_DIR     = Path(__file__).parent
WINDOW_SIZE = 1
AUGMENT_X   = 8        # muestras sintéticas por muestra real
EPOCHS      = 60
BATCH_SIZE  = 32
LR          = 1e-3

# Columnas que siempre dan 0 (giroscopio del dedo medio dañado)
COLS_DROP = ["medio_gx", "medio_gy", "medio_gz"]

# ── Cargar CSVs ────────────────────────────────────────────────────
csv_files = {
    "reposo":  [DATA_DIR / "reposo_20260606_160006.csv"],
    "borrar":  [DATA_DIR / "borrar_20260606_161451.csv"],
    "espacio": [DATA_DIR / "espacio_20260606_162855.csv",
                DATA_DIR / "espacio_20260606_163803.csv"],
}

print("Cargando datos...")
frames_list = []
for gesto, files in csv_files.items():
    dfs = [pd.read_csv(f) for f in files if f.exists()]
    if not dfs:
        print(f"  ⚠ No se encontró CSV para {gesto}")
        continue
    df = pd.concat(dfs, ignore_index=True)
    print(f"  {gesto}: {len(df)} frames")
    frames_list.append(df)

df_all = pd.concat(frames_list, ignore_index=True)
df_all = df_all.drop(columns=COLS_DROP)

FEATURE_COLS = [c for c in df_all.columns if c not in ("timestamp", "gesto")]
CLASES       = sorted(df_all["gesto"].unique())
label_map    = {g: i for i, g in enumerate(CLASES)}
print(f"\nClases: {CLASES}")
print(f"Features: {len(FEATURE_COLS)} × ventana {WINDOW_SIZE} = {len(FEATURE_COLS)*WINDOW_SIZE} inputs")

# ── Crear ventanas deslizantes (sin cruzar gaps > 1.5s) ───────────
def make_windows(df, window=WINDOW_SIZE):
    X, y = [], []
    ts  = df["timestamp"].values
    feat = df[FEATURE_COLS].values
    labs = df["gesto"].values

    for i in range(len(df) - window + 1):
        # No cruzar descansos entre reps
        if (ts[i + window - 1] - ts[i]) > 1.5 * window:
            continue
        # Todas las filas de la ventana deben ser el mismo gesto
        if len(set(labs[i:i+window])) > 1:
            continue
        X.append(feat[i:i+window].flatten())
        y.append(label_map[labs[i]])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

print("\nCreando ventanas...")
X, y = make_windows(df_all)
print(f"  Ventanas totales: {len(X)}")
for cls, idx in label_map.items():
    print(f"  {cls}: {(y==idx).sum()}")

# ── Data augmentation ──────────────────────────────────────────────
print(f"\nData augmentation ×{AUGMENT_X}...")
noise_std = X.std(axis=0) * 0.05
X_aug = [X]
y_aug = [y]
for _ in range(AUGMENT_X):
    noise = np.random.randn(*X.shape).astype(np.float32) * noise_std
    X_aug.append(X + noise)
    y_aug.append(y)

X = np.vstack(X_aug)
y = np.concatenate(y_aug)
print(f"  Total muestras: {len(X)}")

# ── Normalizar ─────────────────────────────────────────────────────
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# ── Split ──────────────────────────────────────────────────────────
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

def to_loader(Xd, yd, shuffle=True):
    ds = TensorDataset(torch.tensor(Xd), torch.tensor(yd))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=shuffle)

tr_loader  = to_loader(X_tr, y_tr)
val_loader = to_loader(X_val, y_val, shuffle=False)

# ── Modelo MLP ─────────────────────────────────────────────────────
INPUT_SIZE = len(FEATURE_COLS) * WINDOW_SIZE
N_CLASES   = len(CLASES)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, 128),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, N_CLASES),
        )
    def forward(self, x):
        return self.net(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = MLP().to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit   = nn.CrossEntropyLoss()

# ── Entrenamiento ──────────────────────────────────────────────────
print(f"\nEntrenando en {device}...")
best_val, best_state = 0.0, None

for epoch in range(1, EPOCHS + 1):
    model.train()
    for xb, yb in tr_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        crit(model(xb), yb).backward()
        opt.step()
    sched.step()

    if epoch % 10 == 0 or epoch == EPOCHS:
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb.to(device)).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total   += len(yb)
        acc = correct / total
        if acc > best_val:
            best_val  = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"  Epoch {epoch:3d} — val_acc: {acc:.3f}  (mejor: {best_val:.3f})")

# ── Evaluación final ───────────────────────────────────────────────
model.load_state_dict(best_state)
model.eval()
all_pred, all_true = [], []
te_loader = to_loader(X_te, y_te, shuffle=False)
with torch.no_grad():
    for xb, yb in te_loader:
        all_pred.extend(model(xb.to(device)).argmax(1).cpu().tolist())
        all_true.extend(yb.tolist())

print("\n── Reporte de clasificación ──────────────────────────")
print(classification_report(all_true, all_pred, target_names=CLASES))

# ── Guardar ────────────────────────────────────────────────────────
torch.save(best_state, OUT_DIR / "modelo.pth")
with open(OUT_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump({
        "clases": CLASES,
        "window_size": WINDOW_SIZE,
        "features": FEATURE_COLS,
        "input_size": INPUT_SIZE,
        "n_clases": N_CLASES,
    }, f, indent=2)

print("\n✓ Guardado: modelo.pth  scaler.pkl  metadata.json")
print(f"✓ Mejor val_acc: {best_val:.3f}")
