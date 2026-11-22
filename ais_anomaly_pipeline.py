# %% [markdown]
# # Case Method Data Mining
# Deteksi Anomali Pola Pergerakan Kapal Berbasis AIS (Bi-LSTM + Isolation Forest)
#
# Versi ini disesuaikan dengan struktur ASLI dataset DTU "Labelled evaluation datasets":
#   - data_*.pkl        : aliran (stream) objek pickle, 1 objek = 1 trajectory (dict)
#                         kunci: mmsi, shiptype, track_length, lat, lon, speed, course, timestamp
#   - datasetInfo_*.pkl : dict metadata; `indicies` = offset byte tiap trajectory,
#                         `outlierLabels` = label per trajectory (1 = abnormal), urutannya sama
#
# Dua mode (atur MODE di CONFIG):
#   "cv_labelled"      : (sementara) cross-validation pada data berlabel saja.
#                        Training hanya pada trajectory NORMAL, tes pada fold yang dipisah per MMSI.
#   "unlabelled_train" : (final) latih pada sampel dataset unlabelled (6 GB), uji pada seluruh data berlabel.
#
# Kebutuhan: pip install numpy pandas scipy scikit-learn matplotlib torch

# %% CONFIG
import os, glob, json, pickle, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from scipy.stats import mannwhitneyu
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

MODE = "cv_labelled"                 # "cv_labelled" | "unlabelled_train"
DATA_DIR = "data"                    # folder berisi file .pkl dataset berlabel (hasil unzip)
VARIANT = "600_43200_120"            # min 600 s, max 43200 s, resample 120 s (521 trajectory, 25 abnormal)
UNLAB_DIR = "data_unlabelled"        # folder dataset unlabelled (hanya mode unlabelled_train)
UNLAB_MAX_TRAJ = 40000               # batas sampel (reservoir sampling) agar muat di RAM
OUT_DIR = "output"; os.makedirs(OUT_DIR, exist_ok=True)

WIN, STRIDE = 30, 5                  # window 30 titik x 120 s = 60 menit
HID, LATENT = 32, 16
EPOCHS, PATIENCE, BATCH, LR = int(os.environ.get("AIS_EPOCHS", 40)), 6, 128, 1e-3
ALPHA = 0.5                          # bobot Bi-LSTM pada skor hybrid
IF_TREES, IF_CONTAM = 300, 0.02
N_FOLDS = 5
N_REPEATS = int(os.environ.get("AIS_REPEATS", 5))   # pengulangan CV dengan pembagian fold berbeda (seed berbeda)
TOP_FRAC = 0.10                      # operator memeriksa 10% trajectory dengan skor tertinggi
BOOT = 1000                          # resample bootstrap untuk interval kepercayaan AUC
ROI = (54.5, 56.0, 13.0, 16.0)       # lat_min, lat_max, lon_min, lon_max (dari datasetInfo)
GRID, LOW_CNT = 0.02, 5              # grid kepadatan jalur (derajat) & ambang sel "jarang dilalui"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() or 1))

# kolom fitur titik: 0 lat, 1 lon, 2 sog, 3 sin(cog), 4 cos(cog), 5 turn_rate, 6 accel, 7 dist_km
SEQ_COLS = [0, 1, 2, 3, 4, 5, 6]

# %% UTILITAS
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def bearing_deg(lat1, lon1, lat2, lon2):
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360

def ang_diff(a, b):
    return (a - b + 180) % 360 - 180

def read_stream(path):
    """Baca file pickle yang berisi banyak objek berurutan (1 objek = 1 trajectory)."""
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

# %% 1. LOAD DATA BERLABEL
def load_labelled(data_dir, variant):
    info_f = glob.glob(os.path.join(data_dir, f"datasetInfo_*_{variant}.pkl"))[0]
    data_f = glob.glob(os.path.join(data_dir, f"data_*_{variant}.pkl"))[0]
    info = pickle.load(open(info_f, "rb"))
    raw = list(read_stream(data_f))
    labels = np.asarray(info["outlierLabels"]).astype(int)
    assert len(raw) == len(labels), "jumlah trajectory != jumlah label"
    return raw, labels, info

# %% 2. PREPROCESSING
def clean_traj(raw, label, tid, st):
    """Bersihkan satu trajectory. Mengembalikan dict atau None bila tidak layak."""
    arr = {k: np.asarray(raw[k], dtype=float) for k in ("lat", "lon", "speed", "course", "timestamp")}
    n = len(arr["lat"])
    if not all(len(v) == n for v in arr.values()) or n < 2:
        st["drop_format"] += 1; return None
    order = np.argsort(arr["timestamp"], kind="stable")
    arr = {k: v[order] for k, v in arr.items()}
    ok = np.r_[True, np.diff(arr["timestamp"]) > 0]                 # buang timestamp duplikat
    ok &= (np.isfinite(arr["lat"]) & np.isfinite(arr["lon"]) & np.isfinite(arr["speed"]) & np.isfinite(arr["timestamp"])
           & (np.abs(arr["lat"]) <= 90) & (np.abs(arr["lon"]) <= 180) & (arr["speed"] >= 0) & (arr["speed"] <= 50))
    st["points_removed"] += int((~ok).sum())
    arr = {k: v[ok] for k, v in arr.items()}
    if len(arr["lat"]) < 2:
        st["drop_invalid"] += 1; return None
    cog = np.where(np.isfinite(arr["course"]), arr["course"] % 360, np.nan)
    miss = np.isnan(cog)
    if miss.any():                                                    # isi COG kosong dari arah posisi
        b = bearing_deg(arr["lat"][:-1], arr["lon"][:-1], arr["lat"][1:], arr["lon"][1:])
        b = np.r_[b, b[-1]]
        cog[miss] = b[miss]
        st["cog_filled_points"] += int(miss.sum()); st["traj_with_cog_filled"] += 1
    t = dict(tid=tid, mmsi=int(raw.get("mmsi", -1)), shiptype=int(raw.get("shiptype", -1)), label=int(label),
             lat=arr["lat"], lon=arr["lon"], sog=arr["speed"], cog=cog, ts=arr["timestamp"])
    t["F"] = point_feats(t)
    return t

def point_feats(t):
    dt = np.diff(t["ts"], prepend=np.nan) / 60.0                      # menit
    dt[~(dt > 0)] = np.nan
    dist = np.r_[np.nan, haversine_km(t["lat"][:-1], t["lon"][:-1], t["lat"][1:], t["lon"][1:])]
    turn = np.r_[np.nan, np.abs(ang_diff(t["cog"][1:], t["cog"][:-1]))] / dt      # derajat/menit
    acc = np.r_[np.nan, np.diff(t["sog"])] / dt                                   # knot/menit
    F = np.c_[t["lat"], t["lon"], t["sog"], np.sin(np.radians(t["cog"])), np.cos(np.radians(t["cog"])),
              np.clip(turn, 0, 90), np.clip(acc, -5, 5), dist]
    return pd.DataFrame(F).bfill().ffill().fillna(0).values.astype("float32")

def preprocess_all(raw_list, labels=None):
    st = dict(n_input=len(raw_list), drop_format=0, drop_invalid=0, points_removed=0,
              cog_filled_points=0, traj_with_cog_filled=0)
    out = []
    for i, r in enumerate(raw_list):
        t = clean_traj(r, -1 if labels is None else labels[i], i, st)
        if t is not None:
            out.append(t)
    st["n_clean"] = len(out)
    return out, st

# %% 3. PROFIL DATASET (output 1) + RINGKASAN PREPROCESSING (output 2)
def dataset_profile(raw, labels, info):
    L = np.array([len(r["lat"]) for r in raw])
    ts_all = np.concatenate([np.asarray(r["timestamp"], float) for r in raw])
    sog_all = np.concatenate([np.asarray(r["speed"], float) for r in raw])
    nan_course = int(sum(np.isnan(np.asarray(r["course"], float)).sum() for r in raw))
    st_ = pd.Series([r["shiptype"] for r in raw]).value_counts().sort_index()
    return {
        "n_trajectory": len(raw), "n_abnormal": int(labels.sum()), "n_normal": int((labels == 0).sum()),
        "prevalence_abnormal": round(float(labels.mean()), 4),
        "n_unique_mmsi": len({r["mmsi"] for r in raw}),
        "points_total": int(L.sum()), "points_per_traj": {"min": int(L.min()), "median": float(np.median(L)), "max": int(L.max())},
        "resample_sec": info.get("resampleFrequency"), "ROI(lat_min,lat_max,lon_min,lon_max)": info.get("ROI"),
        "time_start_utc": str(pd.to_datetime(ts_all.min(), unit="s")), "time_end_utc": str(pd.to_datetime(ts_all.max(), unit="s")),
        "sog_knot": {"min": float(np.nanmin(sog_all)), "mean": float(np.nanmean(sog_all)), "max": float(np.nanmax(sog_all))},
        "missing_course_points": nan_course,
        "shiptype_counts": {int(k): int(v) for k, v in st_.items()},
        "abnormal_by_provided_split": {"note": "split bawaan dataset menaruh seluruh/hampir seluruh abnormal di bagian train; tidak dipakai untuk evaluasi"},
    }

# %% 4. KOMPONEN MODEL
def make_windows(F):
    n = len(F)
    if n < WIN:
        return np.zeros((0, WIN, F.shape[1]), dtype="float32"), np.array([], dtype=int)
    starts = np.arange(0, n - WIN + 1, STRIDE)
    return np.stack([F[s:s + WIN] for s in starts]), starts

class Density:
    """Peta kepadatan jalur normal (grid lat/lon) untuk mendeteksi route deviation."""
    def fit(self, lat, lon):
        self.la = np.arange(ROI[0], ROI[1] + GRID, GRID); self.lo = np.arange(ROI[2], ROI[3] + GRID, GRID)
        self.H, _, _ = np.histogram2d(lat, lon, bins=[self.la, self.lo]); return self
    def frac_low(self, lat2d, lon2d):
        i = np.clip(((lat2d - ROI[0]) / GRID).astype(int), 0, self.H.shape[0] - 1)
        j = np.clip(((lon2d - ROI[2]) / GRID).astype(int), 0, self.H.shape[1] - 1)
        return (self.H[i, j] < LOW_CNT).mean(axis=1)

def window_stats(Wn, dens):
    lat, lon, sog, turn, acc, dist = Wn[:, :, 0], Wn[:, :, 1], Wn[:, :, 2], Wn[:, :, 5], Wn[:, :, 6], Wn[:, :, 7]
    path = dist[:, 1:].sum(1)
    net = haversine_km(lat[:, 0], lon[:, 0], lat[:, -1], lon[:, -1])
    return pd.DataFrame({
        "sog_mean": sog.mean(1), "sog_std": sog.std(1), "sog_max": sog.max(1), "stop_frac": (sog < 0.5).mean(1),
        "turn_mean": turn.mean(1), "turn_max": turn.max(1), "acc_absmax": np.abs(acc).max(1),
        "path_km": path, "net_km": net, "sinuosity": path / (net + 0.05),
        "lat_std": lat.std(1), "lon_std": lon.std(1), "offroute_frac": dens.frac_low(lat, lon)})

class BiLSTMAE(nn.Module):
    """Bi-LSTM autoencoder: encoder 2 arah -> vektor laten -> decoder 2 arah -> rekonstruksi."""
    def __init__(self, n_feat, hid=HID, latent=LATENT):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hid, batch_first=True, bidirectional=True)
        self.to_z = nn.Linear(2 * hid, latent)
        self.from_z = nn.Linear(latent, hid)
        self.dec = nn.LSTM(hid, hid, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hid, n_feat)
    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_z(torch.cat([h[0], h[1]], dim=1))
        d = self.from_z(z).unsqueeze(1).repeat(1, x.size(1), 1)
        return self.out(self.dec(d)[0])

class Detector:
    def fit(self, train, val, verbose=True):
        W_tr = np.concatenate([make_windows(t["F"])[0] for t in train]); W_va = np.concatenate([make_windows(t["F"])[0] for t in val])
        pts = np.concatenate([t["F"] for t in train])
        self.dens = Density().fit(pts[:, 0], pts[:, 1])
        # --- Bi-LSTM AE
        self.sc = StandardScaler().fit(pts[:, SEQ_COLS])
        X_tr, X_va = self._scale(W_tr), self._scale(W_va)
        self.ae = BiLSTMAE(len(SEQ_COLS)).to(DEVICE)
        opt = torch.optim.Adam(self.ae.parameters(), lr=LR); lossf = nn.MSELoss()
        best, best_state, bad, self.hist = 1e9, None, 0, []
        for ep in range(EPOCHS):
            self.ae.train(); perm = torch.randperm(len(X_tr)); tl = 0.0
            for i in range(0, len(X_tr), BATCH):
                b = X_tr[perm[i:i + BATCH]].to(DEVICE)
                opt.zero_grad(); loss = lossf(self.ae(b), b); loss.backward(); opt.step(); tl += loss.item() * len(b)
            self.ae.eval()
            with torch.no_grad():
                vl = lossf(self.ae(X_va.to(DEVICE)), X_va.to(DEVICE)).item()
            self.hist.append((tl / len(X_tr), vl))
            if vl < best - 1e-5:
                best, bad = vl, 0; best_state = {k: v.clone() for k, v in self.ae.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE: break
        self.ae.load_state_dict(best_state)
        if verbose: print(f"   Bi-LSTM: {len(self.hist)} epoch, best val MSE={best:.4f}, windows train/val={len(X_tr)}/{len(X_va)}")
        # --- Isolation Forest (fitur agregat window)
        S_tr, S_va = window_stats(W_tr, self.dens), window_stats(W_va, self.dens)
        self.if_cols = list(S_tr.columns)
        self.if_sc = StandardScaler().fit(S_tr)
        self.iforest = IsolationForest(n_estimators=IF_TREES, contamination=IF_CONTAM, random_state=SEED, n_jobs=-1).fit(self.if_sc.transform(S_tr))
        # --- normalisasi skor memakai data normal VALIDASI (tidak ikut training)
        raw_va = {"bilstm": self._ae_err(X_va), "iforest": self._if_score(S_va)}
        self.ref = {k: (np.median(v), np.percentile(v, 95)) for k, v in raw_va.items()}
        # --- ambang jenis anomali dari distribusi data normal training
        self.thr = {"sin": max(np.percentile(S_tr.sinuosity, 95), 3.0), "turn": np.percentile(S_tr.turn_max, 99),
                    "acc": np.percentile(S_tr.acc_absmax, 99)}
        return self

    def _scale(self, W):
        X = (W[:, :, SEQ_COLS] - self.sc.mean_.astype("float32")) / self.sc.scale_.astype("float32")
        return torch.tensor(X, dtype=torch.float32)

    def _ae_err(self, X):
        self.ae.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(X), 512):
                b = X[i:i + 512].to(DEVICE); out.append(((self.ae(b) - b) ** 2).mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out)

    def _if_score(self, S):
        return -self.iforest.score_samples(self.if_sc.transform(S[self.if_cols]))

    def _norm(self, name, s):
        med, p95 = self.ref[name]
        return np.clip((s - med) / (p95 - med + 1e-9), 0, 10)

    def tag(self, S):
        names = []
        for r in S.itertuples():
            tg = []
            if r.sinuosity > self.thr["sin"] and r.sog_mean > 0.5: tg.append("loitering")
            if r.turn_max > self.thr["turn"] or r.acc_absmax > self.thr["acc"]: tg.append("kinematic")
            if r.offroute_frac > 0.5: tg.append("route_deviation")
            names.append("+".join(tg) if tg else "other")
        return names

    def score(self, t):
        W, starts = make_windows(t["F"])
        if len(W) == 0: return None
        S = window_stats(W, self.dens)
        nb = self._norm("bilstm", self._ae_err(self._scale(W))); ni = self._norm("iforest", self._if_score(S))
        hy = ALPHA * nb + (1 - ALPHA) * ni
        k = int(np.argmax(hy))                                       # window paling anomali
        row = dict(tid=t["tid"], mmsi=t["mmsi"], shiptype=t["shiptype"], label=t["label"], n_win=len(W),
                   bilstm=nb.max(), iforest=ni.max(), hybrid=hy.max(), top_win_start=int(starts[k]),
                   tag=self.tag(S.iloc[[k]])[0])
        row.update({f"f_{c}": float(S.iloc[k][c]) for c in S.columns})
        return row

# %% 5. DATA UNLABELLED (mode final; sampel via reservoir sampling)
def load_unlabelled(dir_, max_traj):
    files = sorted(glob.glob(os.path.join(dir_, "data_*.pkl")))
    assert files, f"tidak ada data_*.pkl di {dir_}"
    rng, res, seen = random.Random(SEED), [], 0
    for fpath in files:
        for r in read_stream(fpath):
            seen += 1
            if len(res) < max_traj: res.append(r)
            else:
                j = rng.randint(0, seen - 1)
                if j < max_traj: res[j] = r
    print(f"Unlabelled: {seen} trajectory dibaca, {len(res)} disampel")
    return res

def split_by_mmsi(trajs, frac_val=0.15, seed=SEED):
    g = np.array(sorted({t["mmsi"] for t in trajs})); np.random.default_rng(seed).shuffle(g)
    vg = set(g[:max(1, int(frac_val * len(g)))])
    return [t for t in trajs if t["mmsi"] not in vg], [t for t in trajs if t["mmsi"] in vg]

# %% 6. EVALUASI
def boot_auc_ci(y, s, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed); pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]; v = []
    for _ in range(n):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        v.append(roc_auc_score(y[idx], s[idx]))
    return np.percentile(v, [2.5, 97.5])

def evaluate(df, boot=True):
    y = df.label.values; k = max(1, int(round(TOP_FRAC * len(y)))); rows = []
    for m in ("bilstm", "iforest", "hybrid"):
        s = df[m].values; top = np.zeros(len(y), bool); top[np.argsort(-s)[:k]] = True
        tp = int((top & (y == 1)).sum()); prec, rec = tp / k, tp / y.sum()
        lo, hi = boot_auc_ci(y, s) if boot else (np.nan, np.nan)
        rows.append({"model": m, "ROC_AUC": roc_auc_score(y, s), "AUC_CI95_low": lo, "AUC_CI95_high": hi,
                     "PR_AUC": average_precision_score(y, s), "PR_AUC_acak(prevalensi)": y.mean(),
                     f"precision@top{int(TOP_FRAC*100)}%": prec, f"recall@top{int(TOP_FRAC*100)}%": rec,
                     f"F1@top{int(TOP_FRAC*100)}%": 0 if tp == 0 else 2 * prec * rec / (prec + rec), "n_diperiksa": k})
    return pd.DataFrame(rows).round(4)

# %% 7. PIPELINE UTAMA
def run():
    raw, labels, info = load_labelled(DATA_DIR, VARIANT)
    prof = dataset_profile(raw, labels, info)
    json.dump(prof, open(f"{OUT_DIR}/01_karakteristik_dataset.json", "w"), indent=2, default=str)
    print(json.dumps({k: prof[k] for k in ("n_trajectory", "n_abnormal", "n_unique_mmsi", "points_per_traj", "time_start_utc", "time_end_utc")}, indent=2))

    trajs, st = preprocess_all(raw, labels)
    kept = [t for t in trajs if len(t["F"]) >= WIN]
    st.update(n_short_dropped=len(trajs) - len(kept), n_final=len(kept),
              n_final_abnormal=int(sum(t["label"] for t in kept)), WIN=WIN)
    json.dump(st, open(f"{OUT_DIR}/02_ringkasan_preprocessing.json", "w"), indent=2)
    print("Preprocessing:", st)
    pd.DataFrame([dict(tid=t["tid"], mmsi=t["mmsi"], shiptype=t["shiptype"], label=t["label"], n_points=len(t["lat"]),
                       durasi_menit=(t["ts"][-1] - t["ts"][0]) / 60, sog_mean=t["sog"].mean(), sog_max=t["sog"].max())
                  for t in kept]).to_csv(f"{OUT_DIR}/02_ringkasan_trajectory.csv", index=False)

    y = np.array([t["label"] for t in kept]); groups = np.array([t["mmsi"] for t in kept]); rows, hist0 = [], None
    if MODE == "cv_labelled":
        ck = f"{OUT_DIR}/_checkpoint_rows.pkl"                       # agar bisa dilanjutkan bila proses terhenti
        sig = json.dumps([VARIANT, WIN, STRIDE, HID, LATENT, EPOCHS, ALPHA, SEED, N_FOLDS, IF_TREES, IF_CONTAM, len(kept)])
        if os.path.exists(ck):                                      # checkpoint hanya dipakai bila konfigurasi identik
            obj = pickle.load(open(ck, "rb"))
            if isinstance(obj, dict) and obj.get("sig") == sig: rows = obj["rows"]
            else: print("Checkpoint lama tidak cocok dengan konfigurasi saat ini, diabaikan.")
        done = {r["repeat"] for r in rows}; part = []
        for rep_i in range(N_REPEATS):
            if rep_i in done: continue
            skf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED + rep_i)
            for f, (tr_i, te_i) in enumerate(skf.split(np.zeros(len(y)), y, groups)):
                normal = [kept[i] for i in tr_i if y[i] == 0]
                tr, va = split_by_mmsi(normal, 0.15, SEED + 100 * rep_i + f)
                print(f"Ulangan {rep_i+1}/{N_REPEATS} - fold {f+1}/{N_FOLDS}: train normal={len(tr)}, val normal={len(va)}, "
                      f"test={len(te_i)} (abnormal={int(y[te_i].sum())})", flush=True)
                det = Detector().fit(tr, va)
                hist0 = det.hist if hist0 is None else hist0
                new_rows = [dict(r, repeat=rep_i) for r in (det.score(kept[i]) for i in te_i) if r]
                rows += new_rows; part += new_rows
            pickle.dump({"sig": sig, "rows": rows}, open(ck, "wb")); part = []
    else:
        unl, _ = preprocess_all(load_unlabelled(UNLAB_DIR, UNLAB_MAX_TRAJ))
        unl = [t for t in unl if len(t["F"]) >= WIN]
        tr, va = split_by_mmsi(unl, 0.1)
        det = Detector().fit(tr, va); hist0 = det.hist
        rows = [dict(r, repeat=0) for r in (det.score(t) for t in kept) if r]

    all_df = pd.DataFrame(rows)
    all_df.to_csv(f"{OUT_DIR}/04_skor_semua_ulangan.csv", index=False)
    k = max(1, int(round(TOP_FRAC * all_df.tid.nunique())))
    all_df["flag_rep"] = all_df.groupby("repeat").hybrid.rank(ascending=False, method="first") <= k
    per = []
    for r_, g in all_df.groupby("repeat"):
        e = evaluate(g.reset_index(drop=True), boot=False); e.insert(0, "repeat", r_); per.append(e)
    per = pd.concat(per); per.to_csv(f"{OUT_DIR}/05_evaluasi_per_ulangan.csv", index=False)

    # skor akhir per trajectory = rata-rata skor out-of-fold dari semua ulangan
    agg = {c: "mean" for c in all_df.columns if c in ("bilstm", "iforest", "hybrid") or c.startswith("f_")}
    agg.update(mmsi="first", shiptype="first", label="first", n_win="first", flag_rep="mean", tag=lambda s_: s_.mode().iat[0])
    df = all_df.groupby("tid").agg(agg).reset_index().rename(columns={"flag_rep": "flag_rate"})
    df = df.sort_values("hybrid", ascending=False).reset_index(drop=True)
    df["flagged"] = False; df.loc[:k - 1, "flagged"] = True
    df.to_csv(f"{OUT_DIR}/04_hasil_deteksi_anomali.csv", index=False)

    pooled = evaluate(df)                                              # skor rata-rata antar ulangan + CI bootstrap
    cols = [c for c in per.columns if c not in ("repeat", "model", "n_diperiksa", "PR_AUC_acak(prevalensi)", "AUC_CI95_low", "AUC_CI95_high")]
    ms = per.groupby("model")[cols].agg(["mean", "std"]); ms.columns = [f"{a}_{b}" for a, b in ms.columns]
    res = ms.reset_index().merge(pooled[["model", "ROC_AUC", "AUC_CI95_low", "AUC_CI95_high", "PR_AUC"]].rename(
        columns={"ROC_AUC": "ROC_AUC_gabungan", "PR_AUC": "PR_AUC_gabungan"}), on="model").round(4)
    res["PR_AUC_acak(prevalensi)"] = round(df.label.mean(), 4)
    print(f"Evaluasi: {N_REPEATS} ulangan x {N_FOLDS} fold"); print(res.set_index("model").T)
    res.to_csv(f"{OUT_DIR}/05_evaluasi_algoritma.csv", index=False)

    # --- analisis faktor penyimpangan (output 6)
    fac = []
    for c in [c for c in df.columns if c.startswith("f_")]:
        a, n_ = df.loc[df.label == 1, c], df.loc[df.label == 0, c]
        fac.append({"fitur": c[2:], "median_normal": n_.median(), "median_abnormal": a.median(),
                    "AUC_tunggal": roc_auc_score(df.label, df[c]), "p_MannWhitney": mannwhitneyu(a, n_).pvalue})
    fac = pd.DataFrame(fac); fac["kekuatan"] = (fac.AUC_tunggal - 0.5).abs()
    fac = fac.sort_values("kekuatan", ascending=False).drop(columns="kekuatan")
    fac = fac.round({"median_normal": 4, "median_abnormal": 4, "AUC_tunggal": 4})          # p-value tidak dibulatkan
    fac.to_csv(f"{OUT_DIR}/06_analisis_faktor.csv", index=False)
    pd.crosstab(df[df.flagged].tag, df[df.flagged].label).to_csv(f"{OUT_DIR}/06_jenis_anomali_terdeteksi.csv")
    print(fac.head(6))

    # --- rekomendasi kapal (output 7)
    rec = (df[df.flagged].groupby(["mmsi", "shiptype"])
           .agg(n_trajectory_ditandai=("tid", "size"), skor_hybrid_maks=("hybrid", "max"), konsistensi_ditandai=("flag_rate", "mean"),
                jenis_penyimpangan=("tag", lambda s: ",".join(sorted(set(s)))), label_sebenarnya_abnormal=("label", "max"))
           .sort_values("skor_hybrid_maks", ascending=False).reset_index())
    rec.to_csv(f"{OUT_DIR}/07_rekomendasi_kapal.csv", index=False)

    # --- grafik
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for m in ("bilstm", "iforest", "hybrid"):
        fpr, tpr, _ = roc_curve(df.label, df[m]); p, r, _ = precision_recall_curve(df.label, df[m])
        ax[0].plot(fpr, tpr, label=f"{m} (AUC={roc_auc_score(df.label, df[m]):.2f})"); ax[1].plot(r, p, label=m)
    ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set(xlabel="FPR", ylabel="TPR", title="ROC (level trajectory)"); ax[0].legend()
    ax[1].axhline(df.label.mean(), color="k", ls="--"); ax[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall"); ax[1].legend()
    plt.tight_layout(); plt.savefig(f"{OUT_DIR}/05_roc_pr.png", dpi=150); plt.close()

    plt.figure(figsize=(5, 4)); plt.boxplot([df[df.label == 0].hybrid, df[df.label == 1].hybrid])
    plt.xticks([1, 2], ["normal", "abnormal"]); plt.ylabel("skor hybrid (maks per trajectory)"); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/06_distribusi_skor.png", dpi=150); plt.close()

    by_tid = {t["tid"]: t for t in kept}; plt.figure(figsize=(7, 6))
    for t in list(by_tid.values())[::3]: plt.plot(t["lon"], t["lat"], color="lightgray", lw=0.5)
    for r in df[(df.label == 1)].itertuples(): plt.plot(by_tid[r.tid]["lon"], by_tid[r.tid]["lat"], color="red", lw=1)
    for r in df[(df.flagged) & (df.label == 0)].itertuples(): plt.plot(by_tid[r.tid]["lon"], by_tid[r.tid]["lat"], color="orange", lw=0.8)
    plt.plot([], [], color="red", label="abnormal (label)"); plt.plot([], [], color="orange", label="ditandai model, label normal")
    plt.legend(); plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("Peta trajectory"); plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/06_peta_trajectory.png", dpi=150); plt.close()

    h = np.array(hist0 if hist0 else [(np.nan, np.nan)]); plt.figure(figsize=(5, 3.5)); plt.plot(h[:, 0], label="train"); plt.plot(h[:, 1], label="validasi")
    plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.tight_layout(); plt.savefig(f"{OUT_DIR}/04_kurva_training_bilstm.png", dpi=150); plt.close()
    return df, res

# %% RUN
if __name__ == "__main__":
    df, res = run()
