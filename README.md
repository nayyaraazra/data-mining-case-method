# Case Method Data Mining: Deteksi Anomali Pergerakan Kapal (AIS)

Bi-LSTM autoencoder + Isolation Forest + hybrid, pada dataset
"Labelled evaluation datasets of AIS Trajectories from Danish Waters" (DTU).

## Isi folder
- `ais_anomaly_pipeline.py`  kode lengkap (preprocessing, fitur, model, evaluasi, laporan hasil)
- `requirements.txt`         library yang dibutuhkan
- `data/`                    tempat file .pkl dataset (tidak disertakan karena ukuran)
- `contoh_hasil/`            hasil dari run lengkap (5 ulangan x 5 fold) yang dipakai di laporan

## Cara menjalankan
1. `pip install -r requirements.txt`
2. Unzip dataset berlabel (21511815.zip) dan taruh file `.pkl` di folder `data/`.
3. `python ais_anomaly_pipeline.py`
   Hasil ditulis ke folder `output/` (dibuat otomatis).

Waktu: sekitar 1 menit per fold di CPU, jadi 5 ulangan x 5 fold sekitar 25 menit.
Tes cepat: `AIS_REPEATS=1 AIS_EPOCHS=2 python ais_anomaly_pipeline.py`
(di Windows PowerShell: `$env:AIS_REPEATS=1; $env:AIS_EPOCHS=2; python ais_anomaly_pipeline.py`)

## Pengaturan (bagian CONFIG di atas file)
| Nama | Fungsi |
| --- | --- |
| `MODE` | `cv_labelled` (default) atau `unlabelled_train` |
| `VARIANT` | varian dataset; default `600_43200_120` (resample 120 detik) |
| `WIN`, `STRIDE` | panjang window (30 titik = 60 menit) dan geseran |
| `N_REPEATS` (env `AIS_REPEATS`) | jumlah ulangan cross-validation |
| `EPOCHS` (env `AIS_EPOCHS`) | maksimum epoch Bi-LSTM |
| `TOP_FRAC` | proporsi trajectory teratas yang ditandai (default 10%) |
| `ALPHA` | bobot Bi-LSTM pada skor hybrid |

Proses disimpan per ulangan di `output/_checkpoint_rows.pkl`, jadi bila terhenti bisa
dijalankan ulang dan melanjutkan. Checkpoint diabaikan otomatis bila konfigurasi berubah.

## Output (di `output/`)
| Berkas | Isi |
| --- | --- |
| `01_karakteristik_dataset.json` | profil dataset |
| `02_ringkasan_preprocessing.json`, `02_ringkasan_trajectory.csv` | hasil preprocessing |
| `04_hasil_deteksi_anomali.csv`, `04_skor_semua_ulangan.csv` | skor dan daftar deteksi |
| `04_kurva_training_bilstm.png` | kurva training |
| `05_evaluasi_algoritma.csv`, `05_evaluasi_per_ulangan.csv`, `05_roc_pr.png` | evaluasi |
| `06_analisis_faktor.csv`, `06_jenis_anomali_terdeteksi.csv`, `06_distribusi_skor.png`, `06_peta_trajectory.png` | faktor dan jenis penyimpangan |
| `07_rekomendasi_kapal.csv` | peringkat kapal untuk diperiksa |

## Struktur dataset (hasil inspeksi)
- `data_*.pkl`: aliran objek pickle, tiap objek = satu trajectory (dict: mmsi, shiptype,
  track_length, lat, lon, speed, course, timestamp).
- `datasetInfo_*.pkl`: metadata; `indicies` = offset byte tiap trajectory,
  `outlierLabels` = label per trajectory (1 = abnormal), urutan sama dengan data.
- Split train/test bawaan tidak dipakai, karena semua abnormal ada di bagian train.
