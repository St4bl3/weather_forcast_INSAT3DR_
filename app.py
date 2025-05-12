import os
import re
import h5py
import numpy as np
from datetime import datetime, timedelta

# Force non-interactive backend so plt.savefig() won’t open a window
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
from flask import Flask, request, render_template, flash, redirect, url_for
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ── Paths & Directories ───────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR  = os.path.join(BASE_DIR, "static")
UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
CKPT_PATH   = os.path.join(BASE_DIR, "checkpoints", "cp-20.weights.h5")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────────
CHANNELS    = ['VIS','MIR','SWIR','WV','TIR1']
TARGET_SIZE = (256, 256)

MEAN = [9.249125480651855, 0.03255389258265495,
        0.35804545879364014, 0.14131774008274078,
        271.2187805175781]
STD  = [7.756808757781982, 0.01960049383342266,
        0.6404346227645874, 0.04700399935245514,
        20.69615936279297]

INDIA_EXTENT = [68.0, 98.0, 6.0, 37.0]

CHANNEL_DESCRIPTIONS = {
    'VIS':  "Visible Reflectance (0.6 μm): Measures reflected sunlight like a daytime photo—used for cloud cover, vegetation health, and land use.",
    'MIR':  "Mid-IR Radiance (3.9 μm): Captures emitted heat and some reflection—ideal for spotting wildfires, volcanic activity, and fog at night.",
    'SWIR': "Shortwave IR (1.6 μm): Sensitive to moisture—used for mapping snow, soil moisture, and distinguishing clouds from ice.",
    'WV':   "Water Vapor (6.3 μm): Tracks upper-tropospheric humidity—essential for understanding storm development and jet streams.",
    'TIR1': "Thermal IR Temp (10.8 μm): Measures surface and cloud-top temperature—used for heat island detection and convection analysis."
}

# ── Flask Setup ───────────────────────────────────────────────────────────────
app = Flask(
    __name__,
    static_folder=STATIC_DIR,
    template_folder=os.path.join(BASE_DIR, "templates")
)
app.secret_key = "replace_with_secure_key"

device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using TensorFlow device: {device}")

# ── Build & Load Model ─────────────────────────────────────────────────────────
def build_model():
    inp = tf.keras.Input(shape=(*TARGET_SIZE, len(CHANNELS)))
    x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv2D(128,3,padding='same',activation='relu')(x)
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
    x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(x)
    out = tf.keras.layers.Conv2D(len(CHANNELS),1,
                                 padding='same',
                                 activation='linear',
                                 dtype='float32')(x)
    model = tf.keras.Model(inp, out)
    model.load_weights(CKPT_PATH)
    return model

with tf.device(device):
    model = build_model()
    model.trainable = False

# ── Preprocessing ─────────────────────────────────────────────────────────────
def load_and_preprocess(fp):
    with h5py.File(fp,'r') as f:
        dn  = {ch: f[f"IMG_{ch}"][0].astype(int) for ch in CHANNELS}
        lut = {
            'VIS':  f['IMG_VIS_ALBEDO'][:],
            'MIR':  f['IMG_MIR_RADIANCE'][:],
            'SWIR': f['IMG_SWIR_RADIANCE'][:],
            'WV':   f['IMG_WV_RADIANCE'][:],
            'TIR1': f['IMG_TIR1_TEMP'][:],
        }
    bands = [lut[ch][dn[ch]].astype(np.float32) for ch in CHANNELS]
    arr   = np.stack(bands, axis=-1)  # H×W×C

    arr_tf = tf.convert_to_tensor(arr)[None]
    arr_rs = tf.image.resize(arr_tf, TARGET_SIZE, method='bilinear')[0].numpy()
    mean   = np.array(MEAN, dtype=np.float32)
    std    = np.array(STD,  dtype=np.float32)
    arr_n  = (arr_rs - mean)/std
    return arr_n[None]

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('upload'))

@app.route('/upload', methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files.get('file')
        if not f or not f.filename.lower().endswith('.h5'):
            flash("Please upload a valid .h5 file.")
            return redirect(request.url)

        # parse date/time from filename
        # expected format: 3RIMG_10MAR2025_0215_L1C_....h5
        m = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", f.filename)
        if m:
            date_str, time_str = m.groups()
            dt_in = datetime.strptime(date_str + time_str, "%d%b%Y%H%M")
        else:
            dt_in = None

        # compute next-day
        dt_out = dt_in + timedelta(days=1) if dt_in else None

        # save upload
        in_fp = os.path.join(UPLOAD_DIR, "input.h5")
        f.save(in_fp)

        # preprocess & predict
        x_in   = load_and_preprocess(in_fp)
        with tf.device(device):
            y_pred = model.predict(x_in)[0]

        mean = np.array(MEAN).reshape(1,1,-1)
        std  = np.array(STD).reshape(1,1,-1)
        y_phys = y_pred*std + mean

        # plot & save
        for i,ch in enumerate(CHANNELS):
            out_fp = os.path.join(STATIC_DIR, f"pred_{ch}.png")
            fig = plt.figure(figsize=(12,8))
            ax  = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent(INDIA_EXTENT, ccrs.PlateCarree())
            ax.coastlines('110m',linewidth=0.5)
            ax.add_feature(cfeature.BORDERS.with_scale('110m'),linewidth=0.5)

            im = ax.imshow(
                y_phys[...,i],
                origin='upper',
                transform=ccrs.PlateCarree(),
                cmap='Greys',
                extent=INDIA_EXTENT
            )
            cbar = plt.colorbar(im,ax=ax,pad=0.02)
            cbar.set_label(ch, rotation=90)

            ax.set_title(f"{ch} Next-Day Forecast", fontsize=20, fontweight='bold')
            ax.text(0.5, -0.15, CHANNEL_DESCRIPTIONS[ch],
                    transform=ax.transAxes,
                    fontsize=14, ha='center')
            fig.savefig(out_fp, dpi=150, bbox_inches='tight')
            plt.close(fig)

        return render_template(
            'results.html',
            channels=CHANNELS,
            descriptions=CHANNEL_DESCRIPTIONS,
            input_datetime=dt_in,
            output_datetime=dt_out
        )

    return render_template('upload.html', channel_info=CHANNEL_DESCRIPTIONS)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)



# import os
# import re
# import h5py
# import numpy as np
# from datetime import datetime, timedelta

# # force non-interactive backend
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

# import tensorflow as tf
# from flask import Flask, request, render_template, flash, redirect, url_for
# import cartopy.crs as ccrs
# import cartopy.feature as cfeature

# # ── Paths & Directories ───────────────────────────────────────────────────────
# BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
# STATIC_DIR  = os.path.join(BASE_DIR, "static")
# UPLOAD_DIR  = os.path.join(BASE_DIR, "uploads")
# CKPT_PATH   = os.path.join(BASE_DIR, "checkpoints", "cp-20.weights.h5")

# os.makedirs(STATIC_DIR, exist_ok=True)
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# # ── Configuration ─────────────────────────────────────────────────────────────
# CHANNELS    = ['VIS','MIR','SWIR','WV','TIR1']
# TARGET_SIZE = (256, 256)

# MEAN = np.array([
#     9.249125480651855, 0.03255389258265495,
#     0.35804545879364014, 0.14131774008274078,
#     271.2187805175781
# ], dtype=np.float32)
# STD  = np.array([
#     7.756808757781982, 0.01960049383342266,
#     0.6404346227645874, 0.04700399935245514,
#     20.69615936279297
# ], dtype=np.float32)

# INDIA_EXTENT = [68.0, 98.0, 6.0, 37.0]

# CHANNEL_DESCRIPTIONS = {
#     'VIS':  "Visible Reflectance (0.6 μm): Measures reflected sunlight like a daytime photo—used for cloud cover, vegetation health, and land use.",
#     'MIR':  "Mid-IR Radiance (3.9 μm): Captures emitted heat and some reflection—ideal for spotting wildfires, volcanic activity, and fog at night.",
#     'SWIR': "Shortwave IR (1.6 μm): Sensitive to moisture—used for mapping snow, soil moisture, and distinguishing clouds from ice.",
#     'WV':   "Water Vapor (6.3 μm): Tracks upper-tropospheric humidity—essential for understanding storm development and jet streams.",
#     'TIR1': "Thermal IR Temp (10.8 μm): Measures surface and cloud-top temperature—used for heat island detection and convection analysis."
# }

# # ── Flask Setup ───────────────────────────────────────────────────────────────
# app = Flask(__name__,
#             static_folder=STATIC_DIR,
#             template_folder=os.path.join(BASE_DIR, "templates"))
# app.secret_key = "change_this_to_secure_key"

# device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"

# # ── Model Loading ─────────────────────────────────────────────────────────────
# def build_model():
#     inp = tf.keras.Input(shape=(*TARGET_SIZE, len(CHANNELS)))
#     x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(inp)
#     x = tf.keras.layers.MaxPooling2D()(x)
#     x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
#     x = tf.keras.layers.Conv2D(128,3,padding='same',activation='relu')(x)
#     x = tf.keras.layers.UpSampling2D()(x)
#     x = tf.keras.layers.Conv2D(64,3,padding='same',activation='relu')(x)
#     x = tf.keras.layers.Conv2D(32,3,padding='same',activation='relu')(x)
#     out = tf.keras.layers.Conv2D(len(CHANNELS),1,
#                                  padding='same',
#                                  activation='linear',
#                                  dtype='float32')(x)
#     model = tf.keras.Model(inp, out)
#     model.load_weights(CKPT_PATH)
#     return model

# with tf.device(device):
#     model = build_model()
#     model.trainable = False

# # ── Preprocessing ─────────────────────────────────────────────────────────────
# def load_and_preprocess(fp):
#     with h5py.File(fp,'r') as f:
#         dn  = {ch: f[f"IMG_{ch}"][0].astype(int) for ch in CHANNELS}
#         lut = {
#             'VIS':  f['IMG_VIS_ALBEDO'][:],
#             'MIR':  f['IMG_MIR_RADIANCE'][:],
#             'SWIR': f['IMG_SWIR_RADIANCE'][:],
#             'WV':   f['IMG_WV_RADIANCE'][:],
#             'TIR1': f['IMG_TIR1_TEMP'][:],
#         }
#     bands = [lut[ch][dn[ch]].astype(np.float32) for ch in CHANNELS]
#     arr   = np.stack(bands, axis=-1)  # H×W×C

#     arr_tf = tf.convert_to_tensor(arr)[None]
#     arr_rs = tf.image.resize(arr_tf, TARGET_SIZE, method='bilinear')[0].numpy()
#     arr_n  = (arr_rs - MEAN)/STD
#     return arr_n[None]  # 1×256×256×5

# # ── Routes ────────────────────────────────────────────────────────────────────
# @app.route('/', methods=['GET'])
# def home():
#     return redirect(url_for('upload'))

# @app.route('/upload', methods=['GET','POST'])
# def upload():
#     if request.method == 'POST':
#         up = request.files.get('file')
#         if not up or not up.filename.lower().endswith('.h5'):
#             flash("Please upload a valid .h5 file.")
#             return redirect(request.url)

#         # parse input timestamp from filename
#         m = re.search(r"3RIMG_(\d{2}[A-Z]{3}\d{4})_(\d{4})_", up.filename)
#         if m:
#             date_str, time_str = m.groups()
#             dt_in = datetime.strptime(date_str + time_str, "%d%b%Y%H%M")
#             dt_out = dt_in + timedelta(days=1)
#         else:
#             dt_in = dt_out = None

#         # save upload
#         in_fp = os.path.join(UPLOAD_DIR, "input.h5")
#         up.save(in_fp)

#         # preprocess input and get its physical units
#         x_norm = load_and_preprocess(in_fp)             # 1×256×256×5
#         x_phys = (x_norm * STD.reshape(1,1,1,-1) +
#                   MEAN.reshape(1,1,1,-1))[0]            # 256×256×5

#         # predict
#         with tf.device(device):
#             y_norm = model.predict(x_norm)[0]           # 256×256×5
#         y_phys = y_norm * STD.reshape(1,1,-1) + MEAN.reshape(1,1,-1)

#         # save comparison plots
#         for i,ch in enumerate(CHANNELS):
#             # input
#             fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(projection=ccrs.PlateCarree()))
#             ax.set_extent(INDIA_EXTENT, ccrs.PlateCarree())
#             ax.coastlines('110m', linewidth=0.5)
#             ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.5)
#             im = ax.imshow(x_phys[...,i], origin='upper', transform=ccrs.PlateCarree(),
#                            cmap='Greys', extent=INDIA_EXTENT)
#             plt.title(f"Input {ch}", fontsize=16)
#             plt.colorbar(im, ax=ax, pad=0.02)
#             fig.savefig(os.path.join(STATIC_DIR, f"input_{ch}.png"),
#                         dpi=150, bbox_inches='tight')
#             plt.close(fig)

#             # predicted
#             fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(projection=ccrs.PlateCarree()))
#             ax.set_extent(INDIA_EXTENT, ccrs.PlateCarree())
#             ax.coastlines('110m', linewidth=0.5)
#             ax.add_feature(cfeature.BORDERS.with_scale('110m'), linewidth=0.5)
#             im = ax.imshow(y_phys[...,i], origin='upper', transform=ccrs.PlateCarree(),
#                            cmap='Greys', extent=INDIA_EXTENT)
#             plt.title(f"Forecast {ch}", fontsize=16)
#             plt.colorbar(im, ax=ax, pad=0.02)
#             fig.savefig(os.path.join(STATIC_DIR, f"pred_{ch}.png"),
#                         dpi=150, bbox_inches='tight')
#             plt.close(fig)

#         return render_template(
#             'results.html',
#             channels=CHANNELS,
#             descriptions=CHANNEL_DESCRIPTIONS,
#             input_datetime=dt_in,
#             output_datetime=dt_out
#         )

#     # GET
#     return render_template('upload.html',
#                            channel_info=CHANNEL_DESCRIPTIONS)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
