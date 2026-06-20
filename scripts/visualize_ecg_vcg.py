"""
ECG ↔ VCG Visualization

Generates a side-by-side animation:
  Left : 12-lead ECG with a moving time cursor
  Right: 3D VCG particle trajectory (current position + fading trail)

No real data needed — synthesises a realistic multi-beat waveform from
a parameterised VCG loop, then projects to 12 leads using the same
pseudo-inverse geometry as lvcg/models/vcg.py.

Dependencies: numpy, matplotlib (pip install matplotlib)
Output      : ecg_vcg.mp4  (requires ffmpeg) or ecg_vcg.gif
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Lead geometry  (Frank X-Y-Z coordinate system)
#     X = left,  Y = inferior,  Z = anterior
# ──────────────────────────────────────────────────────────────────────────────

def _norm(v):
    return v / (np.linalg.norm(v) + 1e-12)

# 12-lead unit direction vectors in Frank XYZ space
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

_U_raw = np.array([
    # Limb leads (frontal plane, Z≈0)
    [ 1.000,  0.000,  0.000],   # I
    [ 0.500,  0.866,  0.000],   # II
    [-0.500,  0.866,  0.000],   # III
    [-0.866, -0.500,  0.000],   # aVR
    [ 0.866, -0.500,  0.000],   # aVL
    [ 0.000,  1.000,  0.000],   # aVF
    # Precordial leads (horizontal plane, Y≈0.3)
    [-0.387,  0.296,  0.874],   # V1
    [-0.196,  0.239,  0.951],   # V2
    [ 0.098,  0.200,  0.975],   # V3
    [ 0.388,  0.296,  0.871],   # V4
    [ 0.687,  0.296,  0.663],   # V5
    [ 0.900,  0.296,  0.318],   # V6
], dtype=np.float64)

# Normalise each row
U_LEADS = np.array([_norm(row) for row in _U_raw])   # [12, 3]


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Synthesise a realistic VCG trajectory
#     We model one heartbeat as a sequence of Gaussian "blobs" in 3D:
#       P-wave  → small excursion, mostly +Y
#       QRS     → large loop, mainly +X, +Y, +Z then swinging back
#       T-wave  → medium excursion, +X +Y +Z
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_pulse(t, t0, sigma, amplitude):
    """Scalar Gaussian bump."""
    return amplitude * np.exp(-0.5 * ((t - t0) / sigma) ** 2)


def synthesise_vcg(fs=500, n_beats=3, hr=72):
    """
    Returns VCG [3, T] representing n_beats heartbeats.
    Axes: [X=left, Y=inferior, Z=anterior]
    """
    beat_len = int(60 / hr * fs)   # samples per beat
    T = n_beats * beat_len
    t = np.arange(T) / fs          # time in seconds

    x = np.zeros(T)
    y = np.zeros(T)
    z = np.zeros(T)

    for b in range(n_beats):
        t0 = b * beat_len / fs      # beat start (seconds)

        # ---- P wave (atrial depolarisation) ----
        tp = t0 + 0.10
        x += gaussian_pulse(t, tp, 0.020, 0.08)
        y += gaussian_pulse(t, tp, 0.020, 0.10)
        z += gaussian_pulse(t, tp, 0.020, 0.03)

        # ---- Q (small negative, septal) ----
        tq = t0 + 0.22
        x += gaussian_pulse(t, tq, 0.012, -0.15)
        y += gaussian_pulse(t, tq, 0.012, -0.05)
        z += gaussian_pulse(t, tq, 0.012,  0.05)

        # ---- R (large positive, main QRS) ----
        tr = t0 + 0.25
        x += gaussian_pulse(t, tr, 0.018,  0.80)
        y += gaussian_pulse(t, tr, 0.018,  0.60)
        z += gaussian_pulse(t, tr, 0.018, -0.20)

        # ---- S (negative, basal forces) ----
        ts = t0 + 0.28
        x += gaussian_pulse(t, ts, 0.014, -0.35)
        y += gaussian_pulse(t, ts, 0.014,  0.15)
        z += gaussian_pulse(t, ts, 0.014,  0.30)

        # ---- T wave (ventricular repolarisation) ----
        tt = t0 + 0.40
        x += gaussian_pulse(t, tt, 0.045,  0.35)
        y += gaussian_pulse(t, tt, 0.045,  0.30)
        z += gaussian_pulse(t, tt, 0.045, -0.10)

    vcg = np.stack([x, y, z], axis=0)   # [3, T]
    return vcg, t


# ──────────────────────────────────────────────────────────────────────────────
# 3.  VCG  →  12-lead ECG  via geometric projection
# ──────────────────────────────────────────────────────────────────────────────

def vcg_to_ecg(vcg, U=U_LEADS):
    """
    vcg : [3, T]
    U   : [L, 3]
    returns ecg [L, T]
    """
    return U @ vcg   # [L, T]


# ──────────────────────────────────────────────────────────────────────────────
# 4.  ECG  →  VCG  via pseudo-inverse  (same formula as lvcg/models/vcg.py)
# ──────────────────────────────────────────────────────────────────────────────

def ecg_to_vcg(ecg, U=U_LEADS, eps=1e-6):
    """
    ecg : [L, T]
    U   : [L, 3]
    returns vcg_hat [3, T]
    """
    UtU = U.T @ U                               # [3, 3]
    UtU_reg = UtU + eps * np.eye(3)
    U_pinv = np.linalg.inv(UtU_reg) @ U.T       # [3, L]
    return U_pinv @ ecg                          # [3, T]


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Animation
# ──────────────────────────────────────────────────────────────────────────────

def make_animation(output="ecg_vcg.mp4", fps=50, trail_sec=0.15, fs=500, n_beats=3):
    vcg_true, t = synthesise_vcg(fs=fs, n_beats=n_beats)
    ecg = vcg_to_ecg(vcg_true)               # [12, T]
    vcg = ecg_to_vcg(ecg)                    # [3, T]  (round-trip, essentially identical)
    T = vcg.shape[1]

    # Down-sample for animation (every stride-th sample → one frame)
    stride = max(1, fs // fps)
    frames = list(range(0, T, stride))
    trail = int(trail_sec * fs)               # samples in the fading trail

    # ── figure layout ──
    fig = plt.figure(figsize=(16, 9), facecolor="#0d1117")
    gs = fig.add_gridspec(12, 2, wspace=0.08, hspace=0.15,
                          left=0.06, right=0.97, top=0.93, bottom=0.04)

    ax_ecg  = [fig.add_subplot(gs[i, 0]) for i in range(12)]
    ax_vcg  = fig.add_subplot(gs[:, 1], projection="3d")

    # ── colour palette ──
    ECG_COLOR  = "#00e5ff"
    CURSOR_COL = "#ff4081"
    TRAIL_BASE = np.array([0.2, 0.8, 1.0])  # RGB for trail
    DOT_COLOR  = "#ffffff"
    AXIS_COL   = "#aaaaaa"
    BG         = "#0d1117"
    GRID_COL   = "#1e2a35"

    # ── ECG panels: static waveform + cursor line ──
    ecg_lines   = []
    cursor_lines = []
    offsets = []
    for i, ax in enumerate(ax_ecg):
        ax.set_facecolor(BG)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        sig = ecg[i]
        # normalise within ±1 per lead
        rng = np.ptp(sig) + 1e-6
        sig_n = (sig - np.mean(sig)) / rng
        ax.plot(t, sig_n, color=ECG_COLOR, lw=0.7, alpha=0.85)
        ax.set_xlim(t[0], t[-1])
        y_pad = 0.3
        ax.set_ylim(-1 - y_pad, 1 + y_pad)

        # Lead label
        ax.text(-0.01, 0.5, LEAD_NAMES[i], transform=ax.transAxes,
                color=AXIS_COL, fontsize=7, ha="right", va="center", fontfamily="monospace")

        # Cursor vertical line
        cline = ax.axvline(x=t[0], color=CURSOR_COL, lw=1.0, alpha=0.9)
        cursor_lines.append(cline)

    ax_ecg[0].set_title("12-lead ECG", color="white", fontsize=11, pad=4, loc="left")

    # ── 3-D VCG panel ──
    ax_vcg.set_facecolor(BG)
    ax_vcg.xaxis.pane.fill = False
    ax_vcg.yaxis.pane.fill = False
    ax_vcg.zaxis.pane.fill = False
    for pane in [ax_vcg.xaxis.pane, ax_vcg.yaxis.pane, ax_vcg.zaxis.pane]:
        pane.set_edgecolor(GRID_COL)

    ax_vcg.tick_params(colors=AXIS_COL, labelsize=6)
    ax_vcg.set_xlabel("X (left)", color=AXIS_COL, fontsize=8, labelpad=2)
    ax_vcg.set_ylabel("Y (inf)", color=AXIS_COL, fontsize=8, labelpad=2)
    ax_vcg.set_zlabel("Z (ant)", color=AXIS_COL, fontsize=8, labelpad=2)
    ax_vcg.set_title("VCG particle trajectory", color="white", fontsize=11, pad=6)

    # Axis limits
    pad = 0.15
    for axis, idx in zip(["x", "y", "z"], [0, 1, 2]):
        lo, hi = vcg[idx].min() - pad, vcg[idx].max() + pad
        getattr(ax_vcg, f"set_{axis}lim")(lo, hi)

    # Ghost loop (full single beat for context)
    beat_t = int(60 / 72 * fs)
    ghost_x, ghost_y, ghost_z = vcg[0, :beat_t], vcg[1, :beat_t], vcg[2, :beat_t]
    ax_vcg.plot(ghost_x, ghost_y, ghost_z, color="#334455", lw=0.8, alpha=0.5, zorder=1)

    # Trail line (will be updated)
    trail_line, = ax_vcg.plot([], [], [], lw=2.0, color=ECG_COLOR, alpha=0.8, zorder=2)
    dot, = ax_vcg.plot([], [], [], "o", color=DOT_COLOR, markersize=6, zorder=3)

    # Time label
    time_text = ax_vcg.text2D(0.02, 0.96, "", transform=ax_vcg.transAxes,
                               color="white", fontsize=9, fontfamily="monospace")

    fig.suptitle("ECG ↔ VCG  |  Synthetic Normal Sinus Rhythm",
                 color="white", fontsize=13, y=0.97)

    # ── animation update ──
    def update(frame_idx):
        n = frames[frame_idx]
        cur_t = t[n]

        # Move cursors
        for cline in cursor_lines:
            cline.set_xdata([cur_t, cur_t])

        # Trail indices
        start = max(0, n - trail)
        xs = vcg[0, start:n+1]
        ys = vcg[1, start:n+1]
        zs = vcg[2, start:n+1]

        trail_line.set_data_3d(xs, ys, zs)

        if len(xs) > 0:
            dot.set_data_3d([xs[-1]], [ys[-1]], [zs[-1]])

        time_text.set_text(f"t = {cur_t:.3f} s")

        # Slowly rotate the 3D view
        ax_vcg.view_init(elev=25, azim=30 + frame_idx * 360 / len(frames))

        return cursor_lines + [trail_line, dot, time_text]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 / fps, blit=False
    )

    # ── save ──
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800,
                                        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        ani.save(output, writer=writer, dpi=120)
        print(f"Saved MP4 → {output}")
    except Exception as e:
        print(f"FFmpeg failed ({e}), falling back to GIF...")
        gif_out = output.replace(".mp4", ".gif")
        ani.save(gif_out, writer="pillow", fps=fps // 2, dpi=80)
        print(f"Saved GIF → {gif_out}")

    plt.close(fig)


if __name__ == "__main__":
    import os
    out_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(out_dir, "..", "ecg_vcg.mp4")
    make_animation(output=output_path, fps=50, trail_sec=0.12, fs=500, n_beats=3)
