#!/usr/bin/env python3
"""Publication-quality architecture diagram for DiffusionUNet.

This figure is aligned with:
  - visplan/training/models/diffusion_unet.py
  - visplan/training/vision_encoder.py
  - visplan/training/diffusion.py

Main goals:
  - Show true overhead-view inputs (start, goal, piece mask).
  - Show coherent model structure and dimensions.
  - Show trajectory output over the overhead scene.
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import binary_dilation


# --------------------------------------------------------------------------- #
# Real sample data for image insets and trajectory overlay
# --------------------------------------------------------------------------- #

# This sample is rendered in a "last-piece completion" style for the output panel.
DATA = np.load("data/voronoi_reassembly/ep1000_pts5-12_seed0/episode_000088.npz")
start_img = DATA["start_image"]
goal_img = DATA["goal_image"]
piece_masks = DATA["piece_masks"]
start_poses = DATA["start_poses"]
goal_poses = DATA["goal_poses"]
trajectories = DATA["trajectories"]
num_pieces = int(DATA["num_pieces"])
outlines = [DATA[f"outline_{i}"] for i in range(num_pieces)]
DATA.close()


def fit_world_to_pixel(start_xy: np.ndarray, masks: np.ndarray):
    """Infer linear world->pixel mapping from start poses and start masks.

    This avoids relying on side_length, which may differ from the render visible range.
    """
    centers = []
    for i in range(masks.shape[0]):
        ys, xs = np.where(masks[i])
        centers.append([xs.mean(), ys.mean()])
    centers = np.array(centers, dtype=np.float32)

    sx, bx = np.polyfit(start_xy[:, 0], centers[:, 0], 1)
    sy, by = np.polyfit(start_xy[:, 1], centers[:, 1], 1)
    return float(sx), float(bx), float(sy), float(by)


MAP_SX, MAP_BX, MAP_SY, MAP_BY = fit_world_to_pixel(start_poses[:, :2], piece_masks)


def world_to_px(xy: np.ndarray):
    """Map world coordinates (x right, y up) to image pixel coordinates."""
    px = MAP_SX * xy[:, 0] + MAP_BX
    py = MAP_SY * xy[:, 1] + MAP_BY
    return px, py


def transform_outline(outline: np.ndarray, pose: np.ndarray):
    """Apply SE(2) pose to local polygon vertices."""
    x, y, theta = float(pose[0]), float(pose[1]), float(pose[2])
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return outline @ R.T + np.array([x, y], dtype=np.float32)


def polygon_to_mask(poly_px: np.ndarray, h: int = 256, w: int = 256):
    """Rasterize a polygon in pixel coordinates to a boolean mask."""
    yy, xx = np.mgrid[0:h, 0:w]
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    return Path(poly_px).contains_points(pts).reshape(h, w)


goal_masks = []
for i in range(num_pieces):
    world_poly = transform_outline(outlines[i], goal_poses[i])
    px, py = world_to_px(world_poly)
    poly_px = np.column_stack([px, py])
    goal_masks.append(polygon_to_mask(poly_px))
goal_masks = np.stack(goal_masks, axis=0)


def choose_final_piece():
    """Select a piece that best illustrates a clean final insertion motion."""
    best_i = 0
    best_score = -1e9
    all_idx = np.arange(num_pieces)
    for i in range(num_pieces):
        start_mask = piece_masks[i]
        area = float(start_mask.sum())
        if area < 40:
            continue
        other_goal_union = np.any(goal_masks[all_idx != i], axis=0)
        overlap = float((start_mask & other_goal_union).sum()) / max(area, 1.0)
        dist = float(np.linalg.norm(goal_poses[i, :2] - start_poses[i, :2]))
        # Prefer larger motion and lower overlap with already-assembled goal pieces.
        score = 4.0 * dist + 1.5 * (1.0 - overlap) + 0.05 * min(area, 600.0) / 600.0
        if score > best_score:
            best_score = score
            best_i = i
    return int(best_i)


PIECE = choose_final_piece()
mask = piece_masks[PIECE]
traj_raw = trajectories[PIECE]  # (32, 3): x, y, theta in metres


# Crop mask inset around the selected piece.
rows, cols = np.where(mask)
pad = 25
r0, r1 = max(0, rows.min() - pad), min(255, rows.max() + pad)
c0, c1 = max(0, cols.min() - pad), min(255, cols.max() + pad)
mask_crop = mask[r0:r1, c0:c1]
mask_crop_rgb = np.stack([mask_crop * 220, mask_crop * 220, mask_crop * 220], axis=-1).astype(np.uint8)
mask_crop_rgb[~mask_crop] = 38

tx, ty = world_to_px(traj_raw[:, :2])

# "Last-piece completion" scene: goal arrangement with this piece removed,
# then overlay the selected piece at its start location.
out_img = goal_img.astype(np.float32).copy()
goal_slot_mask = goal_masks[PIECE]
start_piece_mask = mask

# Empty slot cue in goal layout.
out_img[goal_slot_mask] = out_img[goal_slot_mask] * 0.18 + 26.0
slot_outline = binary_dilation(goal_slot_mask, iterations=2) & ~goal_slot_mask

# Place selected piece in its start pose for the "last piece to insert" story.
out_img[start_piece_mask] = start_img[start_piece_mask]
start_outline = binary_dilation(start_piece_mask, iterations=2) & ~start_piece_mask

out_rgba = np.zeros((256, 256, 4), dtype=float)
out_rgba[start_outline] = [1.0, 0.85, 0.0, 0.88]
out_rgba[slot_outline] = [0.2, 1.0, 0.6, 0.90]
out_img = np.clip(out_img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Visual style
# --------------------------------------------------------------------------- #

C_ENC = "#2563EB"
BG_ENC = "#DBEAFE"

C_DEC = "#C2410C"
BG_DEC = "#FFEDD5"

C_BOT = "#6D28D9"
BG_BOT = "#EDE9FE"

C_COND = "#065F46"
BG_COND = "#D1FAE5"

C_TIME = "#92400E"

C_PROJ = "#374151"
C_RN = "#1E40AF"
C_SKIP = "#6B7280"
C_LINE = "#111827"

TXT_DARK = "#0F172A"
TXT_LIGHT = "white"
FONT = "DejaVu Sans"

# Global visual scaling for readability without reworking layout geometry.
FONT_SCALE = 1.25
ARROW_LW_SCALE = 1.5
ARROW_MS_SCALE = 1.5

FW, FH = 21.0, 8.8
fig = plt.figure(figsize=(FW, FH), dpi=150)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
ax.set_xlim(0.0, FW)
ax.set_ylim(0.0, FH)
ax.axis("off")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def rbox(x, y, w, h, fc, ec="#1E293B", lw=1.1, rad=0.10, z=3, alpha=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle=f"round,pad=0,rounding_size={rad}",
            fc=fc,
            ec=ec,
            lw=lw,
            alpha=alpha,
            zorder=z,
        )
    )


def lbl(x, y, text, size=6.5, col=TXT_LIGHT, bold=False, ha="center", va="center", z=5, style="normal"):
    ax.text(
        x,
        y,
        text,
        ha=ha,
        va=va,
        fontsize=size * FONT_SCALE,
        fontfamily=FONT,
        color=col,
        fontweight="bold" if bold else "normal",
        style=style,
        zorder=z,
    )


def title(x, y, text):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="bottom",
        fontsize=9.0 * FONT_SCALE,
        fontfamily=FONT,
        color=TXT_DARK,
        fontweight="bold",
        zorder=7,
    )


def arr(p0, p1, col=C_LINE, lw=1.25, ms=10, z=5, ls="-"):
    scaled_lw = lw * ARROW_LW_SCALE
    scaled_ms = ms * ARROW_MS_SCALE
    ax.annotate(
        "",
        xy=p1,
        xytext=p0,
        arrowprops=dict(
            arrowstyle="->",
            color=col,
            lw=scaled_lw,
            mutation_scale=scaled_ms,
            linestyle=ls,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=z,
    )


def orthogonal_arrow(points, col=C_LINE, lw=1.25, ms=10, z=5, ls="-"):
    """Draw a polyline route with an arrow head only on the last segment."""
    if len(points) < 2:
        return
    if len(points) > 2:
        xs = [p[0] for p in points[:-1]]
        ys = [p[1] for p in points[:-1]]
        ax.plot(xs, ys, color=col, lw=lw * ARROW_LW_SCALE, ls=ls, zorder=z)
    arr(points[-2], points[-1], col=col, lw=lw, ms=ms, z=z, ls=ls)


def tag(x, y, text, col="#374151", size=5.8, z=6):
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=size * FONT_SCALE,
        fontfamily=FONT,
        color=col,
        zorder=z,
        bbox=dict(fc="white", ec=col, lw=0.7, boxstyle="round,pad=0.2"),
    )


def img_axes(image, x, y, w, h, border_col):
    sub = fig.add_axes([x / FW, y / FH, w / FW, h / FH])
    sub.imshow(image, aspect="auto")
    sub.set_xticks([])
    sub.set_yticks([])
    for spine in sub.spines.values():
        spine.set_edgecolor(border_col)
        spine.set_linewidth(2.0)
    return sub


# --------------------------------------------------------------------------- #
# A. Inputs
# --------------------------------------------------------------------------- #

IX = 0.25
IW = 1.90
IH = 1.90
IGAP = 0.28
input_total_h = 3.0 * IH + 2.0 * IGAP
iy0 = (FH - input_total_h) / 2.0

input_y = [iy0 + 2.0 * (IH + IGAP), iy0 + (IH + IGAP), iy0]
input_imgs = [start_img, goal_img, mask_crop_rgb]
input_cols = [C_ENC, C_ENC, C_PROJ]
input_caps = [
    "Start Scene\n$I_{\\mathrm{start}}$ (3 ch)",
    "Goal Scene\n$I_{\\mathrm{goal}}$ (3 ch)",
    "Piece Mask $M$ (1 ch)",
]

for y, img, c, cap in zip(input_y, input_imgs, input_cols, input_caps):
    img_axes(img, IX, y, IW, IH, c)
    lbl(IX + IW / 2.0, y - 0.16, cap, size=6.0, col="#111111")

title(IX + IW / 2.0, input_y[0] + IH + 0.12, "Inputs")

br_x = IX + IW + 0.08
br_top = input_y[0] + IH
br_bot = input_y[-1]
ax.plot([br_x, br_x + 0.16, br_x + 0.16, br_x], [br_bot, br_bot, br_top, br_top], color=C_LINE, lw=1.4, zorder=5)

inputs_mid_y = 0.5 * (br_top + br_bot)
lbl(br_x + 0.22, inputs_mid_y, "cat", size=7.0, col=C_LINE, ha="left", style="italic")
tag(br_x + 1.02, inputs_mid_y + 0.24, "7 ch")


# --------------------------------------------------------------------------- #
# B. Vision encoder (ResNet-18, 7-channel input)
# --------------------------------------------------------------------------- #

VX, VW = 2.85, 2.95
VY, VH = 0.85, 7.15

rbox(VX, VY, VW, VH, BG_ENC, ec=C_ENC, lw=1.6, rad=0.18, alpha=0.58, z=2)
title(VX + VW / 2.0, VY + VH + 0.12, "Vision Encoder")

vision_layers = [
    ("Conv1  7->64 ch", C_RN),
    ("Layer1  64 ch", C_RN),
    ("Layer2  128 ch", C_RN),
    ("Layer3  256 ch", C_RN),
    ("Layer4  512 ch", C_RN),
    ("AvgPool  -> 512-d", C_PROJ),
    ("Linear  512->256", C_PROJ),
]

vl_h = 0.43
vl_gap = 0.09
vl_x = VX + 0.22
vl_w = VW - 0.44
vl_top = VY + VH - 0.28

for i, (txt, col) in enumerate(vision_layers):
    ly = vl_top - i * (vl_h + vl_gap) - vl_h
    rbox(vl_x, ly, vl_w, vl_h, col, rad=0.07)
    lbl(vl_x + vl_w / 2.0, ly + vl_h / 2.0, txt, size=6.2)
    if i < len(vision_layers) - 1:
        arr((vl_x + vl_w / 2.0, ly), (vl_x + vl_w / 2.0, ly - vl_gap + 0.01), lw=0.95, ms=7)


# Input to vision encoder route.
vision_in_x = VX
vision_in_y = inputs_mid_y
orthogonal_arrow(
    [(br_x + 0.52, inputs_mid_y), (vision_in_x - 0.10, inputs_mid_y), (vision_in_x, vision_in_y)],
    lw=1.45,
)

vision_out = (VX + VW, VY + VH * 0.50)
arr(vision_out, (vision_out[0] + 0.30, vision_out[1]), lw=1.45)
tag(vision_out[0] + 0.60, vision_out[1] + 0.24, "256-d")


# --------------------------------------------------------------------------- #
# C. Conditioning (scene embedding + diffusion timestep)
# --------------------------------------------------------------------------- #

CX, CW = 6.05, 2.35
CY, CH = 0.85, 7.15

rbox(CX, CY, CW, CH, BG_COND, ec=C_COND, lw=1.6, rad=0.18, alpha=0.58, z=2)
title(CX + CW / 2.0, CY + CH + 0.12, "Conditioning")

cb_w = CW - 0.34
cb_h = 0.54
cb_x = CX + 0.17

scene_y = CY + CH - 0.70
tstep_y = scene_y - 0.75
sin_y = tstep_y - 0.75
mlp_y = sin_y - 0.75
concat_y = mlp_y - 0.75

rbox(cb_x, scene_y, cb_w, cb_h, C_ENC, rad=0.08)
lbl(CX + CW / 2.0, scene_y + cb_h / 2.0, "$z_{\\mathrm{scene}}$ (256-d)", size=6.6)

rbox(cb_x, tstep_y, cb_w, cb_h, "#FEF3C7", ec=C_TIME, lw=1.0, rad=0.08)
lbl(CX + CW / 2.0, tstep_y + cb_h / 2.0, "Timestep $t$", size=6.6, col=C_TIME)

rbox(cb_x, sin_y, cb_w, cb_h, C_TIME, rad=0.08)
lbl(CX + CW / 2.0, sin_y + cb_h / 2.0, "Sinusoidal Emb (128-d)", size=6.4)

rbox(cb_x, mlp_y, cb_w, cb_h, C_TIME, rad=0.08)
lbl(CX + CW / 2.0, mlp_y + cb_h / 2.0, "Time MLP (128-d)", size=6.4)

rbox(cb_x, concat_y, cb_w, cb_h, C_COND, ec="#064E3B", rad=0.08)
lbl(CX + CW / 2.0, concat_y + cb_h / 2.0, "Concat $c$ (384-d)", size=6.6)

# Vertical conditioning flow.
arr((CX + CW / 2.0, tstep_y), (CX + CW / 2.0, sin_y + cb_h), col=C_TIME, lw=1.0, ms=8)
arr((CX + CW / 2.0, sin_y), (CX + CW / 2.0, mlp_y + cb_h), col=C_TIME, lw=1.0, ms=8)
arr((CX + CW / 2.0, mlp_y), (CX + CW / 2.0, concat_y + cb_h), col=C_TIME, lw=1.0, ms=8)

# Scene embedding into concat branch.
orthogonal_arrow(
    [
        (cb_x + 0.22, scene_y),
        (cb_x + 0.22, scene_y - 0.08),
        (cb_x - 0.10, scene_y - 0.08),
        (cb_x - 0.10, concat_y + cb_h + 0.10),
        (cb_x + 0.22, concat_y + cb_h + 0.10),
        (cb_x + 0.22, concat_y + cb_h),
    ],
    col=C_ENC,
    lw=1.0,
    ms=8,
)

# Vision output to scene embedding.
orthogonal_arrow(
    [
        (vision_out[0] + 0.30, vision_out[1]),
        (CX - 0.10, vision_out[1]),
        (CX - 0.10, scene_y + cb_h / 2.0),
        (cb_x, scene_y + cb_h / 2.0),
    ],
    lw=1.35,
)

cond_out = (CX + CW, concat_y + cb_h / 2.0)
lbl(cond_out[0] + 0.22, cond_out[1] + 0.18, "FiLM $c$", size=6.3, col=C_COND, ha="center", va="bottom")


# --------------------------------------------------------------------------- #
# D. 1D Temporal U-Net
# --------------------------------------------------------------------------- #

UX, UW = 8.75, 8.35
UY, UH = 0.65, 7.55

rbox(UX, UY, UW, UH, "#F9FAFB", ec="#374151", lw=1.6, rad=0.20, z=2)
title(UX + UW / 2.0, UY + UH + 0.12, "1D Temporal U-Net")

# Geometry anchored by center lines to keep routing clean.
enc_cx = UX + 2.05
dec_cx = UX + UW - 2.05

bw, bh = 2.20, 0.62
sh = 0.34
sw = bw - 0.48

y_input_proj = 6.72
y_e128 = 5.90
y_ds1 = 5.25
y_e256 = 4.30
y_ds2 = 3.65
y_e512 = 2.70
y_ds3 = 2.05
y_bot = 1.22

# Conditioning marker at top of U-Net panel.
tag(UX + UW / 2.0, UY + UH - 0.28, "FiLM cond $c$ (384-d) applied in every FiLMBlock", col=C_COND, size=6.0)
orthogonal_arrow(
    [
        (cond_out[0], cond_out[1]),
        (UX - 0.16, cond_out[1]),
        (UX - 0.16, UY + UH - 0.28),
        (UX + 0.26, UY + UH - 0.28),
    ],
    col=C_COND,
    lw=1.25,
)


def draw_block(center_x, bottom_y, width, height, color, text, txt_size=6.3):
    x = center_x - width / 2.0
    rbox(x, bottom_y, width, height, color, rad=0.08)
    lbl(center_x, bottom_y + height / 2.0, text, size=txt_size)


# Input projection.
draw_block(enc_cx, y_input_proj, bw, bh, C_PROJ, "Input proj. Conv1d 4->128")

arr((UX + 0.12, y_input_proj + bh / 2.0), (enc_cx - bw / 2.0 - 0.04, y_input_proj + bh / 2.0), lw=1.35)
lbl(UX + 0.78, y_input_proj + bh / 2.0 + 0.22, "$x_t$ (32x4)", size=6.2, col="#111111")

# Encoder blocks.
draw_block(enc_cx, y_e128, bw, bh, C_ENC, "2x FiLMBlock (128 ch)")
draw_block(enc_cx, y_e256, bw, bh, C_ENC, "FiLMBlock + 1x1 + FiLMBlock (256 ch)", txt_size=6.0)
draw_block(enc_cx, y_e512, bw, bh, C_ENC, "FiLMBlock + 1x1 + FiLMBlock (512 ch)", txt_size=6.0)

draw_block(enc_cx, y_ds1, sw, sh, C_PROJ, "Downsample (stride 2)", txt_size=5.9)
draw_block(enc_cx, y_ds2, sw, sh, C_PROJ, "Downsample (stride 2)", txt_size=5.9)
draw_block(enc_cx, y_ds3, sw, sh, C_PROJ, "Downsample (stride 2)", txt_size=5.9)

# Bottleneck.
bot_w = UW - 0.92
bot_x = UX + (UW - bot_w) / 2.0
rbox(bot_x, y_bot, bot_w, 0.80, C_BOT, ec="#4C1D95", lw=1.3, rad=0.10)
lbl(UX + UW / 2.0, y_bot + 0.48, "Bottleneck: 2x FiLMBlock (512 ch)", size=7.0, bold=True)
lbl(UX + UW / 2.0, y_bot + 0.22, "Conditioned by $c$ (384-d)", size=6.0, col="#DDD6FE")

# Encoder vertical flow.
arr((enc_cx, y_input_proj), (enc_cx, y_e128 + bh), lw=1.0, ms=8)
arr((enc_cx, y_e128), (enc_cx, y_ds1 + sh), lw=1.0, ms=8)
arr((enc_cx, y_ds1), (enc_cx, y_e256 + bh), lw=1.0, ms=8)
arr((enc_cx, y_e256), (enc_cx, y_ds2 + sh), lw=1.0, ms=8)
arr((enc_cx, y_ds2), (enc_cx, y_e512 + bh), lw=1.0, ms=8)
arr((enc_cx, y_e512), (enc_cx, y_ds3 + sh), lw=1.0, ms=8)
arr((enc_cx, y_ds3), (enc_cx, y_bot + 0.80), lw=1.1, ms=8)

# Decoder blocks.
draw_block(dec_cx, y_e512, bw, bh, C_DEC, "Concat + 1x1 + 2x FiLMBlock (512)", txt_size=6.0)
draw_block(dec_cx, y_e256, bw, bh, C_DEC, "Concat + 1x1 + 2x FiLMBlock (256)", txt_size=6.0)
draw_block(dec_cx, y_e128, bw, bh, C_DEC, "Concat + 1x1 + 2x FiLMBlock (128)", txt_size=6.0)

draw_block(dec_cx, y_ds3, sw, sh, C_PROJ, "Upsample (stride 2)", txt_size=5.9)
draw_block(dec_cx, y_ds2, sw, sh, C_PROJ, "Upsample (stride 2)", txt_size=5.9)
draw_block(dec_cx, y_ds1, sw, sh, C_PROJ, "Upsample (stride 2)", txt_size=5.9)

# Decoder flow.
orthogonal_arrow(
    [
        (UX + UW / 2.0 + 0.20, y_bot + 0.80),
        (dec_cx, y_bot + 0.80),
        (dec_cx, y_ds3),
    ],
    lw=1.1,
    ms=8,
)
arr((dec_cx, y_ds3 + sh), (dec_cx, y_e512), lw=1.0, ms=8)
arr((dec_cx, y_e512 + bh), (dec_cx, y_ds2), lw=1.0, ms=8)
arr((dec_cx, y_ds2 + sh), (dec_cx, y_e256), lw=1.0, ms=8)
arr((dec_cx, y_e256 + bh), (dec_cx, y_ds1), lw=1.0, ms=8)
arr((dec_cx, y_ds1 + sh), (dec_cx, y_e128), lw=1.0, ms=8)

# Skip connections.
skip_levels = [(y_e128 + bh / 2.0, "skip"), (y_e256 + bh / 2.0, "skip"), (y_e512 + bh / 2.0, "skip")]
for sy, txt in skip_levels:
    ax.plot([enc_cx + bw / 2.0, dec_cx - bw / 2.0], [sy, sy], color=C_SKIP, lw=1.2, ls="--", zorder=4)
    arr((dec_cx - bw / 2.0 - 0.10, sy), (dec_cx - bw / 2.0, sy), col=C_SKIP, lw=1.2, ms=8)
    lbl((enc_cx + dec_cx) / 2.0, sy + 0.14, txt, size=5.5, col=C_SKIP, style="italic")

# Output projection and U-Net output.
draw_block(dec_cx, y_input_proj, bw, bh, C_PROJ, "Output proj. GN+SiLU, Conv1d 128->4", txt_size=6.0)
arr((dec_cx, y_e128 + bh), (dec_cx, y_input_proj), lw=1.1, ms=8)

unet_out = (UX + UW, y_input_proj + bh / 2.0)
unet_out_start = (dec_cx + bw / 2.0, y_input_proj + bh / 2.0)
tag(unet_out[0] - 0.34, unet_out[1] + 0.28, "$\\hat{\\epsilon}$ (32x4)")

lbl(
    UX + UW / 2.0,
    UY + 0.24,
    "Iterative denoising: DDPM (T=100) or DDIM (20 steps)",
    size=6.1,
    col="#6B7280",
    style="italic",
)


# --------------------------------------------------------------------------- #
# E. Output trajectory inset (overhead view)
# --------------------------------------------------------------------------- #

OX = UX + UW + 0.12
OW = FW - OX - 0.20

oi_side = min(OW - 0.30, FH - 2.00)
oi_x = OX + 0.16
oi_y = (FH - oi_side) / 2.0

ax_out = img_axes(out_img, oi_x, oi_y, oi_side, oi_side, C_DEC)

pts = np.array([tx, ty]).T.reshape(-1, 1, 2)
segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
lc = LineCollection(segments, cmap="plasma", linewidth=3.0, zorder=3, alpha=0.90)
lc.set_array(np.linspace(0.0, 1.0, len(segments)))
ax_out.add_collection(lc)

ax_out.plot(tx[0], ty[0], "o", color="#22C55E", ms=7, zorder=5, mew=1.4, mec="white")
ax_out.plot(tx[-1], ty[-1], "*", color="#EF4444", ms=10, zorder=5, mew=1.0, mec="white")
ax_out.imshow(out_rgba, aspect="auto", zorder=2)
ax_out.set_xlim(0, 255)
ax_out.set_ylim(255, 0)

title(oi_x + oi_side / 2.0, oi_y + oi_side + 0.12, "Output")
lbl(
    oi_x + oi_side / 2.0,
    oi_y - 0.20,
    "Output Trajectory\n$\\tau$: $(x,y,\\cos\\theta,\\sin\\theta) \\times 32$",
    size=6.2,
    col="#111111",
    va="top",
)

# Route U-Net prediction to output inset center-left.
target_y = oi_y + oi_side * 0.50
orthogonal_arrow(
    [
        unet_out_start,
        (UX + UW + 0.06, unet_out[1]),
        (UX + UW + 0.06, target_y),
        (oi_x, target_y),
    ],
    lw=1.45,
)


# --------------------------------------------------------------------------- #
# Legend and save
# --------------------------------------------------------------------------- #

legend_items = [
    mpatches.Patch(fc=C_ENC, ec="#AAAAAA", label="Encoder FiLM blocks"),
    mpatches.Patch(fc=C_DEC, ec="#AAAAAA", label="Decoder FiLM blocks"),
    mpatches.Patch(fc=C_BOT, ec="#AAAAAA", label="Bottleneck"),
    mpatches.Patch(fc=C_COND, ec="#AAAAAA", label="Conditioning (c)"),
    mpatches.Patch(fc=C_PROJ, ec="#AAAAAA", label="Projection / norm / resize"),
    mpatches.Patch(fc=C_SKIP, ec="#AAAAAA", label="Skip connections"),
]

ax.legend(
    handles=legend_items,
    loc="lower center",
    ncol=6,
    fontsize=6.8 * FONT_SCALE,
    framealpha=0.95,
    bbox_to_anchor=(0.5, 0.005),
    frameon=True,
    handlelength=1.2,
    handleheight=0.9,
)

plt.savefig("figures/arch_diagram.pdf", bbox_inches="tight", dpi=150, facecolor="white")
plt.savefig("figures/arch_diagram.png", bbox_inches="tight", dpi=150, facecolor="white")
print("Saved figures/arch_diagram.pdf + .png")
