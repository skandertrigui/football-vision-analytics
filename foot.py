#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_demo.py — Football Analytics « Broadcast Scouting Feed » (fichier unique, autonome)
=======================================================================================

Pipeline en DEUX PASSES :

  PASSE 1 (analyse)   lecture vidéo -> détection YOLO -> filtre spatial anti-spectateurs
                      -> ByteTrack -> mouvement caméra PTZ (Lucas-Kanade + échelle)
                      -> échantillons de couleur de maillot -> candidats ballon.

  POST-TRAITEMENT     rôles par vote de piste, équipes + arbitre (KMeans, espace Lab),
  (global, hors-ligne) trajectoire du ballon (Viterbi + interpolation + lissage, dans le
                      repère stabilisé), vitesses/distances lissées (homographie),
                      possession avec hystérésis.

  PASSE 2 (rendu)     ellipses au sol, étiquettes semi-transparentes, marqueurs de
                      possession, traînée du ballon, panneaux de statistiques.

Points clés
-----------
1. Caméra PTZ : flot optique Lucas-Kanade sur les bandes latérales (avec contrôle
   aller-retour), échelle (zoom) estimée par le rapport des distances inter-points,
   translation (pan/tilt) estimée après compensation de l'échelle. Les transformations
   sont composées : chaque frame est ramenée dans le repère de la frame 0.
2. Modèles COCO (person / sports ball) ET fine-tunés (player / goalkeeper / referee / ball)
   détectés automatiquement. Filtre spatial (position Y + herbe sous les pieds) contre les
   spectateurs ; arbitre séparé par la couleur si le modèle ne le fournit pas.
3. Cinématique robuste : positions projetées en mètres (ViewTransformer), rejet des
   valeurs aberrantes (Hampel), comblement de trous, lissage gaussien centré, vitesse par
   différences centrées, zone morte anti-bruit, distance cumulée.

Usage
-----
    python main_demo.py --input match.mp4 --output out.mp4
    python main_demo.py --input match.mp4 --model models/best.pt      # modèle fine-tuné
    python main_demo.py --input match.mp4 --calibrate                 # calibrer le terrain
    python main_demo.py --input match.mp4 --court "110,1035,265,275,910,260,1640,915"

Dépendances : ultralytics, opencv-python, numpy, supervision, scikit-learn
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.cluster import KMeans
from ultralytics import YOLO

warnings.filterwarnings("ignore", module="sklearn")

# =============================================================================
# CONFIGURATION
# =============================================================================
DEFAULT_INPUT = "input_video.mp4"
DEFAULT_OUTPUT = "output_demo.mp4"
DEFAULT_MODEL = "yolov8m.pt"      # COCO par défaut ; idéal : modèle fine-tuné (best.pt)

PERSON_CONF = 0.25                # seuil de confiance joueurs / arbitres
BALL_CONF = 0.10                  # seuil (plus bas) pour le ballon : petit objet
DETECTION_IMGSZ = 1280            # résolution d'inférence (960/640 sur CPU)
MIN_BOX_H_RATIO = 0.03            # hauteur mini d'une personne (fraction de H)
MIN_TRACK_LEN = 8                 # pistes plus courtes = masquées (anti-clignotement)

# --- Caméra PTZ (Lucas-Kanade sur les bandes latérales) ---
CAMERA_BAND_RATIO = 0.08          # largeur de chaque bande latérale (fraction de W)
CAMERA_FLOW_WIDTH = 960           # largeur de travail du flot optique
CAMERA_MAX_CORNERS = 300
CAMERA_MIN_POINTS = 12            # points valides minimum pour estimer le mouvement
CAMERA_FB_THRESH = 1.0            # erreur aller-retour max (px, résolution de travail)
CAMERA_MIN_PAIR_DIST = 0.30       # distance mini d'une paire de points (fraction de W)
CAMERA_MAX_PAIRS = 600
CAMERA_MAX_STEP_SCALE = 0.03      # |échelle - 1| max plausible entre 2 frames
CAMERA_MIN_INLIER_RATIO = 0.25    # en dessous : estimation jugée non fiable (coupure, replay)

# --- Filtre spatial anti-spectateurs ---
GRASS_HSV_LO = (35, 50, 40)
GRASS_HSV_HI = (90, 255, 255)
MIN_GRASS_RATIO = 0.08            # part d'herbe minimale dans l'image pour se fier au masque
FIELD_ROW_MIN_FRAC = 0.35         # une ligne « appartient au terrain » si >35 % d'herbe
FIELD_TOP_FALLBACK_RATIO = 0.22   # plan B : pieds au-dessus de 22 % de H = tribunes
MIN_FOOT_GRASS = 0.20             # part d'herbe minimale sous les pieds

# --- Équipes / arbitre (couleur) ---
COLOR_SAMPLES_PER_TRACK = 6
COLOR_SAMPLE_EVERY = 8            # frames entre deux échantillons d'une même piste
REF_MIN_TRACKS = 8                # pistes minimum pour tenter la détection d'arbitre
REF_MAX_WEIGHT_FRAC = 0.15        # l'arbitre = petit cluster (< 15 % du temps de présence)
REF_MIN_LAB_SEP = 22.0            # écart Lab minimal entre cluster arbitre et le plus proche

# --- Ballon ---
BALL_TOP_K = 5                    # candidats conservés par frame
BALL_MAX_SPEED_PX_S_1080 = 2400   # vitesse max plausible (px/s à 1920 px de large)
BALL_EMIS_W = 1.0                 # coût d'émission = (1 - conf) * poids
BALL_MISS_COST = 0.75             # coût d'un « ballon absent »
BALL_MAX_GAP_FRAMES = 25          # trous comblés par interpolation
BALL_SMOOTH_SIGMA = 1.5           # lissage gaussien (frames)
BALL_TRAIL_FRAMES = 18

# --- Possession ---
POSSESSION_DIST_RATIO = 0.60      # distance ballon-pieds max, en fraction de la hauteur du joueur
POSSESSION_MIN_FRAMES = 3         # frames consécutives pour changer de porteur
POSSESSION_LOST_FRAMES = 8        # frames sans porteur avant d'oublier le porteur

# --- Cinématique ---
MAX_PLAUSIBLE_KMH = 38.0
KIN_HAMPEL_HALF_SEC = 0.25        # demi-fenêtre du rejet d'aberrations
KIN_HAMPEL_THR_M = 2.0            # écart max à la médiane locale (m)
KIN_MAX_GAP_SEC = 0.6             # trous de positions comblés
KIN_POS_SIGMA_SEC = 0.25          # lissage gaussien des positions
KIN_SPEED_SIGMA_SEC = 0.20        # lissage gaussien de la vitesse
KIN_DEADBAND_KMH = 0.6            # sous ce seuil : immobile (bruit)

# --- Calibration terrain (vidéo 1920x1080 du projet d'origine) ---
# Ordre : bas-gauche, haut-gauche, haut-droite, bas-droite (dans la frame 0).
DEFAULT_COURT_PIXELS = [[110, 1035], [265, 275], [910, 260], [1640, 915]]
DEFAULT_DEPTH_M = 68.0            # arête bas-gauche -> haut-gauche
DEFAULT_WIDTH_M = 23.32           # arête haut-gauche -> haut-droite

# --- Palette (BGR) & police ---
REFEREE_COLOR = (0, 215, 255)
NEUTRAL_COLOR = (230, 230, 230)
BALL_MARKER_COLOR = (0, 230, 90)
POSSESSION_MARKER_COLOR = (40, 40, 255)
PANEL_COLOR = (18, 18, 18)
FONT = cv2.FONT_HERSHEY_SIMPLEX
DEFAULT_TEAM_COLORS = {1: (255, 140, 30), 2: (60, 60, 255)}

FIELD_ROLES = ("player", "goalkeeper")


# =============================================================================
# 0) UTILITAIRES NUMÉRIQUES
# =============================================================================
def progress(i: int, n: int, t0: float, label: str) -> None:
    if i % 10 == 0 or i == n:
        el = time.time() - t0
        rate = i / el if el > 0 else 0.0
        pct = 100.0 * i / n if n else 0.0
        sys.stdout.write(f"\r{label}: {i}/{n or '?'} ({pct:5.1f}%) — {rate:4.1f} fps")
        sys.stdout.flush()


def smooth_nan(x, sigma: float, keep_nan: bool = True) -> np.ndarray:
    """Lissage gaussien centré (sans retard) tolérant aux NaN (convolution normalisée).
    Accepte (T,) ou (T, k). Avec keep_nan=True, les NaN d'origine restent NaN."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return np.stack([smooth_nan(x[:, k], sigma, keep_nan) for k in range(x.shape[1])], axis=1)
    T = len(x)
    if sigma <= 0 or T < 3:
        return x.copy()
    half = int(min(max(1, round(3 * sigma)), (T - 1) // 2))
    ker = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2)
    ker /= ker.sum()
    valid = ~np.isnan(x)
    num = np.convolve(np.where(valid, x, 0.0), ker, mode="same")
    den = np.convolve(valid.astype(np.float64), ker, mode="same")
    out = np.full(T, np.nan)
    ok = den > 1e-6
    out[ok] = num[ok] / den[ok]
    if keep_nan:
        out[~valid] = np.nan
    return out


def interp_short_gaps(arr, max_gap: int) -> np.ndarray:
    """Interpolation linéaire des trous de NaN de longueur <= max_gap. (T,) ou (T, k)."""
    a = np.array(arr, dtype=np.float64, copy=True)
    if a.ndim == 1:
        return interp_short_gaps(a[:, None], max_gap)[:, 0]
    idx = np.where(~np.isnan(a).any(axis=1))[0]
    for i0, i1 in zip(idx[:-1], idx[1:]):
        gap = i1 - i0 - 1
        if 0 < gap <= max_gap:
            t = (np.arange(i0 + 1, i1) - i0) / float(i1 - i0)
            a[i0 + 1:i1] = a[i0] + (a[i1] - a[i0]) * t[:, None]
    return a


def hampel_reject(P: np.ndarray, half_win: int, thr: float) -> np.ndarray:
    """Met à NaN les points de P (T,2) trop éloignés de la médiane locale (filtre de Hampel)."""
    T = len(P)
    if T < 2 * half_win + 1 or half_win < 1:
        return P
    pad = np.full((T + 2 * half_win, 2), np.nan)
    pad[half_win:half_win + T] = P
    win = sliding_window_view(pad, 2 * half_win + 1, axis=0)          # (T, 2, win)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(win, axis=2)
    dev = np.linalg.norm(P - med, axis=1)
    out = P.copy()
    out[dev > thr] = np.nan
    return out


def bgr_to_lab(bgr) -> np.ndarray:
    px = np.clip(np.asarray(bgr, dtype=np.float64), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(px, cv2.COLOR_BGR2LAB).reshape(3).astype(np.float64)


def display_color(bgr) -> tuple[int, int, int]:
    """Éclaircit les couleurs trop sombres pour rester lisibles sur la pelouse."""
    c = np.clip(np.asarray(bgr, dtype=np.float64), 0, 255)
    luma = 0.114 * c[0] + 0.587 * c[1] + 0.299 * c[2]
    if luma < 110:
        c = np.clip(c + (110 - luma), 0, 255)
    return tuple(int(v) for v in c)


# =============================================================================
# 1) CAMÉRA PTZ : LUCAS-KANADE + COMPENSATION D'ÉCHELLE
# =============================================================================
class CameraMotionEstimator:
    """Estime le mouvement inter-frames de la caméra : échelle s (zoom), translation (tx, ty).

    Modèle de similarité centré sur le centre image c :   p1 = c + s·(p0 − c) + t
      * points suivis dans les bandes latérales (tribunes/panneaux, peu de joueurs) ;
      * contrôle aller-retour du flot (forward-backward) pour éliminer les mauvais suivis ;
      * échelle s = médiane des rapports de distances entre paires de points
        d(p1_i, p1_j) / d(p0_i, p0_j)  (paires écartées -> bruit de localisation négligeable) ;
      * translation t = moyenne des résidus des inliers après compensation de l'échelle.
    Les boîtes de joueurs sont exclues du masque (leur mouvement propre fausserait l'estimation).
    """

    def __init__(self, first_gray: np.ndarray):
        h, w = first_gray.shape
        self.W, self.H = w, h
        self.sc = min(1.0, CAMERA_FLOW_WIDTH / float(w))
        self.sw, self.sh = int(round(w * self.sc)), int(round(h * self.sc))
        band = max(12, int(CAMERA_BAND_RATIO * self.sw))
        self.base_mask = np.zeros((self.sh, self.sw), np.uint8)
        self.base_mask[:, :band] = 255
        self.base_mask[:, -band:] = 255
        self.center = np.array([w / 2.0, h / 2.0])
        self.prev = self._prep(first_gray)
        self.rng = np.random.default_rng(0)
        self.lk = dict(winSize=(21, 21), maxLevel=3,
                       criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

    def _prep(self, gray: np.ndarray) -> np.ndarray:
        if self.sc >= 1.0:
            return gray
        return cv2.resize(gray, (self.sw, self.sh), interpolation=cv2.INTER_AREA)

    def update(self, gray: np.ndarray, exclude_boxes=()) -> tuple[float, float, float]:
        """Retourne (s, tx, ty) du passage frame précédente -> frame courante (px plein format)."""
        cur = self._prep(gray)
        mask = self.base_mask.copy()
        for x1, y1, x2, y2 in exclude_boxes:
            cv2.rectangle(mask, (int(x1 * self.sc) - 4, int(y1 * self.sc) - 4),
                          (int(x2 * self.sc) + 4, int(y2 * self.sc) + 4), 0, -1)
        result = self._estimate(cur, mask)
        self.prev = cur
        return result

    def _estimate(self, cur: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
        identity = (1.0, 0.0, 0.0)
        p0 = cv2.goodFeaturesToTrack(self.prev, maxCorners=CAMERA_MAX_CORNERS, qualityLevel=0.02,
                                     minDistance=6, blockSize=7, mask=mask)
        if p0 is None or len(p0) < CAMERA_MIN_POINTS:
            return identity
        p1, st1, _ = cv2.calcOpticalFlowPyrLK(self.prev, cur, p0, None, **self.lk)
        if p1 is None:
            return identity
        p0b, st2, _ = cv2.calcOpticalFlowPyrLK(cur, self.prev, p1, None, **self.lk)
        if p0b is None:
            return identity
        fb = np.linalg.norm((p0 - p0b).reshape(-1, 2), axis=1)
        good = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < CAMERA_FB_THRESH)
        if good.sum() < CAMERA_MIN_POINTS:
            return identity
        a = p0.reshape(-1, 2)[good] / self.sc          # retour en pixels plein format
        b = p1.reshape(-1, 2)[good] / self.sc

        s = self._scale_from_pairs(a, b)
        res = b - (self.center + s * (a - self.center))
        t = np.median(res, axis=0)

        # Raffinement : inliers autour de la translation médiane (seuil robuste type MAD)
        dev = np.linalg.norm(res - t, axis=1)
        thr = max(0.75, 3.0 * 1.4826 * float(np.median(dev)))
        inl = dev < thr
        if inl.sum() < CAMERA_MIN_INLIER_RATIO * len(a) or inl.sum() < CAMERA_MIN_POINTS:
            return identity                            # estimation peu fiable : on ne bouge pas
        s = self._scale_from_pairs(a[inl], b[inl])
        t = (b[inl] - (self.center + s * (a[inl] - self.center))).mean(axis=0)
        return float(s), float(t[0]), float(t[1])

    def _scale_from_pairs(self, a: np.ndarray, b: np.ndarray) -> float:
        """Échelle = médiane de d(b_i,b_j)/d(a_i,a_j) sur des paires de points éloignées."""
        n = len(a)
        if n < 6:
            return 1.0
        i = self.rng.integers(0, n, CAMERA_MAX_PAIRS)
        j = self.rng.integers(0, n, CAMERA_MAX_PAIRS)
        d0 = np.linalg.norm(a[i] - a[j], axis=1)
        keep = (i != j) & (d0 > CAMERA_MIN_PAIR_DIST * self.W)
        if keep.sum() < 10:
            return 1.0
        d1 = np.linalg.norm(b[i[keep]] - b[j[keep]], axis=1)
        s = float(np.median(d1 / d0[keep]))
        return float(np.clip(s, 1 - CAMERA_MAX_STEP_SCALE, 1 + CAMERA_MAX_STEP_SCALE))

    def step_matrix(self, s: float, tx: float, ty: float) -> np.ndarray:
        """Matrice 3x3 T : coordonnées de la frame précédente -> frame courante."""
        cx, cy = self.center
        return np.array([[s, 0.0, cx * (1 - s) + tx],
                         [0.0, s, cy * (1 - s) + ty],
                         [0.0, 0.0, 1.0]])


# =============================================================================
# 2) HOMOGRAPHIE DU TERRAIN : pixels (frame 0) -> mètres
# =============================================================================
class ViewTransformer:
    """Projette des points image (repère de la frame 0) en mètres sur le terrain."""

    def __init__(self, pixel_vertices, depth_m: float, width_m: float):
        src = np.float32(pixel_vertices)
        dst = np.float32([[0, depth_m], [0, 0], [width_m, 0], [width_m, depth_m]])
        self.matrix = cv2.getPerspectiveTransform(src, dst)

    def __call__(self, points) -> np.ndarray:
        """points : (N,2) -> (N,2) en mètres (NaN si non fini)."""
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2))
        out = cv2.perspectiveTransform(pts, self.matrix).reshape(-1, 2).astype(np.float64)
        out[~np.isfinite(out).all(axis=1)] = np.nan
        return out


# =============================================================================
# 3) MODÈLES DE DÉTECTION : COCO vs FINE-TUNÉS
# =============================================================================
_NAME_TO_ROLE = {
    "ball": "ball", "sports ball": "ball", "soccer ball": "ball", "football": "ball",
    "player": "player", "players": "player", "person": "player",
    "goalkeeper": "goalkeeper", "goal keeper": "goalkeeper", "goalie": "goalkeeper",
    "keeper": "goalkeeper",
    "referee": "referee", "referees": "referee", "ref": "referee",
}


@dataclass
class ModelProfile:
    class_roles: dict                 # id de classe -> "player" | "goalkeeper" | "referee" | "ball"
    generic: bool                     # modèle COCO-like (classe « person », pas de « player »)
    native_referee: bool
    native_goalkeeper: bool

    @property
    def class_ids(self) -> list[int]:
        return sorted(self.class_roles)

    def describe(self) -> str:
        kind = "générique (COCO-like)" if self.generic else "spécialisé (fine-tuné)"
        return (f"Modèle {kind} — arbitre natif : {'oui' if self.native_referee else 'non (couleur)'}, "
                f"gardien natif : {'oui' if self.native_goalkeeper else 'non'}")


def build_model_profile(names) -> ModelProfile:
    """Analyse les noms de classes du modèle et déduit son type et les rôles utilisables."""
    if not isinstance(names, dict):
        names = dict(enumerate(names))
    roles, raw = {}, set()
    for cid, name in names.items():
        n = str(name).lower().replace("_", " ").replace("-", " ").strip()
        raw.add(n)
        if n in _NAME_TO_ROLE:
            roles[int(cid)] = _NAME_TO_ROLE[n]
    specialised = raw & {"player", "players", "goalkeeper", "goal keeper", "goalie", "referee",
                         "referees", "keeper"}
    return ModelProfile(
        class_roles=roles,
        generic=("person" in raw) and not specialised,
        native_referee="referee" in roles.values(),
        native_goalkeeper="goalkeeper" in roles.values(),
    )


class PitchFilter:
    """Heuristique spatiale contre les faux positifs hors terrain (spectateurs, staff en tribune).

    1. Masque d'herbe (HSV) -> position Y du bord supérieur du terrain (lissée dans le temps).
       Une personne dont les PIEDS (bas de boîte) sont au-dessus de ce bord est en tribune.
    2. Contrôle local : il doit y avoir de l'herbe autour des pieds.
    3. Plan B (image sans herbe fiable) : simple seuil Y statique.
    """

    def __init__(self, W: int, H: int, enabled: bool):
        self.enabled, self.W, self.H = enabled, W, H
        self.sc = min(1.0, 480.0 / W)
        self.mask: np.ndarray | None = None
        self.top_y: float | None = None
        self.reliable = False

    def update(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        small = frame if self.sc >= 1.0 else cv2.resize(frame, None, fx=self.sc, fy=self.sc,
                                                        interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, GRASS_HSV_LO, GRASS_HSV_HI)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        self.mask = m > 0
        self.reliable = float(self.mask.mean()) > MIN_GRASS_RATIO
        if not self.reliable:
            return
        rows = np.convolve(self.mask.mean(axis=1), np.ones(5) / 5.0, mode="same")
        idx = np.flatnonzero(rows > FIELD_ROW_MIN_FRAC)
        if idx.size:
            top = idx[0] / self.sc
            self.top_y = top if self.top_y is None else 0.9 * self.top_y + 0.1 * top

    def on_pitch(self, box) -> bool:
        if not self.enabled:
            return True
        x1, y1, x2, y2 = [float(v) for v in box]
        top = self.top_y if (self.top_y is not None and self.reliable) else FIELD_TOP_FALLBACK_RATIO * self.H
        if y2 < top - 0.01 * self.H:                      # pieds au-dessus du terrain
            return False
        if not self.reliable or self.mask is None:
            return True
        bh, sc = y2 - y1, self.sc
        mh, mw = self.mask.shape
        ya, yb = max(0, int((y2 - 0.10 * bh) * sc)), min(mh, int((y2 + 0.06 * bh) * sc) + 1)
        xa, xb = max(0, int(x1 * sc)), min(mw, int(x2 * sc) + 1)
        if yb <= ya or xb <= xa:
            return True
        return float(self.mask[ya:yb, xa:xb].mean()) >= MIN_FOOT_GRASS


# =============================================================================
# 4) PASSE 1 : DÉTECTION + TRACKING + CAMÉRA + COULEURS + CANDIDATS BALLON
# =============================================================================
@dataclass
class Analysis:
    frames_objs: list = field(default_factory=list)   # par frame : [(tid, role_brut, (x1,y1,x2,y2))]
    ball_cands: list = field(default_factory=list)    # par frame : [(cx, cy, conf, w, h)]
    G: np.ndarray | None = None                       # (n,3,3) frame f -> repère frame 0
    pan: np.ndarray | None = None                     # mouvement caméra (px/frame)
    tilt: np.ndarray | None = None
    step_scale: np.ndarray | None = None              # échelle inter-frames
    zoom: np.ndarray | None = None                    # zoom cumulé vs frame 0
    color_samples: dict = field(default_factory=lambda: defaultdict(list))


def make_tracker(fps: float):
    """ByteTrack compatible avec les anciennes et nouvelles versions de supervision."""
    fr = max(1, int(round(fps)))
    try:
        return sv.ByteTrack(track_activation_threshold=PERSON_CONF, lost_track_buffer=30,
                            minimum_matching_threshold=0.8, frame_rate=fr)
    except TypeError:
        return sv.ByteTrack(track_thresh=PERSON_CONF, track_buffer=30, match_thresh=0.8, frame_rate=fr)


def jersey_color(frame: np.ndarray, box) -> np.ndarray | None:
    """Couleur du maillot : KMeans(2) sur le torse (centre, 15-55 % de la hauteur).
    Le cluster « herbe » (teinte verte saturée) est écarté ; sinon, le cluster majoritaire."""
    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    if bh < 24 or bw < 10:
        return None
    tx1, tx2 = max(0, x1 + int(0.2 * bw)), min(w_img, x2 - int(0.2 * bw))
    ty1, ty2 = max(0, y1 + int(0.15 * bh)), min(h_img, y1 + int(0.55 * bh))
    if tx2 - tx1 < 4 or ty2 - ty1 < 4:
        return None
    small = cv2.resize(frame[ty1:ty2, tx1:tx2], (8, 12), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1, 3).astype(np.float32)
    km = KMeans(n_clusters=2, n_init=2, random_state=0).fit(px)
    counts = np.bincount(km.labels_, minlength=2)
    hsv = cv2.cvtColor(np.clip(km.cluster_centers_, 0, 255).astype(np.uint8).reshape(1, 2, 3),
                       cv2.COLOR_BGR2HSV).reshape(2, 3)
    grass = [bool(35 <= h <= 90 and s >= 50) for h, s, _ in hsv]
    k = (1 if grass[0] else 0) if grass[0] != grass[1] else int(np.argmax(counts))
    return km.cluster_centers_[k]


def analyse_video(cap, model, profile: ModelProfile, args, fps, W, H, n_total) -> Analysis:
    """PASSE 1 : une seule lecture de la vidéo, tout ce qui est causal/local."""
    tracker = make_tracker(fps)
    pitch = PitchFilter(W, H, enabled=args.use_spatial_filter)
    cam: CameraMotionEstimator | None = None
    an = Analysis()
    G_cur = np.eye(3)
    G_list, pan, tilt, step = [], [], [], []
    last_sample: dict[int, int] = {}
    total = min(n_total, args.max_frames) if (args.max_frames and n_total) else (args.max_frames or n_total)
    t0, idx = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and idx >= args.max_frames):
            break

        # --- détection (un seul appel, seuil bas pour le ballon) ---
        pitch.update(frame)
        result = model.predict(frame, conf=min(args.conf, BALL_CONF), imgsz=args.imgsz,
                               classes=profile.class_ids, device=args.device, verbose=False)[0]
        det = sv.Detections.from_ultralytics(result)
        cands, people = [], sv.Detections.empty()

        if len(det) > 0:
            roles = np.array([profile.class_roles.get(int(c), "other") for c in det.class_id])
            conf = det.confidence

            # ballon : top-K candidats (la sélection finale se fait globalement, en post-traitement)
            ib = np.where((roles == "ball") & (conf >= BALL_CONF))[0]
            ib = ib[np.argsort(-conf[ib])][:BALL_TOP_K]
            for i in ib:
                x1, y1, x2, y2 = det.xyxy[i]
                cands.append((float((x1 + x2) / 2), float((y1 + y2) / 2), float(conf[i]),
                              float(x2 - x1), float(y2 - y1)))

            # personnes : confiance + taille + filtre spatial
            people = det[np.isin(roles, ["player", "goalkeeper", "referee"]) & (conf >= args.conf)]
            if len(people) > 0:
                keep = np.array([pitch.on_pitch(b) and (b[3] - b[1]) >= MIN_BOX_H_RATIO * H
                                 for b in people.xyxy], dtype=bool)
                people = people[keep]

        # --- caméra PTZ (les joueurs sont exclus du masque de suivi) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if cam is None:
            cam = CameraMotionEstimator(gray)
            s, tx, ty = 1.0, 0.0, 0.0
        else:
            s, tx, ty = cam.update(gray, people.xyxy if len(people) else ())
        G_cur = G_cur @ np.linalg.inv(cam.step_matrix(s, tx, ty))     # frame f -> frame 0
        G_list.append(G_cur.copy())
        pan.append(-tx)             # convention « mouvement de la caméra » = −mouvement de la scène
        tilt.append(-ty)
        step.append(s)

        # --- tracking ---
        tracked = tracker.update_with_detections(people)
        objs = []
        if len(tracked) > 0 and tracked.tracker_id is not None:
            for box, cid, tid in zip(tracked.xyxy, tracked.class_id, tracked.tracker_id):
                if tid is None or int(tid) < 0:
                    continue
                tid = int(tid)
                role = profile.class_roles.get(int(cid), "player")
                objs.append((tid, role, tuple(float(v) for v in box)))
                # échantillonnage de couleur (pas nécessaire pour un arbitre natif)
                if (role != "referee" and len(an.color_samples[tid]) < COLOR_SAMPLES_PER_TRACK
                        and idx - last_sample.get(tid, -10**9) >= COLOR_SAMPLE_EVERY):
                    col = jersey_color(frame, box)
                    if col is not None:
                        an.color_samples[tid].append(col)
                        last_sample[tid] = idx

        an.frames_objs.append(objs)
        an.ball_cands.append(cands)
        idx += 1
        progress(idx, total, t0, "Passe 1 (analyse)")

    print()
    an.G = np.array(G_list)
    an.pan, an.tilt, an.step_scale = np.array(pan), np.array(tilt), np.array(step)
    an.zoom = np.cumprod(an.step_scale) if len(step) else np.array([])
    return an


# =============================================================================
# 5) POST-TRAITEMENT GLOBAL (entre les deux passes)
# =============================================================================
def resolve_track_roles(frames_objs):
    """Rôle d'une piste = vote majoritaire sur toute sa durée (stabilité). + durée de présence."""
    votes: dict[int, Counter] = defaultdict(Counter)
    for objs in frames_objs:
        for tid, role, _ in objs:
            votes[tid][role] += 1
    track_role = {tid: c.most_common(1)[0][0] for tid, c in votes.items()}
    presence = {tid: sum(c.values()) for tid, c in votes.items()}
    return track_role, presence


def track_mean_ref_x(frames_objs, G) -> dict[int, float]:
    """Abscisse moyenne (repère stabilisé) des pieds de chaque piste."""
    acc = defaultdict(list)
    for f, objs in enumerate(frames_objs):
        for tid, _, (x1, _, x2, y2) in objs:
            acc[tid].append(float((G[f] @ np.array([(x1 + x2) / 2.0, y2, 1.0]))[0]))
    return {t: float(np.mean(v)) for t, v in acc.items()}


def assign_teams(color_samples, track_role, presence, mean_x, native_referee):
    """Équipes (et arbitre si le modèle ne le détecte pas) par KMeans en espace Lab.

    * Modèle avec arbitre natif : KMeans(2) sur les joueurs de champ.
    * Sinon : KMeans(3) ; si le plus petit cluster (poids = temps de présence) est très minoritaire
      ET nettement séparé des deux autres -> c'est l'arbitre. Sinon retour à KMeans(2).
    * Les gardiens (maillot distinct) rejoignent l'équipe dont les joueurs sont les plus proches
      en X (position moyenne dans le repère stabilisé).
    Retourne (team_of, team_colors_bgr, track_role_mis_à_jour).
    """
    roles = dict(track_role)
    bgr, lab = {}, {}
    for tid, samples in color_samples.items():
        if samples and roles.get(tid) == "player":
            bgr[tid] = np.median(np.array(samples), axis=0)
            lab[tid] = bgr_to_lab(bgr[tid])
    fit_ids = [t for t in lab if presence.get(t, 0) >= MIN_TRACK_LEN]
    if len(fit_ids) < 2:
        return {t: 1 for t in lab}, dict(DEFAULT_TEAM_COLORS), roles

    X = np.array([lab[t] for t in fit_ids])
    w = np.array([presence[t] for t in fit_ids], dtype=np.float64)
    centers, ref_idx, team_label = None, None, {0: 1, 1: 2}

    if not native_referee and len(fit_ids) >= REF_MIN_TRACKS:
        km3 = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X, sample_weight=w)
        wsum = np.array([w[km3.labels_ == k].sum() for k in range(3)])
        kmin = int(np.argmin(wsum))
        others = [k for k in range(3) if k != kmin]
        c = km3.cluster_centers_
        sep_ref = min(np.linalg.norm(c[kmin] - c[k]) for k in others)
        sep_teams = np.linalg.norm(c[others[0]] - c[others[1]])
        if (wsum[kmin] / wsum.sum() < REF_MAX_WEIGHT_FRAC and sep_ref > REF_MIN_LAB_SEP
                and sep_ref >= 0.6 * sep_teams):
            centers, ref_idx = c, kmin
            team_label = {others[0]: 1, others[1]: 2}
    if centers is None:
        centers = KMeans(n_clusters=2, n_init=10, random_state=0).fit(X, sample_weight=w).cluster_centers_

    # classification de TOUTES les pistes (y compris courtes) par centre le plus proche
    team_of = {}
    for tid, l in lab.items():
        k = int(np.argmin(np.linalg.norm(centers - l, axis=1)))
        if ref_idx is not None and k == ref_idx:
            roles[tid] = "referee"
        else:
            team_of[tid] = team_label[k]

    # couleurs d'affichage : moyenne pondérée des pistes de chaque équipe
    team_colors = {}
    for t in (1, 2):
        ids = [i for i in fit_ids if team_of.get(i) == t]
        team_colors[t] = (display_color(np.average([bgr[i] for i in ids], axis=0,
                                                   weights=[presence[i] for i in ids]))
                          if ids else DEFAULT_TEAM_COLORS[t])

    # gardiens -> équipe de même côté du terrain
    team_x = {t: float(np.mean([mean_x[i] for i in fit_ids if team_of.get(i) == t and i in mean_x]))
              for t in (1, 2) if any(team_of.get(i) == t and i in mean_x for i in fit_ids)}
    for tid, r in roles.items():
        if r == "goalkeeper" and len(team_x) == 2 and tid in mean_x:
            team_of[tid] = min(team_x, key=lambda t: abs(team_x[t] - mean_x[tid]))
    return team_of, team_colors, roles


def prune_inconsistent_ball_segments(obs: np.ndarray, conf: np.ndarray, max_jump: float) -> np.ndarray:
    """Supprime les segments de trajectoire incompatibles avec leurs voisins (faux positifs statiques).

    Un segment = suite de frames consécutives sans saut > max_jump. Deux segments voisins sont
    incompatibles si la distance entre la fin de l'un et le début de l'autre dépasse ce que le
    ballon pourrait parcourir pendant le trou ; on écarte alors celui de plus faible confiance cumulée.
    """
    obs, conf = obs.copy(), conf.copy()
    while True:
        idx = np.where(~np.isnan(obs[:, 0]))[0]
        if len(idx) < 2:
            break
        jumps = np.linalg.norm(np.diff(obs[idx], axis=0), axis=1)
        brk = np.where((np.diff(idx) > 1) | (jumps > max_jump))[0] + 1
        segs = np.split(idx, brk)
        dropped = False
        for a, b in zip(segs[:-1], segs[1:]):
            gap = b[0] - a[-1]
            if np.linalg.norm(obs[b[0]] - obs[a[-1]]) > 1.5 * max_jump * gap:
                loser = a if conf[a].sum() < conf[b].sum() else b
                obs[loser] = np.nan
                conf[loser] = 0.0
                dropped = True
                break
        if not dropped:
            break
    return obs


def stabilise_ball(ball_cands, G, fps, W):
    """Trajectoire du ballon en 2ᵉ passe (algorithme de Viterbi) dans le repère stabilisé.

    États par frame : un des candidats YOLO, ou « absent ». Coût = (1 − confiance) pour un
    candidat, coût fixe pour « absent », pénalité de saut si la vitesse dépasse le plausible.
    Puis : interpolation des petits trous (dans le repère stabilisé -> le panoramique de la caméra
    pendant le trou est pris en compte) et lissage gaussien.
    Retourne (ball_xy_image (n,2), ball_ref (n,2), observé (n,), diamètre_px).
    """
    n = len(ball_cands)
    max_jump = BALL_MAX_SPEED_PX_S_1080 * (W / 1920.0) / fps
    pos, conf, size = [], [], []
    for f, cands in enumerate(ball_cands):
        if cands:
            pts = np.array([[c[0], c[1], 1.0] for c in cands])
            pos.append((G[f] @ pts.T).T[:, :2])
            conf.append(np.array([c[2] for c in cands]))
            size.append(np.array([(c[3] + c[4]) / 2.0 for c in cands]))
        else:
            pos.append(np.zeros((0, 2)))
            conf.append(np.zeros(0))
            size.append(np.zeros(0))

    back, cost_prev = [], None
    for f in range(n):
        m = len(pos[f])
        emis = np.concatenate([(1.0 - conf[f]) * BALL_EMIS_W, [BALL_MISS_COST]])
        if f == 0:
            cost_prev = emis
            back.append(np.zeros(m + 1, dtype=int))
            continue
        pm = len(pos[f - 1])
        trans = np.zeros((pm + 1, m + 1))
        if pm and m:
            d = np.linalg.norm(pos[f - 1][:, None, :] - pos[f][None, :, :], axis=2) / max_jump
            trans[:pm, :m] = np.where(d <= 1.0, 0.3 * d, 1.0 + np.minimum(2.0, d - 1.0))
        trans[:pm, m] = 0.15            # candidat -> absent
        trans[pm, :m] = 0.15            # absent -> candidat
        total = cost_prev[:, None] + trans
        arg = np.argmin(total, axis=0)
        cost_prev = total[arg, np.arange(m + 1)] + emis
        back.append(arg)

    obs = np.full((n, 2), np.nan)
    obs_conf = np.zeros(n)
    sizes = []
    state = int(np.argmin(cost_prev))
    for f in range(n - 1, -1, -1):
        if state < len(pos[f]):
            obs[f] = pos[f][state]
            obs_conf[f] = conf[f][state]
            sizes.append(size[f][state])
        if f > 0:
            state = int(back[f][state])

    obs = prune_inconsistent_ball_segments(obs, obs_conf, max_jump)
    observed = ~np.isnan(obs[:, 0])
    ref = smooth_nan(interp_short_gaps(obs, BALL_MAX_GAP_FRAMES), BALL_SMOOTH_SIGMA)

    Ginv = np.linalg.inv(G)
    xy = np.full((n, 2), np.nan)
    ok = ~np.isnan(ref[:, 0])
    if ok.any():
        hom = np.concatenate([ref[ok], np.ones((ok.sum(), 1))], axis=1)          # (k,3)
        xy[ok] = np.einsum("kij,kj->ki", Ginv[ok], hom)[:, :2]
    diameter = float(np.median(sizes)) if sizes else 12.0
    return xy, ref, observed, diameter


def kinematics_from_positions(P: np.ndarray, fps: float):
    """Vitesse (km/h) et distance cumulée (m) à partir de positions en mètres (T,2, NaN = manquant).

    Chaîne : rejet Hampel -> comblement de trous courts -> lissage gaussien centré -> vitesse par
    différences centrées -> rejet des pics > vmax -> lissage de la vitesse -> zone morte -> intégration.
    """
    T = len(P)
    if T < 3:
        return np.zeros(T), np.zeros(T)
    P = hampel_reject(P, max(2, int(round(KIN_HAMPEL_HALF_SEC * fps))), KIN_HAMPEL_THR_M)
    P = interp_short_gaps(P, int(round(KIN_MAX_GAP_SEC * fps)))
    Ps = smooth_nan(P, KIN_POS_SIGMA_SEC * fps)

    v = np.gradient(Ps, axis=0) * fps                        # m/s (différences centrées)
    speed = np.linalg.norm(v, axis=1) * 3.6                  # km/h
    speed[speed > MAX_PLAUSIBLE_KMH * 1.25] = np.nan         # pics non physiques
    speed = interp_short_gaps(speed, int(round(KIN_MAX_GAP_SEC * fps)))
    speed = smooth_nan(speed, KIN_SPEED_SIGMA_SEC * fps)
    speed = np.clip(speed, 0.0, MAX_PLAUSIBLE_KMH)
    speed[speed < KIN_DEADBAND_KMH] = 0.0
    speed = np.nan_to_num(speed, nan=0.0)
    dist = np.cumsum(speed / 3.6 / fps)
    return speed, dist


def compute_kinematics(frames_objs, track_role, visible, G, transformer, fps):
    """Vitesse instantanée (km/h) et distance cumulée (m) par joueur et par frame.

    Position au sol = milieu du bas de la boîte, ramenée dans le repère stabilisé (G, qui inclut
    le zoom), puis projetée en mètres par l'homographie.
    """
    n = len(frames_objs)
    speed_out = [dict() for _ in range(n)]
    dist_out = [dict() for _ in range(n)]
    rows = defaultdict(list)
    for f, objs in enumerate(frames_objs):
        for tid, _, (x1, _, x2, y2) in objs:
            if tid in visible and track_role.get(tid) in FIELD_ROLES:
                p = G[f] @ np.array([(x1 + x2) / 2.0, y2, 1.0])
                rows[tid].append((f, p[0], p[1]))
    for tid, r in rows.items():
        arr = np.array(r)
        fr = arr[:, 0].astype(int)
        f0 = fr[0]
        P = np.full((fr[-1] - f0 + 1, 2), np.nan)
        P[fr - f0] = transformer(arr[:, 1:3])
        speed, dist = kinematics_from_positions(P, fps)
        for f in fr:
            speed_out[f][tid] = float(speed[f - f0])
            dist_out[f][tid] = float(dist[f - f0])
    return speed_out, dist_out


def compute_possession(frames_objs, track_role, visible, team_of, ball_xy):
    """Porteur du ballon avec hystérésis + cumul de possession par équipe.

    Un joueur est candidat si le ballon est à moins de POSSESSION_DIST_RATIO × sa hauteur de son
    bord inférieur (seuil adaptatif à la perspective). Le porteur ne change qu'après
    POSSESSION_MIN_FRAMES frames consécutives ; l'équipe reste la dernière en possession.
    """
    n = len(frames_objs)
    owner = [None] * n
    team_ctrl = np.zeros(n, dtype=int)
    current, pending, pend_n, lost, last_team = None, None, 0, 0, 0
    for f in range(n):
        bx, by = ball_xy[f]
        cand, best_d = None, np.inf
        if not np.isnan(bx):
            for tid, _, (x1, y1, x2, y2) in frames_objs[f]:
                if tid not in visible or track_role.get(tid) not in FIELD_ROLES:
                    continue
                h = y2 - y1
                d = float(np.hypot(max(x1 - bx, 0.0, bx - x2), by - (y2 - 0.08 * h)))
                if d < POSSESSION_DIST_RATIO * h and d < best_d:
                    cand, best_d = tid, d
        if cand is None:
            lost += 1
            pending, pend_n = None, 0
            if lost > POSSESSION_LOST_FRAMES:
                current = None
        else:
            lost = 0
            if cand == current:
                pending, pend_n = None, 0
            else:
                pend_n = pend_n + 1 if cand == pending else 1
                pending = cand
                if current is None or pend_n >= POSSESSION_MIN_FRAMES:
                    current, pending, pend_n = cand, None, 0
        owner[f] = current
        t = team_of.get(current, 0) if current is not None else 0
        if t:
            last_team = t
        team_ctrl[f] = last_team
    return owner, np.cumsum(team_ctrl == 1), np.cumsum(team_ctrl == 2)


def compute_leaders(speed, dist):
    """Pour chaque frame : (id, valeur) du meilleur sprint et de la plus grande distance jusqu'ici."""
    best_s, best_d = (None, 0.0), (None, 0.0)
    lead_s, lead_d = [], []
    for sp, di in zip(speed, dist):
        for tid, v in sp.items():
            if v > best_s[1]:
                best_s = (tid, v)
        for tid, v in di.items():
            if v > best_d[1]:
                best_d = (tid, v)
        lead_s.append(best_s)
        lead_d.append(best_d)
    return lead_s, lead_d


# =============================================================================
# 6) PASSE 2 : RENDU « BROADCAST SCOUTING FEED »
# =============================================================================
def rounded_rect(img, p1, p2, color, r):
    """Rectangle plein aux coins arrondis."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(1, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)


def blend_layer(frame, drawer, alpha):
    """Dessine sur une copie via drawer(layer) puis fusionne avec transparence."""
    layer = frame.copy()
    drawer(layer)
    return cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0)


def draw_marker(img, tip, color, s):
    """Petit triangle pointant vers le bas (ballon / porteur de balle)."""
    x, y = tip
    hw, h = int(8 * s) + 3, int(14 * s) + 6
    pts = np.array([[x, y], [x - hw, y - h], [x + hw, y - h]], dtype=np.int32)
    cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    cv2.polylines(img, [pts], True, (0, 0, 0), 1, cv2.LINE_AA)


def put_text(img, text, org, scale, thickness, color=(255, 255, 255)):
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)   # contour
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def render_frame(frame, f, ctx):
    H, W = frame.shape[:2]
    s = max(0.5, W / 1920.0)                          # facteur d'échelle de l'interface
    fs = max(0.38, 0.55 * s)
    th = 1 if s < 1.2 else 2
    pad, dot_r = int(6 * s) + 3, int(5 * s) + 2
    owner = ctx["owner"][f]
    track_role, team_of, team_colors = ctx["track_role"], ctx["team_of"], ctx["team_colors"]

    objs = [o for o in ctx["frames_objs"][f] if o[0] in ctx["visible"]]
    objs.sort(key=lambda o: o[2][3])                  # plan arrière -> avant

    fills, arcs, pills = [], [], []
    for tid, _, (x1, y1, x2, y2) in objs:
        role = track_role.get(tid, "player")
        xc, yb, bw = int((x1 + x2) / 2), int(y2), max(int(x2 - x1), 8)
        if role == "referee":
            color, text = REFEREE_COLOR, "REF"
        else:
            color = team_colors.get(team_of.get(tid, 0), NEUTRAL_COLOR)
            spd, dst = ctx["speed"][f].get(tid, 0.0), ctx["dist"][f].get(tid, 0.0)
            text = f"{'GK ' if role == 'goalkeeper' else ''}{tid} / {spd:.1f} km/h / {dst:.0f} m"
        axes = (int(bw * 0.70), max(int(bw * 0.70 * 0.35), 3))
        fills.append(((xc, yb), axes, color))
        arcs.append(((xc, yb), axes, color))

        (tw, thh), _ = cv2.getTextSize(text, FONT, fs, th)
        pw, ph = dot_r * 2 + pad * 3 + tw, thh + 2 * pad
        px1 = int(np.clip(xc - pw // 2, 2, max(2, W - pw - 2)))
        py2 = int(y1) - int(10 * s) - 2
        py1 = max(2, py2 - ph)
        pills.append(dict(box=((px1, py1), (px1 + pw, py1 + ph)), text=text, color=color,
                          thh=thh, has_ball=(tid == owner), xc=xc))

    # 1) remplissage translucide des ellipses au sol
    def _fill(layer):
        for c, a, col in fills:
            cv2.ellipse(layer, c, a, 0, 0, 360, col, -1, cv2.LINE_AA)
    frame = blend_layer(frame, _fill, 0.30)

    # 2) arcs nets (cercle de sélection tactique)
    for c, a, col in arcs:
        cv2.ellipse(frame, c, a, 0.0, -45, 235, col, max(2, int(2.5 * s)), cv2.LINE_AA)

    # 3) traînée du ballon : points du repère stabilisé reprojetés dans la vue courante
    ref = ctx["ball_ref"]
    k0 = max(0, f - BALL_TRAIL_FRAMES)
    Ginv_f = ctx["Ginv"][f]
    trail = []
    for k in range(k0, f + 1):
        if not np.isnan(ref[k, 0]):
            p = Ginv_f @ np.array([ref[k, 0], ref[k, 1], 1.0])
            trail.append((int(p[0]), int(p[1])))
    if len(trail) > 1:
        def _trail(layer):
            for i in range(1, len(trail)):
                if abs(trail[i][0] - trail[i - 1][0]) < W // 4:          # évite les sauts (coupures)
                    cv2.line(layer, trail[i - 1], trail[i], BALL_MARKER_COLOR,
                             max(1, int((1 + 3 * i / len(trail)) * s)), cv2.LINE_AA)
        frame = blend_layer(frame, _trail, 0.55)

    # 4) fonds sombres translucides : étiquettes + panneaux
    ui_w, ui_h = int(470 * s), int(88 * s)
    tl_p1 = (int(20 * s), int(20 * s))
    tl_p2 = (tl_p1[0] + ui_w, tl_p1[1] + ui_h)
    pn_w, pn_h = int(560 * s), int(190 * s)
    br_p1 = (W - pn_w - int(20 * s), H - pn_h - int(20 * s))
    br_p2 = (W - int(20 * s), H - int(20 * s))

    def _panels(layer):
        for p in pills:
            rounded_rect(layer, p["box"][0], p["box"][1], PANEL_COLOR, int(8 * s) + 2)
        rounded_rect(layer, tl_p1, tl_p2, PANEL_COLOR, int(10 * s) + 2)
        rounded_rect(layer, br_p1, br_p2, PANEL_COLOR, int(10 * s) + 2)
    frame = blend_layer(frame, _panels, 0.68)

    # 5) textes et accents nets
    for p in pills:
        (x1, y1), (x2, y2) = p["box"]
        cy = (y1 + y2) // 2
        cv2.circle(frame, (x1 + pad + dot_r, cy), dot_r, p["color"], -1, cv2.LINE_AA)
        cv2.putText(frame, p["text"], (x1 + dot_r * 2 + pad * 2, cy + p["thh"] // 2),
                    FONT, fs, (255, 255, 255), th, cv2.LINE_AA)
        if p["has_ball"]:
            draw_marker(frame, (p["xc"], y1 - 3), POSSESSION_MARKER_COLOR, s)

    # 6) ballon
    bx, by = ctx["ball_xy"][f]
    if not np.isnan(bx):
        r = max(3, int(ctx["ball_diam"] / 2))
        draw_marker(frame, (int(bx), int(by) - r - 3), BALL_MARKER_COLOR, s)

    # 7) HUD haut-gauche : caméra PTZ
    f2, t2 = max(0.45, 0.7 * s), 1 if s < 1.2 else 2
    put_text(frame, f"Camera Pan {ctx['pan'][f]:+.1f}px  Tilt {ctx['tilt'][f]:+.1f}px",
             (tl_p1[0] + int(16 * s), tl_p1[1] + int(34 * s)), f2, t2)
    put_text(frame, f"Camera Zoom x{ctx['zoom'][f]:.3f} ({100 * (ctx['step'][f] - 1):+.2f}%/f)",
             (tl_p1[0] + int(16 * s), tl_p1[1] + int(68 * s)), f2, t2)

    # 8) HUD bas-droite : possession + leaders
    c1, c2 = int(ctx["cum1"][f]), int(ctx["cum2"][f])
    tot = c1 + c2
    p1_pct = 100.0 * c1 / tot if tot else 0.0
    p2_pct = 100.0 * c2 / tot if tot else 0.0
    col1, col2 = team_colors.get(1, NEUTRAL_COLOR), team_colors.get(2, NEUTRAL_COLOR)
    lx = br_p1[0] + int(18 * s)
    for k, (pct, col) in enumerate(((p1_pct, col1), (p2_pct, col2)), start=1):
        y = br_p1[1] + int((34 + 34 * (k - 1)) * s)
        cv2.circle(frame, (lx + int(6 * s), y - int(8 * s)), int(7 * s) + 1, col, -1, cv2.LINE_AA)
        put_text(frame, f"Team {k} Ball Control: {pct:.2f}%", (lx + int(24 * s), y), f2, t2)
    ls, ld = ctx["lead_speed"][f], ctx["lead_dist"][f]
    put_text(frame, "Top speed: --" if ls[0] is None else f"Top speed: #{ls[0]}  {ls[1]:.1f} km/h",
             (lx, br_p1[1] + int(102 * s)), f2, t2)
    put_text(frame, "Top distance: --" if ld[0] is None else f"Top distance: #{ld[0]}  {ld[1]:.0f} m",
             (lx, br_p1[1] + int(136 * s)), f2, t2)
    bar_x1, bar_x2 = lx, br_p2[0] - int(18 * s)
    bar_y1, bar_y2 = br_p2[1] - int(28 * s), br_p2[1] - int(18 * s)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 70), -1)
    if tot:
        split = bar_x1 + int((bar_x2 - bar_x1) * p1_pct / 100.0)
        cv2.rectangle(frame, (bar_x1, bar_y1), (split, bar_y2), col1, -1)
        cv2.rectangle(frame, (split, bar_y1), (bar_x2, bar_y2), col2, -1)
    return frame


# =============================================================================
# 7) OUTIL DE CALIBRATION (optionnel)
# =============================================================================
def calibrate(video_path: str, depth_m: float, width_m: float):
    """Cliquez 4 points d'une zone aux dimensions réelles connues (marquages au sol, frame 0)."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("Impossible de lire la première frame.")
    labels = ["1: bas-gauche", "2: haut-gauche", "3: haut-droite", "4: bas-droite"]
    pts: list[tuple[int, int]] = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    try:
        cv2.namedWindow("calibration", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("calibration", on_mouse)
        while True:
            disp = frame.copy()
            for i, p in enumerate(pts):
                cv2.circle(disp, p, 8, (0, 0, 255), -1)
                cv2.putText(disp, labels[i], (p[0] + 10, p[1] - 10), FONT, 0.8, (0, 255, 255), 2)
            msg = labels[len(pts)] if len(pts) < 4 else "OK — appuyez sur une touche"
            cv2.putText(disp, f"Cliquez: {msg}  (ESC pour quitter)", (20, 40), FONT, 1.0, (0, 255, 0), 2)
            cv2.imshow("calibration", disp)
            key = cv2.waitKey(30) & 0xFF
            if key == 27 or (len(pts) == 4 and key != 255):
                break
        cv2.destroyAllWindows()
    except cv2.error:
        sys.exit("Affichage indisponible (OpenCV headless ?). Utilisez --court avec vos coordonnées.")
    if len(pts) == 4:
        flat = ",".join(f"{x},{y}" for x, y in pts)
        print("\nRelancez avec :")
        print(f'  python main_demo.py --input "{video_path}" --court "{flat}" '
              f"--depth-m {depth_m} --width-m {width_m}")


# =============================================================================
# 8) PROGRAMME PRINCIPAL
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser(description="Football Analytics — Broadcast Scouting Feed")
    ap.add_argument("--input", default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Modèle YOLO (.pt) : COCO ou fine-tuné")
    ap.add_argument("--conf", type=float, default=PERSON_CONF, help="Confiance joueurs/arbitres")
    ap.add_argument("--imgsz", type=int, default=DETECTION_IMGSZ)
    ap.add_argument("--device", default=None, help="ex. cpu, cuda:0, mps")
    ap.add_argument("--spatial-filter", choices=["auto", "on", "off"], default="auto",
                    help="Filtre anti-spectateurs (auto = actif pour les modèles COCO)")
    ap.add_argument("--max-frames", type=int, default=0, help="Limiter le nombre de frames (test)")
    ap.add_argument("--court", default=None,
                    help='4 points "x1,y1,...,x4,y4" (frame 0) : bas-gauche, haut-gauche, haut-droite, bas-droite')
    ap.add_argument("--depth-m", type=float, default=DEFAULT_DEPTH_M, help="Arête bas-gauche -> haut-gauche (m)")
    ap.add_argument("--width-m", type=float, default=DEFAULT_WIDTH_M, help="Arête haut-gauche -> haut-droite (m)")
    ap.add_argument("--calibrate", action="store_true", help="Outil interactif de calibration du terrain")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.calibrate:
        calibrate(args.input, args.depth_m, args.width_m)
        return

    # --- Entrée vidéo ---
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Impossible d'ouvrir la vidéo : {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Vidéo : {W}x{H} @ {fps:.2f} fps, {n_total} frames")

    # --- Calibration terrain (dans le repère de la frame 0) ---
    if args.court:
        vals = [float(v) for v in args.court.split(",")]
        if len(vals) != 8:
            sys.exit("--court attend exactement 8 nombres.")
        court_px = np.array(vals, dtype=np.float32).reshape(4, 2)
    else:
        court_px = np.array(DEFAULT_COURT_PIXELS, dtype=np.float32) * np.array([W / 1920.0, H / 1080.0],
                                                                                dtype=np.float32)
        print("⚠ Calibration par défaut : vitesses/distances approximatives. "
              "Utilisez --calibrate pour des valeurs fiables.")
    transformer = ViewTransformer(court_px, args.depth_m, args.width_m)

    # --- Modèle : détection automatique COCO vs fine-tuné ---
    model = YOLO(args.model)
    profile = build_model_profile(model.names)
    if "ball" not in profile.class_roles.values() or not (set(profile.class_roles.values()) & {"player"}):
        sys.exit(f"Le modèle doit contenir des classes joueur/personne et ballon. Classes : {model.names}")
    args.use_spatial_filter = (args.spatial_filter == "on"
                               or (args.spatial_filter == "auto" and profile.generic))
    print(profile.describe() + f" — filtre spatial : {'actif' if args.use_spatial_filter else 'inactif'}")

    # --- PASSE 1 : analyse ---
    an = analyse_video(cap, model, profile, args, fps, W, H, n_total)
    cap.release()
    n = len(an.frames_objs)
    if n == 0:
        sys.exit("Aucune frame lue.")

    # --- Post-traitement global ---
    print("Post-traitement (rôles, équipes, ballon, cinématique, possession)...")
    Ginv = np.linalg.inv(an.G)
    track_role, presence = resolve_track_roles(an.frames_objs)
    mean_x = track_mean_ref_x(an.frames_objs, an.G)
    team_of, team_colors, track_role = assign_teams(an.color_samples, track_role, presence, mean_x,
                                                    profile.native_referee)
    visible = {t for t, c in presence.items() if c >= MIN_TRACK_LEN}
    ball_xy, ball_ref, ball_obs, ball_diam = stabilise_ball(an.ball_cands, an.G, fps, W)
    speed, dist = compute_kinematics(an.frames_objs, track_role, visible, an.G, transformer, fps)
    owner, cum1, cum2 = compute_possession(an.frames_objs, track_role, visible, team_of, ball_xy)
    lead_speed, lead_dist = compute_leaders(speed, dist)
    n_ref = sum(1 for t in visible if track_role.get(t) == "referee")
    print(f"  {len(visible)} pistes retenues, {n_ref} arbitre(s), "
          f"ballon détecté/suivi sur {100 * np.mean(~np.isnan(ball_xy[:, 0])):.0f}% des frames")

    ctx = dict(frames_objs=an.frames_objs, track_role=track_role, team_of=team_of,
               team_colors=team_colors, visible=visible, speed=speed, dist=dist, owner=owner,
               cum1=cum1, cum2=cum2, lead_speed=lead_speed, lead_dist=lead_dist,
               ball_xy=ball_xy, ball_ref=ball_ref, ball_diam=ball_diam, Ginv=Ginv,
               pan=an.pan, tilt=an.tilt, zoom=an.zoom, step=an.step_scale)

    # --- PASSE 2 : rendu + écriture vidéo ---
    cap = cv2.VideoCapture(args.input)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    if not writer.isOpened():
        sys.exit("Impossible d'initialiser l'écriture de la vidéo de sortie.")
    t0 = time.time()
    for f in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(render_frame(frame, f, ctx))
        progress(f + 1, n, t0, "Passe 2 (rendu)  ")
    print()
    cap.release()
    writer.release()
    print(f"✔ Terminé : {args.output}")


if __name__ == "__main__":
    main()