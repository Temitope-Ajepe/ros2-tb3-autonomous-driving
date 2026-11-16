"""Lane detection with plain OpenCV/NumPy (no ROS imports -> unit-testable offline).

Pipeline
  1. Bird's-eye warp of a trapezoid ROI (roi_src) to a warp_w x warp_h image.
  2. HSV thresholds for the yellow (left) and white (right) lines.
  3. Sliding windows + polynomial fit for each line.
  4. Sanity checks, then lane centre at a look-ahead row.
  5. error in [-1, 1]; > 0 means the lane centre is to the RIGHT of the robot.

If only one line is visible, the centre is inferred using the last measured lane width.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    yellow_lo: Tuple[int, int, int] = (15, 80, 80)
    yellow_hi: Tuple[int, int, int] = (40, 255, 255)
    white_lo: Tuple[int, int, int] = (0, 0, 170)
    white_hi: Tuple[int, int, int] = (179, 70, 255)
    # trapezoid (fractions of the input image): TL, TR, BR, BL as x,y pairs
    roi_src: Tuple[float, ...] = (0.28, 0.55, 0.72, 0.55, 1.0, 0.95, 0.0, 0.95)
    warp_w: int = 320
    warp_h: int = 240
    lane_width_px: float = 160.0   # yellow<->white distance in the warped image
    n_windows: int = 10
    window_margin: int = 35
    min_pix: int = 30
    lookahead_y: float = 0.55      # 0 = top of warped image, 1 = bottom (robot)


@dataclass
class LineFit:
    poly: np.ndarray               # x = poly[0]*y^2 + poly[1]*y + poly[2]
    n_pts: int
    hits: int
    windows: List[Tuple[int, int, int, int]]

    def x_at(self, y: float) -> float:
        return float(np.polyval(self.poly, y))


@dataclass
class Detection:
    error: Optional[float]
    confidence: float              # 1.0 both lines, 0.5 one line, 0.0 none
    center_x: Optional[float]
    lane_width: float
    measured_width: Optional[float]
    debug: Optional[np.ndarray] = None


class LaneDetector:
    def __init__(self, cfg: Optional[DetectorConfig] = None):
        self.cfg = cfg or DetectorConfig()
        self.lane_width = float(self.cfg.lane_width_px)
        self._M_key = None
        self._M = None
        self._src_px = None

    # ------------------------------------------------------------------ warp
    def _get_M(self, w, h):
        c = self.cfg
        key = (w, h, tuple(c.roi_src), c.warp_w, c.warp_h)
        if key != self._M_key:
            r = c.roi_src
            src = np.float32([[r[0] * w, r[1] * h], [r[2] * w, r[3] * h],
                              [r[4] * w, r[5] * h], [r[6] * w, r[7] * h]])
            dst = np.float32([[0, 0], [c.warp_w, 0], [c.warp_w, c.warp_h], [0, c.warp_h]])
            self._M = cv2.getPerspectiveTransform(src, dst)
            self._src_px = src
            self._M_key = key
        return self._M

    # ----------------------------------------------------------------- masks
    def _masks(self, warped):
        c = self.cfg
        hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
        ym = cv2.inRange(hsv, np.array(c.yellow_lo, np.uint8), np.array(c.yellow_hi, np.uint8))
        wm = cv2.inRange(hsv, np.array(c.white_lo, np.uint8), np.array(c.white_hi, np.uint8))
        k = np.ones((3, 3), np.uint8)
        return (cv2.morphologyEx(ym, cv2.MORPH_OPEN, k),
                cv2.morphologyEx(wm, cv2.MORPH_OPEN, k))

    # ------------------------------------------------------------ line fitting
    def _find_base(self, mask) -> Optional[int]:
        H, _ = mask.shape
        hist = (mask[int(H * 0.6):, :] > 0).sum(axis=0).astype(np.float32)
        k = max(3, 2 * self.cfg.window_margin)
        score = np.convolve(hist, np.ones(k, np.float32), mode='same')
        p = int(np.argmax(score))
        return p if score[p] >= self.cfg.min_pix else None

    def _fit_line(self, mask) -> Optional[LineFit]:
        c = self.cfg
        H, W = mask.shape
        x = self._find_base(mask)
        if x is None:
            return None
        nzy, nzx = np.nonzero(mask)
        win_h = max(1, H // c.n_windows)
        shift = 0
        xs, ys, windows, hits = [], [], [], 0
        for i in range(c.n_windows):
            y1 = H - i * win_h
            y0 = max(0, y1 - win_h)
            x0, x1 = max(0, x - c.window_margin), min(W, x + c.window_margin)
            windows.append((x0, y0, x1, y1))
            sel = (nzy >= y0) & (nzy < y1) & (nzx >= x0) & (nzx < x1)
            if np.count_nonzero(sel) >= c.min_pix:
                new_x = int(nzx[sel].mean())
                shift = new_x - x
                x = new_x
                xs.append(nzx[sel])
                ys.append(nzy[sel])
                hits += 1
            else:  # keep following the previous direction through gaps
                x = int(np.clip(x + shift, 0, W - 1))
                shift = int(shift * 0.7)
        if hits < 2:
            return None
        px = np.concatenate(xs).astype(np.float64)
        py = np.concatenate(ys).astype(np.float64)
        if hits >= 4:
            poly = np.polyfit(py, px, 2)
        else:
            p1 = np.polyfit(py, px, 1)
            poly = np.array([0.0, p1[0], p1[1]])
        return LineFit(poly, len(px), hits, windows)

    # ---------------------------------------------------------------- detect
    def detect(self, bgr: np.ndarray, draw: bool = True) -> Detection:
        c = self.cfg
        h, w = bgr.shape[:2]
        M = self._get_M(w, h)
        W, H = c.warp_w, c.warp_h
        warped = cv2.warpPerspective(bgr, M, (W, H), flags=cv2.INTER_LINEAR)
        ymask, wmask = self._masks(warped)

        yel = self._fit_line(ymask)
        whi = self._fit_line(wmask)
        y_look = c.lookahead_y * (H - 1)
        xl = float(np.clip(yel.x_at(y_look), -W, 2 * W)) if yel else None
        xr = float(np.clip(whi.x_at(y_look), -W, 2 * W)) if whi else None

        measured = None
        if xl is not None and xr is not None:
            measured = xr - xl
            if measured < 0.4 * self.lane_width or measured > 1.8 * self.lane_width:
                # implausible pair: keep the better-supported line only
                if yel.n_pts >= whi.n_pts:
                    whi, xr = None, None
                else:
                    yel, xl = None, None
            else:
                self.lane_width = 0.9 * self.lane_width + 0.1 * measured

        lw = self.lane_width
        if xl is not None and xr is not None:
            cx, conf = (xl + xr) / 2.0, 1.0
        elif xl is not None:
            cx, conf = xl + lw / 2.0, 0.5
        elif xr is not None:
            cx, conf = xr - lw / 2.0, 0.5
        else:
            cx, conf = None, 0.0

        err = None if cx is None else float(np.clip((cx - W / 2.0) / (W / 2.0), -1.0, 1.0))
        dbg = self._draw(bgr, warped, ymask, wmask, yel, whi, cx, y_look, err, conf, measured) if draw else None
        return Detection(err, conf, cx, lw, measured, dbg)

    # ------------------------------------------------------------------ draw
    def _draw(self, bgr, warped, ymask, wmask, yel, whi, cx, y_look, err, conf, measured):
        W, H = self.cfg.warp_w, self.cfg.warp_h
        vis = (warped * 0.4).astype(np.uint8)
        vis[ymask > 0] = (0, 255, 255)
        vis[wmask > 0] = (255, 255, 255)
        ys = np.arange(H)
        for fit, col in ((yel, (0, 160, 160)), (whi, (170, 170, 170))):
            if fit is None:
                continue
            for (x0, y0, x1, y1) in fit.windows:
                cv2.rectangle(vis, (x0, y0), (x1, y1), col, 1)
            pts = np.stack([np.polyval(fit.poly, ys), ys], axis=1)
            pts = np.clip(pts, -1000, 1000).astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (0, 0, 255), 2)
        cv2.line(vis, (W // 2, 0), (W // 2, H), (255, 255, 0), 1)
        cv2.line(vis, (0, int(y_look)), (W, int(y_look)), (255, 0, 255), 1)
        if cx is not None:
            cv2.circle(vis, (int(np.clip(cx, 0, W - 1)), int(y_look)), 6, (0, 255, 0), -1)
        txt = 'err=%s conf=%.1f' % ('--' if err is None else '%+.2f' % err, conf)
        cv2.putText(vis, txt, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        if measured is not None:
            cv2.putText(vis, 'width=%.0f' % measured, (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        h, w = bgr.shape[:2]
        orig = bgr.copy()
        cv2.polylines(orig, [self._src_px.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        orig = cv2.resize(orig, (max(1, int(w * H / h)), H))
        return np.hstack([orig, vis])
