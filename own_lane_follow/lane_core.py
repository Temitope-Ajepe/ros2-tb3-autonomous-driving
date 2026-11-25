import cv2
import numpy as np

ROI_TOP = 0.55   # the road area starts at 55% of the picture height
# (row as a fraction of the road area, lane width at that row as a fraction of the picture width)
# rows are tried from far ahead to close; the first row that finds a line is used
ROWS = [(0.50, 0.600), (0.70, 0.741), (0.82, 0.825), (0.30, 0.453)]
MIN_PIX = 3      # ignore a line made of fewer pixels than this
TRAP_INSET = 0.10   # trapezoid: at the top of the road area it is 10% narrower on each side

_last_width = {}   # lane width in pixels, last measured with BOTH lines seen, per row

YELLOW_LO, YELLOW_HI = (20, 100, 100), (35, 255, 255)
WHITE_LO, WHITE_HI = (0, 0, 170), (179, 60, 255)


def _masks(img):
    """Yellow and white masks of the road area, limited to the trapezoid."""
    h, w = img.shape[:2]
    top = int(h * ROI_TOP)
    roi = img[top:, :]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = cv2.inRange(hsv, np.array(YELLOW_LO), np.array(YELLOW_HI))
    white = cv2.inRange(hsv, np.array(WHITE_LO), np.array(WHITE_HI))
    roi_h = roi.shape[0]
    pts = np.array([[int(TRAP_INSET * w), 0], [int((1 - TRAP_INSET) * w), 0],
                    [w - 1, roi_h - 1], [0, roi_h - 1]], np.int32)
    region = np.zeros(yellow.shape, np.uint8)
    cv2.fillPoly(region, [pts], 255)
    return top, roi_h, pts, cv2.bitwise_and(yellow, region), cv2.bitwise_and(white, region)


def _measure(img, update=True):
    """Return (error, how, info). info holds what debug_view needs to draw."""
    h, w = img.shape[:2]
    top, roi_h, pts, yellow, white = _masks(img)
    info = {"top": top, "pts": pts, "row": None, "y": None, "wh": None, "centre": None}

    for frac, width_frac in ROWS:
        row = int(roi_h * frac)
        lane_width = _last_width.get(frac, width_frac * w)
        yellow_x = np.where(yellow[row] > 0)[0]
        white_x = np.where(white[row] > 0)[0]

        y = yellow_x.mean() if len(yellow_x) >= MIN_PIX else None
        # the white line is on the RIGHT: ignore white that is not clearly right of the yellow
        left_limit = (y + 0.5 * lane_width) if y is not None else 0.5 * w
        white_x = white_x[white_x > left_limit]
        wh = white_x.mean() if len(white_x) >= MIN_PIX else None

        if y is not None and wh is not None:
            centre, how = (y + wh) / 2, f"both lines @{frac}"
            if update:
                nominal = width_frac * w
                measured = min(1.6 * nominal, max(0.8 * nominal, wh - y))
                _last_width[frac] = 0.7 * _last_width.get(frac, measured) + 0.3 * measured
        elif y is not None:
            centre, how = y + lane_width / 2, f"yellow only @{frac}"
        elif wh is not None:
            centre, how = wh - lane_width / 2, f"white only @{frac}"
        else:
            continue
        error = max(-1.0, min(1.0, (centre - w / 2) / (w / 2)))
        info.update(row=top + row, y=y, wh=wh, centre=centre)
        return error, how, info

    return None, "no line", info


def lane_error(img):
    """Return (error, how). error is between -1 and +1, or None if no line is seen."""
    error, how, _ = _measure(img)
    return error, how


def steer(error, kp=1.0, speed=0.05):
    return speed, -kp * error


def debug_view(img):
    """Return a copy of img with the detector's view drawn on it (for rqt and the video)."""
    error, how, info = _measure(img, update=False)
    out = img.copy()
    h, w = out.shape[:2]
    shifted = info["pts"] + np.array([0, info["top"]], np.int32)
    cv2.polylines(out, [shifted], True, (0, 255, 0), 1)            # the trapezoid we look at
    r = info["row"]
    if r is not None:
        if info["y"] is not None:
            cv2.circle(out, (int(info["y"]), r), 4, (0, 128, 255), -1)    # yellow line point
        if info["wh"] is not None:
            cv2.circle(out, (int(info["wh"]), r), 4, (255, 255, 0), -1)   # white line point
        cv2.circle(out, (int(info["centre"]), r), 5, (0, 0, 255), -1)     # lane centre = target
        cv2.line(out, (w // 2, r - 8), (w // 2, r + 8), (255, 255, 255), 1)  # picture centre
    label = f"err {error:+.2f}  {how}" if error is not None else "no line"
    cv2.putText(out, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return out
