"""Image-message decoding without cv_bridge (fewer dependencies on the lab PC)."""
import cv2
import numpy as np


def decode_compressed(msg):
    arr = np.frombuffer(msg.data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def decode_raw(msg):
    enc = msg.encoding.lower()
    ch = {'bgr8': 3, 'rgb8': 3, 'mono8': 1}.get(enc)
    if ch is None:
        return None
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    img = buf[:, :msg.width * ch].reshape(msg.height, msg.width, ch)
    if enc == 'rgb8':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    elif enc == 'mono8':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(img)
