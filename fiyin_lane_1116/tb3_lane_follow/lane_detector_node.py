"""ROS2 wrapper around LaneDetector.

Sub : image_topic (CompressedImage by default, or Image if compressed:=false)
Pub : /lane/error            geometry_msgs/Vector3Stamped  x = error [-1,1] (+ = lane centre to the right),
                                                           y = confidence (1 both lines, 0.5 one, 0 none)
      /lane/debug/compressed sensor_msgs/CompressedImage   (only encoded while someone subscribes)
All detector parameters can be changed live with `ros2 param set /lane_detector <name> <value>`.
"""
import rclpy
from geometry_msgs.msg import Vector3Stamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CompressedImage, Image

import cv2

from .lane_detection import DetectorConfig, LaneDetector
from .ros_utils import decode_compressed, decode_raw

CFG_PARAMS = ['yellow_lo', 'yellow_hi', 'white_lo', 'white_hi', 'roi_src', 'warp_w', 'warp_h',
              'lane_width_px', 'n_windows', 'window_margin', 'min_pix', 'lookahead_y']


def _plain(v):
    return tuple(v) if hasattr(v, '__len__') and not isinstance(v, str) else v


class LaneDetectorNode(Node):
    def __init__(self):
        super().__init__('lane_detector')
        d = DetectorConfig()
        self.declare_parameter('image_topic', '/camera/image_raw/compressed')
        self.declare_parameter('compressed', True)
        self.declare_parameter('publish_debug', True)
        for name in CFG_PARAMS:
            default = getattr(d, name)
            self.declare_parameter(name, list(default) if isinstance(default, tuple) else default)

        cfg = DetectorConfig(**{n: _plain(self.get_parameter(n).value) for n in CFG_PARAMS})
        self.detector = LaneDetector(cfg)
        self.publish_debug = self.get_parameter('publish_debug').value
        self.compressed = self.get_parameter('compressed').value
        self.add_on_set_parameters_callback(self._on_params)

        topic = self.get_parameter('image_topic').value
        msg_type = CompressedImage if self.compressed else Image
        self.create_subscription(msg_type, topic, self._on_image, qos_profile_sensor_data)
        self.pub_err = self.create_publisher(Vector3Stamped, '/lane/error', 10)
        self.pub_dbg = self.create_publisher(CompressedImage, '/lane/debug/compressed', 1)
        self.get_logger().info(f'Listening on {topic} ({"compressed" if self.compressed else "raw"})')

    def _on_params(self, params):
        for p in params:
            if p.name in CFG_PARAMS:
                setattr(self.detector.cfg, p.name, _plain(p.value))
                if p.name == 'lane_width_px':
                    self.detector.lane_width = float(p.value)
            elif p.name == 'publish_debug':
                self.publish_debug = p.value
        return SetParametersResult(successful=True)

    def _on_image(self, msg):
        frame = decode_compressed(msg) if self.compressed else decode_raw(msg)
        if frame is None:
            self.get_logger().warn('Could not decode image (unsupported encoding?)', throttle_duration_sec=5.0)
            return
        want_dbg = self.publish_debug and self.pub_dbg.get_subscription_count() > 0
        det = self.detector.detect(frame, draw=want_dbg)

        out = Vector3Stamped()
        out.header = msg.header
        out.vector.x = 0.0 if det.error is None else float(det.error)
        out.vector.y = float(det.confidence)
        self.pub_err.publish(out)

        if want_dbg and det.debug is not None:
            ok, buf = cv2.imencode('.jpg', det.debug, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok:
                dbg = CompressedImage()
                dbg.header = msg.header
                dbg.format = 'jpeg'
                dbg.data = buf.tobytes()
                self.pub_dbg.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
