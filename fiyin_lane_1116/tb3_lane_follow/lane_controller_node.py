"""PD lane-following controller.

Sub : /lane/error  (geometry_msgs/Vector3Stamped: x = error, y = confidence)
Pub : cmd_vel_topic (geometry_msgs/Twist, Humble TurtleBot3 uses Twist)

angular.z = -(kp*e + kd*de/dt)   (e > 0 means lane centre is to the right -> turn clockwise)
linear.x  shrinks with |e| and when only one line is visible.
Safety: if no valid detection for `lost_timeout` seconds, the robot stops. Zero cmd on shutdown.
Toggle at runtime: ros2 param set /lane_controller enabled false
"""
import rclpy
from geometry_msgs.msg import Twist, Vector3Stamped
from rclpy.node import Node


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


class LaneController(Node):
    def __init__(self):
        super().__init__('lane_controller')
        for name, default in [
            ('cmd_vel_topic', '/cmd_vel'), ('enabled', True),
            ('max_linear', 0.10), ('min_linear', 0.04), ('slow_gain', 0.8),
            ('kp', 1.5), ('kd', 0.2), ('max_angular', 1.5),
            ('error_smoothing', 0.6), ('one_line_speed_factor', 0.7),
            ('lost_timeout', 0.6), ('control_rate', 20.0),
        ]:
            self.declare_parameter(name, default)

        self.pub = self.create_publisher(Twist, self.get_parameter('cmd_vel_topic').value, 10)
        self.create_subscription(Vector3Stamped, '/lane/error', self._on_error, 10)
        self.create_timer(1.0 / self.get_parameter('control_rate').value, self._tick)

        self.err_f = 0.0
        self.d_err = 0.0
        self.conf = 0.0
        self.last_good = None
        self.last_t = None
        self.state = 'init'

    def _p(self, n):
        return self.get_parameter(n).value

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_error(self, msg):
        now = self._now()
        self.conf = msg.vector.y
        if self.conf <= 0.0:
            return
        a = clamp(self._p('error_smoothing'), 0.0, 1.0)
        prev = self.err_f
        first = self.last_good is None
        self.err_f = msg.vector.x if first else a * msg.vector.x + (1 - a) * self.err_f
        if self.last_t is not None:
            dt = now - self.last_t
            if 1e-3 < dt < 0.5:
                self.d_err = (self.err_f - prev) / dt
        self.last_t = now
        self.last_good = now

    def _set_state(self, s):
        if s != self.state:
            self.get_logger().info(f'state: {self.state} -> {s}')
            self.state = s

    def _tick(self):
        cmd = Twist()
        now = self._now()
        if not self._p('enabled'):
            self._set_state('disabled')
        elif self.last_good is None or now - self.last_good > self._p('lost_timeout'):
            self._set_state('lost (stopped)')
        else:
            self._set_state('tracking')
            w = -(self._p('kp') * self.err_f + self._p('kd') * self.d_err)
            cmd.angular.z = clamp(w, -self._p('max_angular'), self._p('max_angular'))
            v = self._p('max_linear') * (1.0 - self._p('slow_gain') * min(1.0, abs(self.err_f)))
            if self.conf < 1.0:
                v *= self._p('one_line_speed_factor')
            cmd.linear.x = clamp(v, 0.0, self._p('max_linear')) if v < self._p('min_linear') * 0.5 \
                else clamp(v, self._p('min_linear'), self._p('max_linear'))
        self.pub.publish(cmd)

    def stop(self):
        try:
            self.pub.publish(Twist())
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LaneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
