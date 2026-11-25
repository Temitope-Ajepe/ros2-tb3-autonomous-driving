import rclpy
from rclpy.signals import SignalHandlerOptions
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

from lane_core import lane_error, steer, debug_view


class LaneNode(Node):
    def __init__(self):
        super().__init__("my_lane_node")
        self.declare_parameter("image_topic", "/camera/image")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("enabled", False)      # robot can only move when this is True
        self.was_enabled = False
        self.lost = 0
        self.declare_parameter("kp", 1.5)
        self.declare_parameter("speed", 0.05)

        topic = self.get_parameter("image_topic").value
        self.bridge = CvBridge()
        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        self.err_pub = self.create_publisher(Float32, "/my_lane/error", 10)
        self.dbg_pub = self.create_publisher(Image, "/my_lane/debug", 1)
        self.cmd_pub = self.create_publisher(Twist, self.get_parameter("cmd_vel_topic").value, 10)
        self.get_logger().info(f"listening on {topic}, enabled=False")

    def stop_robot(self):
        for _ in range(3):
            self.cmd_pub.publish(Twist())             # a Twist with all zeros = stop

    def on_image(self, msg):
        enabled = self.get_parameter("enabled").value
        if self.was_enabled and not enabled:
            self.stop_robot()              # switched off -> send zero speed
        self.was_enabled = enabled
        img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        error, how = lane_error(img)

        if error is None:
            self.lost += 1
            if enabled and self.lost > 15:      # about 0.5 s without any line
                self.get_logger().warn("no line seen - stopping", throttle_duration_sec=1.0)
                self.stop_robot()
            return
        self.lost = 0

        out = Float32()
        out.data = float(error)
        self.err_pub.publish(out)
        self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(debug_view(img), "bgr8"))

        if enabled:
            speed, angular = steer(error,
                                   kp=self.get_parameter("kp").value,
                                   speed=self.get_parameter("speed").value)
            cmd = Twist()
            cmd.linear.x = speed
            cmd.angular.z = angular
            self.cmd_pub.publish(cmd)

        self.get_logger().info(f"error={error:+.3f} ({how}) enabled={enabled}",
                               throttle_duration_sec=0.5)


def main():
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = LaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()           # always send a stop before quitting
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
