import json
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory, GripperCommand
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from std_msgs.msg import String

SEARCH_TURN_SPEED = 0.15
ALIGN_GAIN = 0.0015
MAX_ALIGN_SPEED = 0.2
CENTER_X = 640.0
CENTER_TOLERANCE_PX = 60.0
APPROACH_REALIGN_PX = 120.0
APPROACH_LINEAR_SPEED = 0.05
TARGET_BBOX_WIDTH = 180.0
DETECTION_TIMEOUT_SEC = 0.75

GRIPPER_OPEN = 0.01
GRIPPER_CLOSE = -0.01


HOME_POSE = [0.0, 0.0, 0.0, 0.0]
EXTEND_POSE = [0.0, 1.0, -0.1, 0.4]
PLACE_POSE = [0.0, 1.0, -0.1, 0.4]


class BottleAutonomyController(Node):
    def __init__(self):

        # Initialize and Define Node name
        super().__init__('bottle_autonomy_controller')

        self.arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4']
        
        # State Variables        
        self.current_j1 = 0.0
        self.current_j2 = 0.0
        self.current_j3 = 0.0
        self.current_j4 = 0.0

        self.latest_bottle = None
        self.last_detection_time = 0.0
        self.last_bottle_error_x = 0.0
        self.state= "SEARCH"
        self.sequence = []
        self.sequence_deadline = 0.0
        self.return_end_time = 0.0
        self.base_step = ""
        self.base_step_end_time = 0.0
        self.align_settle_until = 0.0
        self.settle_loss_deadline = 0.0

        # Action Clients
        self.arm_action_client = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.gripper_action_client = ActionClient(self, GripperCommand, '/gripper_controller/gripper_cmd')

        # Publishers and Subscribers
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.joint_state_sub = self.create_subscription(JointState, 'joint_states', self.joint_state_callback, 10)
        self.yolo_sub = self.create_subscription(
            String, '/yolo/detections_json', self.yolo_callback, 10
        )

        self.timer = self.create_timer(0.1, self.run_loop)
        self.get_logger().info("Bottle Autonomy Controller has been started.")

    def joint_state_callback(self, msg):
        if 'joint1' in msg.name:
            idx = msg.name.index('joint1')
            self.current_j1 = msg.position[idx]

        if 'joint2' in msg.name:
            idx = msg.name.index('joint2')
            self.current_j2 = msg.position[idx]

        if 'joint3' in msg.name:
            idx = msg.name.index('joint3')
            self.current_j3 = msg.position[idx]

        if 'joint4' in msg.name:
            idx = msg.name.index('joint4')
            self.current_j4 = msg.position[idx]

    def yolo_callback(self, msg):
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Failed to decode YOLO JSON: {exc}")
            return
    
        detections = data.get("detection", data.get("detections", []))
        bottle = self.find_best_bottle(detections)
        if bottle is None:
            return
        
        self.latest_bottle = bottle
        self.last_bottle_error_x = bottle["bbox"]["cx"] - CENTER_X
        self.last_detection_time = time.monotonic()

    def find_best_bottle(self, detections):
        bottle_detections = [
            det for det in detections if det.get("class_name") == "bottle"
        ]
        if not bottle_detections:
            return None
    
        return max(
            bottle_detections,
            key = lambda det: det.get("confidence", 0.0) * det.get("bbox", {}).get("w", 0.0),
        )
    
    def bottle_is_fresh(self):
        return (
            self.latest_bottle is not None
            and (time.monotonic() - self.last_detection_time) < DETECTION_TIMEOUT_SEC
        )

    def run_loop(self):
        if self.state == "SEARCH":
            self.run_search()
        elif self.state == "ALIGN":
            self.run_align()
        elif self.state == "APPROACH":
            self.run_approach()
        elif self.state == "ALIGN_SETTLE":
            self.run_align_settle()
        elif self.state == "PICKUP":
            self.run_sequence()
        elif self.state == "RETURN":
            self.run_return()
        elif self.state == "PLACE":
            self.run_sequence()
        elif self.state == "DONE":
            self.stop_base()

    def run_search(self):
        if self.bottle_is_fresh():
            self.state = "ALIGN"
            self.get_logger().info("Bottle detected, switching to ALIGN")
            return

        if self.last_bottle_error_x > 0.0:
            search_angular = -SEARCH_TURN_SPEED
        else:
            search_angular = SEARCH_TURN_SPEED

        self.publish_base_cmd(0.0, search_angular)
    
    def run_align(self):
        if not self.bottle_is_fresh():
            self.state = "SEARCH"
            self.get_logger().info("Bottle lost, switching to SEARCH")
            return
        
        bbox = self.latest_bottle["bbox"]
        error_x = bbox["cx"] - CENTER_X
        if abs(error_x) <= CENTER_TOLERANCE_PX:
            self.stop_base()
            self.align_settle_until = time.monotonic() + 0.2
            self.settle_loss_deadline = self.align_settle_until + DETECTION_TIMEOUT_SEC
            self.state = "ALIGN_SETTLE"
            self.get_logger().info("Bottle centered, settling before APPROACH")
            return
        
        angular_cmd = -ALIGN_GAIN * error_x
        if abs(error_x) < 160.0:
            angular_cmd *= 0.6

        angular_cmd = max(-MAX_ALIGN_SPEED, min(MAX_ALIGN_SPEED, angular_cmd))
        self.publish_base_cmd(0.0, angular_cmd)

    def run_align_settle(self):
        self.stop_base()

        if not self.bottle_is_fresh():
            if time.monotonic() < self.settle_loss_deadline:
                return

            self.state = "SEARCH"
            self.get_logger().info("Bottle lost after settle grace period, switching to SEARCH")
            return

        if time.monotonic() >= self.align_settle_until:
            self.state = "APPROACH"
            self.get_logger().info("Settle complete, switching to APPROACH")

    def run_approach(self):
        if not self.bottle_is_fresh():
            self.state = "SEARCH"
            self.get_logger().info("Bottle lost during approach, switching to SEARCH")
            return
        
        bbox = self.latest_bottle["bbox"]
        error_x = bbox["cx"] - CENTER_X
    
        if abs(error_x) > APPROACH_REALIGN_PX:
            self.state = "ALIGN"
            return
        
        if bbox["w"] >= TARGET_BBOX_WIDTH:
            self.stop_base()
            self.start_pickup_sequence()
            return
        
        approach_angular = max(-0.1, min(0.1, -0.001 * error_x))
        self.publish_base_cmd(APPROACH_LINEAR_SPEED, approach_angular)

    def start_pickup_sequence(self):
        self.sequence = [
            ("arm", HOME_POSE, 2.0, 2.5),
            ("gripper", GRIPPER_OPEN, 0.0, 1.0),
            ("arm", EXTEND_POSE, 2.0, 2.5),
            ("gripper", GRIPPER_CLOSE, 0.0, 1.0),
            ("arm", HOME_POSE, 2.0, 2.5),
        ]

        self.sequence_deadline = 0.0
        self.state = "PICKUP"
        self.get_logger().info("Starting pickup sequence")

    def start_place_sequence(self):
        self.sequence = [
            ("arm", PLACE_POSE, 2.0, 2.5),
            ("gripper", GRIPPER_OPEN, 0.0, 1.0),
            ("arm", HOME_POSE, 2.0, 2.5),
        ]
        self.sequence_deadline = 0.0
        self.state = "PLACE"
        self.get_logger().info("Starting place sequence")

    def run_sequence(self):
        now = time.monotonic()
        if now < self.sequence_deadline:
            return
        
        if not self.sequence:
            if self.state == "PICKUP":
                self.start_return_sequence()
            elif self.state == "PLACE":
                self.state = "DONE"
                self.get_logger().info("Bottle task complete")
            return
        
        action, payload, duration_sec, settle_sec = self.sequence.pop(0)
        if action == "arm":
            self.send_arm_goal(payload, duration_sec)
        elif action == "gripper":
            self.send_gripper_goal(payload)

        self.sequence_deadline = now + settle_sec

    def start_return_sequence(self):
        self.state = "RETURN"
        self.base_step = "TURN"
        self.base_step_end_time = time.monotonic() + 3.0
        self.get_logger().info("Starting simple timed return to base")
    
    def run_return(self):
        now = time.monotonic()

        if self.base_step == "TURN":
            if now < self.base_step_end_time:
                self.publish_base_cmd(0.0, 0.4)
                return
            
            self.base_step = "DRIVE"
            self.base_step_end_time = now + 4.0
        
        if self.base_step == "DRIVE":
            if now < self.base_step_end_time:
                self.publish_base_cmd(0.08, 0.0)
                return
        
        self.stop_base()
        self.base_step = ""
        self.start_place_sequence()

    def publish_base_cmd(self, linear_x, angular_z):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)

    def stop_base(self):
        self.publish_base_cmd(0.0, 0.0)

    def send_arm_goal(self, positions, duration_sec):
        if not self.arm_action_client.server_is_ready():
            self.get_logger().info("Arm action server not available")
            return
        
        self.get_logger().info(f"Sending arm goal: {positions}")

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = self.arm_joint_names
        point = JointTrajectoryPoint(
            positions=positions, 
            time_from_start=Duration(sec=int(duration_sec), nanosec=int((duration_sec % 1) * 1e9))
        )
        goal.trajectory.points.append(point)
        self.arm_action_client.send_goal_async(goal)

    def send_gripper_goal(self, position):
        if not self.gripper_action_client.server_is_ready():
            self.get_logger().info("Gripper action server not available")
            return
            
        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = 1.0
        self.gripper_action_client.send_goal_async(goal)

    def destroy_node(self):
        self.stop_base()
        super().destroy_node()
        
def main(args=None):
    rclpy.init(args=args)
    node = BottleAutonomyController()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()
