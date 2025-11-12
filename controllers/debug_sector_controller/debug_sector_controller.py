"""
debug_sector_controller.py - Debug sector scoring without robot movement
"""

from controller import Robot, Motor, Supervisor
import math

# Constants (same as main controller)
MAX_SPEED = 6.0
NULL_SPEED = 0.0
MIN_SPEED = 1.0

ARRIVAL_DISTANCE = 0.3  # meters
UWB_SEPARATION = 0.300  # meters between left and right UWB tags

# Hysteresis thresholds for smooth mode transitions
DISTANCE_DIFF_ENTER = 0.25  # Enter turn-in-place when diff > 0.25m
DISTANCE_DIFF_EXIT = 0.05  # Exit arc mode (go straight) when diff < 0.05m
DISTANCE_DIFF_ARC_REENTER = 0.1  # Re-enter arc mode when diff > 0.1m

# Back sensor check
BACK_CHECK_INTERVAL = 30  # Check back sensor every 30 steps
BACK_DISTANCE_MARGIN = 0.05  # Back UWB must be at least 0.05m closer

# LiDAR obstacle avoidance settings
OBSTACLE_THRESHOLD = 0.6  # meters - stop/avoid if obstacle closer
SAFE_CLEARANCE = 1.0  # meters - preferred safe distance
FRONT_SECTOR_ANGLE = 45  # degrees - front sector for obstacle detection
SIDE_SECTOR_ANGLE = 90  # degrees - side sector boundaries

# Sector scoring weights
FREE_SPACE_WEIGHT = 0.6  # Weight for free space score (0-1)
GOAL_ANGLE_WEIGHT = 0.4  # Weight for goal angle score (0-1)

# Waypoint configuration
WAYPOINT_DWELL_TIME = 2.0  # seconds to wait at each waypoint

# GLOBAL WAYPOINT SEQUENCE - Edit this to change route!
WAYPOINT_SEQUENCE = [
    "anchorA",
    "anchorB",
    "anchorC",
    "anchorD",
]


class ControlMode:
    """Control modes for robot navigation"""
    TURN_IN_PLACE = 0
    ARC_STEERING = 1
    FORWARD = 2
    TURN_AROUND = 3
    AVOIDING_OBSTACLE = 4


class DebugSectorController:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize motors (but don't move)
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.stop_robot()
        
        # Initialize LiDAR
        self.lidar = self.robot.getDevice("LDS-01")
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()
            print("LiDAR enabled successfully")
        else:
            print("Warning: LiDAR not found!")
        
        # Get robot and waypoint nodes
        self.robot_node = self.robot.getSelf()
        self.load_waypoints()
        
        # Get UWB nodes
        self.uwb_left_node = self.robot.getFromDef("uwb_left")
        self.uwb_right_node = self.robot.getFromDef("uwb_right")
        self.uwb_back_node = self.robot.getFromDef("uwb_back")
        
        self.have_uwb_nodes = (self.uwb_left_node is not None and 
                               self.uwb_right_node is not None)
        self.have_back_uwb = (self.uwb_back_node is not None)
        
        if not self.have_uwb_nodes:
            print("Warning: UWB nodes not found. Using robot pose fallback.")
        else:
            print("UWB nodes found. Using supervisor positions.")
        
        if not self.have_back_uwb:
            print("Warning: Back UWB node not found.")
        else:
            print("Back UWB node found.")
        
        print(f"\n>>> Debug mode: Sector scoring analysis <<<\n")
    
    def load_waypoints(self):
        """Load all waypoints from the sequence"""
        self.num_waypoints = len(WAYPOINT_SEQUENCE)
        
        if self.num_waypoints == 0:
            print("Error: No waypoints defined in WAYPOINT_SEQUENCE.")
            return False
        
        print(f"Found {self.num_waypoints} waypoints in sequence.")
        
        self.waypoint_nodes = []
        print("\n=== Loading Waypoints ===")
        for i, waypoint_name in enumerate(WAYPOINT_SEQUENCE):
            node = self.robot.getFromDef(waypoint_name)
            if node is None:
                print(f"Error: Waypoint '{waypoint_name}' not found (check DEF names).")
                return False
            self.waypoint_nodes.append(node)
            pos = node.getPosition()
            print(f"  [{i}] {waypoint_name}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print("=========================\n")
        return True
    
    def stop_robot(self):
        """Stop both motors"""
        self.left_motor.setVelocity(NULL_SPEED)
        self.right_motor.setVelocity(NULL_SPEED)
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        return math.sqrt(dx*dx + dy*dy + dz*dz)
    
    def vec_dot(self, a, b):
        """Dot product of two vectors"""
        return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    
    def get_uwb_positions(self):
        """Get positions of left, right, and back UWB tags"""
        if self.have_uwb_nodes:
            left_pos = self.uwb_left_node.getPosition()
            right_pos = self.uwb_right_node.getPosition()
            back_pos = self.uwb_back_node.getPosition() if self.have_back_uwb else None
        else:
            # Fallback: compute from robot pose
            robot_pos = self.robot_node.getPosition()
            robot_orient = self.robot_node.getOrientation()
            
            # Right axis = elements [0, 3, 6] of orientation matrix
            right_axis = [robot_orient[0], robot_orient[3], robot_orient[6]]
            sep = UWB_SEPARATION / 2.0
            
            left_pos = [robot_pos[0] - right_axis[0] * sep,
                       robot_pos[1] - right_axis[1] * sep,
                       robot_pos[2] - right_axis[2] * sep]
            right_pos = [robot_pos[0] + right_axis[0] * sep,
                        robot_pos[1] + right_axis[1] * sep,
                        robot_pos[2] + right_axis[2] * sep]
            back_pos = None
        
        return left_pos, right_pos, back_pos
    
    def calculate_turn_angle(self, left_pos, right_pos, anchor_pos):
        """Calculate signed turn angle to anchor using law of cosines"""
        d_left = self.calculate_distance(left_pos, anchor_pos)
        d_right = self.calculate_distance(right_pos, anchor_pos)
        L = UWB_SEPARATION
        
        cos_turn_angle = (d_left**2 + L**2 - d_right**2) / (2 * d_left * L)
        cos_turn_angle = max(-1.0, min(1.0, cos_turn_angle))  # clamp to avoid domain error
        
        angle_at_left = math.acos(cos_turn_angle)
        turn_angle = math.pi / 2 - angle_at_left
        
        # Determine sign based on which side is closer
        if d_left < d_right:
            turn_angle = abs(turn_angle)  # left side, positive turn
        else:
            turn_angle = -abs(turn_angle)  # right side, negative turn
        
        # Adjust for back direction using back UWB if available
        if self.have_back_uwb and self.uwb_back_node:
            back_pos = self.uwb_back_node.getPosition()
            d_back = self.calculate_distance(back_pos, anchor_pos)
            min_front = min(d_left, d_right)
            if d_back < min_front - BACK_DISTANCE_MARGIN:
                # Anchor is behind, flip the angle
                if turn_angle > 0:
                    turn_angle = math.pi - turn_angle
                else:
                    turn_angle = -math.pi - turn_angle
        
        return turn_angle
    
    def check_obstacles(self, goal_angle_deg):
        """
        Advanced obstacle detection using LiDAR with sector-based path selection.
        Returns: (obstacle_detected, best_navigation_angle_deg)
        """
        if not self.lidar:
            return False, goal_angle_deg
        
        ranges = self.lidar.getRangeImage()
        if not ranges:
            return False, goal_angle_deg
        
        num_points = len(ranges)
        
        # Ensure we have 360 points (or resample if different)
        if num_points != 360:
            # Simple resampling to 360 points
            resampled_ranges = []
            for i in range(360):
                idx = int(i * num_points / 360)
                resampled_ranges.append(ranges[idx])
            ranges = resampled_ranges
            num_points = 360
        
        # Step 1: Create binary obstacle map (0 = free, 1 = obstacle)
        binary_map = []
        obstacle_detected = False
        for i in range(num_points):
            if ranges[i] < OBSTACLE_THRESHOLD and ranges[i] > 0.1:
                binary_map.append(1)  # Obstacle
                obstacle_detected = True
            else:
                binary_map.append(0)  # Free space
        
        if not obstacle_detected:
            return False, goal_angle_deg
        
        # Step 2: Divide into 36 sectors (10 degrees each)
        num_sectors = 36
        points_per_sector = num_points // num_sectors  # 10 points per sector
        
        sector_scores = []
        for sector_idx in range(num_sectors):
            start_idx = sector_idx * points_per_sector
            end_idx = start_idx + points_per_sector
            
            # Count free space (zeros) in this sector
            free_space_score = 0
            for i in range(start_idx, end_idx):
                if binary_map[i] == 0:
                    free_space_score += 1
            
            # Calculate mid-angle of this sector (in degrees)
            sector_mid_angle = (sector_idx * 10) + 5  # e.g., sector 0 = 5°, sector 1 = 15°, etc.
            
            # Calculate goal angle score based on proximity to goal
            angle_diff = abs(sector_mid_angle - goal_angle_deg)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Score based on sector distance from goal
            if angle_diff <= 5:  # Goal is within this sector
                goal_angle_score = 10
            elif angle_diff <= 15:  # Goal is in adjacent sector
                goal_angle_score = 8
            elif angle_diff <= 25:  # Goal is 2 sectors away
                goal_angle_score = 6
            elif angle_diff <= 35:  # Goal is 3 sectors away
                goal_angle_score = 4
            elif angle_diff <= 45:  # Goal is 4 sectors away
                goal_angle_score = 2
            else:
                goal_angle_score = 0
            
            # Total score = weighted combination of free_space_score and goal_angle_score
            total_score = (free_space_score * FREE_SPACE_WEIGHT + 
                          goal_angle_score * GOAL_ANGLE_WEIGHT)
            
            sector_scores.append({
                'sector_idx': sector_idx,
                'mid_angle': sector_mid_angle,
                'free_space_score': free_space_score,
                'goal_angle_score': goal_angle_score,
                'total_score': total_score
            })
        
        # Step 3: Find the sector with the highest total score
        best_sector = max(sector_scores, key=lambda s: s['total_score'])
        best_navigation_angle = best_sector['mid_angle']
        
        # Debug output
        print(f"\n=== Obstacle Avoidance Debug ===")
        print(f"Original Goal Angle: {goal_angle_deg:.1f}°")
        print(f"Best Sector: {best_sector['sector_idx']} (angle: {best_navigation_angle:.1f}°)")
        print(f"  Free Space Score: {best_sector['free_space_score']} × {FREE_SPACE_WEIGHT} = {best_sector['free_space_score'] * FREE_SPACE_WEIGHT:.2f}")
        print(f"  Goal Angle Score: {best_sector['goal_angle_score']} × {GOAL_ANGLE_WEIGHT} = {best_sector['goal_angle_score'] * GOAL_ANGLE_WEIGHT:.2f}")
        print(f"  Total Weighted Score: {best_sector['total_score']:.2f}")
        
        # Print all sector scores in tabular format
        print("\nSector Scores:")
        for i in range(0, num_sectors, 10):
            indices = [str(j+1) for j in range(i, min(i+10, num_sectors))]
            scores = [f"{sector_scores[j]['total_score']:.1f}" for j in range(i, min(i+10, num_sectors))]
            print(" | ".join(indices))
            print(" | ".join(scores))
            print()
        
        return True, best_navigation_angle
    
    def run(self):
        """Debug run - analyze sectors without moving"""
        step_count = 0
        while self.robot.step(self.timestep) != -1 and step_count < 10:  # Run for 10 steps
            # Get current target waypoint
            current_anchor = self.waypoint_nodes[0]  # Just use first waypoint
            anchor_pos = current_anchor.getPosition()
            
            # Get UWB positions
            left_pos, right_pos, back_pos = self.get_uwb_positions()
            
            # Calculate goal angle
            turn_angle = self.calculate_turn_angle(left_pos, right_pos, anchor_pos)
            goal_angle = (math.degrees(turn_angle) + 360) % 360
            
            # Check for obstacles and print sector analysis
            obstacle_detected, avoidance_angle_deg = self.check_obstacles(goal_angle)
            
            if not obstacle_detected:
                print("No obstacles detected.")
            
            # Stop the robot
            self.stop_robot()
            
            step_count += 1
        
        print("Debug complete. Robot stopped.")


# Main entry point
if __name__ == "__main__":
    controller = DebugSectorController()
    controller.run()