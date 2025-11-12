"""
uwb_steer_controller.py - Simplified multi-waypoint navigation with LiDAR obstacle avoidance
"""

from controller import Robot, Motor, Supervisor
import math

# Constants
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
OBSTACLE_THRESHOLD_BASE = 0.8  # meters - base threshold
OBSTACLE_THRESHOLD_MIN = 0.5   # meters - minimum threshold
OBSTACLE_THRESHOLD = OBSTACLE_THRESHOLD_BASE  # Current dynamic threshold
SAFE_CLEARANCE = 1.0  # meters - preferred safe distance
FRONT_SECTOR_ANGLE = 45  # degrees - front sector for obstacle detection
SIDE_SECTOR_ANGLE = 90  # degrees - side sector boundaries

# Sector scoring weights
FREE_SPACE_WEIGHT = 0.8 # Weight for free space score (0-1)
GOAL_ANGLE_WEIGHT = 0.2  # Weight for goal angle score (0-1)

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
    SAFE_MODE = 5


class UWBNavigationController:
    def __init__(self):
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # Initialize motors
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
        
        # State variables
        self.current_mode = ControlMode.TURN_IN_PLACE
        self.current_waypoint_index = 0
        self.waypoint_reached = False
        self.waypoint_arrival_time = 0.0
        
        # Tracking variables
        self.step_counter = 0
        self.steps_per_second = 1000 // self.timestep
        self.prev_avg_dist = 0.0
        self.stuck_counter = 0
        self.stuck_threshold = self.steps_per_second * 3
        
        # Turn around tracking
        self.turn_around_counter = 0
        self.turn_around_duration = self.steps_per_second * 2
        
        # Obstacle avoidance
        self.pre_avoidance_mode = ControlMode.FORWARD
        self.obstacle_threshold = OBSTACLE_THRESHOLD_BASE  # Dynamic threshold
        self.current_speed_reduction = 0.0  # Current speed reduction (0.0 to 3.0)
        self.target_speed_reduction = 0.0   # Target speed reduction
        self.speed_change_rate = 0.05  # Rate of speed change per step
        
        # Safe mode settings
        self.safe_mode_threshold = 120  # Number of obstacles to activate safe mode
        
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
        
        print(f"\n>>> Starting navigation to waypoint 0: {WAYPOINT_SEQUENCE[0]} <<<\n")
    
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
    
    def move_forward(self, speed=MAX_SPEED):
        """Move forward at specified speed"""
        adjusted_speed = max(MIN_SPEED, speed - self.current_speed_reduction)
        self.left_motor.setVelocity(adjusted_speed)
        self.right_motor.setVelocity(adjusted_speed)
    
    def turn_in_place_smart(self, d_left, d_right):
        """Smart turn-in-place: back away the wheel closer to anchor"""
        turn_speed = MAX_SPEED * 0.7
        
        if d_left < d_right:
            self.left_motor.setVelocity(-turn_speed)
            self.right_motor.setVelocity(turn_speed)
        else:
            self.left_motor.setVelocity(turn_speed)
            self.right_motor.setVelocity(-turn_speed)
    
    def turn_around(self, d_left, d_right):
        """Turn 180 degrees in place"""
        turn_speed = MAX_SPEED * 0.6
        
        if d_left < d_right:
            self.left_motor.setVelocity(-turn_speed)
            self.right_motor.setVelocity(turn_speed)
        else:
            self.left_motor.setVelocity(turn_speed)
            self.right_motor.setVelocity(-turn_speed)
    
    def differential_steer_arc(self, goal_angle_deg):
        """Differential drive steering with arc"""
        # Convert goal_angle_deg to signed turn_angle_rad
        if goal_angle_deg <= 180:
            turn_angle_rad = math.radians(goal_angle_deg)
        else:
            turn_angle_rad = math.radians(goal_angle_deg - 360)
        
        base_speed = MAX_SPEED - self.current_speed_reduction
        speed_diff = turn_angle_rad * 4.0
        
        # Clamp speed difference
        speed_diff = max(-base_speed + MIN_SPEED, 
                        min(base_speed - MIN_SPEED, speed_diff))
        
        if turn_angle_rad > 0:
            # Turn left: slow down left wheel
            left_speed = max(MIN_SPEED, base_speed - abs(speed_diff))
            right_speed = base_speed
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
        else:
            # Turn right: slow down right wheel
            left_speed = base_speed
            right_speed = max(MIN_SPEED, base_speed - abs(speed_diff))
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
    
    def check_obstacles(self, goal_angle_deg):
        """
        Advanced obstacle detection using LiDAR with sector-based path selection.
        Dynamically adjusts obstacle threshold based on number of front obstacles.
        Returns: (obstacle_detected, best_navigation_angle_deg, total_obstacles)
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
        
        # FLIP: Map LiDAR indices so that index 180 -> 0°
        # Index 180 = front (0°), 181 = right (359°), 179 = left (1°)
        flipped_ranges = [0] * num_points
        for i in range(num_points):
            flipped_angle = (180 - i) % 360
            flipped_ranges[flipped_angle] = ranges[i]
        ranges = flipped_ranges
        
        # Count obstacles in front sector (0-30° and 330-360°)
        front_obstacles = 0
        for i in range(31):  # 0 to 30 degrees
            if ranges[i] < OBSTACLE_THRESHOLD_BASE and ranges[i] > 0.1:
                front_obstacles += 1
        for i in range(330, 360):  # 330 to 360 degrees
            if ranges[i] < OBSTACLE_THRESHOLD_BASE and ranges[i] > 0.1:
                front_obstacles += 1
        
        # Dynamically adjust obstacle threshold based on front obstacles
        old_threshold = self.obstacle_threshold
        if front_obstacles >= 3:
            # 3 or more obstacles: decrease by 0.1 (more cautious)
            self.obstacle_threshold = max(OBSTACLE_THRESHOLD_MIN, self.obstacle_threshold - 0.1)
        elif front_obstacles >= 2:
            # 2 obstacles: decrease by 0.2 (very cautious)
            self.obstacle_threshold = max(OBSTACLE_THRESHOLD_MIN, OBSTACLE_THRESHOLD_BASE - 0.2)
        else:
            # 0 or 1 obstacle: return to base threshold
            self.obstacle_threshold = OBSTACLE_THRESHOLD_BASE
        
        if old_threshold != self.obstacle_threshold:
            print(f"[THRESHOLD ADJUST] {old_threshold:.2f}m -> {self.obstacle_threshold:.2f}m (front obstacles: {front_obstacles})")
        
        # Adjust sector scoring weights based on number of obstacles
        current_free_space_weight = FREE_SPACE_WEIGHT
        current_goal_angle_weight = GOAL_ANGLE_WEIGHT
        if front_obstacles >= 4:
            current_free_space_weight += 0.05
            current_goal_angle_weight -= 0.05
            if self.step_counter % (self.steps_per_second * 2) == 0:
                print(f"[WEIGHT ADJUST] Obstacles >= 4: FREE_SPACE_WEIGHT={current_free_space_weight:.2f}, GOAL_ANGLE_WEIGHT={current_goal_angle_weight:.2f}")
        
        # Update target speed reduction based on front obstacles
        if front_obstacles >= 2:
            self.target_speed_reduction = 3.0  # Reduce speed by 3.0
        else:
            self.target_speed_reduction = 0.0  # Normal speed
        
        # Gradually adjust current speed reduction towards target
        if self.current_speed_reduction < self.target_speed_reduction:
            self.current_speed_reduction = min(self.target_speed_reduction, 
                                              self.current_speed_reduction + self.speed_change_rate)
        elif self.current_speed_reduction > self.target_speed_reduction:
            self.current_speed_reduction = max(self.target_speed_reduction,
                                              self.current_speed_reduction - self.speed_change_rate)
        
        if abs(self.current_speed_reduction - old_threshold) > 0.1 and self.step_counter % 10 == 0:
            print(f"[SPEED REDUCTION] {self.current_speed_reduction:.2f}m/s")
        
        # Step 1: Create binary obstacle map using dynamic threshold
        binary_map = []
        obstacle_detected = False
        for i in range(num_points):
            if ranges[i] < self.obstacle_threshold and ranges[i] > 0.1:
                binary_map.append(1)  # Obstacle
                obstacle_detected = True
            else:
                binary_map.append(0)  # Free space
        
        if not obstacle_detected:
            return False, goal_angle_deg, 0
        
        # Step 2: Divide into 36 sectors (10 degrees each)
        num_sectors = 36
        points_per_sector = num_points // num_sectors  # 10 points per sector
        
        # First pass: calculate raw free space counts for all sectors
        raw_free_spaces = []
        for sector_idx in range(num_sectors):
            start_idx = sector_idx * points_per_sector
            end_idx = start_idx + points_per_sector
            
            free_count = 0
            for i in range(start_idx, end_idx):
                if binary_map[i] == 0:
                    free_count += 1
            raw_free_spaces.append(free_count)
        
        # Calculate total obstacles
        total_obstacles = sum(binary_map)
        
        # Second pass: calculate weighted free space scores considering neighbors
        sector_scores = []
        for sector_idx in range(num_sectors):
            # Weighted score: current + neighbors with wrap-around
            # i+1 = left neighbor, i-1 = right neighbor (in user's terminology)
            current = raw_free_spaces[sector_idx]
            left1 = raw_free_spaces[(sector_idx + 1) % num_sectors]    # i+1
            right1 = raw_free_spaces[(sector_idx - 1) % num_sectors]   # i-1  
            left2 = raw_free_spaces[(sector_idx + 2) % num_sectors]    # i+2
            right2 = raw_free_spaces[(sector_idx - 2) % num_sectors]   # i-2
            left3 = raw_free_spaces[(sector_idx + 3) % num_sectors]    # i+3
            right3 = raw_free_spaces[(sector_idx - 3) % num_sectors]   # i-3
            left4 = raw_free_spaces[(sector_idx + 4) % num_sectors]    # i+4
            right4 = raw_free_spaces[(sector_idx - 4) % num_sectors]   # i-4
            left5 = raw_free_spaces[(sector_idx + 5) % num_sectors]    # i+5
            right5 = raw_free_spaces[(sector_idx - 5) % num_sectors]   # i-5
            
            free_space_score = (current * 0.2 + 
                              left1 * 0.1 + 
                              right1 * 0.1 + 
                              left2 * 0.1 + 
                              right2 * 0.1 + 
                              left3 * 0.1 + 
                              right3 * 0.1 +
                              left4 * 0.05 + 
                              right4 * 0.05 + 
                              left5 * 0.05 + 
                              right5 * 0.05)
            
            # Calculate mid-angle of this sector (in degrees)
            # Sector 0 = 5°, Sector 1 = 15°, etc.
            sector_mid_angle = (sector_idx * 10) + 5
            
            # Calculate goal angle score based on proximity to goal
            angle_diff = abs(sector_mid_angle - goal_angle_deg)
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            
            # Score based on sector distance from goal
             # Score based on sector distance from goal
            if angle_diff <= 5:  # Goal is within this sector
                goal_angle_score = 10
            elif angle_diff <= 15:  # Goal is in adjacent sector
                goal_angle_score = 9
            elif angle_diff <= 25:  # Goal is 2 sectors away
                goal_angle_score = 8
            elif angle_diff <= 35:  # Goal is 3 sectors away
                goal_angle_score = 7
            elif angle_diff <= 45:  # Goal is 4 sectors away
                goal_angle_score = 6
            elif angle_diff <= 55:  # Goal is 5 sectors away
                goal_angle_score = 5
            elif angle_diff <= 65:  # Goal is 6 sectors away
                goal_angle_score = 4
            elif angle_diff <= 75:  # Goal is 7 sectors away
                goal_angle_score = 3
            elif angle_diff <= 85:  # Goal is 7 sectors away
                goal_angle_score = 2
            elif angle_diff <= 95:  # Goal is 7 sectors away
                goal_angle_score = 1
            else:
                goal_angle_score = 0
            
                        # Total score = weighted combination of free_space_score and goal_angle_score
            total_score = (free_space_score * current_free_space_weight + 
                          goal_angle_score * current_goal_angle_weight)
            
            sector_scores.append({
                'sector_idx': sector_idx,
                'mid_angle': sector_mid_angle,
                'free_space_score': free_space_score,  # Now weighted
                'raw_free_space': current,  # Keep raw count for debugging
                'goal_angle_score': goal_angle_score,
                'total_score': total_score
            })
        
        # Step 3: Find the sector with the highest total score
        best_sector = max(sector_scores, key=lambda s: s['total_score'])
        best_navigation_angle = best_sector['mid_angle']
        
        # Debug output (optional)
        if self.step_counter % (self.steps_per_second * 2) == 0:
            print(f"\n=== Obstacle Avoidance Debug ===")
            print(f"Front Obstacles: {front_obstacles} | Total Obstacles: {total_obstacles} | Threshold: {self.obstacle_threshold:.2f}m")
            print(f"Original Goal Angle: {goal_angle_deg:.1f}°")
            print(f"Best Sector: {best_sector['sector_idx']} (angle: {best_navigation_angle:.1f}°)")
            print(f"  Weighted Free Space Score: {best_sector['free_space_score']:.2f} (raw: {best_sector['raw_free_space']})")
            print(f"  Goal Angle Score: {best_sector['goal_angle_score']} × {current_goal_angle_weight} = {best_sector['goal_angle_score'] * current_goal_angle_weight:.2f}")
            print(f"  Total Weighted Score: {best_sector['total_score']:.2f}")
            
            # Print all sector scores in tabular format
            print("\nSector Scores (Weighted Free Space | Goal | Total):")
            for i in range(0, num_sectors, 10):
                indices = [str(j+1) for j in range(i, min(i+10, num_sectors))]
                weighted_free = [f"{sector_scores[j]['free_space_score']:.1f}" for j in range(i, min(i+10, num_sectors))]
                goals = [f"{sector_scores[j]['goal_angle_score']:2d}" for j in range(i, min(i+10, num_sectors))]
                totals = [f"{sector_scores[j]['total_score']:.1f}" for j in range(i, min(i+10, num_sectors))]
                print("  |  ".join(indices))
                print("  |  ".join(weighted_free))
                print("  |  ".join(goals))
                print("  |  ".join(totals))
                print()
        
        return True, best_navigation_angle, total_obstacles
    
    def avoid_obstacle(self, avoidance_angle_deg):
        """Execute obstacle avoidance maneuver using the best navigation angle"""
        # Convert avoidance_angle_deg to signed angle for steering
        if avoidance_angle_deg <= 180:
            turn_angle_rad = math.radians(avoidance_angle_deg)
        else:
            turn_angle_rad = math.radians(avoidance_angle_deg - 360)
        
        avoidance_speed = MAX_SPEED * 0.5
        
        # Convert angle to differential steering
        turn_factor = abs(turn_angle_rad) / math.radians(90)
        turn_factor = min(1.0, turn_factor)
        
        if turn_angle_rad > 0:
            # Turn left
            left_speed = avoidance_speed * (1 - turn_factor * 0.6)
            right_speed = avoidance_speed
        else:
            # Turn right
            left_speed = avoidance_speed
            right_speed = avoidance_speed * (1 - turn_factor * 0.6)
        
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
    
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
    
    def run(self):
        """Main control loop"""
        while self.robot.step(self.timestep) != -1:
            # Get current target waypoint
            current_anchor = self.waypoint_nodes[self.current_waypoint_index]
            anchor_pos = current_anchor.getPosition()
            
            # Get UWB positions
            left_pos, right_pos, back_pos = self.get_uwb_positions()
            
            # Calculate distances
            d_left = self.calculate_distance(left_pos, anchor_pos)
            d_right = self.calculate_distance(right_pos, anchor_pos)
            d_back = self.calculate_distance(back_pos, anchor_pos) if back_pos else 9999.0
            
            avg_dist = (d_left + d_right) / 2.0
            dist_diff = abs(d_left - d_right)
            
            # Check for stuck condition
            if abs(avg_dist - self.prev_avg_dist) < 0.01:
                self.stuck_counter += 1
            else:
                self.stuck_counter = 0
            self.prev_avg_dist = avg_dist
            
            # Calculate turn angle to goal
            turn_angle = self.calculate_turn_angle(left_pos, right_pos, anchor_pos)
            goal_angle = (math.degrees(turn_angle) + 360) % 360
            
            # Check for obstacles
            obstacle_detected, avoidance_angle_deg, total_obstacles = self.check_obstacles(goal_angle)
            
            # Check if anchor is behind
            anchor_is_behind = False
            if (self.have_back_uwb and 
                (self.step_counter % BACK_CHECK_INTERVAL == 0) and 
                (self.current_mode != ControlMode.TURN_AROUND)):
                min_front = min(d_left, d_right)
                if d_back < min_front - BACK_DISTANCE_MARGIN:
                    anchor_is_behind = True
                    print(f"\n!!! ANCHOR BEHIND !!! Turning around...\n")
            
            # Handle obstacle avoidance mode transitions
            if obstacle_detected and self.current_mode != ControlMode.AVOIDING_OBSTACLE:
                if self.current_mode != ControlMode.TURN_AROUND:
                    print(f"\n!!! OBSTACLE DETECTED !!! Navigating to best sector at {avoidance_angle_deg:.1f}°\n")
                    self.pre_avoidance_mode = self.current_mode
                    self.current_mode = ControlMode.AVOIDING_OBSTACLE
                    # Use the sector-scored angle directly
                    goal_angle = avoidance_angle_deg
            elif not obstacle_detected and self.current_mode == ControlMode.AVOIDING_OBSTACLE:
                print("\n>>> Obstacle cleared, resuming navigation <<<\n")
                self.current_mode = self.pre_avoidance_mode
            
            # Handle safe mode based on total obstacles
            if total_obstacles > self.safe_mode_threshold and self.current_mode != ControlMode.SAFE_MODE:
                print(f"\n!!! SAFE MODE ACTIVATED !!! Total obstacles: {total_obstacles} > {self.safe_mode_threshold}\n")
                self.current_mode = ControlMode.SAFE_MODE
            elif total_obstacles <= self.safe_mode_threshold and self.current_mode == ControlMode.SAFE_MODE:
                print(f"\n>>> SAFE MODE DEACTIVATED !!! Total obstacles: {total_obstacles} <= {self.safe_mode_threshold}\n")
                self.current_mode = ControlMode.TURN_IN_PLACE
            
            # Handle turn-around mode
            if self.current_mode == ControlMode.TURN_AROUND:
                self.turn_around_counter += 1
                if self.turn_around_counter >= self.turn_around_duration:
                    print("Turn-around complete. Resuming navigation.\n")
                    self.current_mode = ControlMode.TURN_IN_PLACE
                    self.turn_around_counter = 0
            else:
                if anchor_is_behind:
                    self.current_mode = ControlMode.TURN_AROUND
                    self.turn_around_counter = 0
            
            # State machine for normal navigation (only if not avoiding, turning around, or in safe mode)
            if self.current_mode not in [ControlMode.TURN_AROUND, ControlMode.AVOIDING_OBSTACLE, ControlMode.SAFE_MODE]:
                next_mode = self.current_mode
                
                if self.current_mode == ControlMode.TURN_IN_PLACE:
                    if dist_diff < DISTANCE_DIFF_EXIT:
                        next_mode = ControlMode.FORWARD
                
                elif self.current_mode == ControlMode.ARC_STEERING:
                    if dist_diff < DISTANCE_DIFF_EXIT:
                        next_mode = ControlMode.FORWARD
                    elif dist_diff > DISTANCE_DIFF_ENTER:
                        next_mode = ControlMode.TURN_IN_PLACE
                
                elif self.current_mode == ControlMode.FORWARD:
                    if dist_diff > DISTANCE_DIFF_ARC_REENTER:
                        next_mode = ControlMode.ARC_STEERING
                    elif dist_diff > DISTANCE_DIFF_ENTER:
                        next_mode = ControlMode.TURN_IN_PLACE
                
                if next_mode != self.current_mode:
                    mode_names = ["TURN-IN-PLACE", "ARC STEERING", "FORWARD", 
                                 "TURN-AROUND", "AVOIDING OBSTACLE", "SAFE MODE"]
                    print(f"[MODE CHANGE] {mode_names[self.current_mode]} -> "
                          f"{mode_names[next_mode]} (dist_diff={dist_diff:.3f}m)")
                    self.current_mode = next_mode
            
            # Debug output
            if self.step_counter % self.steps_per_second == 0:
                print(f"\n--- Status Update (t={self.robot.getTime():.1f} s) ---")
                print(f"Target: [{self.current_waypoint_index}] "
                      f"{WAYPOINT_SEQUENCE[self.current_waypoint_index]} "
                      f"at [{anchor_pos[0]:.3f}, {anchor_pos[1]:.3f}, {anchor_pos[2]:.3f}]")
                print(f"Distance L/R: {d_left:.3f}m / {d_right:.3f}m")
                print(f"Average Distance: {avg_dist:.3f}m")
                print(f"Distance Difference: {dist_diff:.3f}m")
                print(f"Turn Angle: {turn_angle:.3f} rad ({math.degrees(turn_angle):.1f} deg)")
                print(f"Goal Angle: {goal_angle:.1f} deg")
                
                mode_names = ["TURN-IN-PLACE", "ARC STEERING", "FORWARD", 
                             "TURN-AROUND", "AVOIDING OBSTACLE", "SAFE MODE"]
                print(f"Control Mode: {mode_names[self.current_mode]}")
                
                if obstacle_detected:
                    print(f"⚠ Obstacle detected - navigating to {avoidance_angle_deg:.1f}°")
                
                if self.stuck_counter > self.steps_per_second:
                    print(f"WARNING: Robot may be stuck! ({self.stuck_counter} steps)")
            
            # Waypoint arrival logic
            if avg_dist <= ARRIVAL_DISTANCE:
                if not self.waypoint_reached:
                    self.waypoint_reached = True
                    self.waypoint_arrival_time = self.robot.getTime()
                    print(f"\n>>> WAYPOINT {self.current_waypoint_index} REACHED: "
                          f"{WAYPOINT_SEQUENCE[self.current_waypoint_index]} <<<")
                    print(f"Dwelling for {WAYPOINT_DWELL_TIME:.1f} seconds...\n")
                    self.stop_robot()
                else:
                    elapsed = self.robot.getTime() - self.waypoint_arrival_time
                    if elapsed >= WAYPOINT_DWELL_TIME:
                        self.current_waypoint_index += 1
                        if self.current_waypoint_index >= self.num_waypoints:
                            print("\n>>> ALL WAYPOINTS COMPLETED! <<<")
                            print("Mission accomplished. Robot stopping.\n")
                            self.stop_robot()
                            # Keep running but stopped
                            while self.robot.step(self.timestep) != -1:
                                pass
                        else:
                            print(f"\n>>> Proceeding to waypoint {self.current_waypoint_index}: "
                                  f"{WAYPOINT_SEQUENCE[self.current_waypoint_index]} <<<\n")
                            self.waypoint_reached = False
                            self.current_mode = ControlMode.TURN_IN_PLACE
                            self.stuck_counter = 0
                    else:
                        self.stop_robot()
            else:
                # Not at waypoint - navigate
                self.waypoint_reached = False
                
                # Execute control based on current mode
                if self.stuck_counter > self.stuck_threshold:
                    print("UNSTUCK MODE: Recovery maneuver...\n")
                    self.turn_in_place_smart(d_left, d_right)
                    self.stuck_counter = 0
                else:
                    if self.current_mode == ControlMode.AVOIDING_OBSTACLE:
                        # Use the sector-scored avoidance angle with arc steering
                        self.differential_steer_arc(avoidance_angle_deg)
                    elif self.current_mode == ControlMode.TURN_AROUND:
                        self.turn_around(d_left, d_right)
                    elif self.current_mode == ControlMode.SAFE_MODE:
                        # In safe mode, only turn in place
                        self.turn_in_place_smart(d_left, d_right)
                    elif self.current_mode == ControlMode.TURN_IN_PLACE:
                        self.turn_in_place_smart(d_left, d_right)
                    elif self.current_mode == ControlMode.ARC_STEERING:
                        self.differential_steer_arc(goal_angle)
                    elif self.current_mode == ControlMode.FORWARD:
                        self.move_forward()
            
            self.step_counter += 1


# Main entry point
if __name__ == "__main__":
    controller = UWBNavigationController()
    controller.run()