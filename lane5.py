#!/usr/bin/env python3

import numpy as np
import cv2
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
import time
import math

class AdaptivePIDController:
    """PID Controller that adapts to different driving scenarios"""
    
    def __init__(self, speed_mode='medium'):
        # Adjust PID parameters based on speed mode
        if speed_mode in ['fast', 'racing']:
            # More aggressive PID for higher speeds
            self.straight_params = {'kp': 0.35, 'ki': 0.002, 'kd': 0.12}
            self.curve_params = {'kp': 0.5, 'ki': 0.0, 'kd': 0.20}
        elif speed_mode == 'medium':
            # Balanced PID parameters
            self.straight_params = {'kp': 0.30, 'ki': 0.001, 'kd': 0.10}
            self.curve_params = {'kp': 0.45, 'ki': 0.0, 'kd': 0.18}
        else:  # slow
            # Conservative PID parameters
            self.straight_params = {'kp': 0.25, 'ki': 0.001, 'kd': 0.08}
            self.curve_params = {'kp': 0.4, 'ki': 0.0, 'kd': 0.15}
        
        self.current_params = self.straight_params.copy()
        self.dt = 0.1
        
        self.previous_error = 0.0
        self.integral = 0.0
        self.max_integral = 0.3
        
        # Error history for derivative smoothing
        self.error_history = []
        self.history_size = 3
        
    def adapt_to_curve(self, is_curve, curve_sharpness=0.0):
        """Adapt PID parameters based on driving scenario"""
        if is_curve:
            # Blend between straight and curve parameters based on sharpness
            blend = min(1.0, curve_sharpness * 2)
            self.current_params['kp'] = (1-blend) * self.straight_params['kp'] + blend * self.curve_params['kp']
            self.current_params['ki'] = (1-blend) * self.straight_params['ki'] + blend * self.curve_params['ki']
            self.current_params['kd'] = (1-blend) * self.straight_params['kd'] + blend * self.curve_params['kd']
        else:
            self.current_params = self.straight_params.copy()
    
    def update(self, error):
        """Update PID with current parameters"""
        # Smooth error using history
        self.error_history.append(error)
        if len(self.error_history) > self.history_size:
            self.error_history.pop(0)
        smoothed_error = np.mean(self.error_history)
        
        # Proportional term
        proportional = self.current_params['kp'] * smoothed_error
        
        # Integral term with windup protection
        self.integral += smoothed_error * self.dt
        self.integral = np.clip(self.integral, -self.max_integral, self.max_integral)
        integral = self.current_params['ki'] * self.integral
        
        # Derivative term
        derivative = self.current_params['kd'] * (smoothed_error - self.previous_error) / self.dt
        
        output = proportional + integral + derivative
        
        self.previous_error = smoothed_error
        return output
    
    def reset(self):
        self.previous_error = 0.0
        self.integral = 0.0
        self.error_history = []

class AdvancedLaneDetector:
    """Advanced lane detection with curve handling and lookahead"""
    
    def __init__(self):
        # Enhanced color detection for different lighting
        self.yellow_lower = np.array([15, 40, 100])
        self.yellow_upper = np.array([35, 255, 255])
        self.white_lower = np.array([0, 0, 140])
        self.white_upper = np.array([180, 60, 255])
        
        # Multi-point detection parameters
        self.num_scan_lines = 5
        self.lookahead_distance = 0.7  # How far ahead to look (0-1)
        
        # Lane tracking
        self.lane_points_history = []
        self.history_size = 5
        
    def preprocess_image(self, image):
        """Enhanced preprocessing for better lane detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Adaptive thresholding based on image brightness
        brightness = np.mean(hsv[:,:,2])
        if brightness < 100:  # Dark image
            self.white_lower[2] = max(80, self.white_lower[2] - 20)
            self.yellow_lower[2] = max(60, self.yellow_lower[2] - 20)
        elif brightness > 200:  # Bright image
            self.white_lower[2] = min(180, self.white_lower[2] + 20)
            self.yellow_lower[2] = min(140, self.yellow_lower[2] + 20)
        
        # Create masks
        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        white_mask = cv2.inRange(hsv, self.white_lower, self.white_upper)
        combined_mask = cv2.bitwise_or(yellow_mask, white_mask)
        
        # Enhanced morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Gaussian blur for smoothing
        combined_mask = cv2.GaussianBlur(combined_mask, (7, 7), 0)
        
        return combined_mask
    
    def detect_lane_points(self, image):
        """Detect multiple points along the lane for curve analysis"""
        processed = self.preprocess_image(image)
        height, width = processed.shape
        
        lane_points = []
        
        # Scan multiple horizontal lines from near to far
        scan_lines = []
        for i in range(self.num_scan_lines):
            # Scan from bottom (close) to middle-upper (far)
            y_ratio = 0.95 - (i * 0.15)  # 0.95, 0.8, 0.65, 0.5, 0.35
            y_pos = max(int(height * y_ratio), int(height * 0.35))
            scan_lines.append(y_pos)
        
        for i, y_pos in enumerate(scan_lines):
            # Extract horizontal line
            line = processed[y_pos, :]
            
            # Smooth the line to reduce noise
            smoothed_line = cv2.GaussianBlur(line.reshape(1, -1), (15, 1), 0).flatten()
            
            # Find lane center using weighted average
            total_weight = np.sum(smoothed_line)
            if total_weight > 500:  # Enough pixels detected
                weights = smoothed_line.astype(float)
                positions = np.arange(width)
                center_x = int(np.sum(positions * weights) / total_weight)
                
                # Weight points by distance (closer points more important)
                weight = 1.0 / (1.0 + i * 0.3)  # Closer points get higher weight
                lane_points.append({
                    'x': center_x,
                    'y': y_pos,
                    'weight': weight,
                    'distance_index': i
                })
        
        return lane_points, processed
    
    def analyze_curve(self, lane_points, image_width):
        """Analyze lane curvature from detected points"""
        if len(lane_points) < 3:
            return False, 0.0, lane_points[0]['x'] if lane_points else image_width//2
        
        # Extract coordinates
        x_coords = [p['x'] for p in lane_points]
        y_coords = [p['y'] for p in lane_points]
        
        # Calculate curve direction and sharpness
        # Use simple linear regression to find lane direction
        if len(x_coords) >= 2:
            # Calculate slope between far and near points
            dx = x_coords[0] - x_coords[-1]  # Near - far
            dy = y_coords[0] - y_coords[-1]  # Should be positive (near is lower)
            
            if dy != 0:
                slope = dx / dy
                curve_angle = math.atan(slope)
                curve_sharpness = abs(curve_angle)
                
                # Determine if it's a significant curve
                is_curve = curve_sharpness > 0.1  # Threshold for curve detection
                
                # Calculate lookahead point for steering
                # Use weighted average emphasizing closer points
                total_weight = sum(p['weight'] for p in lane_points)
                if total_weight > 0:
                    lookahead_x = sum(p['x'] * p['weight'] for p in lane_points) / total_weight
                else:
                    lookahead_x = x_coords[0]
                
                return is_curve, curve_sharpness, int(lookahead_x)
        
        # Fallback: use closest point
        return False, 0.0, x_coords[0] if x_coords else image_width//2
    
    def get_lane_info(self, image):
        """Get comprehensive lane information"""
        lane_points, processed = self.detect_lane_points(image)
        
        if not lane_points:
            # No lane detected, return safe defaults
            return {
                'center_x': image.shape[1] // 2,
                'is_curve': False,
                'curve_sharpness': 0.0,
                'confidence': 0.0,
                'processed_image': processed,
                'lane_points': []
            }
        
        # Analyze curve
        is_curve, curve_sharpness, lookahead_x = self.analyze_curve(lane_points, image.shape[1])
        
        # Calculate confidence based on number of detected points
        confidence = min(1.0, len(lane_points) / self.num_scan_lines)
        
        # Update history for temporal smoothing
        self.lane_points_history.append(lane_points)
        if len(self.lane_points_history) > self.history_size:
            self.lane_points_history.pop(0)
        
        return {
            'center_x': lookahead_x,
            'is_curve': is_curve,
            'curve_sharpness': curve_sharpness,
            'confidence': confidence,
            'processed_image': processed,
            'lane_points': lane_points
        }

class CurveAwareLaneFollower:
    """Advanced lane following agent that handles curves properly"""
    
    def __init__(self, debug=True, speed_mode='medium'):
        self.lane_detector = AdvancedLaneDetector()
        self.pid_controller = AdaptivePIDController()
        
        # Speed profiles - choose based on preference
        speed_profiles = {
            'slow': {
                'base_speed': 0.12,
                'max_speed': 0.18,
                'min_speed': 0.06,
                'curve_speed_factor': 0.6,
                'max_steering': 0.4,
                'steering_smoothing': 0.7
            },
            'medium': {
                'base_speed': 0.20,  # Increased from 0.12
                'max_speed': 0.28,   # Increased from 0.18
                'min_speed': 0.10,   # Increased from 0.06
                'curve_speed_factor': 0.5,  # Less slowdown in curves
                'max_steering': 0.5,  # Slightly more aggressive steering
                'steering_smoothing': 0.6  # Less smoothing for quicker response
            },
            'fast': {
                'base_speed': 0.30,  # Much faster
                'max_speed': 0.40,   # High max speed
                'min_speed': 0.15,   # Higher minimum
                'curve_speed_factor': 0.4,  # Moderate curve slowdown
                'max_steering': 0.6,  # More aggressive steering
                'steering_smoothing': 0.5  # Quick response
            },
            'racing': {
                'base_speed': 0.40,  # Racing speed
                'max_speed': 0.55,   # Very fast
                'min_speed': 0.20,   # Fast even when slowing down
                'curve_speed_factor': 0.3,  # Minimal curve slowdown
                'max_steering': 0.7,  # Aggressive steering
                'steering_smoothing': 0.4  # Very responsive
            }
        }
        
        # Apply selected speed profile
        profile = speed_profiles.get(speed_mode, speed_profiles['medium'])
        self.base_speed = profile['base_speed']
        self.max_speed = profile['max_speed']
        self.min_speed = profile['min_speed']
        self.curve_speed_factor = profile['curve_speed_factor']
        self.max_steering = profile['max_steering']
        self.steering_smoothing = profile['steering_smoothing']
        
        self.speed_mode = speed_mode
        
        self.debug = debug
        
        # State tracking
        self.last_steering = 0.0
        self.consecutive_no_detection = 0
        self.max_no_detection = 5
        
        # Performance tracking
        self.curve_count = 0
        self.successful_curves = 0
        
    def get_adaptive_action(self, observation):
        """Get action with curve awareness and adaptive control"""
        # Get comprehensive lane information
        lane_info = self.lane_detector.get_lane_info(observation)
        
        center_x = lane_info['center_x']
        is_curve = lane_info['is_curve']
        curve_sharpness = lane_info['curve_sharpness']
        confidence = lane_info['confidence']
        
        # Track detection quality
        if confidence < 0.3:
            self.consecutive_no_detection += 1
        else:
            self.consecutive_no_detection = 0
        
        # Calculate steering error
        image_center_x = observation.shape[1] // 2
        raw_error = (center_x - image_center_x) / (observation.shape[1] // 2)
        
        # Limit error based on confidence
        max_error = 0.6 if confidence > 0.7 else 0.4
        error = np.clip(raw_error, -max_error, max_error)
        
        # Adapt PID controller to current scenario
        self.pid_controller.adapt_to_curve(is_curve, curve_sharpness)
        
        # Get steering from adaptive PID
        steering_raw = self.pid_controller.update(error)
        
        # Apply steering limits
        steering = np.clip(steering_raw, -self.max_steering, self.max_steering)
        
        # Smooth steering to prevent abrupt changes
        smooth_steering = (self.steering_smoothing * self.last_steering + 
                          (1 - self.steering_smoothing) * steering)
        self.last_steering = smooth_steering
        
        # Adaptive speed control
        speed = self.calculate_adaptive_speed(is_curve, curve_sharpness, 
                                            abs(smooth_steering), confidence)
        
        # Track curve performance
        if is_curve and not hasattr(self, '_in_curve'):
            self._in_curve = True
            self.curve_count += 1
        elif not is_curve and hasattr(self, '_in_curve'):
            self._in_curve = False
            self.successful_curves += 1
        
        if self.debug:
            print(f"[{self.speed_mode.upper()}] Center: {center_x}, Error: {error:.3f}, "
                  f"Curve: {is_curve} ({curve_sharpness:.3f}), "
                  f"Confidence: {confidence:.2f}, "
                  f"Steering: {smooth_steering:.3f}, Speed: {speed:.3f}")
            if self.curve_count > 0:
                success_rate = self.successful_curves / self.curve_count * 100
                print(f"  Curves: {self.successful_curves}/{self.curve_count} "
                      f"({success_rate:.1f}% success)")
        
        return np.array([speed, smooth_steering])
    
    def calculate_adaptive_speed(self, is_curve, curve_sharpness, steering_magnitude, confidence):
        """Calculate speed based on current driving scenario"""
        speed = self.base_speed
        
        # Slow down for curves
        if is_curve:
            curve_reduction = curve_sharpness * self.curve_speed_factor
            speed *= (1.0 - curve_reduction)
        
        # Slow down for large steering angles
        steering_reduction = steering_magnitude * 0.3
        speed *= (1.0 - steering_reduction)
        
        # Slow down if detection is poor
        if confidence < 0.5:
            speed *= (0.5 + 0.5 * confidence)
        
        # Emergency slow down if consistently losing lane
        if self.consecutive_no_detection > 3:
            speed *= 0.4
        
        return np.clip(speed, self.min_speed, self.max_speed)
    
    def reset(self):
        """Reset agent state"""
        self.pid_controller.reset()
        self.consecutive_no_detection = 0
        self.last_steering = 0.0
        self.curve_count = 0
        self.successful_curves = 0
        if hasattr(self, '_in_curve'):
            delattr(self, '_in_curve')

def run_curve_aware_simulation(speed_mode='medium'):
    """Run the curve-aware lane following simulation with selectable speed"""
    
    env = DuckietownEnv(
        map_name="loop_empty",  # Good for testing curves
        draw_curve=False,
        draw_bbox=False,
        domain_rand=False,
        frame_skip=1,
        distortion=False,
        camera_width=640,
        camera_height=480,
        max_steps=8000,  # Longer episodes for curve testing
    )
    
    agent = CurveAwareLaneFollower(debug=True, speed_mode=speed_mode)
    
    episode_count = 0
    total_episodes = 3  # Reduced for faster testing
    best_performance = 0
    
    speed_descriptions = {
        'slow': 'SAFE & STABLE (0.12 base speed)',
        'medium': 'BALANCED (0.20 base speed)', 
        'fast': 'AGGRESSIVE (0.30 base speed)',
        'racing': 'MAXIMUM SPEED (0.40 base speed)'
    }
    
    print("CURVE-AWARE LANE FOLLOWING SIMULATION")
    print(f"Speed Mode: {speed_mode.upper()} - {speed_descriptions[speed_mode]}")
    print("="*60)
    
    try:
        for episode in range(total_episodes):
            episode_count += 1
            print(f"\n🚗 Episode {episode_count}/{total_episodes}")
            
            obs = env.reset()
            agent.reset()
            
            episode_reward = 0
            steps = 0
            max_steps = 2000
            
            episode_start_time = time.time()
            
            while steps < max_steps:
                action = agent.get_adaptive_action(obs)
                
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                steps += 1
                
                # Render with error handling
                try:
                    env.render()
                except:
                    pass
                
                # Moderate delay for observation (less delay for faster modes)
                delay_map = {'slow': 0.04, 'medium': 0.03, 'fast': 0.02, 'racing': 0.01}
                time.sleep(delay_map.get(speed_mode, 0.03))
                
                if steps % 200 == 0:
                    elapsed = time.time() - episode_start_time
                    avg_reward = episode_reward / steps
                    print(f"    {steps} steps, {elapsed:.1f}s, "
                          f"Avg reward: {avg_reward:.3f}")
                
                if done:
                    print(f"    Episode ended: {info}")
                    break
            
            # Episode summary
            performance_score = steps + (episode_reward / 10)  # Combined metric
            if performance_score > best_performance:
                best_performance = performance_score
                print(f"    🎉 NEW BEST PERFORMANCE!")
            
            success_rate = (agent.successful_curves / max(1, agent.curve_count)) * 100
            print(f"    📊 Episode {episode_count} Results:")
            print(f"        Steps: {steps}")
            print(f"        Reward: {episode_reward:.2f}")
            print(f"        Curves handled: {agent.successful_curves}/{agent.curve_count} "
                  f"({success_rate:.1f}%)")
            print(f"        Performance score: {performance_score:.1f}")
            
            time.sleep(2)
    
    except KeyboardInterrupt:
        print(f"\n⏹️ Simulation stopped after {episode_count} episodes")
    
    finally:
        env.close()
        print(f"\n📈 Best performance score: {best_performance:.1f}")

def test_detection_only():
    """Test the advanced lane detection without movement"""
    print("Testing advanced lane detection...")
    
    env = DuckietownEnv(map_name="loop_empty", camera_width=640, camera_height=480)
    obs = env.reset()
    
    detector = AdvancedLaneDetector()
    
    for i in range(10):
        lane_info = detector.get_lane_info(obs)
        
        print(f"Frame {i+1}:")
        print(f"  Center: {lane_info['center_x']}")
        print(f"  Is curve: {lane_info['is_curve']}")
        print(f"  Curve sharpness: {lane_info['curve_sharpness']:.3f}")
        print(f"  Confidence: {lane_info['confidence']:.3f}")
        print(f"  Lane points detected: {len(lane_info['lane_points'])}")
        
        # Visualize detection
        debug_img = obs.copy()
        
        # Draw lane points
        for point in lane_info['lane_points']:
            cv2.circle(debug_img, (point['x'], point['y']), 5, (255, 0, 0), -1)
        
        # Draw center and image center
        cv2.circle(debug_img, (lane_info['center_x'], obs.shape[0]//2), 8, (0, 255, 0), -1)
        cv2.line(debug_img, (obs.shape[1]//2, 0), (obs.shape[1]//2, obs.shape[0]), (0, 0, 255), 2)
        
        # Save debug image
        cv2.imwrite(f'advanced_debug_{i}.jpg', cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
        
        if i < 3:  # Move a bit for the first few frames
            obs, _, _, _ = env.step([0.1, 0.0])
    
    env.close()
    print("Advanced detection test complete. Check advanced_debug_*.jpg files")

if __name__ == "__main__":
    print("🏁 Advanced Duckietown Lane Following with Curve Handling")
    print("\nChoose an option:")
    print("1. Test lane detection only")
    print("2. Run simulation - SLOW mode (safe & stable)")
    print("3. Run simulation - MEDIUM mode (balanced)")
    print("4. Run simulation - FAST mode (aggressive)")
    print("5. Run simulation - RACING mode (maximum speed)")
    
    try:
        choice = input("\nEnter choice (1-5, or just press Enter for medium): ").strip()
        
        if choice == "1":
            test_detection_only()
        elif choice == "2":
            print("🐌 Starting SLOW mode simulation...")
            run_curve_aware_simulation('slow')
        elif choice == "3" or choice == "":
            print("⚖️ Starting MEDIUM mode simulation...")
            run_curve_aware_simulation('medium')
        elif choice == "4":
            print("🚀 Starting FAST mode simulation...")
            run_curve_aware_simulation('fast')
        elif choice == "5":
            print("🏎️ Starting RACING mode simulation...")
            print("⚠️ WARNING: This is very fast! The robot might go off track more easily.")
            confirm = input("Continue? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                run_curve_aware_simulation('racing')
            else:
                print("Cancelled. Running medium mode instead...")
                run_curve_aware_simulation('medium')
        else:
            print("Invalid choice. Running medium mode...")
            run_curve_aware_simulation('medium')
            
    except KeyboardInterrupt:
        print("\n⏹️ Exiting...")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nTrying medium mode as fallback...")
        try:
            run_curve_aware_simulation('medium')
        except:
            print("Fallback failed. Please check your environment setup.")
