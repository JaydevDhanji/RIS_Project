#!/usr/bin/env python3
"""
Duckietown Duck Detection - Proper Headless Setup
This version properly configures the headless environment for Docker containers
"""

import os
import sys
import subprocess
import signal
import time

def setup_headless_display():
    """Setup virtual display for headless operation"""
    print("Setting up virtual display...")
    
    # Kill any existing Xvfb processes
    try:
        subprocess.run(['pkill', 'Xvfb'], capture_output=True)
        time.sleep(1)
    except:
        pass
    
    # Start Xvfb
    display_num = 99
    xvfb_process = subprocess.Popen([
        'Xvfb', f':{display_num}', 
        '-screen', '0', '1024x768x24',
        '-ac', '+extension', 'GLX', '+render', '-noreset'
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Set DISPLAY environment variable
    os.environ['DISPLAY'] = f':{display_num}'
    
    # Wait for Xvfb to start
    time.sleep(2)
    
    # Test if display is working
    try:
        result = subprocess.run(['xdpyinfo'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Virtual display :{display_num} started successfully")
            return xvfb_process
        else:
            print(f"✗ Display test failed: {result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("✗ Display test timed out")
        return None
    except FileNotFoundError:
        print("✗ xdpyinfo not found - installing x11-utils might help")
        return xvfb_process  # Continue anyway

def install_missing_packages():
    """Install missing packages that are commonly needed"""
    packages_to_check = [
        'xvfb', 'x11-utils', 'mesa-utils', 'freeglut3-dev'
    ]
    
    missing_packages = []
    for package in packages_to_check:
        result = subprocess.run(['dpkg', '-l', package], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages detected: {missing_packages}")
        print("To install them, run:")
        print(f"apt-get update && apt-get install -y {' '.join(missing_packages)}")
        return False
    
    print("✓ All required packages are installed")
    return True

def main():
    print("=" * 60)
    print("Duckietown Duck Detection - Headless Container Version")
    print("=" * 60)
    
    # Check for required packages
    if not install_missing_packages():
        print("Please install missing packages and retry")
        sys.exit(1)
    
    # Setup virtual display
    xvfb_process = setup_headless_display()
    if xvfb_process is None:
        print("Failed to start virtual display")
        sys.exit(1)
    
    # Clean up function
    def cleanup(signum=None, frame=None):
        print("\nCleaning up...")
        if xvfb_process and xvfb_process.poll() is None:
            xvfb_process.terminate()
            xvfb_process.wait()
        sys.exit(0)
    
    # Register cleanup handlers
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    
    try:
        # Now import and run the actual detection code
        run_detection()
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup()

def run_detection():
    """Main detection routine"""
    import gym
    import gym_duckietown
    import numpy as np
    import cv2
    from collections import deque
    
    # Simple duck detector class
    class SimpleDuckDetector:
        def __init__(self):
            # Basic yellow detection for ducks
            self.lower_yellow = np.array([15, 100, 100])
            self.upper_yellow = np.array([35, 255, 255])
            self.min_area = 200
            
            # Create output directory
            self.output_dir = "duck_detection_results"
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Results will be saved to: {self.output_dir}/")
        
        def detect_ducks(self, image):
            """Simple duck detection"""
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            mask = cv2.inRange(hsv, self.lower_yellow, self.upper_yellow)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    detections.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'center': (x + w//2, y + h//2)
                    })
            
            return detections, mask
        
        def save_results(self, image, detections, mask, step):
            """Save detection results"""
            # Draw detections
            result_img = image.copy()
            for det in detections:
                x, y, w, h = det['bbox']
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(result_img, f"Duck: {det['area']:.0f}", 
                           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert and save
            result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
            original_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(f"{self.output_dir}/step_{step:04d}_original.jpg", original_bgr)
            cv2.imwrite(f"{self.output_dir}/step_{step:04d}_detections.jpg", result_bgr)
            cv2.imwrite(f"{self.output_dir}/step_{step:04d}_mask.jpg", mask)
    
    print("Initializing Duckietown environment...")
    
    try:
        env = gym.make("Duckietown-loop_obstacles-v0")
        print("✓ Environment created successfully!")
    except Exception as e:
        print(f"✗ Environment creation failed: {e}")
        return
    
    detector = SimpleDuckDetector()
    
    try:
        obs = env.reset()
        print(f"✓ Environment reset. Image shape: {obs.shape}")
    except Exception as e:
        print(f"✗ Environment reset failed: {e}")
        env.close()
        return
    
    # Simple movement pattern
    actions = [
        [0.5, 0.0],    # Forward
        [0.5, 0.1],    # Forward + slight right
        [0.5, -0.1],   # Forward + slight left
        [0.3, 0.3],    # Turn right
        [0.3, -0.3],   # Turn left
    ]
    
    step_count = 0
    total_detections = 0
    
    print("Starting duck detection...")
    print("Will run for 100 steps and save results...")
    
    try:
        for step in range(100):
            # Detect ducks
            detections, mask = detector.detect_ducks(obs)
            
            if detections:
                total_detections += len(detections)
                print(f"Step {step}: Found {len(detections)} duck(s)!")
                for i, det in enumerate(detections):
                    print(f"  Duck {i+1}: area={det['area']:.0f}, center={det['center']}")
                
                # Save results when ducks are found
                detector.save_results(obs, detections, mask, step)
            
            # Save some images regardless (every 20 steps)
            if step % 20 == 0:
                detector.save_results(obs, detections, mask, step)
                print(f"Step {step}: Saved debug images")
            
            # Choose action
            action = actions[step % len(actions)]
            
            # Step environment
            try:
                obs, reward, done, info = env.step(action)
                
                # Try to render (may fail silently)
                try:
                    env.render('rgb_array')  # Use rgb_array mode
                except:
                    pass  # Ignore render errors
                
            except Exception as e:
                print(f"Step error at {step}: {e}")
                break
            
            if done:
                print(f"Episode finished at step {step}, resetting...")
                obs = env.reset()
            
            step_count += 1
            time.sleep(0.1)  # Small delay
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"Detection error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        env.close()
        print(f"\n" + "="*50)
        print("DETECTION SUMMARY")
        print("="*50)
        print(f"Total steps: {step_count}")
        print(f"Total duck detections: {total_detections}")
        print(f"Results saved in: {detector.output_dir}/")
        print(f"Check *_detections.jpg files for visual results")
        
        # Create simple summary
        with open(f"{detector.output_dir}/summary.txt", 'w') as f:
            f.write(f"Duck Detection Summary\n")
            f.write(f"Total steps: {step_count}\n")
            f.write(f"Total detections: {total_detections}\n")
            f.write(f"Detection rate: {total_detections/step_count:.2%}\n" if step_count > 0 else "No steps completed\n")

if __name__ == "__main__":
    main()
