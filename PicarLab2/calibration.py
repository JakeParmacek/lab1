#!/usr/bin/env python3
"""
Calibration script to tune turn and forward durations
Run this to find the correct timing values for your robot
"""

import time
from picarx import Picarx

def calibrate_90_turn(px):
    """Find the duration needed for a 90-degree turn"""
    print("\nCalibrating 90-degree turn...")
    print("The robot will attempt turns with different durations.")
    print("Watch and note which duration gives closest to 90 degrees.\n")
    
    test_durations = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    
    for duration in test_durations:
        input(f"Press Enter to test {duration}s turn...")
        
        # Test left turn
        print(f"Testing LEFT turn for {duration}s")
        px.set_dir_servo_angle(-35)
        px.forward(25)
        time.sleep(duration)
        px.stop()
        px.set_dir_servo_angle(0)
        
        time.sleep(2)
        
    best = float(input("Enter best duration for 90-degree turn: "))
    return best

def calibrate_forward_cell(px, cell_size_cm=5):
    """Find the duration needed to move forward one cell"""
    print(f"\nCalibrating forward movement for {cell_size_cm}cm...")
    print("The robot will move forward with different durations.")
    print(f"Measure and find which gets closest to {cell_size_cm}cm.\n")
    
    test_durations = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    for duration in test_durations:
        input(f"Press Enter to test {duration}s forward...")
        
        print(f"Testing forward for {duration}s")
        px.set_dir_servo_angle(0)
        px.forward(30)
        time.sleep(duration)
        px.stop()
        
        time.sleep(2)
    
    best = float(input("Enter best duration for one cell forward: "))
    return best

def main():
    px = Picarx()
    
    try:
        print("="*50)
        print("PiCar-X Movement Calibration")
        print("="*50)
        
        turn_duration = calibrate_90_turn(px)
        forward_duration = calibrate_forward_cell(px)
        
        print("\n" + "="*50)
        print("Calibration Complete!")
        print(f"90-degree turn duration: {turn_duration}s")
        print(f"Forward cell duration: {forward_duration}s")
        print("\nUpdate these values in robot_controller.py:")
        print(f"  self.turn_90_duration = {turn_duration}")
        print(f"  self.cell_forward_duration = {forward_duration}")
        print("="*50)
        
    finally:
        px.stop()

if __name__ == "__main__":
    main()