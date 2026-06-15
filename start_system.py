"""
Start System - Launches main.py and dashboard.py in parallel
main.py runs in the background collecting data and model inference
dashboard.py runs as the front-end interface
"""

import subprocess
import time
import os
import sys

def main():
    """Launch both main.py and dashboard.py"""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=" * 70)
    print("DRIVER MONITORING SYSTEM - STARTUP")
    print("=" * 70)
    print()
    
    # Start main.py in a background process
    print("[1/2] Starting data collection and model pipeline (main.py)...")
    print("      This process collects sensor data, runs vision model, and calculates fatigue metrics...")
    
    main_process = subprocess.Popen(
        [sys.executable, "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    print("      Status: RUNNING (PID: {})".format(main_process.pid))
    time.sleep(3)  # Give main.py time to start
    
    # Start dashboard in foreground
    print()
    print("[2/2] Starting dashboard interface (Streamlit)...")
    print("      Dashboard will open in your default browser...")
    print()
    print("=" * 70)
    print("System is now running!")
    print("=" * 70)
    print()
    
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard/dashboard.py", 
             "--logger.level=warning", "--client.showErrorDetails=false"],
            check=False
        )
    except KeyboardInterrupt:
        print("\n\nShutting down system...")
    finally:
        # Terminate main.py when dashboard closes
        if main_process.poll() is None:  # Still running
            print("Terminating data collection process...")
            main_process.terminate()
            try:
                main_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                main_process.kill()
        
        print("System stopped.")

if __name__ == "__main__":
    main()
