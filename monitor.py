#!/usr/bin/env python3
"""
Monitor script that runs main.py at regular intervals and saves outputs to a log file.
"""

import subprocess
import time
import datetime
import argparse
import sys
import os
from pathlib import Path


def run_main_and_capture():
    """Run main.py and capture its output."""
    try:
        # Run main.py and capture both stdout and stderr
        result = subprocess.run(
            [sys.executable, "main.py"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "Execution timed out after 5 minutes", 1
    except Exception as e:
        return "", f"Error running main.py: {str(e)}", 1


def log_output(output, error, return_code, log_file):
    """Write the output to the log file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"TIMESTAMP: {timestamp}\n")
        f.write(f"RETURN CODE: {return_code}\n")
        f.write(f"{'='*80}\n")
        
        if output:
            f.write("STDOUT:\n")
            f.write(output)
            f.write("\n")
        
        if error:
            f.write("STDERR:\n")
            f.write(error)
            f.write("\n")
        
        f.write(f"{'='*80}\n")
        f.flush()  # Ensure output is written immediately


def main():
    parser = argparse.ArgumentParser(
        description="Run main.py at regular intervals and log outputs"
    )
    parser.add_argument(
        "--interval", 
        type=int, 
        default=5,
        help="Interval in minutes between runs (default: 5)"
    )
    parser.add_argument(
        "--log-file", 
        type=str, 
        default="monitor_log.txt",
        help="Log file path (default: monitor_log.txt)"
    )
    parser.add_argument(
        "--max-runs", 
        type=int, 
        default=None,
        help="Maximum number of runs (default: run indefinitely)"
    )
    
    args = parser.parse_args()
    
    # Convert minutes to seconds
    interval_seconds = args.interval * 60
    
    print(f"Starting monitor with {args.interval} minute intervals")
    print(f"Log file: {args.log_file}")
    print(f"Max runs: {args.max_runs if args.max_runs else 'unlimited'}")
    print(f"Press Ctrl+C to stop\n")
    
    run_count = 0
    
    try:
        while True:
            if args.max_runs and run_count >= args.max_runs:
                print(f"Reached maximum runs ({args.max_runs}). Stopping.")
                break
                
            run_count += 1
            print(f"Run #{run_count} - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            # Run main.py and capture output
            stdout, stderr, return_code = run_main_and_capture()
            
            # Log the output
            log_output(stdout, stderr, return_code, args.log_file)
            
            # Print summary to console
            if return_code == 0:
                print(f"✓ Completed successfully")
            else:
                print(f"✗ Failed with return code {return_code}")
            
            if stderr:
                print(f"  Error: {stderr.strip()}")
            
            print(f"Output logged to {args.log_file}")
            print(f"Next run in {args.interval} minutes...\n")
            
            # Wait for next run (except on the last iteration)
            if not (args.max_runs and run_count >= args.max_runs):
                time.sleep(interval_seconds)
                
    except KeyboardInterrupt:
        print(f"\nMonitor stopped by user after {run_count} runs")
        print(f"Log file: {args.log_file}")


if __name__ == "__main__":
    main() 