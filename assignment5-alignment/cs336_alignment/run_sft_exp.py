#!/usr/bin/env python
"""SFT Experiment Runner - Job Requirements"""
import subprocess
import sys

if __name__ == "__main__":
    # Default: run test mode (64 samples)
    if len(sys.argv) < 2:
        print("Running test mode (64 samples)...")
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--test"]
    elif sys.argv[1] == "test":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--test"]
    elif sys.argv[1] == "128":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--dataset_size", "128", "--lr", "1e-5", "--batch_size", "4"]
    elif sys.argv[1] == "256":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--dataset_size", "256", "--lr", "1e-5", "--batch_size", "4"]
    elif sys.argv[1] == "512":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--dataset_size", "512", "--lr", "1e-5", "--batch_size", "4"]
    elif sys.argv[1] == "1024":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--dataset_size", "1024", "--lr", "1e-5", "--batch_size", "4"]
    elif sys.argv[1] == "2048":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--dataset_size", "2048", "--lr", "1e-5", "--batch_size", "4"]
    elif sys.argv[1] == "4096":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--dataset_size", "1024", "--lr", "1e-5", "--batch_size", "4"]
    
    elif sys.argv[1] == "full":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--lr", "1e-5", "--batch_size", "4"]
    elif sys.argv[1] == "--filtered":
        cmd = ["python", "-m", "cs336_alignment.sft_experiment", "--filtered", "--lr", "1e-5", "--batch_size", "4"]
    else:
        print("Usage: run_sft_exp.py [test|128|256|512|1024|full|--filtered]")
        sys.exit(1)
    
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd)
