#!/usr/bin/env python3
"""
Master script to run all data processing steps
This script orchestrates the entire data processing pipeline.
"""

import subprocess
import sys
from pathlib import Path
import argparse

def run_step(script_path: Path, args: list, step_name: str):
    """Run a single processing step."""
    print(f"\n{'='*60}")
    print(f"Running {step_name}")
    print(f"{'='*60}")
    
    cmd = [sys.executable, str(script_path)] + args
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {step_name} completed successfully!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed!")
        print(f"Error: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run all data processing steps")
    parser.add_argument("--data_dir", type=str, default="finetune_data", 
                       help="Directory containing the pickle files and CSV")
    parser.add_argument("--output_dir", type=str, default="processed_data", 
                       help="Output directory for processed data")
    parser.add_argument("--skip_steps", type=str, nargs="*", default=[],
                       help="Steps to skip (e.g., --skip_steps 2 3)")
    parser.add_argument("--start_from", type=int, default=1,
                       help="Step to start from (1-6)")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define processing steps
    steps = [
        {
            "name": "Step 1: Create base dataset",
            "script": "01_create_base_dataset.py",
            "args": ["--data_dir", str(data_dir), "--output_dir", str(output_dir)]
        },
        {
            "name": "Step 2: Compute pocket vectors",
            "script": "02_compute_pocket_vec.py",
            "args": ["--input_file", str(output_dir / "base_dataset.parquet"), "--output_dir", str(output_dir)]
        },
        {
            "name": "Step 3: Compute evolutionary vectors",
            "script": "03_compute_evo_vec.py",
            "args": ["--input_file", str(output_dir / "dataset_with_pocket_vec.parquet"), "--output_dir", str(output_dir)]
        },
        {
            "name": "Step 4: Compute interaction fingerprints",
            "script": "04_compute_ifp.py",
            "args": ["--input_file", str(output_dir / "dataset_with_evo_vec.parquet"), "--output_dir", str(output_dir)]
        },
        {
            "name": "Step 5: Compute ligand vectors",
            "script": "05_compute_ligand_vec.py",
            "args": ["--input_file", str(output_dir / "dataset_with_ifp.parquet"), "--output_dir", str(output_dir)]
        },
        {
            "name": "Step 6: Create final dataset",
            "script": "06_create_final_dataset.py",
            "args": ["--input_file", str(output_dir / "dataset_with_ligand_vec.parquet"), "--output_dir", str(output_dir)]
        }
    ]
    
    # Run steps
    success_count = 0
    total_steps = len(steps)
    
    for i, step in enumerate(steps, 1):
        # Check if we should skip this step
        if str(i) in args.skip_steps:
            print(f"\n⏭️  Skipping {step['name']}")
            continue
            
        # Check if we should start from this step
        if i < args.start_from:
            print(f"\n⏭️  Skipping {step['name']} (starting from step {args.start_from})")
            continue
        
        # Run the step
        script_path = Path(__file__).parent / step["script"]
        success = run_step(script_path, step["args"], step["name"])
        
        if success:
            success_count += 1
        else:
            print(f"\n❌ Pipeline failed at {step['name']}")
            print("You can resume from the next step using --start_from")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    print(f"Completed steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 All steps completed successfully!")
        print(f"Final dataset available at: {output_dir / 'final_dataset'}")
    else:
        print("⚠️  Pipeline incomplete. Check errors above.")

if __name__ == "__main__":
    main()

