#!/usr/bin/env python3
"""
Plot beta(step) schedule used for KL annealing: linear warm-up from 0 to beta_max over kl_warmup_steps.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt


def beta_schedule(steps: int, beta_max: float, warmup_steps: int):
    steps_arr = np.arange(steps + 1)
    if warmup_steps <= 0:
        beta = np.full_like(steps_arr, fill_value=beta_max, dtype=np.float32)
    else:
        beta = np.minimum(steps_arr, warmup_steps) / float(warmup_steps) * float(beta_max)
    return steps_arr, beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=10000)
    parser.add_argument("--beta_max", type=float, default=1.0)
    parser.add_argument("--kl_warmup_steps", type=int, default=2000)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    steps, beta = beta_schedule(args.total_steps, args.beta_max, args.kl_warmup_steps)

    plt.figure(figsize=(6, 4))
    plt.plot(steps, beta, label=f"beta_max={args.beta_max}, warmup={args.kl_warmup_steps}")
    plt.xlabel("global_step")
    plt.ylabel("beta(step)")
    plt.title("KL Annealing Schedule")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if args.out:
        plt.savefig(args.out, bbox_inches="tight")
        print(f"Saved plot to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()


