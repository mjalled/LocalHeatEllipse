"""Example usage for GP and SR surrogates."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from utils import (
    THETA_DEG_DEFAULT,
    predict_gp_curve,
    predict_sr_theta_curve,
    predict_sr_theta_value,
)


def main():
    re = 175.0
    aoa_deg = 15.0
    ar = 2.0

    gp_mean, gp_std = predict_gp_curve(re, aoa_deg, ar, return_std=True)
    sr18_curve = predict_sr_theta_curve(re, aoa_deg, ar, complexity=18)
    sr30_curve = predict_sr_theta_curve(re, aoa_deg, ar, complexity=30)

    print("Input case:")
    print(f"  Re={re}, angle_of_attack_deg={aoa_deg}, AR={ar}")

    print("\nFirst 5 GP mean values:", np.round(gp_mean[:5], 4))
    print("First 5 GP std values: ", np.round(gp_std[:5], 4))
    print("First 5 SR-18 values:  ", np.round(sr18_curve[:5], 4))
    print("First 5 SR-30 values:  ", np.round(sr30_curve[:5], 4))

    theta_query = 45.0
    sr30_at_45 = predict_sr_theta_value(re, aoa_deg, ar, theta_query, complexity=30)
    print(f"\nSR-30 at theta={theta_query:.1f} deg: {sr30_at_45:.6f}")

    # plot the GP and SR curves
    plt.figure(figsize=(8, 6))
    plt.plot(THETA_DEG_DEFAULT, gp_mean, label="GP mean", color="blue")
    plt.fill_between(THETA_DEG_DEFAULT, gp_mean - gp_std, gp_mean + gp_std, color="blue", alpha=0.2, label="GP std")
    plt.plot(THETA_DEG_DEFAULT, sr18_curve, label="Symbolic Regression SR-18", color="orange")
    plt.plot(THETA_DEG_DEFAULT, sr30_curve, label="Symbolic Regression SR-30", color="green")
    plt.xlabel("Theta (deg)")
    plt.ylabel("Predicted Value")
    plt.title(f"GP and SR Surrogates for Re={re}, AoA={aoa_deg}, AR={ar}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
