import numpy as np
import matplotlib.pyplot as plt
import os

# Project root (this script lives in scripts/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EDL_ROOT = os.path.join(ROOT, "results", "uncertainty", "edl")
MC_ROOT = os.path.join(ROOT, "results", "uncertainty", "mc")

# tag naming: train_test, U=UBFC, P=PURE, M=MMPD
PLOT_CONFIGS = [
    {
        "tag": "P_P",
        "title": "PURE_PURE Calibration Curve Comparison (EDL vs MC Dropout)",
        "path_edl": os.path.join(EDL_ROOT, "edl_P_P", "uncertainty_calibration_data.npz"),
        "path_mc_conf": os.path.join(MC_ROOT, "MC_P_P", "uncertainty_analysis", "Signal_expected_conf.npy"),
        "path_mc_cov": os.path.join(MC_ROOT, "MC_P_P", "uncertainty_analysis", "Signal_observed_freq.npy"),
    },
    {
        "tag": "U_U",
        "title": "UBFC_UBFC Calibration Curve Comparison (EDL vs MC Dropout)",
        "path_edl": os.path.join(EDL_ROOT, "edl_U_U", "uncertainty_calibration_data.npz"),
        "path_mc_conf": os.path.join(MC_ROOT, "MC_U_U", "uncertainty_analysis", "Signal_expected_conf.npy"),
        "path_mc_cov": os.path.join(MC_ROOT, "MC_U_U", "uncertainty_analysis", "Signal_observed_freq.npy"),
    },
    {
        "tag": "P_U",
        "title": "PURE_UBFC Calibration Curve Comparison (EDL vs MC Dropout)",
        "path_edl": os.path.join(EDL_ROOT, "edl_P_U", "uncertainty_calibration_data.npz"),
        "path_mc_conf": os.path.join(MC_ROOT, "MC_P_U", "uncertainty_analysis", "Signal_expected_conf.npy"),
        "path_mc_cov": os.path.join(MC_ROOT, "MC_P_U", "uncertainty_analysis", "Signal_observed_freq.npy"),
    },
    {
        "tag": "U_P",
        "title": "UBFC_PURE Calibration Curve Comparison (EDL vs MC Dropout)",
        "path_edl": os.path.join(EDL_ROOT, "edl_U_P", "uncertainty_calibration_data.npz"),
        "path_mc_conf": os.path.join(MC_ROOT, "MC_U_P", "uncertainty_analysis", "Signal_expected_conf.npy"),
        "path_mc_cov": os.path.join(MC_ROOT, "MC_U_P", "uncertainty_analysis", "Signal_observed_freq.npy"),
    },
]

save_dir = os.path.join(ROOT, "results", "comparison")
os.makedirs(save_dir, exist_ok=True)
save_tag = "_".join(cfg["tag"] for cfg in PLOT_CONFIGS)
save_path_img = os.path.join(save_dir, f"{save_tag}_uncertainty_calibration_comparison.png")
save_path_data = os.path.join(save_dir, f"{save_tag}_uncertainty_calibration_comparison_data.npz")


def load_calibration_data(cfg):
    edl_data = np.load(cfg["path_edl"])
    return {
        "expected_conf_edl": edl_data["expected_confidence"],
        "observed_cov_edl": edl_data["observed_coverage"],
        "ece_edl": edl_data["ece"] if "ece" in edl_data else None,
        "expected_conf_mc": np.load(cfg["path_mc_conf"]),
        "observed_cov_mc": np.load(cfg["path_mc_cov"]),
    }


def plot_calibration(ax, data, title):
    ax.plot(data["expected_conf_edl"], data["observed_cov_edl"], "-", color="tab:blue",
            label="EDL Calibration Curve", linewidth=2)
    ax.plot(data["expected_conf_mc"], data["observed_cov_mc"], "-", color="tab:orange",
            label="MC Dropout Calibration Curve", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Ideal Calibration")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Expected Confidence")
    ax.set_ylabel("Observed Coverage")
    ax.set_title(title, pad=18)
    ax.legend(loc="upper left")
    ax.grid(True)
    ax.tick_params(axis="both", which="major", labelsize=14)


if __name__ == "__main__":
    all_data = {cfg["tag"]: load_calibration_data(cfg) for cfg in PLOT_CONFIGS}

    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
    })

    n_plots = len(PLOT_CONFIGS)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 7.5))
    if n_plots == 1:
        axes = [axes]

    for ax, cfg in zip(axes, PLOT_CONFIGS):
        plot_calibration(ax, all_data[cfg["tag"]], cfg["title"])

    plt.tight_layout(pad=2.0, h_pad=3.0)
    plt.savefig(save_path_img, dpi=300, bbox_inches="tight")

    save_payload = {}
    for cfg in PLOT_CONFIGS:
        tag = cfg["tag"]
        save_payload[f"{tag}_expected_conf_edl"] = all_data[tag]["expected_conf_edl"]
        save_payload[f"{tag}_observed_cov_edl"] = all_data[tag]["observed_cov_edl"]
        save_payload[f"{tag}_expected_conf_mc"] = all_data[tag]["expected_conf_mc"]
        save_payload[f"{tag}_observed_cov_mc"] = all_data[tag]["observed_cov_mc"]
        save_payload[f"{tag}_ece_edl"] = all_data[tag]["ece_edl"]

    np.savez(save_path_data, **save_payload)

    print(f"Calibration comparison image saved to: {save_path_img}")
    print(f"Calibration comparison data saved to: {save_path_data}")
