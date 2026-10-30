#!/usr/bin/env python3

"""
Generate plots from processed Ca2+ imaging data.

Usage
-----
python plot_calcium_data.py /path/to/processed/

Expected merged files
---------------------
merged_timeData.csv
merged_auc_peakMFI.csv

Individual processed CSV files are also used for representative Ca2+ traces.

Outputs
-------
swarmplot.pdf
timeToActivation_eCDF.pdf
IgM_Ionomycin_traces.pdf
pHELD_traces.pdf
sHEL_traces.pdf
"""

import sys
import subprocess
import importlib.util
from pathlib import Path

# =============================================================================
# Package management
# =============================================================================

REQUIRED_PACKAGES = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
}


def install_missing_packages():
    """Install required Python packages if they are not already available."""
    missing = [
        pip_name
        for module_name, pip_name in REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]

    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", *missing]
        )


install_missing_packages()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# Global plotting settings
# =============================================================================

sns.set_theme(
    context="paper",
    style="ticks",
    font_scale=1.0
)

plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.75,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = {
    "control": "black",
    "sHEL": "#532d8c",
    "pHELD": "#f7931d",
}

FIGURE_WIDTH = 8.5
FIGURE_HEIGHT = 11


# =============================================================================
# Input
# =============================================================================

if len(sys.argv) != 2:
    raise SystemExit(
        "\nUsage:\n"
        "  python plot_calcium_data.py <processed_directory>\n\n"
        "Example:\n"
        "  python plot_calcium_data.py "
        "/test/processed/\n"
    )

DATA_DIR = Path(sys.argv[1]).expanduser().resolve()

if not DATA_DIR.is_dir():
    raise NotADirectoryError(
        f"Directory does not exist:\n{DATA_DIR}"
    )


def require_file(filename):
    """Return path to a required file or raise an informative error."""
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{path}"
        )

    return path


TIME_FILE = require_file("merged_timeData.csv")
MFI_FILE = require_file("merged_auc_peakMFI.csv")

time_df = pd.read_csv(TIME_FILE)
mfi_df = pd.read_csv(MFI_FILE)

print(f"Reading data from: {DATA_DIR}")


# =============================================================================
# Utility functions
# =============================================================================

def save_figure(fig, filename):
    """Save PDF and close the figure."""
    output = DATA_DIR / filename

    fig.savefig(
        output,
        format="pdf",
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Saved: {output}")


def subset_condition(df, pattern):
    """Subset dataframe using a condition regex."""
    return df[
        df["condition"].astype(str).str.contains(
            pattern,
            case=False,
            regex=True,
            na=False
        )
    ].copy()


def clean_condition_labels(ax):
    """Make categorical x-axis labels easier to read."""
    labels = []

    for label in ax.get_xticklabels():
        text = label.get_text()
        text = text.replace("_", " ")
        labels.append(text)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )


def find_processed_file(*patterns):
    """
    Find an individual processed CSV using flexible filename matching.

    Each pattern is tested case-insensitively against filenames after common
    separators have been normalized.
    """
    csv_files = [
        p for p in DATA_DIR.glob("*.csv")
        if not p.name.startswith("merged_")
    ]

    def normalize(text):
        return (
            text.lower()
            .replace(":", "_")
            .replace("-", "_")
            .replace(" ", "_")
            .replace(".", "_")
        )

    normalized_patterns = [normalize(p) for p in patterns]

    for file in csv_files:
        name = normalize(file.stem)

        if all(pattern in name for pattern in normalized_patterns):
            return file

    return None


# =============================================================================
# 1. Swarm plots
# =============================================================================

def plot_swarmplots():
    """
    Plot time-to-flux, AUC, and peak MFI for:
      - anti-IgM / Ionomycin
      - soluble HEL
      - particulate HEL
    """

    groups = [
        {
            "name": "IgM / Ionomycin",
            "pattern": r"IgM|Iono",
            "color": COLORS["control"],
        },
        {
            "name": "Soluble HEL",
            "pattern": r"sHEL",
            "color": COLORS["sHEL"],
        },
        {
            "name": "pHELD",
            "pattern": r"pHELD",
            "color": COLORS["pHELD"],
        },
    ]

    metrics = [
        ("time_to_flux", "Time to flux (s)", (-10, 300)),
        ("area", "AUC", (0, 1_250_000)),
        ("max_MFI", "Peak MFI", (700, 7000)),
    ]

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=(7.0, 5.5),
        squeeze=False
    )

    for col, group in enumerate(groups):
        sub_time = subset_condition(time_df, group["pattern"])
        sub_mfi = subset_condition(mfi_df, group["pattern"])

        for row, (metric, ylabel, ylim) in enumerate(metrics):
            ax = axes[row, col]

            data = sub_time if metric == "time_to_flux" else sub_mfi

            if data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes
                )
                continue

            sns.swarmplot(
                data=data,
                x="condition",
                y=metric,
                color=group["color"],
                size=2,
                ax=ax
            )

            ax.set_ylim(*ylim)
            ax.set_xlabel("")

            if col == 0:
                ax.set_ylabel(ylabel)
            else:
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            if row == 0:
                ax.set_title(group["name"])

            if row < 2:
                ax.tick_params(labelbottom=False)
            else:
                clean_condition_labels(ax)

            sns.despine(ax=ax)

    axes[1, 0].set_yticks(
        [0, 500_000, 1_000_000]
    )

    axes[1, 0].set_yticklabels(
        ["0", "0.5e6", "1.0e6"]
    )

    fig.tight_layout(
        h_pad=0.7,
        w_pad=0.8
    )

    save_figure(
        fig,
        "swarmplot.pdf"
    )


# =============================================================================
# 2. Time-to-flux eCDF
# =============================================================================

def plot_ecdf():
    """Plot time-to-flux eCDFs for each major stimulation class."""

    groups = [
        ("pHELD", r"pHELD"),
        ("IgM / Ionomycin", r"IgM|Iono"),
        ("sHEL WT", r"sHEL_"),
        ("sHELD", r"sHELD"),
    ]

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(6.0, 4.5),
        squeeze=False
    )

    axes = axes.flatten()

    for ax, (title, pattern) in zip(axes, groups):
        subset = subset_condition(
            time_df,
            pattern
        )

        if subset.empty:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes
            )
            ax.set_title(title)
            continue

        sns.ecdfplot(
            data=subset,
            x="time_to_flux",
            hue="condition",
            palette="turbo",
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel("Time to flux (s)")
        ax.set_ylabel("Fraction of cells")
        ax.set_xlim(left=0)
        ax.set_ylim(-0.01, 1.01)
        ax.set_xticks([0, 100, 200, 300])
        ax.set_yticks(
            [0, 0.2, 0.4, 0.6, 0.8, 1]
        )

        legend = ax.get_legend()

        if legend is not None:
            legend.set_title("Condition")
            legend.set_frame_on(False)

            for text in legend.get_texts():
                text.set_text(
                    text.get_text().replace("_", " ")
                )

        sns.despine(ax=ax)

    fig.tight_layout()

    save_figure(
        fig,
        "timeToActivation_eCDF.pdf"
    )


# =============================================================================
# Representative Ca2+ traces
# =============================================================================

def plot_trace_group(
    configurations,
    output_filename,
    color,
    ymin,
    ymax,
    dashed_patterns=None,
    xticks=(0, 250)
):
    """
    Plot representative single-cell Ca2+ traces.

    configurations:
        List of dictionaries with:
          label
          patterns
          track_ids
    """

    dashed_patterns = dashed_patterns or []

    n_conditions = len(configurations)
    n_tracks = max(
        len(config["track_ids"])
        for config in configurations
    )

    fig, axes = plt.subplots(
        n_tracks,
        n_conditions,
        figsize=(
            max(2.0, n_conditions * 1.15),
            max(1.5, n_tracks * 0.65)
        ),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for col, config in enumerate(configurations):
        file = find_processed_file(
            *config["patterns"]
        )

        if file is None:
            print(
                f"Warning: could not find file for "
                f"{config['label']}"
            )

            for row in range(n_tracks):
                axes[row, col].axis("off")

            continue

        data = pd.read_csv(file)

        linestyle = (
            (0, (1, 1.5))
            if any(
                pattern.lower() in config["label"].lower()
                for pattern in dashed_patterns
            )
            else "-"
        )

        for row in range(n_tracks):
            ax = axes[row, col]

            if row >= len(config["track_ids"]):
                ax.axis("off")
                continue

            track_id = config["track_ids"][row]

            trace = data[
                data["track_id"] == track_id
            ].copy()

            if trace.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"Track {track_id}\nnot found",
                    ha="center",
                    va="center",
                    fontsize=6,
                    transform=ax.transAxes
                )
            else:
                ax.plot(
                    trace["timepoint"],
                    trace["mean_CH2"],
                    color=color,
                    linewidth=0.8,
                    linestyle=linestyle
                )

            ax.set_ylim(ymin, ymax)
            ax.set_xticks(xticks)
            ax.tick_params(
                length=2,
                labelsize=6
            )

            if row == 0:
                ax.set_title(
                    config["label"],
                    fontsize=7
                )

            if row != n_tracks - 1:
                ax.tick_params(
                    labelbottom=False
                )

            if col != 0:
                ax.tick_params(
                    labelleft=False
                )

            sns.despine(ax=ax)

    for ax in axes[-1, :]:
        if ax.axison:
            ax.set_xlabel(
                "Time (s)",
                fontsize=7
            )

    fig.supylabel(
        r"Mean Ca$^{2+}$ fluorescence",
        fontsize=8
    )

    fig.tight_layout(
        h_pad=0.15,
        w_pad=0.3
    )

    save_figure(
        fig,
        output_filename
    )


# =============================================================================
# 3. anti-IgM and Ionomycin traces
# =============================================================================

def plot_igm_ionomycin_traces():

    configurations = [
        {
            "label": "anti-IgM 0.1 µg/mL",
            "patterns": ("igm", "0_1"),
            "track_ids": [28, 38, 9, 27, 37],
        },
        {
            "label": "anti-IgM 1 µg/mL",
            "patterns": ("igm", "1ug"),
            "track_ids": [4, 11, 17, 16, 10],
        },
        {
            "label": "anti-IgM 10 µg/mL",
            "patterns": ("igm", "10ug"),
            "track_ids": [9, 3, 1, 20, 0],
        },
        {
            "label": "Ionomycin",
            "patterns": ("ionomycin",),
            "track_ids": [22, 30, 61, 63, 13],
        },
    ]

    plot_trace_group(
        configurations=configurations,
        output_filename="IgM_Ionomycin_traces.pdf",
        color="black",
        ymin=600,
        ymax=4000
    )


# =============================================================================
# 4. pHELD traces
# =============================================================================

def plot_pheld_traces():

    configurations = [
        {
            "label": "pHELD 0.01 pM",
            "patterns": ("pheld", "0_01pm"),
            "track_ids": [17, 57, 22, 42, 47],
        },
        {
            "label": "pHELD 0.1 pM",
            "patterns": ("pheld", "0_1pm"),
            "track_ids": [3, 26, 29, 7, 21],
        },
        {
            "label": "pHELD 1 pM",
            "patterns": ("pheld", "1pm"),
            "track_ids": [29, 20, 17, 50, 38],
        },
        {
            "label": "pHELD 10 pM",
            "patterns": ("pheld", "10pm"),
            "track_ids": [38, 44, 18, 61, 54],
        },
    ]

    plot_trace_group(
        configurations=configurations,
        output_filename="pHELD_traces.pdf",
        color=COLORS["pHELD"],
        ymin=600,
        ymax=6100
    )


# =============================================================================
# 5. sHEL traces
# =============================================================================

def plot_shel_traces():

    configurations = [
        {
            "label": "sHEL WT 0.01 µg/mL",
            "patterns": ("shel", "wt", "0_01"),
            "track_ids": [1, 25, 34, 0, 11],
        },
        {
            "label": "sHEL WT 0.1 µg/mL",
            "patterns": ("shel", "wt", "0_1"),
            "track_ids": [8, 26, 19, 15, 13],
        },
        {
            "label": "sHEL WT 1 µg/mL",
            "patterns": ("shel", "wt", "1_ug"),
            "track_ids": [20, 14, 23, 26, 45],
        },
        {
            "label": "sHELD 0.01 µg/mL",
            "patterns": ("sheld", "0_01"),
            "track_ids": [62, 41, 27, 1, 73],
        },
        {
            "label": "sHELD 0.1 µg/mL",
            "patterns": ("sheld", "0_1"),
            "track_ids": [7, 72, 56, 80, 24],
        },
        {
            "label": "sHELD 1 µg/mL",
            "patterns": ("sheld", "1ug"),
            "track_ids": [14, 18, 4, 17, 24],
        },
    ]

    plot_trace_group(
        configurations=configurations,
        output_filename="sHEL_traces.pdf",
        color=COLORS["sHEL"],
        ymin=600,
        ymax=5400,
        dashed_patterns=["sHELD"]
    )


# =============================================================================
# Main
# =============================================================================

def main():

    print("\nGenerating plots...\n")

    plot_swarmplots()
    plot_ecdf()
    plot_igm_ionomycin_traces()
    plot_pheld_traces()
    plot_shel_traces()

    print(
        f"\nFinished. All PDFs were written to:\n{DATA_DIR}\n"
    )


if __name__ == "__main__":
    main()