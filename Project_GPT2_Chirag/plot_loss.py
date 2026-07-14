"""
plot_loss.py
------------
Reads the CSV training log produced by train.py and saves a publication-quality
loss curve image to out/loss_curve.png.

Usage:
    python plot_loss.py
    python plot_loss.py --log_path out/train_log.csv --out_path out/loss_curve.png
"""

import os
import csv
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Plot training/validation loss from train_log.csv")
    parser.add_argument("--log_path", type=str, default="out/train_log.csv",
                        help="Path to CSV log file produced by train.py")
    parser.add_argument("--out_path", type=str, default="out/loss_curve.png",
                        help="Output image file path")
    parser.add_argument("--title", type=str, default="Mini GPT-2 Training Loss",
                        help="Chart title")
    return parser.parse_args()


def load_log(log_path):
    """Parses the CSV log file and returns (iters, train_losses, val_losses, lrs)."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(
            f"Log file not found: '{log_path}'\n"
            "Run train.py first to generate this file."
        )
    iters, train_losses, val_losses, lrs = [], [], [], []
    with open(log_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iters.append(int(row["iter"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            lrs.append(float(row["lr"]))
    return iters, train_losses, val_losses, lrs


def main():
    args = parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend (works without a display)
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        print("matplotlib is required. Install it with: pip install matplotlib")
        return

    iters, train_losses, val_losses, lrs = load_log(args.log_path)

    if not iters:
        print("Log file is empty — run at least one training eval step first.")
        return

    # ── Figure layout: loss on top, lr on bottom ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7),
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor="#0c0818"
    )
    fig.subplots_adjust(hspace=0.35)

    # ── Colour palette matching the web dashboard ──────────────────────────────
    COL_TRAIN = "#8b5cf6"    # purple
    COL_VAL   = "#06b6d4"    # cyan
    COL_LR    = "#ec4899"    # pink
    BG        = "#0c0818"
    GRID      = "#1e1538"
    TEXT      = "#f1ecf9"

    # ── Loss subplot ──────────────────────────────────────────────────────────
    ax1.set_facecolor(BG)
    ax1.plot(iters, train_losses, color=COL_TRAIN, linewidth=2.0,
             label="Train Loss", zorder=3)
    ax1.plot(iters, val_losses,   color=COL_VAL,   linewidth=2.0,
             linestyle="--", label="Val Loss", zorder=3)
    ax1.fill_between(iters, train_losses, alpha=0.08, color=COL_TRAIN)
    ax1.fill_between(iters, val_losses,   alpha=0.08, color=COL_VAL)

    # Annotate final values
    ax1.annotate(f"{train_losses[-1]:.4f}", xy=(iters[-1], train_losses[-1]),
                 xytext=(8, 0), textcoords="offset points",
                 color=COL_TRAIN, fontsize=9, va="center")
    ax1.annotate(f"{val_losses[-1]:.4f}", xy=(iters[-1], val_losses[-1]),
                 xytext=(8, 0), textcoords="offset points",
                 color=COL_VAL, fontsize=9, va="center")

    ax1.set_title(args.title, color=TEXT, fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Cross-Entropy Loss", color=TEXT, fontsize=11)
    ax1.tick_params(colors=TEXT, labelsize=9)
    ax1.spines[:].set_color(GRID)
    ax1.set_facecolor(BG)
    ax1.yaxis.label.set_color(TEXT)
    ax1.grid(True, color=GRID, linestyle="--", linewidth=0.6, zorder=0)
    legend = ax1.legend(facecolor="#1a1035", edgecolor=GRID,
                        labelcolor=TEXT, fontsize=10)

    # ── LR subplot ────────────────────────────────────────────────────────────
    ax2.set_facecolor(BG)
    ax2.plot(iters, lrs, color=COL_LR, linewidth=1.5, label="Learning Rate")
    ax2.fill_between(iters, lrs, alpha=0.1, color=COL_LR)
    ax2.set_ylabel("LR", color=TEXT, fontsize=10)
    ax2.set_xlabel("Iteration", color=TEXT, fontsize=11)
    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.spines[:].set_color(GRID)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0e"))
    ax2.grid(True, color=GRID, linestyle="--", linewidth=0.6)

    fig.patch.set_facecolor(BG)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"Loss curve saved to: {args.out_path}")
    print(f"  Total eval points : {len(iters)}")
    print(f"  Final Train Loss  : {train_losses[-1]:.4f}")
    print(f"  Final Val Loss    : {val_losses[-1]:.4f}")
    print(f"  Best Val Loss     : {min(val_losses):.4f} @ iter {iters[val_losses.index(min(val_losses))]}")


if __name__ == "__main__":
    main()
