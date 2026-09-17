"""
Environment Visualizer & Image Exporter.
Extracts 2D occupancy grids from benchmark .npz archives, renders publication-grade
RGB images of each environment (marking Start, Goal, and Obstacles), and packages
them into compressed .zip archives and multi-grid visual mosaics.
"""

import argparse
import io
import os
import sys
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser(description="Export and Compress Benchmark Environment Images.")
    parser.add_argument("--dataset", type=str, default="both", choices=["test", "train", "both"],
                        help="Which dataset to process: test, train, or both.")
    parser.add_argument("--test_npz", type=str, default=str(DATA_DIR / "test_grids_500.npz"),
                        help="Path to test environments .npz archive.")
    parser.add_argument("--train_npz", type=str, default=str(DATA_DIR / "train_grids_500.npz"),
                        help="Path to training environments .npz archive.")
    parser.add_argument("--output_dir", type=str, default=str(DATA_DIR),
                        help="Output directory for generated .zip archives.")
    parser.add_argument("--extract_folder", action="store_true", default=False,
                        help="Also extract individual PNG files to folders on disk.")
    parser.add_argument("--make_mosaic", action="store_true", default=True,
                        help="Generate a 5x5 mosaic overview image.")
    return parser.parse_args()


def render_single_grid(
    grid: np.ndarray,
    start: Tuple[int, int],
    goal: Tuple[int, int],
    title: str,
    fig_ax=None,
) -> bytes:
    """Renders a single occupancy grid into an in-memory PNG byte stream."""
    if fig_ax is None:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        close_fig = True
    else:
        fig, ax = fig_ax
        ax.clear()
        close_fig = False

    h, w = grid.shape
    color_grid = np.zeros((h, w, 3), dtype=np.float32)
    color_grid[grid == 0] = [0.96, 0.96, 0.98]  # Clean free-space background
    color_grid[grid == 1] = [0.15, 0.18, 0.25]  # Dark slate obstacle blocks

    ax.imshow(color_grid, interpolation="nearest", extent=[0, w, h, 0])

    # Start marker (vivid green)
    ax.plot(start[1] + 0.5, start[0] + 0.5, "o", color="#10b981", markersize=14, markeredgewidth=2, markeredgecolor="white")
    ax.text(start[1] + 0.5, start[0] + 0.5, "S", color="white", fontsize=9, fontweight="bold", ha="center", va="center")

    # Goal marker (vivid red)
    ax.plot(goal[1] + 0.5, goal[0] + 0.5, "*", color="#ef4444", markersize=18, markeredgewidth=1.5, markeredgecolor="white")
    ax.text(goal[1] + 0.5, goal[0] + 0.5, "G", color="white", fontsize=8, fontweight="bold", ha="center", va="center")

    # Subtle grid lines
    ax.set_xticks(np.arange(0, w + 1, 1), minor=True)
    ax.set_yticks(np.arange(0, h + 1, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=0.7)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks([0, 5, 10, w - 1])
    ax.set_yticks([0, 5, 10, h - 1])
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)

    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    if close_fig:
        plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def export_dataset_to_zip(
    npz_path: Path,
    zip_path: Path,
    extract_dir: Optional[Path] = None,
    dataset_name: str = "test",
) -> int:
    """Loads an .npz archive, renders all environments, and saves them to a compressed zip."""
    if not npz_path.exists():
        print(f"Warning: File '{npz_path}' not found. Skipping.")
        return 0

    print(f"Loading '{npz_path.name}'...")
    data = np.load(npz_path)
    grids = data["grids"]
    starts = data["starts"]
    goals = data["goals"]
    num_envs = len(grids)

    print(f"Exporting {num_envs} environments to '{zip_path.name}'...")
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

    if extract_dir:
        extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for idx in range(num_envs):
            grid = grids[idx]
            start = tuple(starts[idx])
            goal = tuple(goals[idx])
            title = f"{dataset_name.capitalize()} Environment #{idx+1:03d} (15x15)"

            png_bytes = render_single_grid(grid, start, goal, title, fig_ax=(fig, ax))
            filename = f"env_{idx+1:03d}.png"
            zf.writestr(filename, png_bytes)

            if extract_dir:
                with open(extract_dir / filename, "wb") as f:
                    f.write(png_bytes)

            if (idx + 1) % 100 == 0 or idx == num_envs - 1:
                print(f"  Processed {idx + 1}/{num_envs} images...")

    plt.close(fig)
    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"✓ Created compressed archive: '{zip_path}' ({zip_size_mb:.2f} MB, {num_envs} images).")
    return num_envs


def generate_mosaic(npz_path: Path, output_image: Path, num_rows: int = 5, num_cols: int = 5, dataset_name: str = "Test"):
    """Generates a high-resolution 5x5 mosaic of 25 benchmark environments."""
    if not npz_path.exists():
        return

    data = np.load(npz_path)
    grids = data["grids"]
    starts = data["starts"]
    goals = data["goals"]
    total_samples = num_rows * num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15), dpi=150)
    for idx in range(total_samples):
        r, c = idx // num_cols, idx % num_cols
        ax = axes[r, c]

        grid = grids[idx]
        start = tuple(starts[idx])
        goal = tuple(goals[idx])
        h, w = grid.shape

        color_grid = np.zeros((h, w, 3), dtype=np.float32)
        color_grid[grid == 0] = [0.96, 0.96, 0.98]
        color_grid[grid == 1] = [0.15, 0.18, 0.25]

        ax.imshow(color_grid, interpolation="nearest", extent=[0, w, h, 0])
        ax.plot(start[1] + 0.5, start[0] + 0.5, "o", color="#10b981", markersize=8, markeredgewidth=1.5, markeredgecolor="white")
        ax.plot(goal[1] + 0.5, goal[0] + 0.5, "*", color="#ef4444", markersize=10, markeredgewidth=1, markeredgecolor="white")

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Map #{idx+1:03d}", fontsize=9, fontweight="bold", pad=4)

    plt.suptitle(f"Sample Benchmark Layouts: {dataset_name} Environments (15x15)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    output_image.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_image, bbox_inches="tight")
    plt.close()
    print(f"✓ Generated visual mosaic: '{output_image}' ({output_image.stat().st_size / 1024:.1f} KB).")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("BENCHMARK ENVIRONMENT VISUALIZATION & ARCHIVE GENERATOR")
    print("=" * 70)

    if args.dataset in ["test", "both"]:
        test_zip = out_dir / "test_env_images_500.zip"
        test_extract = (out_dir / "test_images") if args.extract_folder else None
        export_dataset_to_zip(Path(args.test_npz), test_zip, test_extract, "test")
        if args.make_mosaic:
            generate_mosaic(Path(args.test_npz), RESULTS_DIR / "test_environments_mosaic_25.png", dataset_name="Test")

    if args.dataset in ["train", "both"]:
        train_zip = out_dir / "train_env_images_500.zip"
        train_extract = (out_dir / "train_images") if args.extract_folder else None
        export_dataset_to_zip(Path(args.train_npz), train_zip, train_extract, "train")
        if args.make_mosaic:
            generate_mosaic(Path(args.train_npz), RESULTS_DIR / "train_environments_mosaic_25.png", dataset_name="Training")

    print("\n✓ Environment image extraction and compression completed successfully.")


if __name__ == "__main__":
    main()
