"""Chart generation for saved OLMS analysis outputs."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "olms_matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Charts:
  """Create presentation charts from an existing analysis output directory."""

  COMPONENTS = ["org", "lex", "meaning", "struct"]
  PROMPTS = ["A", "B"]
  COLORS = {"A": "#2F6FAD", "B": "#C65A2E"}
  MARKERS = {"A": "o", "B": "s"}

  def __init__(self, analysis_dir, output_dir):
    self.analysis_dir = Path(analysis_dir)
    self.output_dir = Path(output_dir)
    if not self.analysis_dir.is_dir():
      raise NotADirectoryError(f"Analysis directory does not exist: {self.analysis_dir}")
    if self.output_dir.exists() and any(self.output_dir.iterdir()):
      raise FileExistsError(
        f"Chart output directory is not empty: {self.output_dir}. Choose a new --output directory."
      )
    self.output_dir.mkdir(parents=True, exist_ok=True)

  def primary_rq1(self):
    """Plot essay-level within-prompt instability with bootstrap intervals."""
    within = self._read_csv("within_essay_stability.csv")
    intervals = self._read_csv("within_essay_intervals.csv")
    self._validate_within(within)
    self._validate_intervals(intervals)

    essays = self._essay_order(within)
    fig_height = max(5.5, len(essays) * 0.45 + 1.4)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    y_positions = np.arange(len(essays))
    offsets = {"A": -0.14, "B": 0.14}

    for prompt in self.PROMPTS:
      rows = within[within.prompt == prompt].set_index("essay_file").loc[essays]
      ci = (intervals[(intervals.prompt == prompt) & (intervals.metric == "mean_distance")]
            .set_index("essay_file").loc[essays])
      x = rows["mean_distance"].to_numpy(dtype=float)
      lower = ci["lower"].to_numpy(dtype=float)
      upper = ci["upper"].to_numpy(dtype=float)
      y = y_positions + offsets[prompt]
      ax.hlines(y, lower, upper, color=self.COLORS[prompt], linewidth=1.2, alpha=0.9)
      ax.vlines(lower, y - 0.045, y + 0.045, color=self.COLORS[prompt], linewidth=1.0)
      ax.vlines(upper, y - 0.045, y + 0.045, color=self.COLORS[prompt], linewidth=1.0)
      ax.scatter(
        x, y, marker=self.MARKERS[prompt], color=self.COLORS[prompt],
        s=34, label=f"Prompt {prompt}", zorder=3
      )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([self._short_essay_name(name) for name in essays])
    ax.invert_yaxis()
    ax.set_xlabel("Mean distance from essay-prompt OLMS centroid")
    ax.set_title("RQ1 Within-Essay Stability")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    self._save_figure(fig, "primary_rq1")
    self._write_markdown("primary_rq1", self._primary_rq1_markdown())
    plt.close(fig)
    return self._paths("primary_rq1")

  def diagnostic_rq1(self):
    """Plot component-level same-prompt OLMS centroids by essay."""
    within = self._read_csv("within_essay_stability.csv")
    self._validate_within(within)

    essays = self._essay_order(within)
    fig_height = max(6.0, len(essays) * 0.55 + 1.5)
    fig, axes = plt.subplots(1, len(self.COMPONENTS), figsize=(15, fig_height), sharey=True)
    y_positions = np.arange(len(essays))
    offsets = {"A": -0.13, "B": 0.13}

    for ax, component in zip(axes, self.COMPONENTS):
      column = f"{component}_mean"
      for prompt in self.PROMPTS:
        rows = within[within.prompt == prompt].set_index("essay_file").loc[essays]
        ax.scatter(
          rows[column].to_numpy(dtype=float), y_positions + offsets[prompt],
          color=self.COLORS[prompt], marker=self.MARKERS[prompt], s=34,
          label=f"Prompt {prompt}" if component == self.COMPONENTS[0] else None
        )
      ax.set_title(component.title())
      ax.set_xlabel("Similarity")
      ax.set_xlim(0, 1)
      ax.grid(axis="x", alpha=0.25)

    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels([self._short_essay_name(name) for name in essays])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Essay")
    fig.suptitle("RQ1 OLMS Component Centroids", y=0.995)
    fig.legend(loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    self._save_figure(fig, "diagnostic_rq1")
    self._write_markdown("diagnostic_rq1", self._diagnostic_rq1_markdown())
    plt.close(fig)
    return self._paths("diagnostic_rq1")

  def _read_csv(self, filename):
    path = self.analysis_dir / filename
    if not path.exists():
      raise FileNotFoundError(f"Required analysis file is missing: {path}")
    return pd.read_csv(path, keep_default_na=False)

  def _validate_within(self, frame):
    required = {"essay_file", "prompt", "mean_distance"} | {
      f"{component}_mean" for component in self.COMPONENTS
    }
    missing = required - set(frame.columns)
    if missing:
      raise ValueError(f"within_essay_stability.csv is missing columns: {sorted(missing)}")
    if not set(frame.prompt) == set(self.PROMPTS):
      raise ValueError("within_essay_stability.csv must contain Prompt A and Prompt B rows.")
    for column in ["mean_distance", *[f"{component}_mean" for component in self.COMPONENTS]]:
      values = pd.to_numeric(frame[column], errors="raise")
      if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in {column}.")
      frame[column] = values
    expected_rows = len(frame.essay_file.unique()) * len(self.PROMPTS)
    if len(frame) != expected_rows or frame.duplicated(["essay_file", "prompt"]).any():
      raise ValueError("within_essay_stability.csv must have one row per essay and prompt.")

  def _validate_intervals(self, frame):
    required = {"essay_file", "prompt", "metric", "lower", "upper"}
    missing = required - set(frame.columns)
    if missing:
      raise ValueError(f"within_essay_intervals.csv is missing columns: {sorted(missing)}")
    subset = frame[frame.metric == "mean_distance"]
    if len(subset) == 0:
      raise ValueError("within_essay_intervals.csv must include mean_distance intervals.")
    for column in ["lower", "upper"]:
      values = pd.to_numeric(frame[column], errors="raise")
      if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in interval column {column}.")
      frame[column] = values

  def _essay_order(self, within):
    prompt_a = within[within.prompt == "A"].sort_values("essay_file")
    return prompt_a["essay_file"].tolist()

  def _save_figure(self, fig, name):
    for suffix in ["png", "jpg"]:
      fig.savefig(self.output_dir / f"{name}.{suffix}", dpi=200, bbox_inches="tight")

  def _write_markdown(self, name, text):
    (self.output_dir / f"{name}.md").write_text(text, encoding="utf-8")

  def _paths(self, name):
    return [self.output_dir / f"{name}.{suffix}" for suffix in ["png", "jpg", "md"]]

  @staticmethod
  def _short_essay_name(name):
    stem = Path(name).stem
    parts = stem.split("_")
    return " ".join(parts[2:]) if len(parts) > 2 else stem

  @staticmethod
  def _primary_rq1_markdown():
    return "\n".join([
      "# primary_rq1",
      "",
      "Research Question 1 asks: when the essay and prompt are unchanged, how much does the model response vary across repeated generations?",
      "",
      "This chart plots one point for each essay-prompt condition. The x-axis is `mean_distance`, the average Euclidean distance from each same-prompt OLMS pair vector to that essay-prompt centroid. Lower values mean the repeated responses are more tightly clustered and therefore more stable.",
      "",
      "The horizontal intervals are response-level bootstrap confidence intervals for `mean_distance`. Prompt A and Prompt B are shown separately so their same-prompt stability can be compared within each essay.",
      "",
    ])

  @staticmethod
  def _diagnostic_rq1_markdown():
    return "\n".join([
      "# diagnostic_rq1",
      "",
      "Research Question 1 is primarily about same-prompt response stability. The primary chart answers this with total OLMS distance; this diagnostic chart shows which OLMS components underlie that stability pattern.",
      "",
      "Each panel plots the essay-prompt centroid for one OLMS component: organization, lexical similarity, meaning similarity, and structural similarity. Higher component values mean same-prompt response pairs are more similar on that dimension.",
      "",
      "Comparing the panels shows whether stability is broad across all dimensions or concentrated in particular dimensions. For example, high organization and lexical scores with lower meaning scores would suggest that repeated feedback keeps a similar form while varying more in semantic content.",
      "",
    ])
