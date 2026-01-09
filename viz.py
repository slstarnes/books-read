from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter


# -----------------------------
# Helpers
# -----------------------------

def _prep_reading_df(
    df: pd.DataFrame,
    title_col: str = "Title",
    author_col: str = "Author",
    pages_col: str = "pageCount",
    start_col: str = "Date Started",
    finish_col: str = "Date Read",
) -> pd.DataFrame:
    """
    Clean + enrich reading log with derived fields used by all plots.

    Derived columns:
      - started_dt, finished_dt
      - pages (numeric)
      - duration_days (>=1 if both dates exist; NaN otherwise)
      - pages_per_day (pages / duration_days)
      - finish_date (date, normalized)
      - finish_year, finish_month (YYYY-MM period)
    """
    out = df.copy()

    out["started_dt"] = pd.to_datetime(out[start_col], errors="coerce")
    out["finished_dt"] = pd.to_datetime(out[finish_col], errors="coerce")

    # pages: numeric
    out["pages"] = pd.to_numeric(out[pages_col], errors="coerce")

    # Normalize finished to date (midnight) for grouping/time series
    out["finish_date"] = out["finished_dt"].dt.normalize()

    # Duration in days: require both dates. If started missing, leave NaN.
    dur = (out["finished_dt"] - out["started_dt"]).dt.days
    # If duration is 0 or negative but dates exist, treat as 1 day (a same-day read or bad logging)
    dur = dur.where(dur.isna(), np.maximum(dur, 0))
    dur = dur.replace(0, 1)
    out["duration_days"] = dur

    out["pages_per_day"] = out["pages"] / out["duration_days"]

    # Year and Year-Month for finished date groupings
    out["finish_year"] = out["finish_date"].dt.year
    out["finish_ym"] = out["finish_date"].dt.to_period("M")

    # Basic filters for plotting (keep rows with a finished date)
    out = out.loc[out["finish_date"].notna()].copy()

    # Optional: fill missing titles/authors for labeling
    out[title_col] = out[title_col].fillna("Untitled")
    out[author_col] = out[author_col].fillna("Unknown")

    return out


def _savefig(fig: plt.Figure, outpath: Path, dpi: int = 200) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _nice_date_axis(ax: plt.Axes, years: bool = True) -> None:
    if years:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=(1, 7)))
    ax.grid(True, which="major", axis="both", alpha=0.25)


# -----------------------------
# Plot 1: Cumulative pages read over time
# -----------------------------

def plot_cumulative_pages(
    df: pd.DataFrame,
    outpath: str | Path | None=None,
    *,
    title: str = "Cumulative Pages Read Over Time",
) -> None:
    d = _prep_reading_df(df)

    # Sum pages per finish day; then cumulative
    ts = (
        d.groupby("finish_date")["pages"]
        .sum()
        .sort_index()
        .dropna()
        .cumsum()
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(ts.index, ts.values)
    ax.set_title(title)
    ax.set_xlabel("Date Finished")
    ax.set_ylabel("Cumulative Pages")
    _nice_date_axis(ax)
    if outpath:
        _savefig(fig, Path(outpath))
        return fig


# -----------------------------
# Plot 2: Pages per day (reading speed) over time
# -----------------------------

def plot_pages_per_day(
    df: pd.DataFrame,
    outpath: str | Path | None=None,
    *,
    title: str = "Reading Speed (Pages per Day)",
    rolling_days: int = 90,
) -> None:
    d = _prep_reading_df(df)

    # Use per-book pages/day, plotted at finish_date
    s = d[["finish_date", "pages_per_day"]].dropna().sort_values("finish_date")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(s["finish_date"], s["pages_per_day"], s=18, alpha=0.7)

    # Rolling median over time: resample daily then roll
    daily = (
        s.set_index("finish_date")["pages_per_day"]
        .resample("D")
        .median()
    )
    roll = daily.rolling(rolling_days, min_periods=max(10, rolling_days // 10)).median()
    ax.plot(roll.index, roll.values)

    ax.set_title(title)
    ax.set_xlabel("Date Finished")
    ax.set_ylabel("Pages / Day")
    _nice_date_axis(ax)
    if outpath:
        _savefig(fig, Path(outpath))
        return fig


# -----------------------------
# Plot 3: Duration vs Page Count (commitment map)
# -----------------------------

def plot_duration_vs_pages(
    df: pd.DataFrame,
    outpath: str | Path | None=None,
    *,
    title: str = "Commitment Map: Duration vs Page Count",
    annotate_top_n: int = 8,
) -> None:
    d = _prep_reading_df(df)
    s = d[["Title", "pages", "duration_days"]].dropna()
    s = s[(s["pages"] > 0) & (s["duration_days"] > 0)]

    fig, ax = plt.subplots(figsize=(8.5, 6))
    ax.scatter(s["pages"], s["duration_days"], alpha=0.7)

    ax.set_title(title)
    ax.set_xlabel("Page Count")
    ax.set_ylabel("Days to Finish")
    ax.grid(True, alpha=0.25)

    # Annotate "most extreme" points by (pages * duration) magnitude
    if annotate_top_n and len(s) > 0:
        score = (s["pages"] * s["duration_days"]).sort_values(ascending=False)
        top_idx = score.head(annotate_top_n).index
        for i in top_idx:
            ax.annotate(
                s.loc[i, "Title"],
                (s.loc[i, "pages"], s.loc[i, "duration_days"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
                alpha=0.9,
            )

    if outpath:
        _savefig(fig, Path(outpath))
        return fig


# -----------------------------
# Plot 4: Gantt-style reading timeline
# -----------------------------

def plot_gantt_timeline(
    df: pd.DataFrame,
    outpath: str | Path | None=None,
    *,
    title: str = "Reading Timeline (Gantt Style)",
    max_books: int = 60,
    sort_by: str = "started",   # "started" or "finished"
) -> None:
    d = _prep_reading_df(df)

    # Need at least finished_dt. Start can be missing; if so, set start=finish (1 day bar)
    g = d[["Title", "Author", "started_dt", "finished_dt"]].copy()
    g["started_dt"] = g["started_dt"].fillna(g["finished_dt"])
    g = g.dropna(subset=["finished_dt"])

    # Fix any inversions
    bad = g["finished_dt"] < g["started_dt"]
    g.loc[bad, ["started_dt", "finished_dt"]] = g.loc[bad, ["finished_dt", "started_dt"]].values

    g["duration"] = (g["finished_dt"] - g["started_dt"]).dt.days.replace(0, 1)

    if sort_by == "finished":
        g = g.sort_values("finished_dt")
    else:
        g = g.sort_values("started_dt")

    # Take most recent N (blog-friendly)
    g = g.tail(max_books).reset_index(drop=True)

    fig_h = max(4.5, 0.18 * len(g) + 1.2)
    fig, ax = plt.subplots(figsize=(11, fig_h))

    y = np.arange(len(g))
    left = mdates.date2num(g["started_dt"])
    width = g["duration"].to_numpy()

    ax.barh(y, width, left=left)
    ax.set_yticks(y)

    # Label as "Title — Author" but keep short
    labels = (g["Title"].astype(str) + " — " + g["Author"].astype(str)).tolist()
    ax.set_yticklabels(labels, fontsize=8)

    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, axis="x", alpha=0.25)

    ax.set_title(title)
    ax.set_xlabel("Calendar Date")
    ax.set_ylabel("Books")

    if outpath:
        _savefig(fig, Path(outpath))
        return fig


# -----------------------------
# Plot 5: Heatmap (month x year) by pages finished
# -----------------------------

def plot_month_year_heatmap(
    df: pd.DataFrame,
    outpath: str | Path | None=None,
    *,
    title: str = "Heatmap: Pages Finished by Month & Year",
    value: str = "pages",  # "pages" or "books"
) -> None:
    # Make NaNs visually distinct (e.g., light gray) instead of blending with low values
    cmap = plt.colormaps["viridis"].copy()     # colorful; try "viridis"/"magma"/"plasma" too
    # cmap.set_bad(color="#f2f2f2")            # cells with NaN
    d = _prep_reading_df(df)

    if value == "books":
        pivot = (
            d.assign(n=1)
             .groupby(["finish_year", d["finish_date"].dt.month])["n"]
             .sum()
             .unstack(fill_value=0)
        )
        cbar_label = "Books Finished"
    else:
        pivot = (
            d.groupby(["finish_year", d["finish_date"].dt.month])["pages"]
             .sum()
             .unstack(fill_value=0)
        )
        cbar_label = "Pages Finished"

    # Ensure columns 1..12 exist
    pivot = pivot.reindex(columns=range(1, 13), fill_value=0).sort_index()
    pivot.to_csv("pivot.csv")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    # im = ax.imshow(pivot.values, aspect="auto", cmap=cmap, interpolation="nearest")

    mesh = ax.pcolormesh(
            np.ma.masked_invalid(pivot.values),
            cmap=cmap,
            edgecolors="white",
            linewidth=0.6,
            shading="flat",
        )

    vmax = np.nanpercentile(pivot.values, 95)  # or 95
    mesh.set_clim(vmin=0, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    # ax.set_xticks(np.arange(12))
    ax.set_xticks(np.arange(pivot.shape[1]) + 0.5)
    ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels(pivot.index.astype(int).tolist())

    # Center tick labels on cells
    
    # ax.set_xticklabels(mesh.columns.tolist())
    # ax.set_yticks(np.arange(mesh.shape[0]) + 0.5)
    # ax.set_yticklabels(mesh.index.astype(int).tolist())

    # Put first year at top (to match your current look)
    ax.invert_yaxis()

    # ax.set_yticks(np.arange(len(pivot.index)))
    # ax.set_yticklabels(pivot.index.astype(int).tolist())

    cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(cbar_label)
    if value == "pages":
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    if outpath:
        _savefig(fig, Path(outpath))
        return fig


# -----------------------------
# Plot 6: Books finished per year with page weighting
# -----------------------------

def plot_books_per_year_weighted(
    df: pd.DataFrame,
    outpath: str | Path | None=None,
    *,
    title: str = "Books Finished per Year (Weighted by Length)",
    bins: tuple[int, ...] = (0, 200, 350, 500, 800, 10_000),
) -> None:
    d = _prep_reading_df(df)

    s = d[["finish_year", "pages"]].dropna()
    s = s[s["pages"] > 0].copy()

    # Bin by length
    labels = []
    for i in range(len(bins) - 1):
        labels.append(f"{bins[i]}–{bins[i+1]-1}")

    s["len_bin"] = pd.cut(s["pages"], bins=bins, labels=labels, right=False, include_lowest=True)

    # Count books per year per bin
    counts = (
        s.groupby(["finish_year", "len_bin"])
         .size()
         .unstack(fill_value=0)
         .sort_index()
    )

    # Also compute total pages per year for a line overlay (nice context)
    pages_per_year = s.groupby("finish_year")["pages"].sum().reindex(counts.index)

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)

    bottom = np.zeros(len(counts), dtype=float)
    x = np.arange(len(counts.index))

    for col in counts.columns:
        ax.bar(x, counts[col].values, bottom=bottom, label=str(col))
        bottom += counts[col].values

    ax.set_title(title)
    ax.set_xlabel("Year")
    ax.set_ylabel("Books Finished")
    ax.set_xticks(x)
    ax.set_xticklabels(counts.index.astype(int).tolist())
    ax.grid(True, axis="y", alpha=0.25)

    # Secondary axis for pages per year
    ax2 = ax.twinx()
    ax2.plot(x, pages_per_year.values, linewidth=2)
    ax2.set_ylabel("Total Pages Finished")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))

    # ax.legend(title="Page Count Bin", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)

    leg = ax.legend(
        title="Page Count Bin",
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.7",
        fancybox=True,
    )

    leg.get_frame().set_boxstyle("round,pad=0.4")
    

    ax2.set_zorder(ax.get_zorder() - 1)     # draw ax2 behind ax
    # ax2.patch.set_visible(False)            # keep background transparent

    if outpath:
        _savefig(fig, Path(outpath))
        return fig


# -----------------------------
# Plot 7: Reading efficiency trend (median pages/day by year)
# -----------------------------

def plot_efficiency_trend(
    df: pd.DataFrame,
    outpath: str | Path | None=None,
    *,
    title: str = "Reading Efficiency Trend (Median Pages/Day by Year)",
    min_books_per_year: int = 3,
) -> None:
    d = _prep_reading_df(df)

    s = d[["finish_year", "pages_per_day"]].dropna()
    s = s[s["pages_per_day"] > 0]

    by_year = s.groupby("finish_year")["pages_per_day"].agg(["median", "count"]).sort_index()
    by_year = by_year[by_year["count"] >= min_books_per_year]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(by_year.index.astype(int), by_year["median"].values, marker="o")

    ax.set_title(title)
    ax.set_xlabel("Year Finished")
    ax.set_ylabel("Median Pages/Day")
    ax.grid(True, alpha=0.25)

    if outpath:
        _savefig(fig, Path(outpath))
        return fig


# -----------------------------
# One-shot runner
# -----------------------------

def make_all_reading_plots(
    df: pd.DataFrame,
    outdir: str | Path = "reading_plots",
) -> dict[str, Path]:
    outdir = Path(outdir)
    paths = {
        "cumulative_pages": outdir / "01_cumulative_pages.png",
        "pages_per_day": outdir / "02_pages_per_day.png",
        "duration_vs_pages": outdir / "03_duration_vs_pages.png",
        "gantt_timeline": outdir / "04_gantt_timeline.png",
        "month_year_heatmap": outdir / "05_month_year_heatmap_pages.png",
        "books_per_year_weighted": outdir / "06_books_per_year_weighted.png",
        "efficiency_trend": outdir / "07_efficiency_trend.png",
    }

    plot_cumulative_pages(df, paths["cumulative_pages"])
    plot_pages_per_day(df, paths["pages_per_day"])
    plot_duration_vs_pages(df, paths["duration_vs_pages"])
    plot_gantt_timeline(df, paths["gantt_timeline"])
    plot_month_year_heatmap(df, paths["month_year_heatmap"], value="pages")
    plot_books_per_year_weighted(df, paths["books_per_year_weighted"])
    plot_efficiency_trend(df, paths["efficiency_trend"])

    return paths