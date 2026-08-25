"""Shared loading and cohort logic for NeuroVault dump analyses.

Used by `audit_decoding_uploads.py` and `audit_image_text_corpus.py`.

Two things about these CSVs that will silently corrupt an analysis if missed:

1. **The literal string "None" is a real primary key.** It belongs to the
   `statmaps_cognitiveatlastask` entry named "None / Other". Every task id in
   the dump joins to that table -- zero orphans -- so it is a genuine user
   selection, not a serialised NULL. Pandas' default `na_values` swallows it,
   merging "the user chose None/Other" into "the user answered nothing". In
   february_2024 that is 301,879 maps vs 75,452. Everything here is read with
   `keep_default_na=False` and blankness is tested explicitly.

2. **Neuroscout is roughly half the database.** Owner 5761 is the Neuroscout
   mass-upload account (naturalistic-stimuli re-analyses pushed from a separate
   platform): 3,889 collections and 289,248 statistic maps in february_2024,
   51.7% of all maps. It is not user-contributed content and is excluded by
   default, matching the convention in `large-scale-ibma/get_nv_*.py`.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Convention shared with large-scale-ibma/get_nv_collections.py
NEUROSCOUT_OWNER_ID = "5761"
REST_EYES_OPEN_ID = "trm_4c8a834779883"
REST_EYES_CLOSED_ID = "trm_54e69c642d89b"

BLANK = "(blank)"
NONE_OTHER = "None"  # PK of the "None / Other" task entry
SIZE_CAP = 1000  # collections larger than this are "mass uploads"


def data_dir():
    return Path(os.environ.get("NV_DATA_DIR", "february_2024"))


def out_dir(name):
    d = Path("audit_output") / data_dir().name / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def read(table, usecols=None):
    """Read a table as raw strings, or None if this dump lacks it."""
    path = data_dir() / f"{table}.csv"
    if not path.exists():
        print(f"  ! {table} absent from {data_dir().name}")
        return None
    return pd.read_csv(path, usecols=usecols, dtype=str,
                       keep_default_na=False, low_memory=False)


def blank(s):
    return s.fillna("").astype(str).str.strip() == ""


def is_none_other(s):
    return s.fillna("").astype(str).str.strip() == NONE_OTHER


def answered(s):
    """Genuinely answered: neither blank nor the None/Other escape value."""
    return ~blank(s) & ~is_none_other(s)


def cat(s):
    return s.fillna("").astype(str).str.strip().replace("", BLANK)


def num(s):
    return pd.to_numeric(s.replace("", np.nan), errors="coerce")


def report(where, name, obj):
    """Print a result and persist it next to the other tables."""
    print(f"\n----- {name} -----")
    print(obj)
    save(where, name, obj)
    return obj


def save(where, name, obj):
    """Persist a result and return it, so a notebook cell can render it."""
    (obj.to_frame() if isinstance(obj, pd.Series) else obj).to_csv(where / f"{name}.csv")
    return obj


def setup_plots():
    """Consistent, readable figure defaults."""
    import matplotlib as mpl

    # Under Jupyter the inline backend is already configured. Run as a plain
    # script, a GUI backend would block on plt.show(), so force Agg.
    try:
        get_ipython()  # noqa: F821
    except NameError:
        mpl.use("Agg")

    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "figure.figsize": (9, 4.5), "figure.dpi": 110,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-",
        "axes.axisbelow": True, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })
    return plt


def barh(ax, series, title, xlabel="maps", fmt="{:,.0f}", color="#4C78A8",
         highlight=None, highlight_color="#E45756"):
    """Horizontal bar chart with value labels, largest at top."""
    s = series.sort_values()
    colors = [highlight_color if (highlight and i in highlight) else color
              for i in s.index]
    ax.barh(s.index.astype(str), s.values, color=colors)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    span = max(s.values.max(), 1)
    for y, v in enumerate(s.values):
        ax.text(v + span * 0.01, y, fmt.format(v), va="center", fontsize=8.5)
    ax.set_xlim(0, span * 1.15)
    return ax


def load(exclude_neuroscout=True):
    """Return (statistic maps, collections), both cohort-annotated.

    Collections carry `stratum` (one of temporary / doi_backed / no_doi),
    `is_mass_upload` and `n_items`. Maps inherit those plus `add_year`.
    """
    d = data_dir()
    print(f"Loading {d}  (exclude_neuroscout={exclude_neuroscout})")

    coll = read("statmaps_collection",
                usecols=["id", "name", "DOI", "paper_url", "journal_name",
                         "add_date", "coordinate_space", "owner_id"])
    item = read("statmaps_basecollectionitem",
                usecols=["id", "name", "description", "add_date",
                         "collection_id", "is_valid"])
    statmap = read("statmaps_statisticmap")
    image = read("statmaps_image",
                 usecols=["basecollectionitem_ptr_id", "target_template_image"])

    n_coll_all, n_map_all = len(coll), len(statmap)

    # Merge chain as established in `data_access.py`.
    image_merged = image.merge(item, left_on="basecollectionitem_ptr_id",
                              right_on="id", how="left")
    sm = statmap.merge(image_merged, left_on="image_ptr_id",
                       right_on="basecollectionitem_ptr_id", how="left")

    # Cohorts, by precedence, so every map lands in exactly one stratum.
    coll["is_neuroscout"] = coll["owner_id"] == NEUROSCOUT_OWNER_ID
    coll["is_temporary"] = coll["name"].fillna("").str.lower().str.contains(
        "temporary collection")
    coll["has_doi"] = ~blank(coll["DOI"])
    coll["stratum"] = np.where(coll["is_temporary"], "temporary",
                        np.where(coll["has_doi"], "doi_backed", "no_doi"))

    n_items = item.groupby("collection_id").size().rename("n_items")
    coll = coll.merge(n_items, left_on="id", right_index=True, how="left")
    coll["n_items"] = coll["n_items"].fillna(0).astype(int)
    coll["is_mass_upload"] = coll["n_items"] > SIZE_CAP

    sm = sm.merge(
        coll[["id", "stratum", "is_mass_upload", "has_doi", "is_temporary",
              "is_neuroscout", "n_items"]],
        left_on="collection_id", right_on="id", how="left", suffixes=("", "_coll"))
    sm["stratum"] = sm["stratum"].fillna("unknown_collection")
    for c in ["is_mass_upload", "is_neuroscout", "is_temporary"]:
        sm[c] = sm[c].fillna(False).astype(bool)

    if exclude_neuroscout:
        ns_maps, ns_coll = int(sm["is_neuroscout"].sum()), int(coll["is_neuroscout"].sum())
        sm = sm[~sm["is_neuroscout"]].copy()
        coll = coll[~coll["is_neuroscout"]].copy()
        print(f"  excluded Neuroscout (owner {NEUROSCOUT_OWNER_ID}): "
              f"{ns_coll} collections, {ns_maps} maps "
              f"({100*ns_maps/n_map_all:.1f}% of all maps)")

    for c in ["map_type", "modality", "analysis_level", "is_thresholded"]:
        sm[c] = cat(sm[c])
    sm["add_year"] = pd.to_datetime(sm["add_date"], errors="coerce",
                                    format="mixed", utc=True).dt.year
    sm["n_subj"] = num(sm["number_of_subjects"])

    print(f"  analysing {len(coll)} collections / {len(sm)} maps "
          f"(dump totals: {n_coll_all} / {n_map_all})")
    assert sm["stratum"].notna().all()
    return sm, coll


def task_names():
    """Cognitive Atlas task id -> name, excluding the None/Other pseudo-task."""
    t = read("statmaps_cognitiveatlastask")
    return t[t["cog_atlas_id"] != NONE_OTHER].copy()
