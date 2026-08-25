# ### Auditing the NeuroVault Archived Data
#
# Characterises the contents of a NeuroVault dump: what kinds of images are in
# there, how well annotated they are, and which subsets are usable for
# image-based meta-analysis or for training text/image models.
#
# Runs against any dump in this repo. Extract the archive first, then either
# edit `DATA_DIR` below or set `NV_DATA_DIR`:
#
# +
# #!tar -xzf february_2024.tar.gz
# #!NV_DATA_DIR=november_2022 python audit_neurovault.py
# -
#
# Two things to know before reading any number out of this:
#
# 1. **Read section B first.** Raw corpus-wide counts are dominated by a handful
#    of mass uploads and are misleading on their own.
# 2. **These CSVs contain the literal string `"None"`**, which is a real primary
#    key in `statmaps_cognitiveatlastask` for the entry named `"None / Other"`.
#    Pandas' default `na_values` swallows it, silently merging "the user chose
#    None/Other" into "the user answered nothing" -- two very different facts,
#    and the difference is ~300k maps. Everything below is therefore read with
#    `keep_default_na=False` and blankness is tested explicitly.

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DATA_DIR = Path(os.environ.get("NV_DATA_DIR", "february_2024"))
OUT_DIR = Path("audit_output") / DATA_DIR.name
SIZE_CAP = 1000  # collections with more items than this are "mass uploads"

OUT_DIR.mkdir(parents=True, exist_ok=True)

BLANK = "(blank)"
NONE_OTHER = "None"  # the literal PK of the "None / Other" task entry


def load(table, usecols=None):
    """Read a table as raw strings, or None if this dump lacks it.

    Dumps differ: november_2022 has django_content_type but no NIDM tables;
    february_2024 is the reverse.
    """
    path = DATA_DIR / f"{table}.csv"
    if not path.exists():
        print(f"  ! {table} absent from {DATA_DIR.name}")
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


def cat(series):
    out = series.fillna("").astype(str).str.strip()
    return out.replace("", BLANK)


def num(series):
    return pd.to_numeric(series.replace("", np.nan), errors="coerce")


def report(name, obj):
    print(f"\n----- {name} -----")
    print(obj)
    out = obj.to_frame() if isinstance(obj, pd.Series) else obj
    out.to_csv(OUT_DIR / f"{name}.csv")
    return obj


# ## Load

print(f"Loading {DATA_DIR}")

collection = load(
    "statmaps_collection",
    usecols=["id", "name", "DOI", "paper_url", "journal_name", "add_date",
             "coordinate_space", "owner_id", "group_comparison"],
)
item = load(
    "statmaps_basecollectionitem",
    usecols=["id", "name", "description", "add_date", "collection_id", "is_valid"],
)
statmap = load("statmaps_statisticmap")
image = load("statmaps_image", usecols=["basecollectionitem_ptr_id", "target_template_image"])
task = load("statmaps_cognitiveatlastask")

# Merge chain as established in `data_access.py`: image joins to
# basecollectionitem on `basecollectionitem_ptr_id`, then statisticmap joins to
# that on `image_ptr_id`.

image_merged = image.merge(item, left_on="basecollectionitem_ptr_id", right_on="id",
                           how="left", suffixes=("_img", "_item"))
sm = statmap.merge(image_merged, left_on="image_ptr_id",
                   right_on="basecollectionitem_ptr_id", how="left")

print(f"collections={len(collection)}  items={len(item)}  statmaps={len(sm)}")

# `django_content_type` is missing from newer dumps, so polymorphic type cannot
# be resolved through it. Type is instead implied by membership of the
# type-specific tables, which is what the merge above does.


# ## B. Provenance strata
#
# Assigned per collection, by precedence, so every statistic map lands in
# exactly one stratum:
#
# 1. `temporary` -- auto-named "<user>'s temporary collection". This is the
#    Neurosynth decoder upload flow: a throwaway collection created so a user
#    could get a map decoded. No marker field exists; the generated name is the
#    only reliable signal.
# 2. `no_doi` -- a real collection, but no DOI recorded.
# 3. `doi_backed` -- has a DOI.
#
# `is_mass_upload` is orthogonal and flags collections above `SIZE_CAP` items.

coll = collection.copy()
coll["is_temporary"] = coll["name"].fillna("").str.lower().str.contains("temporary collection")
coll["has_doi"] = ~blank(coll["DOI"])
coll["stratum"] = np.where(coll["is_temporary"], "temporary",
                    np.where(coll["has_doi"], "doi_backed", "no_doi"))

n_items = item.groupby("collection_id").size().rename("n_items")
coll = coll.merge(n_items, left_on="id", right_index=True, how="left")
coll["n_items"] = coll["n_items"].fillna(0).astype(int)
coll["is_mass_upload"] = coll["n_items"] > SIZE_CAP

sm = sm.merge(
    coll[["id", "stratum", "is_mass_upload", "has_doi", "is_temporary", "n_items"]],
    left_on="collection_id", right_on="id", how="left", suffixes=("", "_coll"),
)
# Maps whose collection is absent from the collection table (deleted/private)
# get their own stratum so the partition stays exact.
sm["stratum"] = sm["stratum"].fillna("unknown_collection")
sm["is_mass_upload"] = sm["is_mass_upload"].fillna(False).astype(bool)

report("B_collections_per_stratum", coll["stratum"].value_counts())
report("B_maps_per_stratum", sm["stratum"].value_counts())
report("B_provenance_fields", pd.Series({
    "collections": len(coll),
    "with_DOI": int(coll["has_doi"].sum()),
    "with_paper_url": int((~blank(coll["paper_url"])).sum()),
    "with_journal_name": int((~blank(coll["journal_name"])).sum()),
    "temporary": int(coll["is_temporary"].sum()),
    "empty (0 items)": int((coll["n_items"] == 0).sum()),
    "mass_uploads": int(coll["is_mass_upload"].sum()),
}))
report("B_collection_size_summary", coll["n_items"].describe(
    percentiles=[.5, .75, .9, .95, .99]))

top5 = coll.nlargest(5, "n_items")[["id", "name", "n_items"]]
top5_share = 100 * top5["n_items"].sum() / coll["n_items"].sum()
print(f"\ntop-5 collections hold {top5_share:.1f}% of all items")
report("B_largest_collections", top5)

assert coll["stratum"].value_counts().sum() == len(coll)
assert sm["stratum"].value_counts().sum() == len(sm)


# ## A. Inventory
#
# Reported both across all maps and excluding mass uploads, because the two tell
# different stories.

for col in ["map_type", "modality", "analysis_level", "is_thresholded"]:
    sm[col] = cat(sm[col])

nomass = sm[~sm["is_mass_upload"]]


def both_views(col):
    return pd.DataFrame({
        "all_maps": sm[col].value_counts(),
        "excl_mass_uploads": nomass[col].value_counts(),
    }).fillna(0).astype(int)


report("A_map_type", both_views("map_type"))
report("A_modality", both_views("modality"))
report("A_analysis_level", both_views("analysis_level"))
report("A_maptype_x_analysislevel", pd.crosstab(sm["map_type"], sm["analysis_level"]))
report("A_thresholded_x_analysislevel", pd.crosstab(sm["is_thresholded"], sm["analysis_level"]))
report("A_target_template", cat(sm["target_template_image"]).value_counts().head(15))
report("A_coordinate_space", cat(coll["coordinate_space"]).value_counts())

sm["add_year"] = pd.to_datetime(sm["add_date"], errors="coerce", format="mixed",
                                utc=True).dt.year
report("A_uploads_by_year", pd.crosstab(sm["add_year"], sm["stratum"]))


# ## C. The decoder upload pile
#
# Profile of the `temporary` stratum: what the Neurosynth decoding flow actually
# captured, and where it failed to. This is the subset the upload-flow redesign
# turns on.

temp = sm[sm["stratum"] == "temporary"]
temp_coll = coll[coll["is_temporary"]]

report("C_summary", pd.Series({
    "temporary collections": len(temp_coll),
    "  of which empty": int((temp_coll["n_items"] == 0).sum()),
    "  of which have a DOI": int(temp_coll["has_doi"].sum()),
    "statistic maps within": len(temp),
    "thresholded (violates flow instructions)": int((temp["is_thresholded"] == "t").sum()),
    "unthresholded (compliant)": int((temp["is_thresholded"] == "f").sum()),
    "threshold status unstated": int((temp["is_thresholded"] == BLANK).sum()),
    "analysis_level unstated": int((temp["analysis_level"] == BLANK).sum()),
    "task answered": int(answered(temp["cognitive_paradigm_cogatlas_id"]).sum()),
    "task = None/Other": int(is_none_other(temp["cognitive_paradigm_cogatlas_id"]).sum()),
    "task blank": int(blank(temp["cognitive_paradigm_cogatlas_id"]).sum()),
    "has a description": int((~blank(temp["description"])).sum()),
}))
report("C_map_type", temp["map_type"].value_counts())
report("C_modality", temp["modality"].value_counts())
report("C_by_year", temp["add_year"].value_counts().sort_index())


# ## D. Annotation quality
#
# The headline: `cognitive_paradigm` has *three* states, not two. Collapsing
# None/Other into missing (as any default pandas read does) hides the fact that
# a majority of uploads take the escape hatch rather than leaving it empty.

cog = sm["cognitive_paradigm_cogatlas_id"]
report("D_cog_paradigm_three_states", pd.Series({
    "real task selected": int(answered(cog).sum()),
    "explicitly None / Other": int(is_none_other(cog).sum()),
    "left blank": int(blank(cog).sum()),
    "total": len(sm),
}))
report("D_cog_paradigm_by_stratum", pd.DataFrame({
    "n_maps": sm.groupby("stratum").size(),
    "real_task": answered(cog).groupby(sm["stratum"]).sum(),
    "none_other": is_none_other(cog).groupby(sm["stratum"]).sum(),
    "blank": blank(cog).groupby(sm["stratum"]).sum(),
}))

fill = pd.DataFrame({
    "n_maps": sm.groupby("stratum").size(),
    "cog_paradigm": answered(cog).groupby(sm["stratum"]).mean(),
    "cog_contrast": answered(sm["cognitive_contrast_cogatlas_id"]).groupby(sm["stratum"]).mean(),
    "contrast_definition": answered(sm["contrast_definition"]).groupby(sm["stratum"]).mean(),
    "number_of_subjects": num(sm["number_of_subjects"]).notna().groupby(sm["stratum"]).mean(),
    "description": (~blank(sm["description"])).groupby(sm["stratum"]).mean(),
})
report("D_fill_rate_by_stratum", fill.round(3))

# List-position bias, over real task selections only (the None/Other pseudo-task
# is excluded -- it isn't a position in the list users scroll).
#
# Caveat to carry into any write-up: the dump does not record the order the
# picker presented tasks in, so alphabetical rank is a *proxy* for it.
sel = cog[answered(cog)].value_counts().rename("n_selected")
tasks = task[task["cog_atlas_id"] != NONE_OTHER].copy()
tasks["alpha_rank"] = tasks["name"].str.lower().rank(method="first")
tasks = tasks.merge(sel, left_on="cog_atlas_id", right_index=True, how="left")
tasks["n_selected"] = tasks["n_selected"].fillna(0).astype(int)

rho, p = spearmanr(tasks["alpha_rank"], tasks["n_selected"])
first10 = tasks.nsmallest(10, "alpha_rank")["n_selected"].sum()
total_sel = tasks["n_selected"].sum()
print(f"\nlist-position bias (proxy: alphabetical rank)")
print(f"  spearman rho={rho:.3f} p={p:.3g}")
print(f"  first 10 alphabetical tasks hold {100*first10/max(total_sel,1):.1f}% "
      f"of {total_sel} selections; uniform would be {100*10/len(tasks):.1f}%")
report("D_task_usage_top30", tasks.sort_values("n_selected", ascending=False)
       [["name", "cog_atlas_id", "alpha_rank", "n_selected"]].head(30))
report("D_task_vocabulary_use", pd.Series({
    "tasks in vocabulary (excl. None/Other)": len(tasks),
    "never selected": int((tasks["n_selected"] == 0).sum()),
    "selected once": int((tasks["n_selected"] == 1).sum()),
    "selections concentrated in top 10 tasks (%)":
        round(100 * tasks.nlargest(10, "n_selected")["n_selected"].sum() / max(total_sel, 1), 1),
}))


# ## E. Text/image pair tiers
#
# How many maps carry enough accompanying text to train or evaluate a
# text-to-image model, at increasing levels of richness.

# `name` and `description` come from basecollectionitem; image and
# basecollectionitem share no column names, so the merge applied no suffix.
has_name = ~blank(sm["name"])
has_desc = ~blank(sm["description"])
has_task = answered(cog)
has_contrast = answered(sm["cognitive_contrast_cogatlas_id"])
has_cdef = answered(sm["contrast_definition"])

report("E_text_tiers", pd.Series({
    "total maps": len(sm),
    "name only": int(has_name.sum()),
    "name + description": int((has_name & has_desc).sum()),
    "name + real task": int((has_name & has_task).sum()),
    "name + desc + real task": int((has_name & has_desc & has_task).sum()),
    "name + desc + task + contrast": int((has_name & has_desc & has_task & has_contrast).sum()),
    "has contrast_definition free text": int(has_cdef.sum()),
    "richest tier, excl. mass uploads":
        int((has_name & has_desc & has_task & ~sm["is_mass_upload"]).sum()),
}))
report("E_text_tiers_by_stratum", pd.DataFrame({
    "n_maps": sm.groupby("stratum").size(),
    "name+desc": (has_name & has_desc).groupby(sm["stratum"]).sum(),
    "name+desc+task": (has_name & has_desc & has_task).groupby(sm["stratum"]).sum(),
}))


# ## F. Per-algorithm IBMA coverage
#
# The baseline requirement for image-based meta-analysis is an unthresholded
# group-level Z or T map with a known sample size. Reported per collection as
# well as per map, since a meta-analysis needs distinct *studies*, not images.

sm["n_subj"] = num(sm["number_of_subjects"])
crit = {
    "unthresholded": sm["is_thresholded"] == "f",
    "group level": sm["analysis_level"] == "G",
    "Z or T map": sm["map_type"].isin(["Z", "T"]),
    "N subjects known": sm["n_subj"].notna(),
}
all_crit = np.logical_and.reduce(list(crit.values()))

rows = {"ALL FOUR": [int(all_crit.sum()), sm.loc[all_crit, "collection_id"].nunique()]}
for label, mask in crit.items():
    rows[f"only {label}"] = [int(mask.sum()), sm.loc[mask, "collection_id"].nunique()]
# Sensitivity: which single requirement costs the most?
for label in crit:
    relaxed = np.logical_and.reduce([m for k, m in crit.items() if k != label])
    rows[f"all except {label}"] = [int(relaxed.sum()), sm.loc[relaxed, "collection_id"].nunique()]

report("F_ibma_coverage", pd.DataFrame(rows, index=["maps", "collections"]).T)
report("F_ibma_eligible_by_stratum",
       sm.loc[all_crit].groupby("stratum").agg(maps=("image_ptr_id", "size"),
                                               collections=("collection_id", "nunique")))


# ## G. Quality field distributions
#
# These four fields are already computed by NeuroVault and can flag likely-bad
# images with no new analysis -- the basis for a pre-computed QA pass.

qa_cols = ["brain_coverage", "perc_bad_voxels", "perc_voxels_outside"]
for c in qa_cols:
    sm[c] = num(sm[c])

report("G_qa_distributions",
       sm[qa_cols].describe(percentiles=[.05, .25, .5, .75, .95]).round(2))
report("G_qa_by_stratum", sm.groupby("stratum")[qa_cols].median().round(2))
report("G_not_mni", cat(sm["not_mni"]).value_counts())
report("G_qa_flags", pd.Series({
    "coverage < 50%": int((sm["brain_coverage"] < 50).sum()),
    "bad voxels > 25%": int((sm["perc_bad_voxels"] > 25).sum()),
    "voxels outside > 25%": int((sm["perc_voxels_outside"] > 25).sum()),
    "flagged not_mni": int((cat(sm["not_mni"]) == "t").sum()),
    "QA fields entirely missing": int(sm[qa_cols].isna().all(axis=1).sum()),
}))

print(f"\nWrote {len(list(OUT_DIR.glob('*.csv')))} tables to {OUT_DIR}/")
