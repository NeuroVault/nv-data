# NeuroVault Public Data Archive

This repository contains archived data from NeuroVault.org.

The data has been dumped from a subset production database concerned with the "statsmaps" application, and censoring has been performed to remove user related information.

## Tables

Within the statsmaps subset, the following tables are included:

### Collections
* `statmaps_collection`
This is the main table that indexes NeuroVault Collections. Critically the `id` column is the primary key for the table, and is used to link to other tables (refered to in other tables as `collection_id`)
* `statmaps_collection_contributors`
Association of collections with `user_id`


### Collection Items / Images
* `statmaps_basecollectionitem`
This is the main table that indexes NeuroVault Collection Items. Critically the `id` column is the primary key for the table, and is used to link to other tables as `basecollectionitem_ptr_id`
* `statmaps_image`
Images are a type of Collection Item. This table contains the image specific information. The column `basecollectionitem_ptr_id` in this table corresponds to `statmaps_basecollectionitem.id`
* `statmaps_statisticmap`
StatisticMaps are a type of Images, with additional meta-data, such as `smoothness_fwhm`, `cognitive_paradigm_cogatlas_id`.
The column `image_ptr_id'` in this table corresponds to `statmaps_basecollectionitem.id` & `statmaps_image.basecollectionitem_ptr_id`
* `statmaps_atlas`
Atlases are a type of Images, with an additional reference to `label_description_file`. the column `image_ptr_id` refers to `statmaps_basecollectionitem.id` & `statmaps_image.basecollectionitem_ptr_id`.

### Cogntive Atlas
* `statmaps_cognitiveatlastask`
Mapping of `cog_atlas_id` (referenced in `statmaps_statisticmap`) to task names
* `statmaps_cognitiveatlascontrast`
Mapping of `cog_atlas_id` referenced in `statmaps_statisticmap`) to contrast names

### Other tables
* `statmaps_collection_communities`
Grouping of collections into "communities". 
* `statmaps_communities`
Listing of communities. This is specific to two communities, "Developmental Neuroscience" and "Nutritional Neuroscience"

## Importing into a SQLite database

In order to quickly explore the contents of these CSV files, it may be convenient to import them into an SQLite database.
The 'create_db.sh' script creates such a database (with the primary and foreign key constraints mentioned above) and fills it.
It can be used like this:

```
tar xzf november_2022.tar.gz
cd november_2022
../create_db.sh

sqlite3 neurovault.sqlite3
```

## Gotchas when loading these CSVs

Two properties of the data that will silently distort an analysis:

* **The literal string `"None"` is a real primary key.** It belongs to the
  `statmaps_cognitiveatlastask` row named "None / Other". Every
  `cognitive_paradigm_cogatlas_id` in the dump joins to that table, with no
  orphans, so `"None"` records a genuine user selection rather than a missing
  value. Pandas' default `na_values` converts it to `NaN`, merging *"the user
  chose None/Other"* with *"the user answered nothing"* — in `february_2024`
  that is 301,879 rows versus 75,452. Read with `keep_default_na=False` and test
  for blank strings explicitly.

* **Owner `5761` is the Neuroscout mass-upload account**, a bulk push from a
  separate platform: 3,889 collections and 289,248 statistic maps in
  `february_2024`, or 51.7% of all maps. Most analyses of user-contributed
  content should exclude it.

Also note that `statmaps_collection.description` contains newlines, so line
counts do not equal row counts — use a real CSV parser.

Analyses built on these dumps live in the companion `nv-audit` repository.
