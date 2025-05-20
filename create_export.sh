#!/bin/bash
############################################################################################
# For use on neurovault production server to generate tar backups used in this repository. #
############################################################################################

set -euxo pipefail

tables=('statmaps_atlas' 'statmaps_basecollectionitem' 'statmaps_cognitiveatlascontrast' 'statmaps_cognitiveatlastask' 'statmaps_collection_communities' 'statmaps_collection_contributors' 'statmaps_collection' 'statmaps_community' 'statmaps_statisticmap' 'statmaps_image' 'statmaps_nidmresults' 'statmaps_nidmresultstatisticmap')

if [[ -z "$PGPASSWORD" || -z "$PGHOST" || -z  "$PGUSER" ]]; then
    echo "Set PGPASSWORD, PGHOST and PGUSER environment variables."
    exit 126
fi

cd "$(dirname "$0")"
mkdir -p scratch/

for table in ${tables[@]}; do
    psql -c "COPY (select * from $table) TO STDOUT WITH CSV HEADER" > ./scratch/$table.csv
done

dname=$(date +%F)_neurovault_data
tar --transform "s/scratch/$dname/" -czf $dname.tar.gz scratch
