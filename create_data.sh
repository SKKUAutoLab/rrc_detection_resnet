#!/bin/bash

dataset_name=$1
if [ "$dataset_name" == "" ]
then
  dataset_name="KITTI"
fi

working_dir=$(cd $( dirname ${BASH_SOURCE[0]} ) && pwd )

redo=1
root_dir="$HOME/data/$dataset_name/"
mapfile="$working_dir/data/$dataset_name/labelmap_voc.prototxt"
anno_type="detection"
db="lmdb"
min_dim=0
max_dim=0
width=0
height=0

extra_cmd="--encode-type=png --encoded"
if [ $redo ]
then
  extra_cmd="$extra_cmd --redo"
fi
for subset in training testing
do
  python $working_dir/scripts/create_annoset.py --anno-type=$anno_type --label-map-file=$mapfile --min-dim=$min_dim --max-dim=$max_dim --resize-width=$width --resize-height=$height --check-label $extra_cmd $root_dir $working_dir/data/$dataset_name/$subset.txt $root_dir/$db/$dataset_name"_"$subset"_"$db $working_dir/data/$dataset_name/lmdb
done
