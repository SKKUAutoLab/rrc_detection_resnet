#!/bin/bash

dataset_name=$1
if [ "$dataset_name" == "" ]
then
  dataset_name="KITTI"
fi
  
root_dir="$HOME/data/$dataset_name/"   #your path to dataset
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
label_dir="label_2"                #path to labels
img_dir="image_2"
for dataset in training testing
do
  dst_file=$bash_dir/$dataset.txt
  if [ -f $dst_file ]
  then
    rm -f $dst_file
  fi

  echo "Create list for $dataset_name $img_dir $dataset..."
  dataset_file=./main/$dataset.txt

  img_list_file=$bash_dir/$dataset"_img.txt"
  cp $dataset_file $img_list_file
  sed -i "s/^/$dataset\/$img_dir\//g" $img_list_file
  sed -i "s/$/.png/g" $img_list_file

  label_list_file=$bash_dir/$dataset"_label.txt"
  cp $dataset_file $label_list_file
  sed -i "s/^/$dataset\/$label_dir\/xml\//g" $label_list_file
  sed -i "s/$/.xml/g" $label_list_file

  paste -d' ' $img_list_file $label_list_file >> $dst_file

  rm -f $label_list_file
  rm -f $img_list_file

  # Generate image name and size infomation.
  if [ "$dataset" == "testing" ]
  then
    $bash_dir/../../build/tools/get_image_size $root_dir $dst_file $bash_dir/$dataset"_name_size.txt"
  fi

  # Shuffle trainval file.
  if [ $dataset == "training" ]
  then
    rand_file=$dst_file.random
    cat $dst_file | perl -MList::Util=shuffle -e 'print shuffle(<STDIN>);' > $rand_file
    mv $rand_file $dst_file
  fi
done
