#!/usr/bin/env bash

base_path=$1
output_path=$2

echo "Base Path" $base_path
echo "Output Path" $output_path

mkdir -p $output_path

start=`date +%s`

for f in $base_path/*.mp4; do
  echo "Processing" $f
#  echo "$output_path/$f"
  name=${f##*/}
#  echo $name
  if [ ! -f $output_path/$name ]; then
      echo "File not found!"
      ffmpeg -i $f -vcodec libx264 "$output_path/$name"

  fi

done

end=`date +%s`
runtime=$((end-start))
hours=$((runtime / 3600)); minutes=$(( (runtime % 3600) / 60 )); seconds=$(( (runtime % 3600) % 60 ));

echo "Runtime: $hours:$minutes:$seconds (hh:mm:ss)"


# MPEG format 673M
# X264 format 290M