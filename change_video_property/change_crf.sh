#!/usr/bin/env bash
# The range of the CRF scale is 0–51, where 0 is lossless, 23 is the default, and 51 is worst quality possible.
# A lower value generally leads to higher quality, and a subjectively sane range is 17–28. Consider 17 or 18 to be
# visually lossless or nearly so; it should look the same or nearly the same as the input but it isn't
# technically lossless.
# crf 16, 22, 28, 34, 40, 46

base_path=$1
output_path=$2

echo "Base Path" $base_path
echo "Output Path" $output_path

mkdir -p $output_path

start=`date +%s`

for f in $base_path/*.MP4; do
  echo "Processing" $f
#  echo "$output_path/$f"
  name=${f##*/}
#  echo $name
  if [ ! -f $output_path/$name ]; then
      echo "File not found!"
      ffmpeg -i $f -vcodec libx264 -crf 34 "$output_path/$name"

  fi

done

end=`date +%s`
runtime=$((end-start))
hours=$((runtime / 3600)); minutes=$(( (runtime % 3600) / 60 )); seconds=$(( (runtime % 3600) % 60 ));

echo "Runtime: $hours:$minutes:$seconds (hh:mm:ss)"