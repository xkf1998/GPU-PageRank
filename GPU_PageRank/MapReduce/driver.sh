#!/bin/bash
set -x

echo "Starting the test"

data_dir="gpu/input/"
output_dir="gpu/output/"
log_dir="/scratch/xl3341/output/"
for i in `seq 10 10 100`
do
    for j in "t1w" "t10w" "t25w" "t50w" "t75w" "t100w"
    do
        # hadoop fs -rm -r ${output_dir}res_${i}_${j}
        # hadoop jar PageRank-1.0-SNAPSHOT.jar PageRank $data_dir$j ${output_dir}res_${i}_${j} $i 2>&1 | tee ${log_dir}log_${i}_${j}.log
        echo ${output_dir}res_${i}_${j}
    done
done

echo "Finished the test"