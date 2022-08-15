#!/bin/bash
set -x
input="/scratch/xl3341"
output="/scratch/xl3341/output"

echo "======START======"

rm -rf $output/*
rm -rf build/*.out

g++ ./pagerank.cpp -o build/pagerank.out
nvcc ./pagerank_sync.cu -o build/pagerank_sync.out
nvcc ./pagerank_async.cu -o build/pagerank_async.out
nvcc ./pagerank_async_unified.cu -o build/pagerank_async_unified.out
nvcc ./pagerank_async_atomicAdd.cu -o build/pagerank_async_atomicAdd.out

echo "compilation done"

filenames=(t1w t10w t25w t50w t75w t100w)
thread_per_block=(64 128 256 512 1024)
epsilons=(0.0001 0.00001 0.000001 0.0000001 0.00000001)
iterations=(10 25 50 75 100)
for filename in ${filenames[@]}; do
    inputfile="${input}/${filename}"
    for iteration in ${iterations[@]}; do
        echo -e "\n[CPU]:"
        ./build/pagerank.out -f $inputfile -o "${output}/CPU_${filename}_I_${iteration}" -i $iteration
    done 
    for thread in ${thread_per_block[@]}; do
        for iteration in ${iterations[@]}; do
            echo -e "\n[GPU Sync]:"
            nvprof ./build/pagerank_sync.out -f $inputfile -o "${output}/GPU_SYNC_${filename}_I_${iteration}" -i $iteration -t $thread
        done 

        for epsilon in ${epsilons[@]}; do
            echo -e "\n[GPU Async]:"
            nvprof ./build/pagerank_async.out -f $inputfile -o "$output/GPU_ASYNC_${filename}_${epsilon}" -e $epsilon -t $thread

            echo -e "\n[GPU Async unified memory]:"
            nvprof ./build/pagerank_async_unified.out -f $inputfile -o "$output/GPU_ASYNC_UNI_${filename}_${epsilon}" -e $epsilon -t $thread

            echo -e "\n[GPU Async atomic add]:"
            nvprof ./build/pagerank_async_atomicAdd.out -f $inputfile -o "$output/GPU_ASYNC_ATOM_${filename}_${epsilon}" -e $epsilon -t $thread
        done
    done
done

echo "=======END======="