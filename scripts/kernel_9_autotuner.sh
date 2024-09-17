#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
NUM_THREADS_VALUES=(256)

cd "$(dirname "$0")"
cd "../build"

OUTPUT="../benchmark_results/kernel_9_autotune_results.txt"
BEST_CONFIG="../benchmark_results/best_kernel_9_config.cmake"

# Clear the output and best config file
echo "" > $OUTPUT
echo "# Best Kernel 9 Autotune Configuration" > $BEST_CONFIG

# Set GPU to use
export DEVICE="0"

TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} ))"
CONFIG_NUM=0

# Variables to store the best configuration
best_gflops=0.0
best_config=""

# Loop through all combinations of parameters
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          for nt in ${NUM_THREADS_VALUES[@]}; do
            echo ""
            CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

            # Skip configurations that don't fulfill preconditions
            config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"
            if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK != 0"
              continue
            fi
            if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN != 0"
              continue
            fi
            if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BN % (16 * TN) != 0"
              continue
            fi
            if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
              echo "QUANTIZATION: Skipping $config because BM % (16 * TM) != 0"
              continue
            fi
            if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) != 0"
              continue
            fi
            if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
              echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) != 0"
              continue
            fi

            echo "($CONFIG_NUM/$TOTAL_CONFIGS): $config" | tee -a $OUTPUT

            # FIXME: It doesn't prop
            if ! make clean; then
              echo "Failed to clean build directory" | tee -a $OUTPUT
              continue
            fi

            if ! cmake -DCMAKE_BUILD_TYPE=Release \
                      -DTUNING=ON \
                      -DK9_BK=$bk \
                      -DK9_TM=$tm \
                      -DK9_TN=$tn \
                      -DK9_BM=$bm \
                      -DK9_BN=$bn \
                      -DK9_NUM_THREADS=$nt \
                      ..; then
              echo "CMake configuration failed for $config" | tee -a $OUTPUT
              continue
            fi

            if ! make; then
              echo "Compilation failed for $config" | tee -a $OUTPUT
              continue
            fi

            # Run the benchmark and capture the output
            result=$(timeout -v 15 ./sgemm 9 2>&1 | tee -a $OUTPUT)

            if [ $? -eq 124 ]; then
              echo "Execution timed out for $config" | tee -a $OUTPUT
              continue
            fi

            # Extract the best GFLOPS from the output
            gflops=$(echo "$result" | grep -oP 'performance: \(\s*\K[0-9]+(\.[0-9]+)?(?=\) GFLOPS)' | sort -nr | head -1)

            # If a valid GFLOPS value is found, compare with the best one
            if [[ ! -z "$gflops" ]]; then
              gflops=$(echo "$gflops" | awk '{print $1}')
              echo "Current GFLOPS: $gflops for $config" | tee -a $OUTPUT

              # Update the best configuration if this one is better
              if (( $(echo "$gflops > $best_gflops" | bc -l) )); then
                best_gflops=$gflops
                best_config="BK=$bk; BM=$bm; BN=$bn; TM=$tm; TN=$tn; NUM_THREADS=$nt"
                echo "New best configuration: $best_config with GFLOPS: $best_gflops" | tee -a $OUTPUT
                echo "set(K9_BK $bk CACHE STRING \"K9_BK\")" > $BEST_CONFIG
                echo "set(K9_BM $bm CACHE STRING \"K9_BM\")" >> $BEST_CONFIG
                echo "set(K9_BN $bn CACHE STRING \"K9_BN\")" >> $BEST_CONFIG
                echo "set(K9_TM $tm CACHE STRING \"K9_TM\")" >> $BEST_CONFIG
                echo "set(K9_TN $tn CACHE STRING \"K9_TN\")" >> $BEST_CONFIG
                echo "set(K9_NUM_THREADS $nt CACHE STRING \"K9_NUM_THREADS\")" >> $BEST_CONFIG
              fi
            else
              echo "No valid GFLOPS value found for $config" | tee -a $OUTPUT
            fi
          done
        done
      done
    done
  done
done

# Output the best configuration
echo "Best configuration: $best_config with GFLOPS: $best_gflops" | tee -a $OUTPUT
