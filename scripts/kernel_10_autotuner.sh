#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
WM_VALUES=(32 64 128 256)
WN_VALUES=(32 64 128 256)
WNITER_VALUES=(1 2 4 8)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
NUM_THREADS_VALUES=(128 256)

cd "$(dirname "$0")"
cd "../build"

OUTPUT="../benchmark_results/kernel_10_autotune_results.txt"
BEST_CONFIG="../benchmark_results/best_kernel_10_config.cmake"

# Clear the output and best config file
echo "" >$OUTPUT
echo "# Best Kernel 10 Autotune Configuration" >$BEST_CONFIG

# Set GPU to use
export DEVICE="0"
WARPSIZE=32

TOTAL_CONFIGS="$((${#BK_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#WM_VALUES[@]} * ${#WN_VALUES[@]} * ${#WNITER_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#NUM_THREADS_VALUES[@]}))"
CONFIG_NUM=0

# Variables to store the best configuration
best_gflops=0.0
best_config=""

# Loop through all combinations of parameters
for BK in "${BK_VALUES[@]}"; do
  for BM in "${BM_VALUES[@]}"; do
    for BN in "${BN_VALUES[@]}"; do
      for WM in "${WM_VALUES[@]}"; do
        for WN in "${WN_VALUES[@]}"; do
          for WN_ITER in "${WNITER_VALUES[@]}"; do
            for TM in "${TM_VALUES[@]}"; do
              for TN in "${TN_VALUES[@]}"; do
                for NUM_THREADS in "${NUM_THREADS_VALUES[@]}"; do
                  echo ""
                  CONFIG_NUM=$((CONFIG_NUM + 1))
                  config="BK=$BK BM=$BM BN=$BN WM=$WM WN=$WN WN_ITER=$WN_ITER TM=$TM TN=$TN NUM_THREADS=$NUM_THREADS"

                  # Skip configurations that don't fulfill preconditions
                  NUM_WARPS=$((NUM_THREADS / 32))
                  if ! ((BN % WN == 0 && BM % WM == 0)); then
                    echo "Skipping $config due to preconditions not met: BN % WN and BM % WM must be 0."
                    continue
                  fi
                  if ! (((BN / WN) * (BM / WM) == NUM_WARPS)); then
                    echo "Skipping $config due to preconditions not met: (BN / WN) * (BM / WM) must equal NUM_WARPS."
                    continue
                  fi
                  if ! (((WM * WN) % (WARPSIZE * TM * TN * WN_ITER) == 0)); then
                    echo "Skipping $config due to preconditions not met: (WM * WN) % (WARPSIZE * TM * TN * WN_ITER) must be 0."
                    continue
                  fi
                  WM_ITER=$(((WM * WN) / (WARPSIZE * TM * TN * WN_ITER)))
                  if ! ((WM % WM_ITER == 0 && WN % WN_ITER == 0)); then
                    echo "Skipping $config due to preconditions not met: WM % WM_ITER and WN % WN_ITER must be 0."
                    continue
                  fi
                  if ! (((NUM_THREADS * 4) % BK == 0)); then
                    echo "Skipping $config due to preconditions not met: (NUM_THREADS * 4) % BK must be 0."
                    continue
                  fi
                  if ! (((NUM_THREADS * 4) % BN == 0)); then
                    echo "Skipping $config due to preconditions not met: (NUM_THREADS * 4) % BN must be 0."
                    continue
                  fi
                  if ! ((BN % (16 * TN) == 0)); then
                    echo "Skipping $config due to preconditions not met: BN must be a multiple of 16 * TN."
                    continue
                  fi
                  if ! ((BM % (16 * TM) == 0)); then
                    echo "Skipping $config due to preconditions not met: BM must be a multiple of 16 * TM."
                    continue
                  fi
                  if ! (((BM * BK) % (4 * NUM_THREADS) == 0)); then
                    echo "Skipping $config due to preconditions not met: (BM * BK) % (4 * NUM_THREADS) must be 0."
                    continue
                  fi
                  if ! (((BN * BK) % (4 * NUM_THREADS) == 0)); then
                    echo "Skipping $config due to preconditions not met: (BN * BK) % (4 * NUM_THREADS) must be 0."
                    continue
                  fi

                  echo "($CONFIG_NUM/$TOTAL_CONFIGS): $config" | tee -a $OUTPUT

                  if ! make clean; then
                    echo "Compilation failed for $config" | tee -a $OUTPUT
                    continue
                  fi

                  if ! cmake -DCMAKE_BUILD_TYPE=Release \
                    -DTUNING=ON \
                    -DK10_BK=$BK \
                    -DK10_BM=$BM \
                    -DK10_BN=$BN \
                    -DK10_WM=$WM \
                    -DK10_WN=$WN \
                    -DK10_WNITER=$WN_ITER \
                    -DK10_TM=$TM \
                    -DK10_TN=$TN \
                    -DK10_NUM_THREADS=$NUM_THREADS \
                    ..; then
                    echo "CMake configuration failed for $config" | tee -a $OUTPUT
                    continue
                  fi

                  if ! make; then
                    echo "Compilation failed for $config" | tee -a $OUTPUT
                    continue
                  fi

                  # Run the benchmark and capture the output
                  result=$(timeout -v 15 ./sgemm 10 2>&1 | tee -a $OUTPUT)

                  if [ $? -eq 124 ]; then
                    echo "Execution timed out for $config" | tee -a $OUTPUT
                    continue
                  fi

                  # Extract the best GFLOPS from the output
                  gflops=$(echo "$result" | grep -oP 'performance: \(\s*\K[0-9]+(\.[0-9]+)?(?=\) GFLOPS)' | sort -nr | head -1)

                  if [[ ! -z "$gflops" ]]; then
                    gflops=$(echo "$gflops" | awk '{print $1}')
                    echo "Current GFLOPS: $gflops for $config" | tee -a $OUTPUT

                    # Update the best configuration if this one is better
                    if (($(echo "$gflops > $best_gflops" | bc -l))); then
                      best_gflops=$gflops
                      best_config="BK=$BK; BM=$BM; BN=$BN; WM=$WM; WN=$WN; WN_ITER=$WN_ITER; TM=$TM; TN=$TN; NUM_THREADS=$NUM_THREADS"
                      echo "New best configuration: $best_config with GFLOPS: $best_gflops" | tee -a $OUTPUT

                      echo "set(K10_BK $BK CACHE STRING \"K10_BK\")" >$BEST_CONFIG
                      echo "set(K10_BM $BM CACHE STRING \"K10_BM\")" >>$BEST_CONFIG
                      echo "set(K10_BN $BN CACHE STRING \"K10_BN\")" >>$BEST_CONFIG
                      echo "set(K10_WM $WM CACHE STRING \"K10_WM\")" >>$BEST_CONFIG
                      echo "set(K10_WN $WN CACHE STRING \"K10_WN\")" >>$BEST_CONFIG
                      echo "set(K10_WNITER $WN_ITER CACHE STRING \"K10_WNITER\")" >>$BEST_CONFIG
                      echo "set(K10_TM $TM CACHE STRING \"K10_TM\")" >>$BEST_CONFIG
                      echo "set(K10_TN $TN CACHE STRING \"K10_TN\")" >>$BEST_CONFIG
                      echo "set(K10_NUM_THREADS $NUM_THREADS CACHE STRING \"K10_NUM_THREADS\")" >>$BEST_CONFIG
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
    done
  done
done

# Output the best configuration
echo "Best configuration: $best_config with GFLOPS: $best_gflops" | tee -a $OUTPUT
