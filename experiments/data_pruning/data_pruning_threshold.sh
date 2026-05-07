ROOT=$HOME/scratch/2023-difficulty/metrics/lottery_8ae667adac9b690a88b522e85f1388c8
# FRACTION=("0.5" "0.4" "0.3")
FRACTION=("0.1")
OFFSET=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9")
REPLICATE=($(seq 1 1 5))

parallel --delay=3 --jobs=3  \
    python -m experiments.data_pruning.data_pruning_threshold  \
        --score_file=$ROOT/avg_gradmetrics/el2n_20ep0it.npz  \
        --subset_fraction={1}  \
        --subset_offset={2}  \
        --subset_invert  \
        --save_file=$ROOT/"data_pruning/el2n_20ep0it_"{1}-{2}"_avg.npy"  \
    ::: ${FRACTION[@]}  \
    ::: ${OFFSET[@]}  \

parallel --delay=3 --jobs=3  \
    python -m experiments.data_pruning.data_pruning_threshold  \
        --score_file=$ROOT/"replicate_"{3}"/level_0/main/gradmetrics/el2n_20ep0it.npz"  \
        --subset_fraction={1}  \
        --subset_offset={2}  \
        --subset_invert  \
        --save_file=$ROOT/"data_pruning/el2n_20ep0it_"{1}-{2}"_replicate_"{3}".npy"  \
  ::: ${FRACTION[@]}  \
    ::: ${OFFSET[@]}  \
  ::: ${REPLICATE[@]}  \
