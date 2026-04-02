#!/bin/bash
# Run in-domain ablation experiments on KITTI-360 seq 0000 and 0009.
# CENet trained on KITTI-360, tested on KITTI-360 (in-domain).
# Each sequence: baseline (no height kernel) vs with height kernel.
#
# Usage (inside container):
#   source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation_indomain.sh
#
# Or run individual experiments:
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation_indomain.sh exp1   # seq 0000 only
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation_indomain.sh exp2   # seq 0009 only

set -e

DATA_ROOT="/media/sgarimella34/hercules-collect1/datasets"
SEQ_0000="kitti360/2013_05_28_drive_0000_sync"
SEQ_0009="kitti360/2013_05_28_drive_0009_sync"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_experiment() {
    local seq="$1"
    local config_no_height="$2"
    local config_with_height="$3"
    local label="$4"

    local eval_dir_a="${DATA_ROOT}/${seq}/evaluations/indomain_no_height"
    local eval_dir_b="${DATA_ROOT}/${seq}/evaluations/indomain_with_height"

    echo ""
    echo "######################################################################"
    echo "  ${label}"
    echo "  Sequence: ${seq}"
    echo "  Labels: cenet_kitti360_softmax (in-domain)"
    echo "######################################################################"

    # --- Run A: baseline ---
    echo ""
    echo "  [A] OSM-BKI baseline (no height kernel)"
    echo "  Output: ${eval_dir_a}"
    if [ -d "${eval_dir_a}" ]; then
        echo "  Clearing previous results..."
        rm -rf "${eval_dir_a}"
    fi
    mkdir -p "${eval_dir_a}"

    echo "  Launching..."
    ros2 launch osm_bki kitti360_launch.py data_config:="${config_no_height}"
    echo "  Done. Scan files: $(ls "${eval_dir_a}"/*.txt 2>/dev/null | wc -l)"

    sleep 5

    # --- Run B: with height kernel ---
    echo ""
    echo "  [B] OSM-BKI + height kernel"
    echo "  Output: ${eval_dir_b}"
    if [ -d "${eval_dir_b}" ]; then
        echo "  Clearing previous results..."
        rm -rf "${eval_dir_b}"
    fi
    mkdir -p "${eval_dir_b}"

    echo "  Launching..."
    ros2 launch osm_bki kitti360_launch.py data_config:="${config_with_height}"
    echo "  Done. Scan files: $(ls "${eval_dir_b}"/*.txt 2>/dev/null | wc -l)"

    # --- Evaluate ---
    echo ""
    echo "  Evaluating ${label}..."
    python3 "${SCRIPT_DIR}/evaluate_ablation.py" compare \
        --dir-a "${eval_dir_a}" --label-a "Baseline (no height)" \
        --dir-b "${eval_dir_b}" --label-b "+ Height kernel" \
        --output "${DATA_ROOT}/${seq}/evaluations"

    echo ""
    echo "  ${label} complete."
    echo "----------------------------------------------------------------------"
}

exp1() {
    run_experiment "${SEQ_0000}" \
        "kitti360_0000_indomain_no_height" \
        "kitti360_0000_indomain_with_height" \
        "Experiment 1: KITTI-360 seq 0000 (in-domain)"
}

exp2() {
    run_experiment "${SEQ_0009}" \
        "kitti360_0009_indomain_no_height" \
        "kitti360_0009_indomain_with_height" \
        "Experiment 2: KITTI-360 seq 0009 (in-domain)"
}

# Main
case "${1:-all}" in
    exp1)
        exp1
        ;;
    exp2)
        exp2
        ;;
    all)
        exp1
        echo ""
        echo "======================================================================"
        echo "  Experiment 1 finished. Starting Experiment 2..."
        echo "======================================================================"
        echo ""
        exp2
        echo ""
        echo "======================================================================"
        echo "  ALL EXPERIMENTS COMPLETE"
        echo "======================================================================"
        ;;
    *)
        echo "Usage: $0 {exp1|exp2|all}"
        echo "  exp1 - KITTI-360 seq 0000, in-domain labels"
        echo "  exp2 - KITTI-360 seq 0009, in-domain labels"
        echo "  all  - Run both experiments sequentially"
        exit 1
        ;;
esac
