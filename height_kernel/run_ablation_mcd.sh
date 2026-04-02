#!/bin/bash
# Run MCD ablation experiments 3 and 4.
# Exp 3: MCD kth_day_09 with cross-domain labels (CENet trained on KITTI-360)
# Exp 4: MCD kth_day_09 with in-domain labels (CENet trained on MCD)
# Each: baseline (no height kernel) vs with height kernel.
#
# Usage (inside container):
#   source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation_mcd.sh

set -e

DATA_ROOT="/media/sgarimella34/hercules-collect1/datasets"
SEQ="mcd/kth_day_09"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_experiment() {
    local config_no_height="$1"
    local config_with_height="$2"
    local eval_suffix_a="$3"
    local eval_suffix_b="$4"
    local label="$5"

    local eval_dir_a="${DATA_ROOT}/${SEQ}/evaluations/${eval_suffix_a}"
    local eval_dir_b="${DATA_ROOT}/${SEQ}/evaluations/${eval_suffix_b}"

    echo ""
    echo "######################################################################"
    echo "  ${label}"
    echo "######################################################################"

    # --- Run A: baseline ---
    echo ""
    echo "  [A] Baseline (no height kernel)"
    echo "  Output: ${eval_dir_a}"
    if [ -d "${eval_dir_a}" ]; then
        echo "  Clearing previous results..."
        rm -rf "${eval_dir_a}"
    fi
    mkdir -p "${eval_dir_a}"

    echo "  Launching..."
    ros2 launch osm_bki mcd_with_osm_launch.py data_config:="${config_no_height}"
    echo "  Done. Scan files: $(ls "${eval_dir_a}"/*.txt 2>/dev/null | wc -l)"

    sleep 5

    # --- Run B: with height kernel ---
    echo ""
    echo "  [B] + Height kernel"
    echo "  Output: ${eval_dir_b}"
    if [ -d "${eval_dir_b}" ]; then
        echo "  Clearing previous results..."
        rm -rf "${eval_dir_b}"
    fi
    mkdir -p "${eval_dir_b}"

    echo "  Launching..."
    ros2 launch osm_bki mcd_with_osm_launch.py data_config:="${config_with_height}"
    echo "  Done. Scan files: $(ls "${eval_dir_b}"/*.txt 2>/dev/null | wc -l)"

    # --- Evaluate ---
    echo ""
    echo "  Evaluating ${label}..."
    python3 "${SCRIPT_DIR}/evaluate_ablation.py" compare \
        --dir-a "${eval_dir_a}" --label-a "Baseline (no height)" \
        --dir-b "${eval_dir_b}" --label-b "+ Height kernel" \
        --output "${DATA_ROOT}/${SEQ}/evaluations"

    echo ""
    echo "  ${label} complete."
    echo "----------------------------------------------------------------------"
}

exp3() {
    run_experiment \
        "mcd_kitti360_no_height" \
        "mcd_kitti360_with_height" \
        "kitti360labels_no_height" \
        "kitti360labels_with_height" \
        "Experiment 3: MCD kth_day_09, cross-domain labels (CENet-KITTI360)"
}

exp4() {
    run_experiment \
        "mcd_mcd_no_height" \
        "mcd_mcd_with_height" \
        "mcdlabels_no_height" \
        "mcdlabels_with_height" \
        "Experiment 4: MCD kth_day_09, in-domain labels (CENet-MCD)"
}

case "${1:-all}" in
    exp3) exp3 ;;
    exp4) exp4 ;;
    all)
        exp3
        echo ""
        echo "======================================================================"
        echo "  Experiment 3 finished. Starting Experiment 4..."
        echo "======================================================================"
        exp4
        echo ""
        echo "======================================================================"
        echo "  ALL MCD EXPERIMENTS COMPLETE"
        echo "======================================================================"
        ;;
    *) echo "Usage: $0 {exp3|exp4|all}"; exit 1 ;;
esac
