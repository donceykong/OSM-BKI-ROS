#!/bin/bash
# Run ablation experiment: OSM-BKI baseline vs OSM-BKI + height kernel
# Execute inside the Docker container after sourcing ROS2.
#
# Usage (inside container):
#   source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh
#
# Or run individual steps:
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh run_a
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh run_b
#   bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh evaluate

set -e

DATA_ROOT="/media/sgarimella34/hercules-collect1/datasets"
SEQ="kitti360/2013_05_28_drive_0000_sync"
EVAL_DIR_A="${DATA_ROOT}/${SEQ}/evaluations/osm_prior_no_height"
EVAL_DIR_B="${DATA_ROOT}/${SEQ}/evaluations/osm_prior_with_height"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Launch ros2 and wait for it to exit (node exits automatically when visualize=false).
run_and_wait() {
    local config="$1"
    echo "  Launching with data_config:=${config} ..."
    ros2 launch osm_bki kitti360_launch.py data_config:="${config}"
    echo "  Done."
}

run_a() {
    echo "=============================================="
    echo "  Run A: OSM-BKI baseline (no height kernel)"
    echo "=============================================="
    echo "  Output: ${EVAL_DIR_A}"
    echo ""

    # Clean previous results
    if [ -d "${EVAL_DIR_A}" ]; then
        echo "  Clearing previous Run A results..."
        rm -rf "${EVAL_DIR_A}"
    fi
    mkdir -p "${EVAL_DIR_A}"

    run_and_wait kitti360_no_height

    echo ""
    echo "  Run A complete. Results in ${EVAL_DIR_A}"
    echo "  Scan files: $(ls "${EVAL_DIR_A}"/*.txt 2>/dev/null | wc -l)"
}

run_b() {
    echo "=============================================="
    echo "  Run B: OSM-BKI + height kernel"
    echo "=============================================="
    echo "  Output: ${EVAL_DIR_B}"
    echo ""

    # Clean previous results
    if [ -d "${EVAL_DIR_B}" ]; then
        echo "  Clearing previous Run B results..."
        rm -rf "${EVAL_DIR_B}"
    fi
    mkdir -p "${EVAL_DIR_B}"

    run_and_wait kitti360_with_height

    echo ""
    echo "  Run B complete. Results in ${EVAL_DIR_B}"
    echo "  Scan files: $(ls "${EVAL_DIR_B}"/*.txt 2>/dev/null | wc -l)"
}

evaluate() {
    echo "=============================================="
    echo "  Evaluating ablation results"
    echo "=============================================="

    # Check that both dirs have results
    COUNT_A=$(ls "${EVAL_DIR_A}"/*.txt 2>/dev/null | wc -l)
    COUNT_B=$(ls "${EVAL_DIR_B}"/*.txt 2>/dev/null | wc -l)
    echo "  Run A scans: ${COUNT_A}"
    echo "  Run B scans: ${COUNT_B}"

    if [ "${COUNT_A}" -eq 0 ]; then
        echo "ERROR: No results in ${EVAL_DIR_A}. Run 'run_a' first."
        exit 1
    fi
    if [ "${COUNT_B}" -eq 0 ]; then
        echo "ERROR: No results in ${EVAL_DIR_B}. Run 'run_b' first."
        exit 1
    fi

    python3 "${SCRIPT_DIR}/evaluate_ablation.py" compare \
        --dir-a "${EVAL_DIR_A}" --label-a "OSM-BKI (no height)" \
        --dir-b "${EVAL_DIR_B}" --label-b "OSM-BKI + height kernel" \
        --output "${DATA_ROOT}/${SEQ}/evaluations"
}

# Main
case "${1:-all}" in
    run_a)
        run_a
        ;;
    run_b)
        run_b
        ;;
    evaluate)
        evaluate
        ;;
    all)
        run_a
        echo ""
        echo "  Waiting 5 seconds before Run B..."
        sleep 5
        run_b
        echo ""
        evaluate
        ;;
    *)
        echo "Usage: $0 {run_a|run_b|evaluate|all}"
        exit 1
        ;;
esac
