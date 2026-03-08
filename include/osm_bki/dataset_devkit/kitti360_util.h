#pragma once

/**
 * KITTI360 dataset support.
 * Pose format matches scripts/kitti360/visualize_sem_map_KITTI360.py:
 *   velodyne_poses.txt: one line per frame — frame_index (int) then 12 or 16 floats (3x4 or 4x4 row-major).
 * MCDData::read_lidar_poses_kitti360() parses this format; the rest of the pipeline uses MCDData (mcd_util.h).
 */
#include "mcd_util.h"
