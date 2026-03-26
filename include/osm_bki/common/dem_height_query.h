#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace osm_bki {

/**
 * @brief Loads a precomputed elevation grid (DEM or DSM) in KITTI-360 local
 *        frame and provides fast bilinear-interpolation queries.
 *
 * The grid file is produced by height_kernel/precompute_dem_grid.py.
 * Stored values are already offset by kitti_first_z so that:
 *     height_above_ground = point_z - query(point_x, point_y)
 *
 * Binary format (little-endian):
 *   Header (32 bytes):
 *     char[4]  magic     = "DEMG"
 *     int32    version   = 1
 *     float32  origin_x
 *     float32  origin_y
 *     float32  cell_size
 *     int32    cols
 *     int32    rows
 *     float32  nodata  (NaN)
 *   Data:
 *     float32[rows * cols]  row-major, row 0 = min Y
 */
class DEMHeightQuery {
public:
    DEMHeightQuery() = default;

    /// Load grid from binary file.  Returns false on failure.
    bool load(const std::string &path) {
        FILE *fp = std::fopen(path.c_str(), "rb");
        if (!fp) return false;

        // Read header (32 bytes)
        struct Header {
            char magic[4];
            int32_t version;
            float origin_x;
            float origin_y;
            float cell_size;
            int32_t cols;
            int32_t rows;
            float nodata;
        } hdr;
        static_assert(sizeof(Header) == 32, "Header must be 32 bytes");

        if (std::fread(&hdr, sizeof(Header), 1, fp) != 1) {
            std::fclose(fp);
            return false;
        }
        if (std::memcmp(hdr.magic, "DEMG", 4) != 0 || hdr.version != 1) {
            std::fclose(fp);
            return false;
        }

        origin_x_ = hdr.origin_x;
        origin_y_ = hdr.origin_y;
        cell_size_ = hdr.cell_size;
        cols_ = hdr.cols;
        rows_ = hdr.rows;
        inv_cell_ = 1.0f / cell_size_;

        size_t n = static_cast<size_t>(rows_) * static_cast<size_t>(cols_);
        data_.resize(n);
        if (std::fread(data_.data(), sizeof(float), n, fp) != n) {
            std::fclose(fp);
            data_.clear();
            return false;
        }
        std::fclose(fp);
        loaded_ = true;
        return true;
    }

    bool is_loaded() const { return loaded_; }
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    float cell_size() const { return cell_size_; }

    /// Query the grid at a point in KITTI-360 local frame.
    /// Returns the stored elevation (already offset by first_z).
    /// Returns NaN if out of bounds.
    float query(float x, float y) const {
        if (!loaded_) return std::nanf("");

        // Map to fractional grid coordinates (cell centers at +0.5)
        float gx = (x - origin_x_) * inv_cell_ - 0.5f;
        float gy = (y - origin_y_) * inv_cell_ - 0.5f;

        // Bilinear interpolation
        int ix = static_cast<int>(std::floor(gx));
        int iy = static_cast<int>(std::floor(gy));

        if (ix < 0 || ix >= cols_ - 1 || iy < 0 || iy >= rows_ - 1)
            return std::nanf("");

        float fx = gx - static_cast<float>(ix);
        float fy = gy - static_cast<float>(iy);

        float v00 = at(iy, ix);
        float v10 = at(iy, ix + 1);
        float v01 = at(iy + 1, ix);
        float v11 = at(iy + 1, ix + 1);

        // If any corner is NaN, return NaN
        if (std::isnan(v00) || std::isnan(v10) || std::isnan(v01) || std::isnan(v11))
            return std::nanf("");

        float v = v00 * (1 - fx) * (1 - fy)
                + v10 * fx * (1 - fy)
                + v01 * (1 - fx) * fy
                + v11 * fx * fy;
        return v;
    }

    /// Returns height above ground: point_z - grid_elevation.
    /// NaN if out of bounds.
    float height_above_ground(float x, float y, float z) const {
        float elev = query(x, y);
        if (std::isnan(elev)) return std::nanf("");
        return z - elev;
    }

private:
    float at(int row, int col) const {
        return data_[static_cast<size_t>(row) * static_cast<size_t>(cols_)
                      + static_cast<size_t>(col)];
    }

    bool loaded_{false};
    float origin_x_{0.f};
    float origin_y_{0.f};
    float cell_size_{1.f};
    float inv_cell_{1.f};
    int32_t cols_{0};
    int32_t rows_{0};
    std::vector<float> data_;
};

}  // namespace osm_bki
