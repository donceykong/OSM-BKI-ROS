#pragma once

#include "bkioctomap.h"

namespace osm_bki {

	/*
     * @brief Bayesian Generalized Kernel Inference on Bernoulli distribution
     * @param dim dimension of data (2, 3, etc.)
     * @param T data type (float, double, etc.)
     * @ref Nonparametric Bayesian inference on multivariate exponential families
     */
    template<int dim, typename T>
    class SemanticBKInference {
    public:
        /// Eigen matrix type for training and test data and kernel
        using MatrixXType = Eigen::Matrix<T, -1, dim, Eigen::RowMajor>;
        using MatrixKType = Eigen::Matrix<T, -1, -1, Eigen::RowMajor>;
        using MatrixDKType = Eigen::Matrix<T, -1, 1>;
        using MatrixYType = Eigen::Matrix<T, -1, 1>;

        /// nc = num_class: both the number of output classes and the label index range [0, nc).
        /// Only training labels in [0, nc-1] are used; predicted semantics are in [0, nc-1]. Set nc >= max_label_id + 1.
        SemanticBKInference(int nc, T sf2, T ell) : nc(nc), sf2(sf2), ell(ell), trained(false) { }

        /*
         * @brief Fit BGK Model
         * @param x input vector (3N, row major)
         * @param y target vector (N)
         */
        void train(const std::vector<T> &x, const std::vector<T> &y) {
            assert(x.size() % dim == 0 && (int) (x.size() / dim) == y.size());
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), x.size() / dim, dim);
            MatrixYType _y = Eigen::Map<const MatrixYType>(y.data(), y.size(), 1);
            this->y_vec = y;
            this->w_vec.clear();
            this->y_soft.clear();
            this->w_class.clear();
            train(_x, _y);
        }

        void train(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &w) {
            assert(x.size() % dim == 0 && (int) (x.size() / dim) == y.size());
            assert(w.size() == y.size());
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), x.size() / dim, dim);
            MatrixYType _y = Eigen::Map<const MatrixYType>(y.data(), y.size(), 1);
            this->y_vec = y;
            this->w_vec = w;
            this->y_soft.clear();
            this->w_class.clear();
            train(_x, _y);
        }

        /// Train with hard labels, per-point weights, and per-class kernel weights.
        void train(const std::vector<T> &x, const std::vector<T> &y, const std::vector<T> &w,
                   const std::vector<std::vector<T>> &w_class) {
            assert(x.size() % dim == 0 && (int) (x.size() / dim) == y.size());
            assert(w.size() == y.size() && w_class.size() == y.size());
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), x.size() / dim, dim);
            MatrixYType _y = Eigen::Map<const MatrixYType>(y.data(), y.size(), 1);
            this->y_vec = y;
            this->w_vec = w;
            this->w_class = w_class;
            this->y_soft.clear();
            train(_x, _y);
        }

        /*
         * @brief Fit BGK Model
         * @param x input matrix (NX3)
         * @param y target matrix (NX1)
         */
        void train(const MatrixXType &x, const MatrixYType &y) {
            this->x = MatrixXType(x);
            this->y = MatrixYType(y);
            this->y_soft.clear();
            trained = true;
        }

        /// Train with soft (multiclass probability) labels for counting sensor model.
        /// y_soft is (n_points x nc); when non-empty, predict_csm uses these instead of one-hot from y_vec.
        void train_soft(const std::vector<T> &x, const std::vector<std::vector<T>> &y_soft) {
            assert(x.size() % dim == 0);
            size_t n = x.size() / dim;
            assert(y_soft.size() == n);
            for (size_t i = 0; i < n; ++i)
                assert((int)y_soft[i].size() == nc);
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), n, dim);
            this->x = _x;
            this->y_vec.clear();
            this->w_vec.clear();
            this->y_soft = y_soft;
            this->w_class.clear();
            trained = true;
        }

        /// Train with soft labels and per-point weights (soft count = prob * weight).
        void train_soft(const std::vector<T> &x, const std::vector<std::vector<T>> &y_soft, const std::vector<T> &w) {
            assert(x.size() % dim == 0);
            size_t n = x.size() / dim;
            assert(y_soft.size() == n && w.size() == n);
            for (size_t i = 0; i < n; ++i)
                assert((int)y_soft[i].size() == nc);
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), n, dim);
            this->x = _x;
            this->y_vec.clear();
            this->w_vec = w;
            this->y_soft = y_soft;
            this->w_class.clear();
            trained = true;
        }

        /// Train with soft labels, per-point weights, and per-point per-class kernel weights.
        /// w_class[i][k] multiplies into class k's contribution for point i (e.g. OSM semantic kernel).
        void train_soft(const std::vector<T> &x, const std::vector<std::vector<T>> &y_soft,
                        const std::vector<T> &w, const std::vector<std::vector<T>> &w_class) {
            assert(x.size() % dim == 0);
            size_t n = x.size() / dim;
            assert(y_soft.size() == n && w.size() == n);
            assert(w_class.size() == n);
            for (size_t i = 0; i < n; ++i) {
                assert((int)y_soft[i].size() == nc);
                assert((int)w_class[i].size() == nc);
            }
            MatrixXType _x = Eigen::Map<const MatrixXType>(x.data(), n, dim);
            this->x = _x;
            this->y_vec.clear();
            this->w_vec = w;
            this->y_soft = y_soft;
            this->w_class = w_class;
            trained = true;
        }

       
      void predict(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars) {
          assert(xs.size() % dim == 0);
          MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
          assert(trained == true);
          MatrixKType Ks;

          covSparse(_xs, x, Ks);

          ybars.resize(_xs.rows());
          for (int r = 0; r < _xs.rows(); ++r)
            ybars[r].resize(nc);

            bool has_weights = !w_vec.empty();
            bool has_class_weights = !w_class.empty();
            MatrixYType _y_vec = Eigen::Map<const MatrixYType>(y_vec.data(), y_vec.size(), 1);
            for (int k = 0; k < nc; ++k) {
              for (size_t i = 0; i < y_vec.size(); ++i) {
                if (y_vec[i] == k) {
                  T w = has_weights ? w_vec[i] : static_cast<T>(1);
                  T wc = has_class_weights ? w_class[i][k] : static_cast<T>(1);
                  _y_vec(i, 0) = w * wc;
                } else
                  _y_vec(i, 0) = 0;
              }

            MatrixYType _ybar;
            _ybar = (Ks * _y_vec);

            for (int r = 0; r < _ybar.rows(); ++r)
              ybars[r][k] = _ybar(r, 0);
          }
      }

      void predict_soft(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars) {
          assert(xs.size() % dim == 0);
          MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
          assert(trained == true);
          MatrixKType Ks;

          covSparse(_xs, x, Ks);

          ybars.resize(_xs.rows());
          for (int r = 0; r < _xs.rows(); ++r)
              ybars[r].resize(nc);

          bool has_weights = !w_vec.empty();
          bool has_class_weights = !w_class.empty();
          MatrixYType _y_vec(y_soft.size(), 1);
          for (int k = 0; k < nc; ++k) {
              for (size_t i = 0; i < y_soft.size(); ++i) {
                  T w = has_weights ? w_vec[i] : static_cast<T>(1);
                  T wc = has_class_weights ? w_class[i][k] : static_cast<T>(1);
                  _y_vec(i, 0) = y_soft[i][k] * w * wc;
              }
              MatrixYType _ybar = Ks * _y_vec;
              for (int r = 0; r < _ybar.rows(); ++r)
                  ybars[r][k] = _ybar(r, 0);
          }
      }

      void predict_csm(const std::vector<T> &xs, std::vector<std::vector<T>> &ybars) {
          assert(xs.size() % dim == 0);
          MatrixXType _xs = Eigen::Map<const MatrixXType>(xs.data(), xs.size() / dim, dim);
          assert(trained == true);
          MatrixKType Ks;

          covCountingSensorModel(_xs, x, Ks);

          ybars.resize(_xs.rows());
          for (int r = 0; r < _xs.rows(); ++r)
            ybars[r].resize(nc);

          bool use_soft = !y_soft.empty();
          bool has_weights = !w_vec.empty();
          bool has_class_weights = !w_class.empty();
          MatrixYType _y_vec(use_soft ? y_soft.size() : y_vec.size(), 1);
          for (int k = 0; k < nc; ++k) {
            if (use_soft) {
              for (size_t i = 0; i < y_soft.size(); ++i) {
                T w = has_weights ? w_vec[i] : static_cast<T>(1);
                T wc = has_class_weights ? w_class[i][k] : static_cast<T>(1);
                _y_vec(i, 0) = y_soft[i][k] * w * wc;
              }
            } else {
              for (size_t i = 0; i < y_vec.size(); ++i) {
                if (y_vec[i] == k) {
                  T w = has_weights ? w_vec[i] : static_cast<T>(1);
                  T wc = has_class_weights ? w_class[i][k] : static_cast<T>(1);
                  _y_vec(i, 0) = w * wc;
                } else
                  _y_vec(i, 0) = 0;
              }
            }
            MatrixYType _ybar;
            _ybar = (Ks * _y_vec);
            for (int r = 0; r < _ybar.rows(); ++r)
              ybars[r][k] = _ybar(r, 0);
          }
      }

        
    private:
        /*
         * @brief Compute Euclid distances between two vectors.
         * @param x input vector
         * @param z input vecotr
         * @return d distance matrix
         */
        void dist(const MatrixXType &x, const MatrixXType &z, MatrixKType &d) const {
            d = MatrixKType::Zero(x.rows(), z.rows());
            for (int i = 0; i < x.rows(); ++i) {
                d.row(i) = (z.rowwise() - x.row(i)).rowwise().norm();
            }
        }

        /*
         * @brief Matern3 kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         */
        void covMaterniso3(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(1.73205 / ell * x, 1.73205 / ell * z, Kxz);
            Kxz = ((1 + Kxz.array()) * exp(-Kxz.array())).matrix() * sf2;
        }

        /*
         * @brief Sparse kernel.
         * @param x input vector
         * @param z input vector
         * @return Kxz covariance matrix
         * @ref A sparse covariance function for exact gaussian process inference in large datasets.
         */
        void covSparse(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
            dist(x / ell, z / ell, Kxz);
            Kxz = (((2.0f + (Kxz * 2.0f * 3.1415926f).array().cos()) * (1.0f - Kxz.array()) / 3.0f) +
                  (Kxz * 2.0f * 3.1415926f).array().sin() / (2.0f * 3.1415926f)).matrix() * sf2;

            // Clean up for values with distance outside length scale
            // Possible because Kxz <= 0 when dist >= ell
            for (int i = 0; i < Kxz.rows(); ++i)
            {
                for (int j = 0; j < Kxz.cols(); ++j)
                    if (Kxz(i,j) < 0.0)
                        Kxz(i,j) = 0.0f;
            }
        }

        void covCountingSensorModel(const MatrixXType &x, const MatrixXType &z, MatrixKType &Kxz) const {
          Kxz = MatrixKType::Ones(x.rows(), z.rows());
        }

        T sf2;    // signal variance
        T ell;    // length-scale
        int nc;   // number of classes

        MatrixXType x;   // temporary storage of training data
        MatrixYType y;   // temporary storage of training labels
        std::vector<T> y_vec;
        std::vector<T> w_vec;  // per-point weights (empty = all 1.0)
        std::vector<std::vector<T>> y_soft;  // soft labels (n x nc); when non-empty, predict_csm uses these
        std::vector<std::vector<T>> w_class; // per-point per-class weights (n x nc); empty = all 1.0

        bool trained;    // true if bgkinference stored training data
    };

    typedef SemanticBKInference<3, float> SemanticBKI3f;

}
