#pragma once
#include "dmlc/logging.h"
namespace dmlc {
namespace linear {

/**
 * @brief \f$ \lambda_1 * \|x\|_1 + \lambda_2 * \|x\|_2^2 \f$
 */
template <typename T>
class L1L2 {
 public:
  L1L2() : lambda1_(0), lambda2_(0) { }

  void set_lambda1(T lambda1) {
    CHECK_GE(lambda1, 0);
    lambda1_ = lambda1;
  }
  void set_lambda2(T lambda2) {
    CHECK_GE(lambda2, 0);
    lambda2_ = lambda2;
  }

  /**
   * \brief Solve the proximal operator:
   * \f$ \arg\min_x 0.5 * \beta * (x - z / \beta )^2 + h(x)\f$
   * where h denotes this penatly.
   *
   * No virtual function for performance consideration.
   *
   * \param z In proximal gradient descent, we have z = eta * w - grad
   * \param eta an estimation of the second order gradient, or the inverse of
   * the learning rate. It is often
   * approximated by sqrt(t) or sqrt(\sum_i grad_i^2)
   * \return the new w
   */
  inline T Solve(T z, T eta) {
    // soft-thresholding
    CHECK_GT(eta, 0);
    // LOG(INFO) << "Current lambda_l1 " << lambda1_ << "\n";
    if (z <= lambda1_ && z >= -lambda1_) return 0;
    return (z > 0 ? z - lambda1_ : z + lambda1_) / (eta + lambda2_);
  }

    inline T Solve_with_occurence(T z, T eta, T l1_factor, T l2_factor) {
      // soft-thresholding
      CHECK_GT(eta, 0);
      // LOG(INFO) << "Current lambda_l1 " << lambda1_ << "\n"
        T l1 = lambda1_ * l1_factor;
        T l2 = lambda2_ * l2_factor;
      if (z <= l1 && z >= -l1) return 0;
      return (z > 0 ? z - l1 : z + l1) / (eta + l2);
    }

 private:
  T lambda1_, lambda2_;
};

}  // namespace linear
}  // namespace dmlc
