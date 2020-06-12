#include <mrfmap/KeyframeSelector.h>
#include <iostream>

// Borrowing from GTSAM codebase...
Eigen::Vector3f LogmapSO3(const Eigen::Matrix3f& R) {
  using std::sin;
  using std::sqrt;

  // note switch to base 1
  const double &R11 = R(0, 0), R12 = R(0, 1), R13 = R(0, 2);
  const double &R21 = R(1, 0), R22 = R(1, 1), R23 = R(1, 2);
  const double &R31 = R(2, 0), R32 = R(2, 1), R33 = R(2, 2);

  // Get trace(R)
  const double tr = R.trace();

  Eigen::Vector3f omega;

  // when trace == -1, i.e., when theta = +-pi, +-3pi, +-5pi, etc.
  // we do something special
  if (std::abs(tr + 1.0) < 1e-10) {
    if (std::abs(R33 + 1.0) > 1e-10)
      omega = (M_PI / sqrt(2.0 + 2.0 * R33)) * Eigen::Vector3f(R13, R23, 1.0 + R33);
    else if (std::abs(R22 + 1.0) > 1e-10)
      omega = (M_PI / sqrt(2.0 + 2.0 * R22)) * Eigen::Vector3f(R12, 1.0 + R22, R32);
    else
      // if(std::abs(R.r1_.x()+1.0) > 1e-10)  This is implicit
      omega = (M_PI / sqrt(2.0 + 2.0 * R11)) * Eigen::Vector3f(1.0 + R11, R21, R31);
  } else {
    double magnitude;
    const double tr_3 = tr - 3.0;  // always negative
    if (tr_3 < -1e-7) {
      double theta = acos((tr - 1.0) / 2.0);
      magnitude = theta / (2.0 * sin(theta));
    } else {
      // when theta near 0, +-2pi, +-4pi, etc. (trace near 3.0)
      // use Taylor expansion: theta \approx 1/2-(t-3)/12 + O((t-3)^2)
      magnitude = 0.5 - tr_3 * tr_3 / 12.0;
    }
    omega = magnitude * Eigen::Vector3f(R32 - R23, R13 - R31, R21 - R12);
  }
  return omega;
}

inline Eigen::Matrix3f skewSymmetric(double wx, double wy, double wz) {
  return (Eigen::Matrix3f() << 0.0, -wz, +wy, +wz, 0.0, -wx, -wy, +wx, 0.0).finished();
}
template <class Derived>
inline Eigen::Matrix3f skewSymmetric(const Eigen::MatrixBase<Derived>& w) {
  return skewSymmetric(w(0), w(1), w(2));
}

Eigen::VectorXf LogmapSE3(const Eigen::MatrixXf& p) {
  const Eigen::Vector3f w = LogmapSO3(p.block<3, 3>(0, 0));
  const Eigen::Vector3f T = p.block<3, 1>(0, 3);
  const double t = w.norm();
  if (t < 1e-10) {
    Eigen::VectorXf log = Eigen::VectorXf::Zero(6);
    log << w, T;
    return log;
  } else {
    const Eigen::Matrix3f W = skewSymmetric(w / t);
    // Formula from Agrawal06iros, equation (14)
    // simplified with Mathematica, and multiplying in T to avoid matrix math
    const double Tan = tan(0.5 * t);
    const Eigen::Vector3f WT = W * T;
    const Eigen::Vector3f u = T - (0.5 * t) * WT + (1 - t / (2. * Tan)) * (W * WT);
    Eigen::VectorXf log = Eigen::VectorXf::Zero(6);
    log << w, u;
    return log;
  }
}

float KeyframeSelector::compute_distance(float& rot, float& trans, const Eigen::MatrixXf& first, const Eigen::MatrixXf& second) {
  Eigen::MatrixXf delta = first.inverse() * second;
  Eigen::VectorXf tangent = LogmapSE3(delta);
  rot = tangent.head<3>().norm();
  trans = tangent.tail<3>().norm();
  return rot + trans;
}

bool KeyframeSelector::is_keyframe(const Eigen::MatrixXf& pose) {
  // Compute distance from previous keyframe
  float min_trans = 1.0e10f, min_rot = 1.0e10f;
  std::vector<Eigen::MatrixXf>::reverse_iterator pose_iter = kf_poses_.rbegin();
  for (int i=0; pose_iter != kf_poses_.rend(); ++pose_iter, ++i) {
    float rot = 0.0f, trans = 0.0f;
    compute_distance(rot, trans, pose, *pose_iter);
    if (trans < translation_thresh_ && rot < rotation_thresh_) {
      // This pose is close to a previous keyframe, bail
      return false;
    } else {
      if (trans < min_trans) min_trans = trans;
      if (rot < min_rot) min_rot = rot;
    }
  }
  // If we're here then we're not within the thresholds for any kf, add to kfs and return true
  kf_poses_.push_back(pose);
  return true;
}