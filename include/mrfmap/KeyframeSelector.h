#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
class KeyframeSelector {
 private:
  float rotation_thresh_, translation_thresh_;
  std::vector<Eigen::MatrixXf> kf_poses_;

 public:
  KeyframeSelector(float rotation_thresh = 0.1f, float translation_thresh = 0.1f) : rotation_thresh_(rotation_thresh), translation_thresh_(translation_thresh) {
  }
  void set_thresh(float rot_thresh, float trans_thresh){
    rotation_thresh_ = rot_thresh;
    translation_thresh_ = trans_thresh;
  }
  bool is_keyframe(const Eigen::MatrixXf&);
  float compute_distance(float&, float&, const Eigen::MatrixXf&, const Eigen::MatrixXf&);
};
