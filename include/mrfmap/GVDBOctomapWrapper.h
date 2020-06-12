#pragma once

#include <mrfmap/GVDBMapLikelihoodEstimator.h>
#include <mrfmap/GVDBParams.h>
#include <octomap/octomap.h>

class GVDBOctomapWrapper {
 private:
  std::shared_ptr<octomap::OcTree> map_;
  uint num_cams_;
  gvdb_params_t params_;

 public:
  std::shared_ptr<GVDBMapLikelihoodEstimator> octo_;

  GVDBOctomapWrapper(gvdb_params_t params) : params_(params), num_cams_(0) {
    init();
  }

  ~GVDBOctomapWrapper() {
    DEBUG(std::cout << "[GVDBOctomapWrapper] Deleting object!\n");
    map_.reset();
    octo_.reset();
  }

  void init();

  void add_camera_with_depth(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& img);

  void add_camera(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr);

  void push_to_gvdb_volume();

  uint get_num_cams() { return num_cams_; }

  float get_likelihood_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& depth_img, Eigen::Ref<GVDBImage::MRow> like_img) {
    return octo_->get_likelihood_image_at_pose(pose, depth_img, like_img);
  }

  float get_accuracy_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& depth_img, Eigen::Ref<GVDBImage::MRow> acc_img) {
    return octo_->get_accuracy_image_at_pose(pose, depth_img, acc_img);
  }

  void get_diagnostic_image(uint id, Eigen::Ref<GVDBImage::MRow> eigen_img, int mode) {
    octo_->get_diagnostic_image(id, eigen_img);
  }

  void get_diagnostic_image_at_pose(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr, Eigen::Ref<GVDBImage::MRow> eigen_img) {
    octo_->get_diagnostic_image_at_pose(pose, img_ptr, eigen_img);
  }

  void set_selected_x_y(uint x, uint y) { octo_->set_selected_x_y(x, y); }

  Eigen::MatrixXf get_occupied_bricks(int lev) {
    return octo_->get_occupied_bricks(lev);
  }
  void get_alphas(Eigen::Ref<GVDBImage::MRow> pts) {
    octo_->get_alphas(pts);
  }
};