#pragma once

#include <mrfmap/GVDBInference.h>
#include <mrfmap/GVDBMapLikelihoodEstimator.h>
#include <mrfmap/GVDBOctomapWrapper.h>
#include <mrfmap/GVDBParams.h>
#include <pangolin/handler/handler_image.h>
#include <pangolin/pangolin.h>

#include <Eigen/Core>
#include <atomic>
#include <bitset>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> RMatXf;
typedef Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> RMatXi;
typedef Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> RMatXU8;

extern gvdb_params_t gvdb_params_;

class PangolinViewer {
 private:
  std::string window_name_;
  std::shared_ptr<pangolin::OpenGlRenderState> cam_;
  std::vector<Eigen::MatrixXf> kf_images_;
  std::vector<Eigen::MatrixXf> kf_poses_;

  Eigen::MatrixXf latest_pose_;
  Eigen::MatrixXf latest_img_;

  std::shared_ptr<GVDBInference> inf_;
  std::shared_ptr<GVDBOctomapWrapper> octo_wrapper_;
  std::vector<std::shared_ptr<GVDBImage>> gvdb_images_;
  std::shared_ptr<std::thread> gui_thread_, mrfmap_thread_, octomap_thread_;

  std::timed_mutex data_mutex_;
  std::condition_variable mrfmap_cv_, octomap_cv_;
  bool init_, new_mrf_data_, new_octo_data_, was_following_;
  std::atomic_bool should_quit_;
  float injected_noise_;
  const bool view_only_mode_, mrfmap_only_mode_, render_likelihoods_;
  std::function<void()> exit_callback_;

 public:
  std::mutex mrfmap_mutex_, octomap_mutex_;
  // Version with external MRFMaps and Octomap pointers
  explicit PangolinViewer(std::string name, std::shared_ptr<GVDBInference> inf, std::shared_ptr<GVDBOctomapWrapper> octo_wrapper) : window_name_(name), inf_(inf), octo_wrapper_(octo_wrapper), latest_pose_(Eigen::Matrix4f::Identity()), latest_img_(Eigen::MatrixXf::Zero(gvdb_params_.rows, gvdb_params_.cols)), init_(false), new_mrf_data_(false), new_octo_data_(false), was_following_(false), should_quit_(false), injected_noise_(0.0f), view_only_mode_(true), mrfmap_only_mode_(false), render_likelihoods_(true), exit_callback_(std::function<void()>()) {
    init();
  }

  // Version with only external MRFMaps (and no Octomap)
  explicit PangolinViewer(std::string name, std::shared_ptr<GVDBInference> inf) : window_name_(name), inf_(inf), octo_wrapper_(NULL), latest_pose_(Eigen::Matrix4f::Identity()), latest_img_(Eigen::MatrixXf::Zero(gvdb_params_.rows, gvdb_params_.cols)), init_(false), new_mrf_data_(false), new_octo_data_(false), was_following_(false), should_quit_(false), injected_noise_(0.0f), view_only_mode_(false), mrfmap_only_mode_(true), render_likelihoods_(true), exit_callback_(std::function<void()>()) {
    init();
  }

  // Make PangolinViewer handle inference by spawning separate threads
  explicit PangolinViewer(std::string name, bool mrfmap_only_mode = true, bool render_likelihoods = true) : window_name_(name), latest_pose_(Eigen::Matrix4f::Identity()), latest_img_(Eigen::MatrixXf::Zero(gvdb_params_.rows, gvdb_params_.cols)), init_(false), new_mrf_data_(false), new_octo_data_(false), was_following_(false), should_quit_(false), injected_noise_(0.0f), view_only_mode_(false), mrfmap_only_mode_(mrfmap_only_mode), render_likelihoods_(render_likelihoods), exit_callback_(std::function<void()>()) {
    init();
  }

  ~PangolinViewer() {
    gui_thread_->join();
    if (!view_only_mode_) {
      should_quit_ = true;
      mrfmap_cv_.notify_one();
      mrfmap_thread_->join();
      if (!mrfmap_only_mode_) {
        octomap_cv_.notify_one();
        octomap_thread_->join();
      }
    }
  }

  void init();
  void run();
  void mrfmap_worker_runner();
  void octomap_worker_runner();

  void add_keyframe(const Eigen::MatrixXf &pose, const Eigen::MatrixXf &image);
  void add_frame(const Eigen::MatrixXf &pose, const Eigen::MatrixXf &image);
  void set_latest_pose(const Eigen::MatrixXf &pose) { latest_pose_ = pose; }
  void set_kf_pose(uint index, const Eigen::MatrixXf &pose);
  void signal_new_data();
  void register_exit_callback(const std::function<void()> &callback) { exit_callback_ = callback; }
  void signal_gui_end() { exit_callback_(); };
};

namespace pangolin {
struct MyHandler3D : Handler3D {
  MyHandler3D(OpenGlRenderState &cam_state,
              int cols,
              int rows,
              const std::function<void(int, int, int)> &callback,
              AxisDirection enforce_up = AxisNone,
              float trans_scale = 0.01f,
              float zoom_fraction = PANGO_DFLT_HANDLER3D_ZF)
      : Handler3D(cam_state, enforce_up, trans_scale, zoom_fraction), cols_(cols), rows_(rows), external_clicked_callback(callback){};
  void Mouse(View &display,
             MouseButton button,
             int x,
             int y,
             bool pressed,
             int button_state) {
    if (pressed) {
      if (external_clicked_callback) {
        external_clicked_callback(static_cast<int>(1.0f * (x - display.v.l) * cols_ / display.v.w),
                                  static_cast<int>(1.0f * (display.v.t() - y) * rows_ / display.v.h),
                                  button_state);
      }
    }
  }
  std::function<void(int, int, int)> external_clicked_callback;
  int cols_, rows_;
};
}  // namespace pangolin