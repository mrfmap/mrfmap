#include <Eigen/Core>
#include <Eigen/Geometry>
// GVDB library
#include <cuda_runtime_api.h>
#include <mrfmap/GVDBInference.h>
#include <mrfmap/GVDBOctomapWrapper.h>
#include <mrfmap/KeyframeSelector.h>
#include <mrfmap/Viewer.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

gvdb_params_t gvdb_params_;

template <class Matrix>
void read_binary(const std::string& filename, Matrix& matrix) {
  std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);
  typename Matrix::Index rows = 0, cols = 0;
  in.read((char*)(&rows), sizeof(typename Matrix::Index));
  in.read((char*)(&cols), sizeof(typename Matrix::Index));
  matrix.resize(rows, cols);
  in.read((char*)matrix.data(), rows * cols * sizeof(typename Matrix::Scalar));
  in.close();
}
int main(int argc, const char** argv) {
  int num_cams_ = 12;
  std::string params_file = CONFIG_PATH;

  gvdb_params_.load_from_file(params_file);

  KeyframeSelector selector(0.1f, 0.1f);
  PangolinViewer viewer(std::string("MRFMap Viewer"), false);

  YAML::Node params = YAML::LoadFile(params_file);
  YAML::Node rosviewer_params_node = params["viewer_params_nodes"];
  YAML::Node cam_in_body_node = rosviewer_params_node["cam_in_body"];
  Eigen::MatrixXf cam_in_body = Eigen::Matrix4f::Identity(), pose = Eigen::Matrix4f::Identity();
  Eigen::MatrixXf depth_img = Eigen::MatrixXf::Zero(gvdb_params_.rows, gvdb_params_.cols);

  if (cam_in_body_node) {
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        cam_in_body(i, j) = cam_in_body_node[i * 4 + j].as<float>();
      }
    }
  }

  for (uint cam_id = 0; cam_id < num_cams_; ++cam_id) {
    // Load image
    std::string data_path = DATA_PATH;
    read_binary(data_path + "pose_" + std::to_string(cam_id) + ".bin", pose);
    pose = pose * cam_in_body;
    read_binary(data_path + "depth_" + std::to_string(cam_id) + ".bin", depth_img);

    std::shared_ptr<GVDBImage> image_ptr = std::make_shared<GVDBImage>(depth_img);
    if (selector.is_keyframe(pose)) {
      viewer.add_keyframe(pose, depth_img);
    } else {
      viewer.add_frame(pose, depth_img);
    }
    sleep(1);
  }
  return 0;
}
