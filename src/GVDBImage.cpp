#include <mrfmap/GVDBImage.h>

extern gvdb_params_t gvdb_params_;

GVDBImage::GVDBImage(const Eigen::MatrixXf &image) : params_(gvdb_params_), d_img_(NULL) {
  gpuAlloc(image);
}

GVDBImage::GVDBImage(const Eigen::MatrixXf &image, const gvdb_params_t &params) : params_(params), d_img_(NULL) {
  gpuAlloc(image);
}

void GVDBImage::gpuAlloc(const Eigen::MatrixXf &image) {
  // Allocate GPU memory for this image
  cuMemAlloc(&d_img_, params_.rows * params_.cols * sizeof(float));
#ifndef USE_DEPTH_IMAGE_BUFFER
  // Set nans to -1
  MRow image_clean = (image.array().isFinite()).select(image, -1.0f);

  // Copy over image data directly as depth framebuffer
  MRow flipped = image.colwise().reverse();
  flipped = (flipped.array().isFinite()).select(flipped, 0.001f);
  float far_plane = Z_MAX / params_.res, near_plane = 1.0f;
  // Scale flipped
  flipped /= params_.res;
  flipped = far_plane * (1.0f - near_plane / flipped.array()) / (far_plane - near_plane);

  // Copy to GPU
  cuMemcpyHtoD(d_img_, flipped.data(), params_.rows * params_.cols * sizeof(float));
#else
  // Set nans to -1, and don't bother about scaling, since we do direct access.
  MRow image_clean = (image.array().isFinite()).select(image, -1.0f);
  MRow flipped = image_clean.colwise().reverse();
  // // Clip outer periphery if realsense?
  if (params_.rows == 480 && params_.cols == 848) {
    flipped.leftCols<24>().array() = -1;
    flipped.rightCols<24>().array() = -1;
    flipped.topRows<20>().array() = -1;
    flipped.bottomRows<20>().array() = -1;
  }
  // Copy to GPU
  cuMemcpyHtoD(d_img_, flipped.data(), params_.rows * params_.cols * sizeof(float));
#endif
  // Also let's save all the reprojected points so that they can be easily transformed
  // when populating maps
  // Ok now this was a depth image; reproject to 3D

  reproj_points_ = MRow::Zero(params_.rows * params_.cols, 3);
  uint num_points = -1;
  for (int v = 0; v < params_.rows; ++v) {
    for (int u = 0; u < params_.cols; ++u) {
      float d = image_clean(v, u);
      float foverz = d / params_.K(0, 0);

      if (d > 0) {
        reproj_points_.row(++num_points) << foverz * (u - params_.K(0, 2)), foverz * (v - params_.K(1, 2)), d;
      }
    }
  }
  reproj_points_.conservativeResize(num_points, Eigen::NoChange);

#ifdef DEBUG_IMAGE
  cv::Mat img;
  eigen2cv(image, img);
  double min;
  double max;
  cv::minMaxIdx(img, &min, &max);
  cv::Mat adjMap;
  cv::convertScaleAbs(img, adjMap, 255 / max);
// cv::imshow("orig_img", adjMap);
// cv::waitKey(100);
#endif
}