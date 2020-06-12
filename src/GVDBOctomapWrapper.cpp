#include <mrfmap/GVDBOctomapWrapper.h>

void GVDBOctomapWrapper::init() {
  map_ = std::make_shared<octomap::OcTree>(params_.res);
  map_->setClampingThresMax(0.9999);
  map_->setClampingThresMin(0.0001);
  map_->setOccupancyThres(params_.occ_thresh);

  octomap::point3d bbxmax(params_.dims[0] * 0.5f,
                          params_.dims[1] * 0.5f,
                          params_.dims[2]);
  octomap::point3d bbxmin(-params_.dims[0] * 0.5f,
                          -params_.dims[1] * 0.5f,
                          0.0f);
  map_->setBBXMax(bbxmax);
  map_->setBBXMin(bbxmin);
  map_->useBBXLimit(true);
  octo_ = std::make_shared<GVDBMapLikelihoodEstimator>(params_);
}

void GVDBOctomapWrapper::add_camera_with_depth(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& img) {
  // Create a GVDBImage; this should just hang around till the camera object is deleted.
  octo_->push_ctx();
  std::shared_ptr<GVDBImage> gvdb_img = std::make_shared<GVDBImage>(img);
  add_camera(pose, gvdb_img);
  // Delete the temporary GVDBImage
  gvdb_img.reset();
  octo_->pop_ctx();
}

void GVDBOctomapWrapper::add_camera(const Eigen::MatrixXf& pose, const std::shared_ptr<GVDBImage>& img_ptr) {
  octomap::Pointcloud cloud;
  // Since I can't resize the cloud can't really memcpy...
  cloud.reserve(img_ptr->get_reproj_points().rows());
  for (int i = 0; i < img_ptr->get_reproj_points().rows(); ++i) {
    Eigen::Vector3f transformed_pt = pose.block<3, 3>(0, 0) * img_ptr->get_reproj_points().row(i).transpose() + pose.block<3, 1>(0, 3);
    cloud.push_back(transformed_pt[0],
                    transformed_pt[1],
                    transformed_pt[2]);
  }
  map_->insertPointCloud(cloud, octomap::point3d(pose(0, 3), pose(1, 3), pose(2, 3)));
  ++num_cams_;
}

void GVDBOctomapWrapper::push_to_gvdb_volume() {
  octo_->push_ctx();
  // And also push this to the map likelihood viewer?
  // First, pull out the indices and the alphas
  octomap::OcTree::tree_iterator t_it = map_->begin_tree();
  std::vector<float> coords, occupancies;
  for (; t_it != map_->end(); ++t_it) {
    if (t_it.isLeaf()) {
      if (map_->isNodeOccupied(*t_it)) {
        Eigen::Vector3f p(t_it.getX(), t_it.getY(), t_it.getZ());
        if ((p[0] > -0.5f * params_.dims[0] && p[0] <= 0.5f * params_.dims[0]) &&
            (p[1] > -0.5f * params_.dims[1] && p[1] <= 0.5f * params_.dims[1]) &&
            (p[2] > 0 && p[2] <= params_.dims[2])) {
          coords.push_back(p[0]);
          coords.push_back(p[1]);
          coords.push_back(p[2]);
          occupancies.push_back(t_it->getOccupancy());
        }
      }
    }
  }
  Eigen::Map<GVDBMapLikelihoodEstimator::PointMats> locs(coords.data(), coords.size() / 3, 3);
  Eigen::Map<GVDBMapLikelihoodEstimator::RMatXf> occs(occupancies.data(), occupancies.size(), 1);
  // Perform scaling to gvdb
  locs /= params_.res;
  locs.leftCols<2>().rowwise() += params_.dims.head<2>().transpose() * 0.5f / params_.res;

  octo_->load_map(locs, occs);
  octo_->pop_ctx();
}
