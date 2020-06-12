#include <mrfmap/GVDBInference.h>
#include <mrfmap/GVDBMapLikelihoodEstimator.h>
#include <mrfmap/GVDBOctomapWrapper.h>
#include <mrfmap/KeyframeSelector.h>
#include <mrfmap/Viewer.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include <string>

PYBIND11_MAKE_OPAQUE(std::vector<float>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<uint>>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<std::vector<float>>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<uint>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::vector<float>>);
PYBIND11_MAKE_OPAQUE(std::vector<uint>);
namespace py = pybind11;

gvdb_params_t gvdb_params_;

void set_from_python(py::object obj) {
  gvdb_params_t *cls = obj.cast<gvdb_params_t *>();
  gvdb_params_ = *cls;
  LOG(std::cout << gvdb_params_ << "\n";);
}

PYBIND11_MODULE(GVDBPyModules, m) {
  m.doc() = "Python Wrapper for Occupancy inference using GVDB";  // optional module
                                                                  // docstring
  pybind11::bind_vector<std::vector<float>>(m, std::string("VecFloat"));
  pybind11::bind_vector<std::vector<int>>(m, std::string("VecInt"));
  pybind11::bind_vector<std::vector<uint>>(m, std::string("VecUInt"));
  pybind11::bind_vector<std::vector<std::vector<uint>>>(m, std::string("VecVecUInt"));
  pybind11::bind_vector<std::vector<std::vector<float>>>(m, std::string("VecVecFloat"));
  pybind11::bind_vector<std::vector<std::vector<std::vector<uint>>>>(m, std::string("VecVecVecUInt"));
  pybind11::bind_vector<std::vector<std::vector<std::vector<float>>>>(m, std::string("VecVecVecFloat"));

  pybind11::class_<gvdb_params_t>(m, "gvdb_params")
      .def(pybind11::init<>())
      .def_readwrite("prior", &gvdb_params_t::prior)
      .def_readwrite("res", &gvdb_params_t::res)
      .def_readwrite("dims", &gvdb_params_t::dims)
      .def_readwrite("const_sigmasq_poly", &gvdb_params_t::const_sigmasq_poly)
      .def_readwrite("occ_thresh", &gvdb_params_t::occ_thresh)
      .def_readwrite("max_iters", &gvdb_params_t::max_iters)
      .def_readwrite("dims", &gvdb_params_t::dims)
      .def_readwrite("rows", &gvdb_params_t::rows)
      .def_readwrite("cols", &gvdb_params_t::cols)
      .def_readwrite("K", &gvdb_params_t::K)
      .def_readwrite("use_polys", &gvdb_params_t::use_polys)
      .def_readwrite("lookup_n", &gvdb_params_t::lookup_n)

      .def("load_from_file", &gvdb_params_t::load_from_file)
      .def("set_from_python", &set_from_python);

  pybind11::class_<GVDBImage, std::shared_ptr<GVDBImage>>(m, "GVDBImage")
      .def(pybind11::init<const Eigen::MatrixXf &>())
      .def(pybind11::init<const Eigen::MatrixXf &, const gvdb_params_t &>());

  pybind11::class_<GVDBInference, std::shared_ptr<GVDBInference>>(m, "GVDBInference")
      .def(pybind11::init<bool, bool>())
      .def("add_camera", &GVDBInference::add_camera)
      .def("add_camera_with_depth", &GVDBInference::add_camera_with_depth)
      .def("perform_inference", &GVDBInference::perform_inference)
      .def("perform_inference_dryrun", &GVDBInference::perform_inference_dryrun)
      .def("set_pose", &GVDBInference::set_pose)
      .def("get_likelihood_image", &GVDBInference::get_likelihood_image)
      .def("get_expected_depth_image", &GVDBInference::get_expected_depth_image)
      .def("compute_likelihood", &GVDBInference::compute_likelihood)
      .def("set_selected_x_y", &GVDBInference::set_selected_x_y)
      .def("set_selected_voxel", &GVDBInference::set_selected_voxel)
      .def("set_owned_context", &GVDBInference::set_owned_context)
      .def("get_voxel_coords", &GVDBInference::get_voxel_coords)
      .def("get_diagnostic_image", &GVDBInference::get_diagnostic_image)
      .def("get_diagnostic_image_at_pose", &GVDBInference::get_diagnostic_image_at_pose)
      .def("get_occupied_bricks", &GVDBInference::get_occupied_bricks)
      .def("get_indices_and_alphas", &GVDBInference::get_alphas)
      .def("get_accuracy_image_at_pose", &GVDBInference::get_accuracy_image_at_pose)
      .def("get_likelihood_image_at_pose", &GVDBInference::get_likelihood_image_at_pose);

  pybind11::class_<GVDBMapLikelihoodEstimator, std::shared_ptr<GVDBMapLikelihoodEstimator>>(m, "GVDBMapLikelihoodEstimator")
      .def(pybind11::init<gvdb_params_t, bool>())
      .def("add_camera", &GVDBMapLikelihoodEstimator::add_camera)
      .def("activate_camera", &GVDBMapLikelihoodEstimator::activate_camera)
      .def("load_map", &GVDBMapLikelihoodEstimator::load_map)
      .def("compute_likelihood", &GVDBMapLikelihoodEstimator::compute_likelihood)
      .def("get_rendered_image", &GVDBMapLikelihoodEstimator::get_rendered_image)
      .def("get_rendered_image_at_pose", &GVDBMapLikelihoodEstimator::get_rendered_image_at_pose)
      .def("set_selected_x_y", &GVDBMapLikelihoodEstimator::set_selected_x_y)
      .def("set_acc_thresh", &GVDBMapLikelihoodEstimator::set_acc_thresh)
      .def("get_diagnostic_image", &GVDBMapLikelihoodEstimator::get_diagnostic_image)
      .def("get_diagnostic_image_at_pose", &GVDBMapLikelihoodEstimator::get_diagnostic_image_at_pose)
      .def("get_accuracy_image", &GVDBMapLikelihoodEstimator::get_accuracy_image)
      .def("get_accuracy_image_at_pose", &GVDBMapLikelihoodEstimator::get_accuracy_image_at_pose)
      .def("get_accuracy_image_at_pose_with_depth", &GVDBMapLikelihoodEstimator::get_accuracy_image_at_pose_with_depth)
      .def("get_simple_accuracy_at_pose", &GVDBMapLikelihoodEstimator::get_simple_accuracy_at_pose)
      .def("get_simple_accuracy_image_at_pose_with_depth", &GVDBMapLikelihoodEstimator::get_simple_accuracy_image_at_pose_with_depth)
      .def("get_likelihood_image_at_pose", &GVDBMapLikelihoodEstimator::get_likelihood_image_at_pose)
      .def("get_likelihood_image_at_pose_with_depth", &GVDBMapLikelihoodEstimator::get_likelihood_image_at_pose_with_depth)
      .def("get_occ_thresh_image_at_pose", &GVDBMapLikelihoodEstimator::get_occ_thresh_image_at_pose)
      .def("get_occupied_bricks", &GVDBMapLikelihoodEstimator::get_occupied_bricks)
      .def("get_indices_and_alphas", &GVDBMapLikelihoodEstimator::get_alphas)
      .def("get_likelihood_image", &GVDBMapLikelihoodEstimator::get_likelihood_image);

  pybind11::class_<PangolinViewer>(m, "PangolinViewer")
      .def(pybind11::init<std::string>())
      .def(pybind11::init<std::string, std::shared_ptr<GVDBInference>, std::shared_ptr<GVDBOctomapWrapper>>())
      .def(pybind11::init<std::string, std::shared_ptr<GVDBInference>>())
      .def("add_frame", &PangolinViewer::add_frame)
      .def("add_keyframe", &PangolinViewer::add_keyframe)
      .def("set_latest_pose", &PangolinViewer::set_latest_pose)
      .def("set_kf_pose", &PangolinViewer::set_kf_pose);

  pybind11::class_<GVDBOctomapWrapper, std::shared_ptr<GVDBOctomapWrapper>>(m, "GVDBOctomapWrapper")
      .def(pybind11::init<gvdb_params_t>())
      .def("add_camera", &GVDBOctomapWrapper::add_camera)
      .def("push_to_gvdb_volume", &GVDBOctomapWrapper::push_to_gvdb_volume)
      .def("get_accuracy_image_at_pose", &GVDBOctomapWrapper::get_accuracy_image_at_pose)
      .def("get_likelihood_image_at_pose", &GVDBOctomapWrapper::get_likelihood_image_at_pose)
      .def("set_selected_x_y", &GVDBOctomapWrapper::set_selected_x_y)
      .def("get_diagnostic_image", &GVDBOctomapWrapper::get_diagnostic_image)
      .def("get_diagnostic_image_at_pose", &GVDBOctomapWrapper::get_diagnostic_image_at_pose)
      .def("get_occupied_bricks", &GVDBOctomapWrapper::get_occupied_bricks)
      .def("get_alphas", &GVDBOctomapWrapper::get_alphas);

  pybind11::class_<KeyframeSelector>(m, "KeyframeSelector")
      .def(pybind11::init<float, float>())
      .def("compute_distance", &KeyframeSelector::compute_distance)
      .def("is_keyframe", &KeyframeSelector::is_keyframe);
}