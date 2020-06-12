#pragma once
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <iostream>
#include <mrfmap/GVDBCommon.cuh>
#include <vector>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajorBit> RowMatXf;

struct gvdb_params_t {
  const float hfov = 0.01f;
  float res;
  std::vector<float> const_sigmasq_poly, const_bias_poly;
  float occ_thresh, acc_thresh;
  float prior;
  uint max_iters;
  Eigen::Vector3f dims;
  int rows;
  int cols;
  Eigen::MatrixXf K;

  bool use_polys;
  int lookup_n;
  std::vector<float> depth_sigma_lookup;
  std::vector<float> depth_bias_lookup;
  Eigen::VectorXi gvdb_map_config = (Eigen::VectorXi(5) << 1, 3, 3, 3, 3).finished();
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  gvdb_params_t() {
    res = 0.01f;
    occ_thresh = 0.1f;
    acc_thresh = 0.5f;
    prior = 0.01f;
    max_iters = 1;
    dims << 1.0f, 1.0f, 1.0f;
    rows = 1;
    cols = 1;
    K = Eigen::Matrix3f::Identity();
    K(0, 0) = K(1, 1) = (cols / 2.0f) / tanf(hfov / 2.0f);
    K(0, 2) = cols * 0.5f;
    K(1, 2) = rows * 0.5f;

    use_polys = false;
    lookup_n = 3;  // Quadratic polynomial
    const_sigmasq_poly.resize(3);
    const_bias_poly.resize(3);
    const_sigmasq_poly = {1.5f * res, 0.f, 0.0f};
    const_bias_poly = {.0f, .0f, .0f};
    depth_bias_lookup.reserve(LOOKUP_NUM);
    depth_sigma_lookup.reserve(LOOKUP_NUM);

    populate_lookup_tables();
  }

  gvdb_params_t(const gvdb_params_t &other) : res(other.res),
                                              const_sigmasq_poly(other.const_sigmasq_poly),
                                              const_bias_poly(other.const_bias_poly),
                                              occ_thresh(other.occ_thresh),
                                              acc_thresh(other.acc_thresh),
                                              prior(other.prior),
                                              max_iters(other.max_iters),
                                              dims(other.dims),
                                              rows(other.rows),
                                              cols(other.cols),
                                              K(other.K),
                                              use_polys(other.use_polys),
                                              lookup_n(other.lookup_n),
                                              depth_sigma_lookup(other.depth_sigma_lookup),
                                              depth_bias_lookup(other.depth_bias_lookup),
                                              gvdb_map_config(other.gvdb_map_config) {
  }

  gvdb_params_t &operator=(const gvdb_params_t &other) {
    res = other.res;
    const_sigmasq_poly = other.const_sigmasq_poly;
    const_bias_poly = other.const_bias_poly;
    occ_thresh = other.occ_thresh;
    acc_thresh = other.acc_thresh;
    prior = other.prior;
    max_iters = other.max_iters;
    dims = other.dims;
    rows = other.rows;
    cols = other.cols;
    K = other.K;
    use_polys = other.use_polys;
    lookup_n = other.lookup_n;
    depth_sigma_lookup = other.depth_sigma_lookup;
    depth_bias_lookup = other.depth_bias_lookup;
    gvdb_map_config = other.gvdb_map_config;
  }

  void populate_lookup_tables() {
    int n_us = static_cast<int>(ceilf(1.0f * cols / LOOKUP_PX));
    int n_vs = static_cast<int>(ceilf(1.0f * rows / LOOKUP_PX));
    depth_sigma_lookup.resize(n_us * n_vs * lookup_n);
    depth_bias_lookup.resize(n_us * n_vs * lookup_n);
    for (int i = 0; i < n_us * n_vs; ++i) {
      for (int j = 0; j < lookup_n; ++j) {
        depth_sigma_lookup[i * lookup_n + j] = const_sigmasq_poly[j];
        depth_bias_lookup[i * lookup_n + j] = const_bias_poly[j];
      }
    }
  }

  void load_from_file(const std::string &file_path) {
    try {
      std::cout << "Trying to read file " << file_path << "\n";
      YAML::Node params = YAML::LoadFile(file_path);
      YAML::Node gvdb_params_node = params["gvdb_params"];
      if (!gvdb_params_node) {
        std::cerr
            << "[gvdb_params] Could not read [gvdb_params]!";
        exit(-1);
      } else {
        YAML::Node rows_node = gvdb_params_node["rows"];
        if (rows_node) {
          rows = rows_node.as<float>();
        }
        YAML::Node cols_node = gvdb_params_node["cols"];
        if (cols_node) {
          cols = cols_node.as<float>();
        }
        YAML::Node dims_node = gvdb_params_node["dims"];
        if (dims_node) {
          for (int i = 0; i < 3; ++i) {
            dims[i] = dims_node[i].as<float>();
          }
        }
        if (gvdb_params_node["res"]) {
          res = gvdb_params_node["res"].as<float>();
        }

        if (gvdb_params_node["use_polys"]) {
          use_polys = gvdb_params_node["use_polys"].as<bool>();
          if (use_polys) {
            lookup_n = gvdb_params_node["poly_degree"].as<int>();
            std::string sigma_file = gvdb_params_node["depth_sigma_lookup"].as<std::string>();
            std::string bias_file = gvdb_params_node["depth_bias_lookup"].as<std::string>();
            auto pos = file_path.rfind("/");
            if (pos != std::string::npos) {
              sigma_file = file_path.substr(0, pos + 1) + sigma_file;
              bias_file = file_path.substr(0, pos + 1) + bias_file;
            }
            std::cout << "Trying to read " + sigma_file << "\n";
            YAML::Node sigma_node = YAML::LoadFile(sigma_file);
            int n_us = static_cast<int>(ceilf(1.0f * cols / LOOKUP_PX));
            int n_vs = static_cast<int>(ceilf(1.0f * rows / LOOKUP_PX));
            depth_sigma_lookup.resize(n_us * n_vs * lookup_n);
            for (int i = 0; i < n_us * n_vs; ++i) {
              for (int j = 0; j < lookup_n; ++j) {
                depth_sigma_lookup[i * lookup_n + j] = sigma_node[i * lookup_n + j].as<float>();
              }
            }
            std::cout << "Trying to read " + bias_file << "\n";
            YAML::Node bias_node = YAML::LoadFile(bias_file);
            depth_bias_lookup.resize(n_us * n_vs * lookup_n);
            for (int i = 0; i < n_us * n_vs; ++i) {
              for (int j = 0; j < lookup_n; ++j) {
                depth_bias_lookup[i * lookup_n + j] = bias_node[i * lookup_n + j].as<float>();
              }
            }
          } else {
            if (gvdb_params_node["const_sigmasq_poly"]) {
              lookup_n = gvdb_params_node["poly_degree"].as<int>();
              const_sigmasq_poly.resize(lookup_n);
              for (int j = 0; j < lookup_n; ++j) {
                const_sigmasq_poly[j] = gvdb_params_node["const_sigmasq_poly"][j].as<float>();
              }
              if (gvdb_params_node["const_bias_poly"]) {
                const_bias_poly.resize(lookup_n);
                for (int j = 0; j < lookup_n; ++j) {
                  const_bias_poly[j] = gvdb_params_node["const_bias_poly"][j].as<float>();
                }
              }
              populate_lookup_tables();
            }
          }
        } else {
          std::cout << "[gvdb_params] Please configure use_polys in config file. Exiting...\n";
          exit(-1);
        }
        if (gvdb_params_node["prior"]) {
          prior = gvdb_params_node["prior"].as<float>();
        }
        if (gvdb_params_node["occ_thresh"]) {
          occ_thresh = gvdb_params_node["occ_thresh"].as<float>();
        }
        if (gvdb_params_node["acc_thresh"]) {
          acc_thresh = gvdb_params_node["acc_thresh"].as<float>();
        }
        if (gvdb_params_node["max_iters"]) {
          max_iters = gvdb_params_node["max_iters"].as<int>();
        }
        YAML::Node K_node = gvdb_params_node["K"];
        if (K_node) {
          for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
              K(i, j) = K_node[i * 3 + j].as<float>();
            }
          }
        }
        YAML::Node map_config_node = gvdb_params_node["gvdb_map_config"];
        if (map_config_node) {
          for (int i = 0; i < 5; ++i) {
            gvdb_map_config[i] = map_config_node[i].as<int>();
          }
        }
      }
    } catch (const std::runtime_error &e) {
      std::cerr << e.what() << std::endl;
      exit(-1);
    }
  }
};

inline std::ostream &operator<<(std::ostream &o, const gvdb_params_t &params) {
  Eigen::Map<const RowMatXf> sigma_look = Eigen::Map<const RowMatXf>(params.depth_sigma_lookup.data(), (params.cols / LOOKUP_PX) * (params.rows / LOOKUP_PX), params.lookup_n);
  o << "\tres::" << params.res << "\n"
    << "\tconst_sigmasq_poly::" << params.const_sigmasq_poly[0] << ", " << params.const_sigmasq_poly[1] << ", " << params.const_sigmasq_poly[2] << "\n"
    << "\tconst_bias_poly::" << params.const_bias_poly[0] << ", " << params.const_bias_poly[1] << ", " << params.const_bias_poly[2] << "\n"
    << "\tocc_thresh::" << params.occ_thresh << "\n"
    << "\tacc_thresh::" << params.acc_thresh << "\n"
    << "\tprior::" << params.prior << "\n"
    << "\tmax_iters::" << params.max_iters << "\n"
    << "\tdims::" << params.dims.transpose() << "\n"
    << "\trows::" << params.rows << "\n"
    << "\tcols::" << params.cols << "\n"
    << "\tK::\n"
    << params.K << "\n"
    << "\tPolynomial Degree::" << params.lookup_n << "\n"
    // << "Sigma Lookup ::\n"
    // << sigma_look << "\n"
    << "\tGVDB Config::" << params.gvdb_map_config.transpose() << "\n";
  return o;
}