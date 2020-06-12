#include <codetimer/codetimer.h>
#include <mrfmap/Viewer.h>

#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>

inline Eigen::Matrix4f get_opengl_cam_pose_in_world(const pangolin::OpenGlRenderState& cam, const Eigen::Matrix4f& latest_pose, bool follow) {
  Eigen::MatrixXf world_in_gl = pangolin::ToEigen<float>(cam.GetModelViewMatrix());
  if (follow) {
    // Then we need to also use the last pose
    world_in_gl = world_in_gl * latest_pose.inverse();
  }
  Eigen::MatrixXf gl_in_world = world_in_gl.inverse();
  Eigen::Matrix4f opengl_in_cam;
  opengl_in_cam << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
  Eigen::MatrixXf pose_in_world = gl_in_world * opengl_in_cam.inverse();
  return pose_in_world;
}

void PangolinViewer::init() {
  pangolin::CreateWindowAndBind(window_name_, 1920, 720);
  glEnable(GL_DEPTH_TEST);
  // unset the current context from the main thread
  pangolin::GetBoundWindow()->RemoveCurrent();

  // If we're also managing the inference pointers
  if (!view_only_mode_) {
    inf_ = std::make_shared<GVDBInference>(true, false);
    // inf_->set_selected_x_y(209,120);
    // inf_->set_selected_voxel(141021);
    if (!mrfmap_only_mode_) {
      octo_wrapper_ = std::make_shared<GVDBOctomapWrapper>(gvdb_params_);
      octomap_thread_ = std::make_shared<std::thread>(&PangolinViewer::octomap_worker_runner, this);
    }
    mrfmap_thread_ = std::make_shared<std::thread>(&PangolinViewer::mrfmap_worker_runner, this);
  }

  // CREATE A THREAD WOOP
  gui_thread_ = std::make_shared<std::thread>(&PangolinViewer::run, this);
}

void PangolinViewer::mrfmap_worker_runner() {
  std::cout << "Starting MRFMap thread...\n";
  // Bind the context to this thread
  cuCtxSetCurrent(inf_->gvdb_->getContext());
  while (!should_quit_) {
    std::unique_lock<std::mutex> lock(mrfmap_mutex_);
    mrfmap_cv_.wait(lock, [this] { return new_mrf_data_ || should_quit_; });
    if (should_quit_) {
      break;
    }
    DEBUG(std::cout << "New MRFMap keyframe!!\n");
    // We have new data! Pull in the latest gvdb image and add it!
    inf_->add_camera(kf_poses_.back(), gvdb_images_.back());
    inf_->perform_inference();
    new_mrf_data_ = false;
  }
  std::cout << "MRFmap thread ended!\n";
}

void PangolinViewer::octomap_worker_runner() {
  std::cout << "Starting OctoMap thread...\n";
  // Bind the context to this thread
  cuCtxSetCurrent(octo_wrapper_->octo_->gvdb_->getContext());
  while (!should_quit_) {
    std::unique_lock<std::mutex> lock(octomap_mutex_);
    octomap_cv_.wait(lock, [this] { return new_octo_data_ || should_quit_; });
    if (should_quit_) {
      break;
    }
    DEBUG(std::cout << "New OctoMap keyframe!!\n");
    // We have new data! Pull in the latest gvdb image and add it!
    octo_wrapper_->add_camera(kf_poses_.back(), gvdb_images_.back());
    octo_wrapper_->push_to_gvdb_volume();
    new_octo_data_ = false;
  }
  std::cout << "Octomap thread ended!\n";
}

void PangolinViewer::run() {
  std::cout << "Starting PangoViewer thread!\n";
  // fetch the context and bind it to this thread
  pangolin::BindToContext(window_name_);

  // we manually need to restore the properties of the context
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // Select glformat and gltype
  GLint glformat = GL_LUMINANCE;  // Can be GL_LUMINANCE for 1 chan, GL_RGB for 3 chan, GL_RGBA for 4 chan
  GLenum gltype = GL_FLOAT;       // Can be GL_UNSIGNED_BYTE for 8bit, GL_UNSIGNED_SHORT for 16, and GL_FLOAT for 32 bit

  pangolin::OpenGlMatrix proj = pangolin::ProjectionMatrix(
      gvdb_params_.cols,
      gvdb_params_.rows,
      gvdb_params_.K(0, 0),
      gvdb_params_.K(1, 1),
      gvdb_params_.K(0, 2),
      gvdb_params_.K(1, 2), 1.0, Z_MAX / gvdb_params_.res);
  cam_ = std::make_shared<pangolin::OpenGlRenderState>(
      proj,
      pangolin::ModelViewLookAt(5, 0, 4, 0, 0, 0, pangolin::AxisZ));
  pangolin::OpenGlRenderState cam_img(
      proj,
      pangolin::ModelViewLookAt(0, 0, 1, 0, 0, 0, pangolin::AxisY));

  int num_cams = 0, cam_id = 0;
  pangolin::DataLog log, debug_log;
  std::vector<std::string> labels = {"t", "MRFMap"};
  if (!mrfmap_only_mode_) {
    labels.push_back("Octomap");
  }
  log.SetLabels(labels);
  pangolin::Plotter plotter(&log, 0, 100.0, 0, 1, 0.001, 0.5f);
  plotter.SetBounds(0.0, 1.0, 0.0, 1.0);
  plotter.ClearSeries();
  plotter.AddSeries("$0", "$1");
  if (!mrfmap_only_mode_) {
    plotter.AddSeries("$0", "$2");
  }

  std::vector<std::string> debug_labels = {"distance", "alphas", "vis_is", "w_is"};
  debug_log.SetLabels(debug_labels);
  pangolin::Plotter plotter2(&debug_log, 0, 10.0, 0, 1, 0.01, 0.05f);
  plotter2.SetBounds(0.0, 1.0, 0.0, 1.0);
  plotter2.ClearSeries();
  plotter2.AddSeries("$0", "$1");
  plotter2.AddSeries("$0", "$2");
  plotter2.AddSeries("$0", "$3");
  // Create Interactive View in window
  pangolin::View& view_map = pangolin::CreateDisplay()
                                 //  .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
                                 .SetAspect(1.0f * gvdb_params_.cols / gvdb_params_.rows)
                                 .SetHandler(new pangolin::Handler3D(*cam_));

  // Create view for depth images
  pangolin::View& view_depth = pangolin::CreateDisplay()
                                   .SetAspect(1.0f * gvdb_params_.cols / gvdb_params_.rows)
                                   .SetHandler(new pangolin::Handler3D(cam_img));

  // Create view for likelihood images
  pangolin::View& view_octo = pangolin::CreateDisplay()
                                  .SetAspect(1.0f * gvdb_params_.cols / gvdb_params_.rows)
                                  .SetHandler(new pangolin::Handler3D(*cam_));

  // Create views for accuracy images
  pangolin::View& view_acc_mrf = pangolin::CreateDisplay()
                                     .SetAspect(1.0f * gvdb_params_.cols / gvdb_params_.rows);
  // Use a mouse feedback handler
  pangolin::MyHandler3D mouse_handler =
      pangolin::MyHandler3D(cam_img,
                            gvdb_params_.cols,
                            gvdb_params_.rows,
                            [&](int x, int y, int state) {

#ifdef ENABLE_DEBUG
                              // Call the diagnostic method here
                              std::unique_lock<std::mutex> lock(mrfmap_mutex_, std::defer_lock);
                              if (lock.try_lock()) {
                                // Set the x and y
                                inf_->set_selected_x_y(x, y);
                                // Also reset selected node
                                // inf_->set_selected_voxel(0);
                                Eigen::RowVectorXf debug_img = Eigen::RowVectorXf::Zero(gvdb_params_.rows * gvdb_params_.cols);
                                inf_->get_diagnostic_image(cam_id, debug_img);
                                //Extract data from image
                                const int stride = 10;  // Each voxel entry is 10*sizeof(float) wide
                                int num_voxels = static_cast<int>(debug_img.tail<1>()[0]);

                                Eigen::Map<Eigen::RowVectorXf, 0, Eigen::InnerStride<stride>> z_distances(debug_img.data(), num_voxels);
                                Eigen::Map<Eigen::RowVectorXf, 0, Eigen::InnerStride<stride>> alphas(debug_img.data() + 1, num_voxels);
                                Eigen::Map<Eigen::RowVectorXf, 0, Eigen::InnerStride<stride>> w_is(debug_img.data() + 4, num_voxels);
                                Eigen::Map<Eigen::RowVectorXf, 0, Eigen::InnerStride<stride>> vis_is(debug_img.data() + 5, num_voxels);

                                Eigen::Map<Eigen::Matrix<unsigned long, 1, -1>, 0, Eigen::InnerStride<stride / 2>> node_ids(reinterpret_cast<unsigned long*>(debug_img.data()) + 1, num_voxels);

                                debug_log.Clear();
                                for (int i = 0; i < num_voxels; ++i) {
                                  debug_log.Log(z_distances[i], alphas[i], vis_is[i], w_is[i]);
                                  std::cout << "a:" << alphas[i] << "::" << node_ids[i] << "\t";
                                }
                                std::cout << "\n";
                              }
#endif
                            });
  view_acc_mrf.SetHandler(&mouse_handler);

  pangolin::View& view_acc_octo = pangolin::CreateDisplay()
                                      .SetAspect(1.0f * gvdb_params_.cols / gvdb_params_.rows)
                                      .SetHandler(new pangolin::Handler3D(cam_img));

  // Tile the views
  if (!mrfmap_only_mode_) {
    if (render_likelihoods_) {
      pangolin::Display("multi")
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0)
          .SetLayout(pangolin::Layout::LayoutEqual)
          .AddDisplay(view_depth)
          .AddDisplay(view_map)
          .AddDisplay(view_octo)
          .AddDisplay(view_acc_mrf)
          .AddDisplay(view_acc_octo)
          .AddDisplay(plotter);
    } else {
      pangolin::Display("multi")
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0)
          .SetLayout(pangolin::Layout::LayoutEqual)
          .AddDisplay(view_depth)
          .AddDisplay(view_map)
          .AddDisplay(view_octo);
    }
  } else {
    if (render_likelihoods_) {
      pangolin::Display("multi")
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0)
          .SetLayout(pangolin::Layout::LayoutEqual)
          .AddDisplay(view_depth)
          .AddDisplay(view_map)
          .AddDisplay(view_acc_mrf)
          .AddDisplay(plotter)
          .AddDisplay(plotter2);
    } else {
      pangolin::Display("multi")
          .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0)
          .SetLayout(pangolin::Layout::LayoutEqual)
          .AddDisplay(view_depth)
          .AddDisplay(view_map);
    }
  }

  pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
  pangolin::Var<bool> menuFollowCamera("menu.FollowCamera", false, true);
  pangolin::Var<bool> menuMapWireframe("menu.MapWireframe", false, true);
  pangolin::Var<int> menuLevel("menu.WireLevel", -1, -1, 4);
  pangolin::Var<float> menuThresh("menu.WireThresh", 0.5, 0.0, 1.0);
  pangolin::Var<bool> menuRenderLikelihood("menu.RenderLikelihood", false, true);
  pangolin::Var<bool> menuRenderAccuracy("menu.RenderAccuracy", true, true);
  pangolin::Var<bool> menuRenderCloud("menu.RenderCloud", true, true);
  pangolin::Var<bool> menuRenderMap("menu.RenderMap", true, true);
  pangolin::Var<bool> menuRenderPoses("menu.RenderPoses", true, true);
  pangolin::Var<int> menuImageId("menu.DisplayImage", 0);
  pangolin::Var<float> menuNoise("menu.InjectedNoise", 0.0, 0.0, 1.0);
  pangolin::Var<bool> menuReset("menu.Reset", false, false);
  pangolin::RegisterKeyPressCallback('n', [&]() { cam_id = cam_id >= num_cams - 1 ? 0 : cam_id + 1; std::unique_lock<std::timed_mutex> lock(data_mutex_, std::defer_lock); if(lock.try_lock()){ latest_pose_ = kf_poses_[cam_id]; latest_img_ = kf_images_[cam_id];} });
  pangolin::RegisterKeyPressCallback('p', [&]() { cam_id = cam_id <= 0 ? num_cams - 1 : cam_id - 1; std::unique_lock<std::timed_mutex> lock(data_mutex_, std::defer_lock); if(lock.try_lock()){ latest_pose_ = kf_poses_[cam_id]; latest_img_ = kf_images_[cam_id];} });

  // Allocate memory for displaying raw depth images
  // unsigned char* imageArray = new unsigned char[3 * gvdb_params_.cols * gvdb_params_.rows];
  pangolin::GlTexture imageTexture(gvdb_params_.cols, gvdb_params_.rows, glformat, false, 0, glformat, gltype);
  pangolin::GlTexture colorized_depth_tex(gvdb_params_.cols, gvdb_params_.rows, GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);
  pangolin::GlTexture likeTexture(gvdb_params_.cols, gvdb_params_.rows, glformat, false, 0, glformat, gltype);
  pangolin::GlTexture accOctoTexture(gvdb_params_.cols, gvdb_params_.rows, glformat, false, 0, glformat, gltype);
  pangolin::GlTexture accMRFTexture(gvdb_params_.cols, gvdb_params_.rows, glformat, false, 0, glformat, gltype);
  pangolin::GlTexture renderTexture(gvdb_params_.cols, gvdb_params_.rows, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);
  pangolin::GlTexture octoTexture(gvdb_params_.cols, gvdb_params_.rows, GL_RGBA, false, 0, GL_RGBA, GL_UNSIGNED_BYTE);

  pangolin::GlSlProgram voxel_render_prog, pointcloud_render_prog;
  std::string shader_str = "point_voxels";
  if (!voxel_render_prog.AddShaderFromFile(pangolin::GlSlVertexShader, SHADER_DIR + shader_str + ".vert")) {
    throw std::runtime_error("could not compile Vertex shader!");
  }
  if (!voxel_render_prog.AddShaderFromFile(pangolin::GlSlGeometryShader, SHADER_DIR + shader_str + ".geom")) {
    throw std::runtime_error("could not compile Geometry shader!");
  }
  if (!voxel_render_prog.AddShaderFromFile(pangolin::GlSlFragmentShader, SHADER_DIR + shader_str + ".frag")) {
    throw std::runtime_error("could not compile Fragment shader!");
  }
  voxel_render_prog.Link();

  shader_str = "pointcloud";
  pointcloud_render_prog.AddShaderFromFile(pangolin::GlSlVertexShader, SHADER_DIR + shader_str + ".vert");
  pointcloud_render_prog.AddShaderFromFile(pangolin::GlSlFragmentShader, SHADER_DIR + shader_str + ".frag");
  pointcloud_render_prog.Link();

  if (!voxel_render_prog.Valid() || !pointcloud_render_prog.Valid()) {
    throw std::runtime_error("Shaders are not valid!");
  }

  // Pregenerate cloud_VAO and cloud_VBO for pointcloud and map wireframe cubes
  GLuint cloud_VAO, cloud_VBO, wireframe_VAO, wireframe_VBO;
  glGenBuffers(1, &cloud_VBO);
  glGenVertexArrays(1, &cloud_VAO);
  glGenBuffers(1, &wireframe_VBO);
  glGenVertexArrays(1, &wireframe_VAO);

  glBindBuffer(GL_ARRAY_BUFFER, wireframe_VBO);
  glBindVertexArray(wireframe_VAO);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE,
                        sizeof(float) * 3, (void*)0);
  glBindVertexArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  float mrf_acc = 0.0f, octo_acc = 0.0f;
  float mrf_like_val = 0.0f, octo_like_val = 0.0f;

  std::cout << "Starting loop!\n";
  auto lastTime = std::chrono::system_clock::now(), startTime = std::chrono::system_clock::now(), lastLogTime = std::chrono::system_clock::now();
  Eigen::Matrix3f K_inv = Eigen::Matrix3f(gvdb_params_.K).inverse();
  int nbFrames = 0;
  std::shared_ptr<GVDBImage> gvdb_img;
  Eigen::Matrix4f latest_pose_temp;
  Eigen::MatrixXf latest_img_temp;
  int num_vertices = 0;  // Number of vertices in wireframe grid

  while (!pangolin::ShouldQuit()) {
    bool have_latest_data = false;
    // Measure speed
    auto currentTime = std::chrono::system_clock::now();
    nbFrames++;
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    if (std::chrono::duration<double>(currentTime - lastTime).count() >= 1.0) {  // If last prinf() was more than 1sec ago
      // printf and reset
      printf("%f ms/frame Running for ::%f s\r", 1000.0 / double(nbFrames),
             std::chrono::duration<double>(currentTime - startTime).count());

      nbFrames = 0;
      lastTime = std::chrono::system_clock::now();
    }

    {
      std::unique_lock<std::timed_mutex> lock(data_mutex_, std::defer_lock);
      if (lock.try_lock()) {
        num_cams = kf_images_.size();
        latest_pose_temp = latest_pose_;
        latest_img_temp = latest_img_;
        have_latest_data = true;
      } else {
        // DEBUG(std::cout << "OMG didn't get data lock!\n");
        // continue;
      }
    }
    menuImageId = cam_id;
    // Clear screen and activate view to render into
    if(!mrfmap_only_mode_){
      view_octo.Activate();
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    }
    view_map.Activate(*cam_);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    

    if (num_cams > 0) {
      if (!init_) {
        // If this is the first time we've come inside here, move the model view matrix to this pose
        Eigen::Matrix4f opengl_in_cam;
        opengl_in_cam << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
        Eigen::Matrix4f opengl_in_world = latest_pose_ * opengl_in_cam;
        Eigen::Matrix4f view_mat = opengl_in_world.inverse();
        // Move view mat back along the z axis
        view_mat(2, 3) -= 2.0f;
        cam_->SetModelViewMatrix(view_mat);
        cuCtxSetCurrent(inf_->gvdb_->getContext());
        init_ = true;
        std::cout << "Viewer initialized!\n";
      }
      if (have_latest_data) {
        // Only if we have new data do we need to create a GVDBImage...
        gvdb_img = std::make_shared<GVDBImage>(latest_img_temp);
      }
      if (!view_only_mode_) {
        injected_noise_ = menuNoise;
      } else {
        menuNoise = 0.0f;
      }

      // Make buttons sticky.
      if (pangolin::Pushed(menuFollowCamera)) menuFollowCamera = !menuFollowCamera;
      if (pangolin::Pushed(menuRenderLikelihood)) menuRenderLikelihood = !menuRenderLikelihood;
      if (pangolin::Pushed(menuRenderAccuracy)) menuRenderAccuracy = !menuRenderAccuracy;
      if (pangolin::Pushed(menuMapWireframe)) menuMapWireframe = !menuMapWireframe;
      if (pangolin::Pushed(menuRenderCloud)) menuRenderCloud = !menuRenderCloud;
      if (pangolin::Pushed(menuRenderMap)) menuRenderMap = !menuRenderMap;
      if (pangolin::Pushed(menuRenderPoses)) menuRenderPoses = !menuRenderPoses;

      if (menuFollowCamera) {
        cam_->Follow(latest_pose_temp);
        was_following_ = true;
      } else {
        if (was_following_) {
          cam_->Unfollow();
          was_following_ = false;
        }
      }

      if (render_likelihoods_ && menuRenderLikelihood) {
        menuRenderAccuracy = false;
        {
          std::unique_lock<std::mutex> lock(mrfmap_mutex_, std::defer_lock);
          RMatXf like_img = RMatXf::Zero(gvdb_params_.rows, gvdb_params_.cols);
          bool got_data = false;
          if (lock.try_lock()) {
            // inf_->get_likelihood_image(cam_id, like_img);
            mrf_like_val = inf_->get_likelihood_image_at_pose(latest_pose_temp, gvdb_img, like_img);
            got_data = true;
          }
          std::string view_like_text = "MRF Likelihood";
          if (got_data) {
            // Upload image data to GPU
            view_acc_mrf.Activate(cam_img);
            likeTexture.Upload(reinterpret_cast<unsigned char*>(like_img.data()), glformat, gltype);
            glDepthMask(GL_FALSE);  // Reset to default use of depth mask
            likeTexture.RenderToViewportFlipY();
            glDepthMask(GL_TRUE);  // Reset to default use of depth mask
          } else {
            view_like_text = "(BUSY)";
          }
          glPushAttrib(GL_CURRENT_BIT);
          glColor3f(1.0, 0.0, 0.0);
          pangolin::GlFont::I().Text(view_like_text.c_str()).DrawWindow(view_acc_mrf.v.r() - 0.8f * view_like_text.length() * pangolin::GlFont::I().Height(), view_acc_mrf.v.t() - 1.1 * pangolin::GlFont::I().Height());
          glPopAttrib();
        }

        // Show likelihood image for octo
        if (!mrfmap_only_mode_) {
          std::unique_lock<std::mutex> lock(mrfmap_mutex_, std::defer_lock);
          RMatXf like_img = RMatXf::Zero(gvdb_params_.rows, gvdb_params_.cols);
          bool got_data = false;
          if (lock.try_lock()) {
            octo_like_val = octo_wrapper_->octo_->get_likelihood_image_at_pose(latest_pose_temp, gvdb_img, like_img);
            got_data = true;
          }
          std::string view_like_text = "Octo Likelihood";
          if (got_data) {
            view_acc_octo.Activate(cam_img);
            likeTexture.Upload(reinterpret_cast<unsigned char*>(like_img.data()), glformat, gltype);
            likeTexture.RenderToViewportFlipY();

            glDepthMask(GL_FALSE);  // Reset to default use of depth mask
            likeTexture.RenderToViewportFlipY();
            glDepthMask(GL_TRUE);  // Reset to default use of depth mask
          } else {
            view_like_text = "(BUSY)";
          }
          glPushAttrib(GL_CURRENT_BIT);
          glColor3f(1.0, 0.0, 0.0);
          pangolin::GlFont::I().Text(view_like_text.c_str()).DrawWindow(view_acc_octo.v.r() - 0.8f * view_like_text.length() * pangolin::GlFont::I().Height(), view_acc_octo.v.t() - 1.1 * pangolin::GlFont::I().Height());
          glPopAttrib();
        }

        if (!mrfmap_only_mode_) {
          float dt = std::chrono::duration<double>(std::chrono::system_clock::now() - lastLogTime).count();     //s
          float dtStart = std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();  //s
                                                                                                                // if (dt > 0.1) {
          log.Log(dtStart, logf(-mrf_like_val), logf(-octo_like_val));
          lastLogTime = std::chrono::system_clock::now();
          // }
        } else {
          float dt = std::chrono::duration<double>(std::chrono::system_clock::now() - lastLogTime).count();     //s
          float dtStart = std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();  //s
                                                                                                                // if (dt > 0.1) {
          log.Log(dtStart, logf(-mrf_like_val));
          lastLogTime = std::chrono::system_clock::now();
          // }
        }
      }

      if (render_likelihoods_ && menuRenderAccuracy) {
        menuRenderLikelihood = false;
        Camera3D* cam = new Camera3D;
        cam->mNear = 0.1f;
        cam->mFar = Z_MAX / gvdb_params_.res;
        Eigen::Matrix4f pose_in_world = get_opengl_cam_pose_in_world(*cam_, latest_pose_temp, menuFollowCamera);
        TMat view_mat = TMat::Zero(4, 4), proj_mat = TMat::Zero(4, 4);
        convert_pose_to_opengl_proj_mat(pose_in_world, view_mat, proj_mat);
        cam->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
        cam->updateFrustum();

        {
          std::unique_lock<std::mutex> lock(mrfmap_mutex_, std::defer_lock);
          RMatXf acc_img_mrf = RMatXf::Zero(gvdb_params_.rows, gvdb_params_.cols);
          bool got_data = false;
          if (lock.try_lock()) {
            mrf_acc = inf_->get_accuracy_image_at_pose(latest_pose_temp, gvdb_img, acc_img_mrf);
            got_data = true;
          }
          std::string view_like_text = "MRF Accuracy";
          if (got_data) {
            view_acc_mrf.Activate(cam_img);
            accMRFTexture.Upload(reinterpret_cast<unsigned char*>(acc_img_mrf.data()), glformat, gltype);
            glDepthMask(GL_FALSE);  // Reset to default use of depth mask
            accMRFTexture.RenderToViewportFlipY();
            glDepthMask(GL_TRUE);  // Reset to default use of depth mask
          } else {
            view_like_text = "(BUSY)";
          }
          glPushAttrib(GL_CURRENT_BIT);
          glColor3f(1.0, 0.0, 0.0);
          pangolin::GlFont::I().Text(view_like_text.c_str()).DrawWindow(view_acc_mrf.v.r() - 0.8f * view_like_text.length() * pangolin::GlFont::I().Height(), view_acc_mrf.v.t() - 1.1 * pangolin::GlFont::I().Height());
          glPopAttrib();
        }

        if (!mrfmap_only_mode_) {
          std::unique_lock<std::mutex> lock(octomap_mutex_, std::defer_lock);
          RMatXf acc_img_octo = RMatXf::Zero(gvdb_params_.rows, gvdb_params_.cols);
          bool got_data = false;
          if (lock.try_lock()) {
            octo_acc = octo_wrapper_->octo_->get_accuracy_image_at_pose(latest_pose_temp, gvdb_img, acc_img_octo);
            got_data = true;
          } else {
            DEBUG(std::cout << "Octomap is busy inferring...\n");
          }
          std::string view_like_text = "Octo Accuracy";
          if (got_data) {
            view_acc_octo.Activate(cam_img);
            accOctoTexture.Upload(reinterpret_cast<unsigned char*>(acc_img_octo.data()), glformat, gltype);

            glDepthMask(GL_FALSE);  // Reset to default use of depth mask
            accOctoTexture.RenderToViewportFlipY();
            glDepthMask(GL_TRUE);  // Reset to default use of depth mask
          } else {
            view_like_text = "(BUSY)";
          }
          glPushAttrib(GL_CURRENT_BIT);
          glColor3f(1.0, 0.0, 0.0);
          pangolin::GlFont::I().Text(view_like_text.c_str()).DrawWindow(view_acc_octo.v.r() - 0.8f * view_like_text.length() * pangolin::GlFont::I().Height(), view_acc_octo.v.t() - 1.1 * pangolin::GlFont::I().Height());
          glPopAttrib();
        }

        if (!mrfmap_only_mode_) {
          float dt = std::chrono::duration<double>(std::chrono::system_clock::now() - lastLogTime).count();     //s
          float dtStart = std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();  //s
                                                                                                                // if (dt > 0.1) {
          log.Log(dtStart, mrf_acc, octo_acc);
          lastLogTime = std::chrono::system_clock::now();
          // }
        } else {
          float dt = std::chrono::duration<double>(std::chrono::system_clock::now() - lastLogTime).count();     //s
          float dtStart = std::chrono::duration<double>(std::chrono::system_clock::now() - startTime).count();  //s
                                                                                                                // if (dt > 0.1) {
          log.Log(dtStart, mrf_acc);
          lastLogTime = std::chrono::system_clock::now();
          // }
        }
      }

      if (menuRenderCloud) {
        view_map.Activate(*cam_);
        // Simply render the reprojected points of the current GVDBImage
        glBindVertexArray(cloud_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, cloud_VBO);

        // Send the reprojected points up, but saved in column order
        Eigen::MatrixXf cam_points = gvdb_img->get_reproj_points().transpose();
        // Transform these to world space given current pose
        Eigen::MatrixXf world_points = (latest_pose_temp * cam_points.colwise().homogeneous()).colwise().hnormalized();

        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * world_points.cols() * world_points.rows(), world_points.data(), GL_STATIC_DRAW);
        GLint vPosLoc = 0;
        glVertexAttribPointer(vPosLoc, 3, GL_FLOAT, GL_FALSE,
                              sizeof(float) * 3, (void*)0);
        glEnableVertexAttribArray(vPosLoc);
        // Enable billboard
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_POINT_SPRITE);
        pointcloud_render_prog.SaveBind();
        pangolin::OpenGlMatrix mvp = cam_->GetProjectionModelViewMatrix();
        pointcloud_render_prog.SetUniform("mvp", mvp);

        glDrawArrays(GL_POINTS, 0, world_points.cols());
        // Also, if we're showing the octomap as well, render it to its viewport
        if (!mrfmap_only_mode_) {
          view_octo.Activate();
          glDrawArrays(GL_POINTS, 0, world_points.cols());
          view_map.Activate();
        }

        glDisable(GL_POINT_SPRITE);
        glDisable(GL_PROGRAM_POINT_SIZE);

        // Reset
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        pointcloud_render_prog.Unbind();
      }

      if (menuRenderMap) {
        // First set the current scene camera to be at this pose
        Camera3D* cam = new Camera3D;
        cam->mNear = 0.1f;
        cam->mFar = Z_MAX / gvdb_params_.res;
        Eigen::Matrix4f pose_in_world = get_opengl_cam_pose_in_world(*cam_, latest_pose_temp, menuFollowCamera);
        TMat view_mat = TMat::Zero(4, 4), proj_mat = TMat::Zero(4, 4);
        convert_pose_to_opengl_proj_mat(pose_in_world, view_mat, proj_mat);
        cam->setMatrices(view_mat.data(), proj_mat.data(), Vector3DF(0.f, 0.f, 0.f));
        cam->updateFrustum();
        RMatXU8 world_img = RMatXU8::Zero(gvdb_params_.rows, 4 * gvdb_params_.cols);
        bool updated = false;
        {
          std::unique_lock<std::mutex> lock(mrfmap_mutex_, std::defer_lock);
          if (lock.try_lock()) {
            inf_->map_->push_ctx();
            inf_->map_->gvdb_.getScene()->mCamera = cam;
            // Set the dbuf to 255 to disable using the depth buffer
            int prev_buf = inf_->map_->gvdb_.getScene()->getDepthBuf();
            inf_->map_->gvdb_.getScene()->SetDepthBuf(255);
            inf_->map_->gvdb_.SetModule();
            cudaCheck(cuCtxSynchronize(), "PangolinViewer", "menuRenderMap", "cuCtxSynchronize0", "", false);
            inf_->gvdb_->Render(SHADE_VOLUME, ALPHAS, 0);
            cudaCheck(cuCtxSynchronize(), "PangolinViewer", "menuRenderMap", "cuCtxSynchronize1", "", false);
            inf_->gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(world_img.data()));
            cudaCheck(cuCtxSynchronize(), "PangolinViewer", "menuRenderMap", "cuCtxSynchronize2", "", false);
            inf_->map_->gvdb_.getScene()->SetDepthBuf(prev_buf);
            inf_->map_->gvdb_.SetModule(inf_->map_->gvdb_module_);
            inf_->map_->pop_ctx();
            updated = true;
          } else {
            DEBUG(std::cout << "MRFMap busy, continuing...\n");
          }
        }
        if (updated) {
          renderTexture.Upload(reinterpret_cast<unsigned char*>(world_img.data()), GL_RGBA, GL_UNSIGNED_BYTE);
          updated = false;
        }
        // Also, if we're showing the octomap as well, render it to its viewport
        if (!mrfmap_only_mode_) {
          std::unique_lock<std::mutex> lock(octomap_mutex_, std::defer_lock);
          if (lock.try_lock()) {
            octo_wrapper_->octo_->gvdb_->getScene()->mCamera = cam;
            // Set the dbuf to 255 to disable using the depth buffer
            octo_wrapper_->octo_->gvdb_->getScene()->SetDepthBuf(255);
            octo_wrapper_->octo_->gvdb_->SetModule();
            octo_wrapper_->octo_->gvdb_->Render(SHADE_VOLUME, OCCUPANCY_CHANNEL, 0);
            cudaCheck(cuCtxSynchronize(), "PangolinViewer", "menuRenderMap", "cuCtxSynchronize", "", false);
            octo_wrapper_->octo_->gvdb_->ReadRenderBuf(0, reinterpret_cast<uchar*>(world_img.data()));
            cudaCheck(cuCtxSynchronize(), "PangolinViewer", "menuRenderMap", "cuCtxSynchronize", "", false);
            octo_wrapper_->octo_->gvdb_->SetModule(octo_wrapper_->octo_->gvdb_module_);
            updated = true;
          } else {
            DEBUG(std::cout << "Octomap busy, continuing...\n");
          }

          if (updated) {
            octoTexture.Upload(reinterpret_cast<unsigned char*>(world_img.data()), GL_RGBA, GL_UNSIGNED_BYTE);
          }
        }
        delete cam;
      }

      if (menuMapWireframe) {
        std::unique_lock<std::mutex> lock(mrfmap_mutex_, std::defer_lock);

        view_map.Activate(*cam_);
        // Obtain brick coordinates
        Vector3DF clrs[10];
        clrs[0] = Vector3DF(0.0f, 0.0f, 1.0f);  // blue
        clrs[1] = Vector3DF(0.0f, 1.0f, 0.0f);  // green
        clrs[2] = Vector3DF(1.0f, 0.0f, 0.0f);  // red
        clrs[3] = Vector3DF(1.0f, 1.0f, 0.0f);  // yellow
        clrs[4] = Vector3DF(1.0f, 0.0f, 1.0f);  // purple
        clrs[5] = Vector3DF(0.0f, 1.0f, 1.0f);  // aqua
        clrs[6] = Vector3DF(1.0f, 0.5, 0.0f);   // orange
        clrs[7] = Vector3DF(0.0f, 0.5, 1.0f);   // green-blue
        clrs[8] = Vector3DF(0.7f, 0.7f, 0.7f);  // grey
        Node* node;
        Vector3DF bmin, bmax;
        int lev = menuLevel;
        // for (int lev = 0; lev < 5; lev++) {  // draw all levels

        int node_cnt = inf_->map_->gvdb_.getNumNodes(std::max(0, lev));

        Eigen::Matrix4f opengl_in_cam, cam_in_opengl;
        opengl_in_cam << 1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 1;
        cam_in_opengl = opengl_in_cam.inverse();
        if (lock.try_lock()) {
          Eigen::MatrixXf centers = inf_->map_->get_occupied_bricks(std::max(0, lev));
          // Offset to center of volume due to requirements of gvdb to only have positive coordinates
          Eigen::Vector3f scaled_map_center_coords = gvdb_params_.dims * 0.5f / gvdb_params_.res;
          //Don't shift z coordinate
          scaled_map_center_coords[2] = 0.0f;

          // Try out pulling data about the individual occupied voxels
          //   inf_->map_->gvdb_.SetModule(inf_->map_->gvdb_module_);
          if (lev == -1) {
            RMatXi voxels_mask(centers.rows(), inf_->gvdb_->getVoxCnt(0) / (8 * sizeof(int)));
            inf_->map_->get_occupied_voxels_mask(voxels_mask, menuThresh);

            // Extract only voxels that have been classified
            std::vector<float> occupied_voxels;
            int r = inf_->gvdb_->getRes(0);
            for (size_t i = 0; i < voxels_mask.rows(); ++i) {
              for (size_t j = 0; j < voxels_mask.cols(); ++j) {
                std::bitset<8 * sizeof(int)> bits(voxels_mask(i, j));
                for (size_t k = 0; k < 8 * sizeof(int); ++k) {
                  if (bits[k] == 1) {
                    // Get the index in brick space
                    int index = j * (8 * sizeof(int)) + k;
                    // Get coordinates
                    int x = index / (r * r);
                    int y = index / r - x * r;
                    int z = index % r;

                    Eigen::Vector3f delta =
                        {static_cast<float>(x - (r - 1) / 2),
                         static_cast<float>(y - (r - 1) / 2),
                         static_cast<float>(z - (r - 1) / 2)};
                    Eigen::Vector3f mid_pt = centers.row(i);
                    Eigen::Vector3f vox_center = mid_pt + delta;
                    vox_center -= scaled_map_center_coords;
                    vox_center *= gvdb_params_.res;
                    occupied_voxels.push_back(vox_center[0]);
                    occupied_voxels.push_back(vox_center[1]);
                    occupied_voxels.push_back(vox_center[2]);
                  }
                }
              }
            }

            glBindBuffer(GL_ARRAY_BUFFER, wireframe_VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * occupied_voxels.size(), occupied_voxels.data(), GL_STATIC_DRAW);
            num_vertices = occupied_voxels.size() / 3;
            glBindBuffer(GL_ARRAY_BUFFER, 0);
          } else {
            RMatXf test(centers.rows(), centers.cols());
            for (int i = 0; i < centers.rows(); ++i) {
              Eigen::Vector3f mid_pt = centers.row(i);
              mid_pt -= scaled_map_center_coords;
              mid_pt *= gvdb_params_.res;
              test.row(i) = mid_pt;
            }
            Eigen::MatrixXf brick_centers = test.transpose();
            // Just send out the brick centers
            glBindBuffer(GL_ARRAY_BUFFER, wireframe_VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * brick_centers.rows() * brick_centers.cols(), brick_centers.data(), GL_STATIC_DRAW);
            num_vertices = brick_centers.cols();
            glBindBuffer(GL_ARRAY_BUFFER, 0);
          }
        }  // try lock end

        voxel_render_prog.SaveBind();

        pangolin::OpenGlMatrix mvp = cam_->GetProjectionModelViewMatrix();
        voxel_render_prog.SetUniform("mvp", mvp);
        voxel_render_prog.SetUniform("vColor", clrs[lev + 1].x, clrs[lev + 1].y, clrs[lev + 1].z);
        if (lev == -1) {
          voxel_render_prog.SetUniform("voxSize", gvdb_params_.res, gvdb_params_.res, gvdb_params_.res);
        } else {
          float brick_size = gvdb_params_.res;
          for (int l = 0; l <= lev; ++l)
            brick_size *= inf_->map_->gvdb_.getRes(l);
          voxel_render_prog.SetUniform("voxSize", brick_size, brick_size, brick_size);
        }

        glBindVertexArray(wireframe_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, wireframe_VBO);
        glDrawArrays(GL_POINTS, 0, num_vertices);
        // Reset
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        voxel_render_prog.Unbind();
      }

      {
        // Also render bounds of the volume
        voxel_render_prog.SaveBind();
        pangolin::OpenGlMatrix mvp = cam_->GetProjectionModelViewMatrix();
        voxel_render_prog.SetUniform("mvp", mvp);
        voxel_render_prog.SetUniform("voxSize", gvdb_params_.dims[0], gvdb_params_.dims[1], gvdb_params_.dims[2]);

        glBindBuffer(GL_ARRAY_BUFFER, wireframe_VBO);
        // Offset to center of volume due to requirements of gvdb to only have positive coordinates
        Eigen::Vector3f scaled_map_center_coords = gvdb_params_.dims * 0.5f / gvdb_params_.res;
        //Don't shift z coordinate
        scaled_map_center_coords[2] = 0.0f;
        Eigen::Vector3f map_center = {0.0f, 0.0f, gvdb_params_.dims[2] / 2.0f};
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 3, map_center.data(), GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glBindVertexArray(wireframe_VAO);
        glBindBuffer(GL_ARRAY_BUFFER, wireframe_VBO);
        glDrawArrays(GL_POINTS, 0, 1);
        // Also, if we're showing the octomap as well, render it to its viewport
        if (!mrfmap_only_mode_) {
          view_octo.Activate();
          glDrawArrays(GL_POINTS, 0, 1);
          view_map.Activate();
        }
        // Reset
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
        voxel_render_prog.Unbind();
      }
      // Display current depth image
      view_depth.Activate(cam_img);
      std::string view_depth_text = "Depth Image";
      //Upload image data to GPU
      RMatXf grayscale = latest_img_temp;
      grayscale = (grayscale.array().isFinite()).select(grayscale, 0.0f);
      grayscale /= Z_MAX;  // grayscale.maxCoeff();
      grayscale = (grayscale.array() <= 1.0f).select(1.0f - grayscale.array(), 1.0f) * 255;
      cv::Mat normalized_depth_img(gvdb_params_.rows, gvdb_params_.cols, CV_32FC1, grayscale.data());
      cv::Mat normalized_8bit, lut_image;
      normalized_depth_img.convertTo(normalized_8bit, CV_8UC1);
      cv::applyColorMap(normalized_8bit, lut_image, cv::COLORMAP_JET);
      colorized_depth_tex.Upload(lut_image.data, GL_BGR, GL_UNSIGNED_BYTE);
      glDepthMask(GL_FALSE);  // Because otherwise the text is covered over...
      colorized_depth_tex.RenderToViewportFlipY();

      glPushAttrib(GL_CURRENT_BIT);
      glColor3f(0.0, 0.0, 1.0);
      pangolin::GlFont::I().Text(view_depth_text.c_str()).DrawWindow(view_depth.v.r() - 0.8f * view_depth_text.length() * pangolin::GlFont::I().Height(), view_depth.v.t() - 1.1 * pangolin::GlFont::I().Height());
      glDepthMask(GL_TRUE);  // Reset to default use of depth mask
      glPopAttrib();

      // Now render map view
      view_map.Activate(*cam_);
      // pangolin::glDrawColouredCube();
      pangolin::glDrawAxis(1.0f);

      glPushAttrib(GL_CURRENT_BIT);
      glColor3f(0.0f, 0.0f, 1.0f);
      pangolin::glDrawFrustum(K_inv, gvdb_params_.cols, gvdb_params_.rows, latest_pose_temp, 0.3f);
      if (menuRenderPoses) {
        // Also draw the frustums of the keyframes
        glColor3f(0.5f, 0.5f, 0.5f);
        for (int i = 0; i < num_cams; ++i) {
          pangolin::glDrawFrustum(K_inv, gvdb_params_.cols, gvdb_params_.rows, Eigen::Matrix4f(kf_poses_[i]), 0.2f);
        }
      }
      glPopAttrib();
      glDepthMask(GL_FALSE);
      if (menuRenderMap)
        renderTexture.RenderToViewportFlipY();
      glPushAttrib(GL_CURRENT_BIT);
      glColor3f(1.0f, 0.0f, 0.0f);
      std::string view_map_text = "MRFMap";
      pangolin::GlFont::I().Text(view_map_text.c_str()).DrawWindow(view_map.v.r() - 1.0f * view_map_text.length() * pangolin::GlFont::I().Height(), view_map.v.t() - 1.1 * pangolin::GlFont::I().Height());
      glPopAttrib();

      // Switch to octomap view
      if (!mrfmap_only_mode_) {
        view_octo.Activate(*cam_);
        // And render frustums here too!
        glPushAttrib(GL_CURRENT_BIT);
        glColor3f(0.0f, 0.0f, 1.0f);
        pangolin::glDrawFrustum(K_inv, gvdb_params_.cols, gvdb_params_.rows, latest_pose_temp, 0.3f);
        if (menuRenderPoses) {
          // Also draw the frustums of the keyframes
          glColor3f(0.5f, 0.5f, 0.5f);
          for (int i = 0; i < num_cams; ++i) {
            pangolin::glDrawFrustum(K_inv, gvdb_params_.cols, gvdb_params_.rows, Eigen::Matrix4f(kf_poses_[i]), 0.2f);
          }
        }
        glPopAttrib();

        glDepthMask(GL_FALSE);
        octoTexture.RenderToViewportFlipY();
        glPushAttrib(GL_CURRENT_BIT);
        glColor3f(1.0f, 0.0f, 0.0f);
        std::string view_octo_text = "Octomap";
        pangolin::GlFont::I().Text(view_octo_text.c_str()).DrawWindow(view_octo.v.r() - 1.0f * view_octo_text.length() * pangolin::GlFont::I().Height(), view_octo.v.t() - 1.1 * pangolin::GlFont::I().Height());
        glPopAttrib();
      }
      glDepthMask(GL_TRUE);
    }
    // Swap frames and Process Events
    pangolin::FinishFrame();

    if (pangolin::Pushed(menuReset)) {
      if (!view_only_mode_) {
        // Reset both the inference objects
        {  // Stop the inference threads
          should_quit_ = true;
          mrfmap_cv_.notify_one();
          octomap_cv_.notify_one();
          // Wait for them to join
          mrfmap_thread_->join();
          octomap_thread_->join();

          std::lock_guard<std::timed_mutex> dataguard(data_mutex_);
          std::lock_guard<std::mutex> mrfguard(mrfmap_mutex_);
          std::lock_guard<std::mutex> octoguard(octomap_mutex_);

          should_quit_ = false;
          // Clear all the data we had stored
          kf_images_.clear();
          kf_poses_.clear();
          gvdb_images_.clear();
          new_mrf_data_ = false;
          new_octo_data_ = false;
          init_ = false;

          // Reset the inference pointers
          inf_ = std::make_shared<GVDBInference>(true, false);
          if (!mrfmap_only_mode_) {
            octo_wrapper_ = std::make_shared<GVDBOctomapWrapper>(gvdb_params_);
          }

          // Spawn them off again
          mrfmap_thread_ = std::make_shared<std::thread>(&PangolinViewer::mrfmap_worker_runner, this);
          octomap_thread_ = std::make_shared<std::thread>(&PangolinViewer::octomap_worker_runner, this);
        }
      }
    }
  }
  // Delete all the VAOs and VBOs
  glDeleteBuffers(1, &cloud_VBO);
  glDeleteBuffers(1, &wireframe_VBO);
  glDeleteVertexArrays(1, &cloud_VAO);
  glDeleteVertexArrays(1, &wireframe_VAO);

  // unset the current context from the main thread
  pangolin::GetBoundWindow()->RemoveCurrent();
  if (!view_only_mode_) {
    // Signal other threads to quit!
    should_quit_ = true;
    mrfmap_cv_.notify_one();
    if (!mrfmap_only_mode_) {
      octomap_cv_.notify_one();
    }
  }
  pangolin::QuitAll();
  if (exit_callback_) {
    signal_gui_end();
  }
  std::cout << "Quitting pango thread!!\n";
}

void PangolinViewer::add_keyframe(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& image) {
  std::lock_guard<std::timed_mutex> guard(data_mutex_);
  latest_pose_ = pose;
  if (!view_only_mode_) {
    // Corrupt the incoming image with gaussian multiplicative noise
    int r = image.rows(), c = image.cols();
    latest_img_ = image.array() *
                  ((Eigen::MatrixXf::Random(r, c) + Eigen::MatrixXf::Constant(r, c, 1.0f)) * 0.5f * injected_noise_ + Eigen::MatrixXf::Constant(r, c, 1.0f)).array();
  } else {
    latest_img_ = image;
  }
  kf_images_.push_back(latest_img_);
  kf_poses_.push_back(pose);
  if (!view_only_mode_) {
    // Create a GVDBImage
    std::shared_ptr<GVDBImage> image_ptr = std::make_shared<GVDBImage>(latest_img_);
    gvdb_images_.push_back(image_ptr);
    if (gvdb_images_.back()->get_reproj_points().size() == 0) {
      std::cout << "YIKES!!!!\n";
    }
    DEBUG(std::cout << "Added a new keyframe!\n");
    signal_new_data();
  }
}

void PangolinViewer::add_frame(const Eigen::MatrixXf& pose, const Eigen::MatrixXf& image) {
  std::lock_guard<std::timed_mutex> guard(data_mutex_);
  latest_pose_ = pose;
  if (!view_only_mode_) {
    // Corrupt the incoming image with gaussian multiplicative noise
    int r = image.rows(), c = image.cols();
    latest_img_ = image.array() *
                  ((Eigen::MatrixXf::Random(r, c) + Eigen::MatrixXf::Constant(r, c, 1.0f)) * 0.5f * injected_noise_ + Eigen::MatrixXf::Constant(r, c, 1.0f)).array();
    signal_new_data();
  } else {
    latest_img_ = image;
  }
}

void PangolinViewer::set_kf_pose(uint index, const Eigen::MatrixXf& pose) {
  {
    std::unique_lock<std::mutex> lock(mrfmap_mutex_);
    inf_->set_pose(index, pose);
    inf_->perform_inference();
  }
  {
    std::lock_guard<std::timed_mutex> guard(data_mutex_);
    kf_poses_[index] = pose;
  }
}

void PangolinViewer::signal_new_data() {
  // Set the worker threads to keep spinning on new data until no
  // gvdbimage is in queue
  {
    std::unique_lock<std::mutex> lock(mrfmap_mutex_, std::defer_lock);
    if (lock.try_lock()) {
      if (inf_->cameras_.size() < gvdb_images_.size()) {
        new_mrf_data_ = true;
        mrfmap_cv_.notify_one();
      }
    }
  }

  if (!mrfmap_only_mode_) {
    std::unique_lock<std::mutex> lock(octomap_mutex_, std::defer_lock);
    if (lock.try_lock()) {
      if (octo_wrapper_->get_num_cams() < gvdb_images_.size()) {
        new_octo_data_ = true;
        octomap_cv_.notify_one();
      }
    }
  }
}