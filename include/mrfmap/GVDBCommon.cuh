#pragma once

#define Z_MAX 10.0f
#define Z_MIN 0.1f
#define F_EPS 1e-4f
#define LOGODDS_F_EPS -9.2102403669758495f

#define LIKE_EPS 1e-3f

#define LOOKUP_NUM 3096 //Max 27*48*2 = 2592 floats for kinect, 24*32*3 = 2304 for realsense640, 24*43*3=3096 for realsense848
#define LOOKUP_PX 20

#define LEGACY 1

#define GVDB_SKIP_DEPTH_BUFFER_TEST
#define GVDB_USE_DEPTH_IMAGE_BUFFER

// Some ghetto logging because I can't cleanly pull out streams from within device kernels
#define ENABLE_LOG
// #define ENABLE_DEBUG
// #define SINGLE_RAY

// Enable this to get per-image channels for proper BP
#define FULL_INFERENCE 1

#ifdef ENABLE_LOG
#define LOG(x) x
#else
#define LOG(x) (void)0
#endif
#ifdef ENABLE_DEBUG
#define DEBUG(x) x
#else
#define DEBUG(x) (void)0
#endif

enum CHANNELS {
  POS_LOG_MSG_SUM,
  CUM_LENS,
#ifndef FULL_INFERENCE
  WEIGHTED_INCOMING,
#endif
  ALPHAS
};