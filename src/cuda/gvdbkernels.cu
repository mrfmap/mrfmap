
#include <cuda_fp16.h>
#include <stdio.h>
#include "cuda_math.cuh"
typedef unsigned char uchar;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned long ulong;
typedef unsigned long long uint64;

#include <mrfmap/GVDBCommon.cuh>

//-------------------------------- GVDB Data Structure
#define CUDA_PATHWAY

#include "cuda_gvdb_scene.cuh"  // GVDB Scene

#include "cuda_gvdb_nodes.cuh"  // GVDB Node structure

#include "cuda_gvdb_geom.cuh"  // GVDB Geom helpers

#include "cuda_gvdb_dda.cuh"  // GVDB DDA

#include "cuda_gvdb_raycast.cuh"  // GVDB Raycasting

#include "cuda_gvdb_operators.cuh"
//--------------------------------
//-------------------------------- GVDB Data Structure

// Some ugly global device members because I don't want to modify the gvdb
// library API Maybe implement a helper cuda file later with host wrappers to
// set these?
__constant__ float g_res_;
__constant__ float g_occ_thresh_;
__constant__ float g_acc_thresh_;
__constant__ float g_prob_prior_;
__constant__ float g_logodds_prob_prior_;

__device__ int g_selected_x_, g_selected_y_;
__device__ uint64 g_selected_vox_id_;

__constant__ int g_lookup_u_, g_lookup_v_, g_lookup_n_;
__constant__ float g_depth_bias_lookup_[LOOKUP_NUM];
__constant__ float g_depth_sigma_lookup_[LOOKUP_NUM];

inline __device__ float get_depth_sigma(float depth, int u, int v) {
  float* poly = g_depth_sigma_lookup_ + (v / LOOKUP_PX) * (g_lookup_u_ * g_lookup_n_) + (u / LOOKUP_PX) * g_lookup_n_;
  return (g_lookup_n_ == 3 ? depth * depth * poly[2] + depth * poly[1] + poly[0]
                           : depth * poly[1] + poly[0]);
}
inline __device__ float get_depth_bias(float depth, int u, int v) {
  float* poly = g_depth_bias_lookup_ + (v / LOOKUP_PX) * (g_lookup_u_ * g_lookup_n_) + (u / LOOKUP_PX) * g_lookup_n_;
  return (g_lookup_n_ == 3 ? depth * depth * poly[2] + depth * poly[1] + poly[0]
                           : depth * poly[1] + poly[0]);
}

inline __device__ float depth_potential(const float& query,
                                        const float& measurement,
                                        const int x, const int y) {
  // The purpose of this potential is this:
  // If matter exists at this ground truth depth, how well does the measurement
  // correspond with it? To get this, we first utilise the sensor bias to
  // determine what the mean sensor measurement will be at this location,
  // and then evaluate the likelihood using the predicted noise value.
  float predicted_bias = get_depth_bias(query, x, y);
  float diff = fabsf(measurement - (query + predicted_bias));
  float sigma = 3.0f*get_depth_sigma(query, x, y);
  // if (diff > 3.0f*sqrtf(*g_sigma_sq_))
  //   return 0.0f;
  float inv_sigma = __frcp_rz(sigma);
  return 0.398942f * __expf(-0.5f * powf(diff * inv_sigma, 2)) * inv_sigma;
}

// Need to make extern "C" so that we can lookup symbol from generated PTX
extern "C" __global__ void gvdbOpDistToSphere(VDBInfo* gvdb, int3 res,
                                              uchar chan) {
  GVDB_VOX
  float3 wpos;
  if (!getAtlasToWorld(gvdb, vox, wpos)) return;
  float v = sqrtf(wpos.x * wpos.x + wpos.y * wpos.y + wpos.z * wpos.z) / 100.0f;
  // DEBUG(printf("\nWorld x::%f y::%f z::%f", wpos.x, wpos.y, wpos.z);)
  // distance to sphere
  surf3Dwrite(v, gvdb->volOut[chan], vox.x * sizeof(float), vox.y, vox.z);
}

extern "C" __global__ void shade_by_z(VDBInfo* gvdb, int3 res, uchar chan) {
  GVDB_VOX
  int3 atlas_res = gvdb->atlas_res;
  unsigned long int atlas_id = vox.z * atlas_res.x * atlas_res.y +
                               vox.y * atlas_res.x + vox.x;

  float3 wpos;
  if (!getAtlasToWorld(gvdb, vox, wpos)) return;

  float v = 1.0f;
  if (chan == 1) {
    float* atlas_mem = (float*)(gvdb->atlas_dev_mem[1]) + atlas_id;
    *atlas_mem = v;

  } else {
    float* atlas_mem = (float*)(gvdb->atlas_dev_mem[1]) + atlas_id;
    surf3Dwrite(*atlas_mem, gvdb->volOut[chan], vox.x * sizeof(float), vox.y,
                vox.z);
  }
}

__device__ void rayLengthTraversedBrick(VDBInfo* gvdb, uchar chan, int nodeid,
                                        float3 t, float3 pos, float3 dir,
                                        float3& pStep, float3& hit,
                                        float3& norm, float4& clr) {
  float3 vmin;
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  float3 p;
  t.x += gvdb->epsilon;  // make sure we start inside
  p = (pos + t.x * dir - vmin) / gvdb->vdel[0];
  hit = p * gvdb->vdel[0] + vmin;  // Return brick hit

  // Determine the length traversed within this voxel
  float res = static_cast<float>(gvdb->res[0]);
  float3 t3 = rayBoxIntersect(p, dir, make_float3(0.f, 0.f, 0.f),
                              make_float3(res, res, res));
  if (t3.z == NOHIT) {
    hit.z = NOHIT;
  } else {
    float length = t3.x;
    if (length == 0.0f)  // This happens if vmin is actually further away along
                         // ray direction
      length = t3.y;
    // First coordinate of length has the corresponding t value till the
    // bounding box Set clr.w to perform ray termination
    clr = make_float4(fmin(fabs(length / (1.73f * res)), 1.0f), 0.0, 0.0, 0.0);
  }
}

__device__ void raySimpleAccuracyBrick(VDBInfo* gvdb, uchar chan, int nodeid,
                                       float3 t, float3 pos, float3 dir,
                                       float3& pStep, float3& hit,
                                       float3& norm, float4& clr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 vmin;
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  t.x += gvdb->epsilon;                             // make sure we start inside
  float multiplier = g_res_ * fabsf(dot(scn.dir_vec, dir));
  float distance = t.x * multiplier;
  float depth = getLinearDepth(SCN_DBUF);
  float3 o = make_float3(node->mValue);  // gives us the atlas data coordinates
  /// Initialize brick coordinates, DDA and ray traversal
  float3 p, tDel, tSide, mask;  // 3DDA variables
  PREPARE_DDA_LEAF
  // Iterate till the end of the brick, or max iters, whichever is first
  float prev_tx = 0.0f;
  int iter = 0;
  for (; iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 &&
         p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0];
       iter++) {
    NEXT_DDA;  // Update the value of t, and store it in t.y
    // Access the alpha value at this voxel
    uint3 vox = {static_cast<uint>(p.x + o.x),
                 static_cast<uint>(p.y + o.y),
                 static_cast<uint>(p.z + o.z)};
    float length = t.y - prev_tx;  // Updated t minus prev_t
    float p_occ =
        tex3D<float>(gvdb->volIn[chan], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
    distance += length * multiplier;
    // Threshold test: If current alpha is greater than threshold param, bail.
    if (p_occ >= g_acc_thresh_) {
      // Is the current ray distance within 1 voxel distance of true depth?
      clr.x = fabsf(distance - depth) <= g_res_? 1 : 2; // Sends out true positive or false positive
      clr.w = -1;
      return;
    }
    prev_tx = t.y;
    STEP_DDA
  }
  // Bail test: If we're still raytracing, and we just crossed the depth, bail.
  if (distance >= depth) {
    clr.x = 0;  // No measurement was made, and we've gone beyond the measured depth (false negative)
    clr.w = -1;
  }
}

__device__ void rayLikelihoodBrick(VDBInfo* gvdb, uchar chan, int nodeid,
                                   float3 t, float3 pos, float3 dir,
                                   float3& pStep, float3& hit,
                                   float3& norm, float4& clr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 vmin;
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  t.x += gvdb->epsilon;                             // make sure we start inside
  float multiplier = g_res_ * fabsf(dot(scn.dir_vec, dir));
  float s = t.x * multiplier;
  float3 o = make_float3(node->mValue);  // gives us the atlas data coordinates
  /// Initialize brick coordinates, DDA and ray traversal
  float3 p, tDel, tSide, mask;  // 3DDA variables
  PREPARE_DDA_LEAF
  // Iterate till the end of the brick, or max iters, whichever is first
  float prev_tx = 0.0f, prev_s = s;
  float sum = 0.0f, cum_alphas = 0.0f;
  float prob_z_given_b = 0.0f, prev_vis = 1.0f;  //, b_i = 1.0f;

  if (clr.w != 1.0f) {
    // cumprod = clr.x;
    prev_vis = clr.x;
    // cumsum = clr.y;
    cum_alphas = clr.y;
    sum = clr.z;
    prev_s = norm.x;
  }

  int iter = 0;
  float depth = getLinearDepth(SCN_DBUF);
  float predicted_depth_minus_measured = s - depth;
  for (; iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 &&
         p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0];
       iter++) {
    NEXT_DDA;  // Update the value of t, and store it in t.y
    // Access the alpha value at this voxel
    uint3 vox = {static_cast<uint>(p.x + o.x),
                 static_cast<uint>(p.y + o.y),
                 static_cast<uint>(p.z + o.z)};
    float length = t.y - prev_tx;  // Updated t minus prev_t
    float p_occ =
        tex3D<float>(gvdb->volIn[chan], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);

    // b_i = alpha * cumprod;
    // Also compute the visibility terms
    // Eq. 3.17 of Crispell thesis, $\alpha = - \frac{ln(1 - p(Q^{\hat{l}}))}{\hat{l}} $ where $\hat{l}$ is a canonical length segment
    float alpha_i = -1.0f / (1.0f) * logf(1.0f - p_occ);
    cum_alphas += -alpha_i * length;

    float vis_i = expf(cum_alphas);
    float w_i = (prev_vis - vis_i) / length;
    
    s += length * multiplier;
    // float3 wpos;
    // getAtlasToWorld(gvdb, vox, wpos);
    // float3 ray_to_cam_center = wpos - pos;
    // // float vox_s = fabsf(dot(scn.dir_vec, ray_to_cam_center)) * g_res_;
    predicted_depth_minus_measured = s + get_depth_bias(s, x, y) - depth;

    if (s < Z_MIN) {
      prob_z_given_b = 0.0f;
    } else if (fabsf(predicted_depth_minus_measured) <= 3.0f * get_depth_sigma(s, x, y)) {
      float ray_mu_d_i = depth_potential(s, depth, x, y);
      prob_z_given_b = (1.0f - LIKE_EPS) * 0.5f * (ray_mu_d_i + depth_potential(prev_s, depth, x, y));
    } else {
      prob_z_given_b = LIKE_EPS;  //epsilon/(*num_voxels)
    }
    // For the accuracy kernel
    if (w_i > hit.x) {
      hit.x = w_i;  // Set the new peak to be the current w_i
      // Is the peak within 1 sigma of measured depth?
      hit.y = (fabsf(predicted_depth_minus_measured) <= get_depth_sigma(s, x, y))? 1 : 0;
      // Is the measured depth between peak s_i and s_{i-1}?
      // hit.y = (predicted_depth_minus_measured >= 0 && prev_s + get_depth_bias(prev_s, x, y) < depth) ? 1 : 0;
#ifdef ENABLE_DEBUG
      if (x == g_selected_x_ && y == g_selected_y_) {
        DEBUG(printf("> %f ", hit.y));
      }
#endif
    }
    // For the occ threshold kernel
    if (predicted_depth_minus_measured > 0 && norm.y == -1) {
      norm.y = p_occ >= g_occ_thresh_;
    }
    // cumprod *= 1.0f - alpha;
    // We want to normalize the p_z_given_bs by the area they cover, so accum
    // cumsum += prob_z_given_b * length * multiplier;
    // sum += prob_z_given_b * b_i;

    sum += (prev_vis - vis_i) * prob_z_given_b;

  #ifdef ENABLE_DEBUG
    if (x == g_selected_x_ && y == g_selected_y_) {
      DEBUG(printf("s::%f depth::%f pocc::%f alpha::%f w_i::%f vis_i::%f bias::%f sigma::%f pdmm::%f prev::%f\n", s, depth, p_occ, alpha_i, w_i, vis_i, get_depth_bias(s, x, y), get_depth_sigma(s, x, y), predicted_depth_minus_measured, prev_s + get_depth_bias(prev_s, x, y) );)
    }
  #endif
    prev_vis = vis_i;
    prev_tx = t.y;
    prev_s = s;
    STEP_DDA
  }

  // clr.x = cumprod;
  clr.x = prev_vis;
  // clr.y = cumsum;
  clr.y = cum_alphas;
  clr.z = sum;
  clr.w = 0.5f;
  norm.x = predicted_depth_minus_measured + depth;
}

__device__ void rayExpectedDepthBrick(VDBInfo* gvdb, uchar chan, int nodeid,
                                      float3 t, float3 pos, float3 dir,
                                      float3& pStep, float3& hit,
                                      float3& norm, float4& clr) {
  float3 vmin;
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  t.x += gvdb->epsilon;                             // make sure we start inside
  float multiplier = g_res_ * fabsf(dot(scn.dir_vec, dir));
  float distance = t.x * multiplier;
  float3 o = make_float3(node->mValue);  // gives us the atlas data coordinates
  /// Initialize brick coordinates, DDA and ray traversal
  float3 p, tDel, tSide, mask;  // 3DDA variables
  PREPARE_DDA_LEAF
  // Iterate till the end of the brick, or max iters, whichever is first
  float prev_tx = 0.0f;
  float prev_vis = 1.0f, sum = 0.0f, cum_alphas = 0.0f;

  if (clr.w != 1.0f) {
    prev_vis = clr.x;
    cum_alphas = clr.y;
    sum = clr.z;
  }

  int iter = 0;
  for (; iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 &&
         p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0];
       iter++) {
    NEXT_DDA;  // Update the value of t, and store it in t.y

    // Access the alpha value at this voxel
    uint3 vox = {static_cast<uint>(p.x + o.x),
                 static_cast<uint>(p.y + o.y),
                 static_cast<uint>(p.z + o.z)};
    float length = t.y - prev_tx;  // Updated t minus prev_t
    float p_occ =
        tex3D<float>(gvdb->volIn[ALPHAS], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);

    distance += length * multiplier;
    float3 wpos;
    getAtlasToWorld(gvdb, vox, wpos);
    float3 ray_to_cam_center = wpos - pos;
    float vox_distance = fabsf(dot(scn.dir_vec, ray_to_cam_center)) * g_res_;

    float alpha_i = -1.0f / (1.0f) * logf(1.0f - p_occ);
    cum_alphas += -alpha_i * length;
    float vis_i = expf(cum_alphas);
    // float w_i = (prev_vis - vis_i) / length;
    // sum += (prev_vis - vis_i) * ( 1.0f + 1.0f/alpha_i) * multiplier;
    sum += (prev_vis - vis_i) * vox_distance;

    prev_vis = vis_i;
    prev_tx = t.y;
    STEP_DDA
  }

  clr.x = prev_vis;
  clr.y = cum_alphas;
  clr.z = sum;
  clr.w = 0.5f;
}

inline __host__ __device__ float3 reflect3(float3 i, float3 n) {
  return i - 2.0f * n * dot(n, i);
}

extern "C" __global__ void get_channel_kernel(VDBInfo* gvdb, uint3 offs, int3 res, float* buf, int chan) {
  uint3 t = threadIdx;
  float val = -1.0f;
  uint3 vox = make_uint3(t.x + offs.x, t.y + offs.y, t.z + offs.z);
  if (gvdb->use_tex_mem[chan]) {
    val =
        tex3D<float>(gvdb->volIn[chan], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
  } else {
    uint64 atlas_id = vox.z * res.x * res.y +
                      vox.y * res.x + vox.x;
    val = *((float*)(gvdb->atlas_dev_mem[chan]) + atlas_id);
  }
  buf[(t.x * res.y + t.y) * res.x + t.z] = val;
}

extern "C" __global__ void get_occupied_voxel_mask_kernel(VDBInfo* gvdb, uint3 offs, int3 res, int* mask, int chan, float threshold) {
  uint3 t = threadIdx;
  float val = -1.0f;
  int index = (t.x * res.x + t.y) * res.y + t.z;
  uint3 vox = make_uint3(t.x + offs.x, t.y + offs.y, t.z + offs.z);
  if (gvdb->use_tex_mem[chan]) {
    val =
        tex3D<float>(gvdb->volIn[chan], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
  } else {
    uint64 atlas_id = vox.z * res.x * res.y +
                      vox.y * res.x + vox.x;
    val = *((float*)(gvdb->atlas_dev_mem[chan]) + atlas_id);
  }
  // Set mask bit to 1 if val > thresh
  if (val >= threshold) {
    // Get a word (32 bits) and then OR the corresponding bit in place
    atomicOr(&mask[index / (8 * sizeof(int))], 1 << (index % (8 * sizeof(int))));
  }
}

// Computes messages from each voxel node to all the factors that go through it
extern "C" __global__ void compute_gvdb_message_to_factors(VDBInfo* gvdb,
                                                           int3 res,
                                                           uchar dry_run) {
  GVDB_VOX
  int3 atlas_res = gvdb->atlas_res;
  unsigned long int atlas_id = vox.z * atlas_res.x * atlas_res.y +
                               vox.y * atlas_res.x + vox.x;
  float* cum_len = (float*)(gvdb->atlas_dev_mem[CUM_LENS]) + atlas_id;
  // We send the channel id in the dry_run variable
  half2* weighted_incoming =
      reinterpret_cast<half2*>(gvdb->atlas_dev_mem[dry_run]) + atlas_id;

  if (*cum_len > 0.1f) {
    // The new message is the weighted average of the incoming
    // messages from the rays based on the segment lengths
    float in_pos_log_msg_sum = tex3D<float>(gvdb->volIn[POS_LOG_MSG_SUM], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
    half2 pos_log_msg_sum = (*(reinterpret_cast<half2*>(&in_pos_log_msg_sum)));

#ifndef FULL_INFERENCE
    half2* weighted_incoming =
        reinterpret_cast<half2*>(gvdb->atlas_dev_mem[WEIGHTED_INCOMING]) + atlas_id;
#ifdef LEGACY
    // Which mode of incoming messages has a higher number of cumulative length?
    // Just choose the average of that as the message sent back to all rays
    float len_0s = __low2float(*weighted_incoming);
    float average_incoming_non_empty = __high2float(*weighted_incoming) / (*cum_len - len_0s);

    float cell_new_msg = 0.0f;
    if (len_0s > *cum_len - len_0s) {
      // More empty node messages
      cell_new_msg = F_EPS;
    } else {
      cell_new_msg = average_incoming_non_empty;
    }
    // Now we clamp and move to log space for numerical hygiene
    cell_new_msg = clamp(cell_new_msg, F_EPS, 1.0f - F_EPS);
    float new_log_msg = logf(cell_new_msg) - logf(1.0f - cell_new_msg);
    
    float updated_pos_log_msg_sum = new_log_msg + g_logodds_prob_prior_;
    half2 updated_pos_log_msg_sum_half2 = __floats2half2_rn(average_incoming_non_empty, updated_pos_log_msg_sum);
#else
    // The correct sent message is 1/( 1 + e^-(sum incoming log_odds except ray in question))
    // Thus, we only store the sum of the incoming log_odds in pos_log_msg_sum, and modify it
    //  in the ray traversal method
    float len_0s = __low2float(*weighted_incoming);
    float cell_incoming_log_msg_sum = __high2float(*weighted_incoming) + len_0s * LOGODDS_F_EPS;
    float average_incoming_non_empty = __high2float(*weighted_incoming) / (*cum_len - len_0s + F_EPS);

    float cell_new_msg = 1.0f / (1.0f + expf(-average_incoming_non_empty));
    float new_log_msg = cell_incoming_log_msg_sum;

    // // Add some momentum damping
    float lambda = 0.0f;
    float updated_pos_log_msg_sum = lambda * __high2float(pos_log_msg_sum) + (1.0f - lambda) * cell_incoming_log_msg_sum + g_logodds_prob_prior_;
    // float updated_pos_log_msg_sum = __high2float(pos_log_msg_sum) + cell_incoming_log_msg_sum;
    // Clamp this to a certain threshold after which removing a max individual message hardly impacts
    // the overall sum. Choosing it to be 100x max individual message (LOGODDS_F_EPS)
    updated_pos_log_msg_sum = clamp(updated_pos_log_msg_sum, 100.0f * LOGODDS_F_EPS, -100.0f * LOGODDS_F_EPS);

    half2 updated_pos_log_msg_sum_half2 = __floats2half2_rn(average_incoming_non_empty, updated_pos_log_msg_sum);
#endif
#else  // FULL_INFERENCE is defined...

    float average_incoming_non_empty = __high2float(*weighted_incoming) / (*cum_len);
    float cell_new_msg = average_incoming_non_empty;
    cell_new_msg = clamp(cell_new_msg, F_EPS, 1.0f - F_EPS);
    float new_log_msg = logf(cell_new_msg) - logf(1.0f - cell_new_msg);
    float old_log_msg = __low2float(*weighted_incoming);  // Low 16 bits store the previous weighted incoming
    float updated_pos_log_msg_sum = __high2float(pos_log_msg_sum) + new_log_msg - old_log_msg;
    half2 updated_pos_log_msg_sum_half2 = __floats2half2_rn(average_incoming_non_empty, updated_pos_log_msg_sum);
    // Also cache this new_log_msg, and reset weighted_incoming
    *weighted_incoming = half2(new_log_msg, 0.0f);
#endif
    float neg = clamp(1.0f / (1.0f + expf(updated_pos_log_msg_sum)), F_EPS, 1.0f - F_EPS);
    float positive = 1.0f - neg;

#ifndef FULL_INFERENCE
    if (!dry_run) {
#endif
      surf3Dwrite(positive / (positive + neg), gvdb->volOut[ALPHAS], vox.x * sizeof(float),
                  vox.y, vox.z);
      float temp = *(reinterpret_cast<float*>(&updated_pos_log_msg_sum_half2));
      surf3Dwrite(temp, gvdb->volOut[POS_LOG_MSG_SUM],
                  vox.x * sizeof(float), vox.y, vox.z);
#ifndef FULL_INFERENCE
    }
#endif
    if (g_selected_vox_id_ != 0 && atlas_id == g_selected_vox_id_) {
      (printf("\n<====factors====\ncell_new_msg:%f new_log_msg:%f prev_avg_incoming_non0::%f prev_pos_log_msg_sum:%f updated:%f alpha:%f cum_len:%f len_0s::%f incoming_non0s::%f avg_incoming_non0::%f updated_pos_log_msg_sum::%f atlas_id::%lu\n====factors======>\n",
              cell_new_msg, new_log_msg, __low2float(pos_log_msg_sum), __high2float(pos_log_msg_sum),
              updated_pos_log_msg_sum, positive / (positive + neg), *cum_len, __low2float(*weighted_incoming), __high2float(*weighted_incoming), __low2float(updated_pos_log_msg_sum_half2), __high2float(updated_pos_log_msg_sum_half2), atlas_id));
      //   DEBUG(printf("\t vox :: %u %u %u\n", vox.x, vox.y, vox.z););
    }
  } else {
    if (g_selected_vox_id_ != 0 && atlas_id == g_selected_vox_id_) {
      float in_pos_log_msg_sum = tex3D<float>(gvdb->volIn[POS_LOG_MSG_SUM], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
      half2 pos_log_msg_sum = (*(reinterpret_cast<half2*>(&in_pos_log_msg_sum)));
      printf("\n!!!!!tiny voxel!!!!! low:%f high:%f\n", __low2float(pos_log_msg_sum), __high2float(pos_log_msg_sum));
    }
  }
}

extern "C" __global__ void reset_empty_voxels(VDBInfo* gvdb,
                                              int3 res,
                                              uchar chan) {
  GVDB_VOX
  int3 atlas_res = gvdb->atlas_res;
  unsigned long int atlas_id = vox.z * atlas_res.x * atlas_res.y +
                               vox.y * atlas_res.x + vox.x;
  float* cum_len = (float*)(gvdb->atlas_dev_mem[CUM_LENS]) + atlas_id;
  half2* weighted_incoming =
      reinterpret_cast<half2*>(gvdb->atlas_dev_mem[chan]) + atlas_id;
  if (*cum_len < 0.05f) {
    // We should only send the standard prior out for this voxel
    surf3Dwrite(0.0f, gvdb->volOut[ALPHAS], vox.x * sizeof(float), vox.y, vox.z);
    half2 pos_log_msg_prior_half2 = __floats2half2_rn(0.0f, g_logodds_prob_prior_);
    float assigned_val = *(reinterpret_cast<float*>(&pos_log_msg_prior_half2));
    surf3Dwrite(assigned_val, gvdb->volOut[POS_LOG_MSG_SUM], vox.x * sizeof(float), vox.y, vox.z);
    // Also 0 incoming messages...
    *weighted_incoming = half2(0.0f, 0.0f);
  }
}

__device__ void rayComputeCumLens(VDBInfo* gvdb, uchar chan, int nodeid,
                                  float3 t, float3 pos, float3 dir,
                                  float3& pStep, float3& hit, float3& norm,
                                  float4& clr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 vmin;
  int3 atlas_res = gvdb->atlas_res;
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  t.x += gvdb->epsilon;                             // make sure we start inside
  float multiplier = g_res_ * fabsf(dot(scn.dir_vec, dir));
  float distance = t.x * multiplier;
  float3 o = make_float3(node->mValue);  // gives us the atlas data coordinates
  /// Initialize brick coordinates, DDA and ray traversal
  float3 p, tDel, tSide, mask;  // 3DDA variables
  PREPARE_DDA_LEAF;
  // Iterate till the end of the brick, or max iters, whichever is first
  float prev_tx = 0.0f;
  for (int iter = 0;
       iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 &&
       p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0];
       iter++) {
    NEXT_DDA;  // Update the value of t, and store it in t.y
    uint3 vox = {static_cast<uint>(p.x + o.x),
                 static_cast<uint>(p.y + o.y),
                 static_cast<uint>(p.z + o.z)};
    float length = t.y - prev_tx;  // Updated t minus prev_t
    prev_tx = t.y;
    // This was the length of this segment traversed within the cell
    distance += length * multiplier;
    unsigned long int atlas_id = vox.z * atlas_res.x * atlas_res.y +
                                 vox.y * atlas_res.x + vox.x;
    float* atlas_mem = (float*)(gvdb->atlas_dev_mem[chan]) + atlas_id;
    float prev_value;
    // Overload this kernel to also subtract
    if (hit.x < 0.0f) {
      prev_value = atomicAdd(atlas_mem, -length);
    } else {  //Add the traversed length to the voxel's cum_len
      prev_value = atomicAdd(atlas_mem, length);
    }
#ifdef DEBUG
    if (g_selected_vox_id_ != 0 && g_selected_vox_id_ == atlas_id) {
      printf(">>>Updating cum_len for vox id %lu. t:%.3f Was: %.3f and new length is ::%.3f\n", atlas_id, t.y, prev_value, prev_value + (hit.x < 0 ? -1 : 1) * length);
    }
#endif
    STEP_DDA;
  }

  float depth = getLinearDepth(SCN_DBUF);
  if (distance + get_depth_bias(distance, x, y) >= depth + 3.0f * get_depth_sigma(distance, x, y)) {
    hit.z = 1.0f;
  }
}

__device__ void rayDiagnostics(VDBInfo* gvdb, uchar chan, int nodeid,
                               float3 t, float3 pos, float3 dir,
                               float3& pStep, float3& hit,
                               float3& norm, float4& clr) {
  float3 vmin;
  int3 atlas_res = gvdb->atlas_res;
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  t.x += gvdb->epsilon;                             // make sure we start inside
  float3 o = make_float3(node->mValue);             // gives us the atlas data coordinates
  float multiplier = g_res_ * fabsf(dot(scn.dir_vec, dir));
  float distance = t.x * multiplier;
  /// Initialize brick coordinates, DDA and ray traversal
  float3 p, tDel, tSide, mask;  // 3DDA variables
  PREPARE_DDA_LEAF
  // Iterate till the end of the brick, or max iters, whichever is first
  float prev_tx = 0.0f, cumprod = 1.0f, cum_alphas = 0.0f, prev_vis = 1.0f;
  int stride = 10;
  int current_voxel_counter = 0;
  float old_t = t.x;

  if (clr.w != 1.0f) {
    cum_alphas = clr.x;
    current_voxel_counter = clr.y;
    // cumprod = clr.z;
    prev_vis = clr.z;
  }

  int iter = 0;
  float depth = getLinearDepth(SCN_DBUF);

  for (; iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 &&
         p.x < gvdb->res[0] && p.y < gvdb->res[0] && p.z < gvdb->res[0];
       iter++) {
    NEXT_DDA;  // Update the value of t, and store it in t.y

    // Access the alpha value at this voxel
    uint3 vox = {static_cast<uint>(p.x + o.x),
                 static_cast<uint>(p.y + o.y),
                 static_cast<uint>(p.z + o.z)};
    uint64 atlas_id = vox.z * atlas_res.x * atlas_res.y +
                      vox.y * atlas_res.x + vox.x;
    float length = t.y - prev_tx;  // Updated t minus prev_t
    old_t += length;
    distance += length * multiplier;
    float p_occ =
        tex3D<float>(gvdb->volIn[chan], vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
    // float3 wpos;
    // getAtlasToWorld(gvdb, vox, wpos);
    // float3 ray_to_cam_center = wpos - pos;
    // float vox_distance = fabsf(dot(scn.dir_vec, ray_to_cam_center)) * g_res_;
    *((float*)(scn.outbuf) + current_voxel_counter * stride) = distance;
    // *((float*)(scn.outbuf) + current_voxel_counter * stride) = vox_distance;
    // if (x == g_selected_x_ && y == g_selected_y_) {
    //   DEBUG(printf("distance is %f vox_distance is %f\n", distance, vox_distance));
    // }
    *((float*)(scn.outbuf) + current_voxel_counter * stride + 1) = p_occ;
    // Since the atlas_id is 64 bytes, we need to store it across two floats
    *((uint64*)(scn.outbuf) + current_voxel_counter * (stride / 2) + 1) = atlas_id;
    // Add b_i to this list
    // float b_i = p_occ * cumprod;
    // *((float*)(scn.outbuf) + current_voxel_counter * stride + 4) = b_i;

    // Also compute the visibility terms
    float alpha_i = -1.0f / (1.0f) * logf(1.0f - p_occ);
    cum_alphas += -alpha_i * length;

    float vis_i = expf(cum_alphas);
    *((float*)(scn.outbuf) + current_voxel_counter * stride + 5) = vis_i;

    // Compute w_i
    float w_i = (prev_vis - vis_i) / length;
    *((float*)(scn.outbuf) + current_voxel_counter * stride + 4) = w_i;

    // Going to store the coordinates of this brick (node)
    *((uint64*)(scn.outbuf) + current_voxel_counter * (stride / 2) + 3) = nodeid;
    // int x = blockIdx.x * blockDim.x + threadIdx.x;  // Pixel coordinates
    // int y = blockIdx.y * blockDim.y + threadIdx.y;
    // if( x == g_selected_x_ && y == g_selected_y_){
    //   float3 W = pos + old_t * dir;
    //   printf("\n t:%f dist:%f nodeid is %lu and W: %f %f %f, vmin: %f %f %f, vdel is %f p:%f %f %f", t.x, distance, nodeid, W.x, W.y, W.z, vmin.x, vmin.y, vmin.z, gvdb->vdel[0], p.x, p.y, p.z);
    // }
    // And store the brick coordinates of this voxel
    // *((float*)(scn.outbuf) + current_voxel_counter * stride + 8) = (int)p.x * gvdb->res[0] * gvdb->res[1]
    // +(int)p.y * gvdb->res[0] + (int) p.z;
    *((uchar*)(scn.outbuf) + current_voxel_counter * stride * 4 + 8 * 4) = static_cast<uchar>(p.x);
    *((uchar*)(scn.outbuf) + current_voxel_counter * stride * 4 + 8 * 4 + 1) = static_cast<uchar>(p.y);
    *((uchar*)(scn.outbuf) + current_voxel_counter * stride * 4 + 8 * 4 + 2) = static_cast<uchar>(p.z);

    ++current_voxel_counter;
    prev_vis = vis_i;
    cumprod *= 1.0f - p_occ;
    prev_tx = t.y;
    STEP_DDA
  }

  clr.x = cum_alphas;
  clr.y = current_voxel_counter;
  // clr.z = cumprod;
  clr.z = prev_vis;
  clr.w = 0.5f;
}

// Step 1. Use the depth image points to activate volume (think of smart
// extension for increasing sigma with ground truth distance) Step 2. Further
// ray trace traversals use custom volume render kernel?

// On the meta level - there are two ways we can go about this:
// a. Scatter: Each ray updates the voxels (atomically, ew)
// b. Gather: For every voxel, determine the rays that go through it - can be
// made faster by doing a first pass to determine the rays going through a particular brick?
// Anyway, moving on with a....

//-------------------------------------
// PRE PASS
//-------------------------------------
// Method to compute the accumulated taus in the first pass
__device__ void rayStartComputingMessagesFromBrickPre(
    VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir,
    float3& pStep, float3& hit, float3& norm, float4& clr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 vmin;
  int3 atlas_res = gvdb->atlas_res;
  int brick_dim = gvdb->res[0];
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  t.x += gvdb->epsilon;                             // make sure we start inside
  float3 o = make_float3(node->mValue);             // gives us the atlas data coordinates

  // Ugly solution - use a global defined device member.
  float multiplier = g_res_ * fabsf(dot(scn.dir_vec, dir));

  // float prod_mu_o_0s = 1.0f;
  float prev_vis = 1.0f, cum_alphas = 0.0f, accum_taus = 0.0f, distance = t.x * multiplier;
  int current_voxel_counter = 0, prev_outgoing_index = hit.x;  // Index of the first non-zero outgoing message from the previous iteration

  if (clr.x != 0.0f) {
    accum_taus = clr.x;
    // prod_mu_o_0s = clr.y;
    // Visibility cache from previous brick
    cum_alphas = norm.x;
    prev_vis = norm.y;
    current_voxel_counter = norm.z;
  }

  float depth = getLinearDepth(SCN_DBUF);

  // Initialize brick coordinates, DDA and ray traversal
  float3 p, tDel, tSide, mask;  // 3DDA variables
  PREPARE_DDA_LEAF
  // Iterate till the end of the brick, or max iters, whichever is first
  /**
   * compute_message_to_node_pre
   */
  float prev_tx = 0.0f;
  for (int iter = 0;
       iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 &&
       p.x < brick_dim && p.y < brick_dim && p.z < brick_dim;
       iter++) {
    // Iterate through all the voxels in the current queue and
    // Compute their respective occupancy probabilities
    NEXT_DDA;  // Update the value of t, and store it in t.y
    uint3 vox = {static_cast<uint>(p.x + o.x),
                 static_cast<uint>(p.y + o.y),
                 static_cast<uint>(p.z + o.z)};
    unsigned long int atlas_id = vox.z * atlas_res.x * atlas_res.y +
                                 vox.y * atlas_res.x + vox.x;
    float length = t.y - prev_tx;  // Updated t minus prev_t
    float in_pos_log_msg_sum = (tex3D<float>(gvdb->volIn[POS_LOG_MSG_SUM],
                                             vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f));
    half2 pos_log_msg_sum_half2 = (*(reinterpret_cast<half2*>(&in_pos_log_msg_sum)));
    float pos_log_msg_sum = __high2float(pos_log_msg_sum_half2);

#ifndef FULL_INFERENCE
#ifdef LEGACY
    pos_log_msg_sum = g_logodds_prob_prior_;
#else
    // The correct sent message is 1/( 1 + e^-(sum incoming log_odds except ray in question))
    // pos_log_msg_sum has sum of all incoming messages including what this ray sent previous iter
    // Therefore subtract our contribution

    // Obtain the old outgoing mode (based on the voxel index compared to the first non-0 outgoing message)
    float last_outgoing_msg = 0.0f;
    if (current_voxel_counter < prev_outgoing_index && length > 0.1f) {
      // We're in the empty message mode, set it to F_EPS
      last_outgoing_msg = LOGODDS_F_EPS;
    } else {
      // We just assume that the last_outgoing_message was the average value
      last_outgoing_msg = __low2float(pos_log_msg_sum_half2);
    }
    pos_log_msg_sum -= length * last_outgoing_msg;
#endif
#else  // FULL_INFERENCE is declared...
    // Remove the sent message in the previous iteration from the accumulated
    half2* last_outgoing = reinterpret_cast<half2*>(gvdb->atlas_dev_mem[chan]) + atlas_id;
    float old_msg = __low2float(*last_outgoing);
    float old_log_msg = logf(old_msg) - logf(1.0f - old_msg);
    pos_log_msg_sum -= old_msg;
#endif
    // Now convert this to a negative probability vector
    // Clamp this to be between F_eps and 1.0f-F_eps; These are the mu_o_0s
    float neg_normalized = clamp(1.0f / (1.0f + expf(pos_log_msg_sum)), F_EPS, 1.0f - F_EPS);

    // We need to do one forward pass through the array to compute the
    // reverse accum taus. The choice im making here is to recompute taus
    // instead of caching them like I did on the CPU

    // In piecewise continuous interp, this is a region of constant occlusion, get dip in visibility
    // caused by traversing this amount of length through it
    float alpha_i = -1.0f / (1.0f) * logf(neg_normalized);
    cum_alphas += -alpha_i * length;
    float vis_i = expf(cum_alphas);

    prev_tx = t.y;
    distance += length * multiplier;
    // float3 wpos;
    // getAtlasToWorld(gvdb, vox, wpos);
    // float3 ray_to_cam_center = wpos - pos;
    // float vox_distance = fabsf(dot(scn.dir_vec, ray_to_cam_center)) * g_res_;
    // float ray_mu_d_i = depth_potential(vox_distance, depth);
    float ray_mu_d_i = depth_potential(distance, depth, x, y);

    // float tau_i = prod_mu_o_0s * (1.0f - neg_normalized) * ray_mu_d_i;
    float tau_i = vis_i * (1.0f - neg_normalized) * ray_mu_d_i;
    accum_taus += tau_i;
    // prod_mu_o_0s *= neg_normalized;
    prev_vis = vis_i;
    ++current_voxel_counter;
    STEP_DDA
  }

  // if (x == g_selected_x_ && y == g_selected_y_) {
  //   DEBUG(printf(
  //       "distance at end is %f and t.x is %f accum_taus is %f prod_mu_0s is "
  //       "%f limit is %f\n",
  //       distance, t.x, accum_taus, prod_mu_o_0s,
  //       depth + 3.0f * sqrtf(g_sigma_sq_)));
  // }
  clr.x = accum_taus;
  // clr.y = prod_mu_o_0s;
  norm.x = cum_alphas;
  norm.y = prev_vis;
  norm.z = current_voxel_counter;
  clr.z = distance + get_depth_bias(distance, x, y);
  if (distance + get_depth_bias(distance, x, y) > depth + 3.0f * get_depth_sigma(distance, x, y)) {
    clr.w = 0.0f;
  }
}

//-------------------------------------
// POST PASS
//-------------------------------------
__device__ void rayStartComputingMessagesFromBrickPost(
    VDBInfo* gvdb, uchar chan, int nodeid, float3 t, float3 pos, float3 dir,
    float3& pStep, float3& hit, float3& norm, float4& clr) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  float3 vmin;
  int3 atlas_res = gvdb->atlas_res;
  int brick_dim = gvdb->res[0];
  VDBNode* node = getNode(gvdb, 0, nodeid, &vmin);  // Get the VDB leaf node
  t.x += gvdb->epsilon;                             // make sure we start inside
  float3 o = make_float3(node->mValue);             // gives us the atlas data coordinates

  // Ugly solution - use a global defined device member.
  float multiplier = g_res_ * fabsf(dot(scn.dir_vec, dir));

  // float prod_mu_o_0s = 1.0f;
  float accum_taus = 0.0f, accum_rev_taus = 0.0f,
        distance = t.x * multiplier, cum_alphas = 0.0f, prev_vis = 1.0f;

  int current_voxel_counter = 0, prev_outgoing_index = hit.x;  // Index of the first non-zero outgoing message from the previous iteration
  if (clr.w == 0.0f) {
    accum_rev_taus = clr.x;
    clr.w = 0.5f;
    norm.z = 0;
  } else {
    accum_taus = clr.x;
    // prod_mu_o_0s = clr.y;
    accum_rev_taus = clr.z;
    // Visibility cache from previous brick
    cum_alphas = norm.x;
    prev_vis = norm.y;
    current_voxel_counter = norm.z;
  }

  float depth = getLinearDepth(SCN_DBUF);

  // Initialize brick coordinates, DDA and ray traversal
  float3 p, tDel, tSide, mask;  // 3DDA variables
  PREPARE_DDA_LEAF
  // Iterate till the end of the brick, or max iters, whichever is first
  /**
   * compute_message_to_node_pre
   */
  float prev_tx = 0.0f;
  for (int iter = 0;
       iter < MAX_ITER && p.x >= 0 && p.y >= 0 && p.z >= 0 &&
       p.x < brick_dim && p.y < brick_dim && p.z < brick_dim;
       iter++) {
    // Iterate through all the voxels in the current queue and
    // Compute their respective occupancy probabilities
    NEXT_DDA;  // Update the value of t, and store it in t.y
    uint3 vox = {static_cast<uint>(p.x + o.x),
                 static_cast<uint>(p.y + o.y),
                 static_cast<uint>(p.z + o.z)};
    unsigned long int atlas_id = vox.z * atlas_res.x * atlas_res.y +
                                 vox.y * atlas_res.x + vox.x;
    float length = t.y - prev_tx;  // Updated t minus prev_t
    float in_pos_log_msg_sum = (tex3D<float>(gvdb->volIn[POS_LOG_MSG_SUM],
                                             vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f));
    half2 pos_log_msg_sum_half2 = (*(reinterpret_cast<half2*>(&in_pos_log_msg_sum)));
    float pos_log_msg_sum = __high2float(pos_log_msg_sum_half2);
    float pos_log_msg_sum_pre = pos_log_msg_sum;
    // TODO: Last outgoing message is saved in our 2D buffer
    // float last_outgoing_msg = tex3D<float>(gvdb->volIn[CURR_LOG_MSG],
    //                                        vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
    float last_outgoing_msg = 0.0f;
#ifndef FULL_INFERENCE
#ifdef LEGACY
    // pos_log_msg_sum -= last_outgoing_msg;
    pos_log_msg_sum = g_logodds_prob_prior_;
#else
    // The correct sent message is 1/( 1 + e^-(sum incoming log_odds except ray in question))
    // pos_log_msg_sum has sum of all incoming messages including what this ray sent previous iter
    // Therefore subtract our contribution

    // Obtain the old outgoing mode (based on the voxel index compared to the first non-0 outgoing message)
    if (current_voxel_counter < prev_outgoing_index && length > 0.1f) {
      // We're in the empty message mode, set it to F_EPS
      last_outgoing_msg = LOGODDS_F_EPS;
    } else {
      // We just assume that the last_outgoing_message was the average value
      last_outgoing_msg = __low2float(pos_log_msg_sum_half2);
    }
    pos_log_msg_sum -= length * last_outgoing_msg;
#endif
#else  // FULL_INFERENCE is declared...
    // Remove the sent message in the previous iteration from the accumulated
    half2* last_outgoing =
        reinterpret_cast<half2*>(gvdb->atlas_dev_mem[chan]) + atlas_id;
    float old_msg = __low2float(*last_outgoing);
    float old_log_msg = logf(old_msg) - logf(1.0f - old_msg);
    pos_log_msg_sum -= old_msg;
#endif
    // Now convert this to a negative probability vector
    // Clamp this to be between F_eps and 1.0f-F_eps; These are the mu_o_0s
    float neg_normalized = clamp(1.0f / (1.0f + expf(pos_log_msg_sum)), F_EPS, 1.0f - F_EPS);
    // Now evaluate the outgoing messages
    prev_tx = t.y;
    distance += length * multiplier;

    // In piecewise continuous interp, this is a region of constant occlusion, get dip in visibility
    // caused by traversing this amount of length through it
    float alpha_i = -1.0f / (1.0f) * logf(neg_normalized);
    cum_alphas += -alpha_i * length;
    float vis_i = expf(cum_alphas);

    // float3 wpos;
    // getAtlasToWorld(gvdb, vox, wpos);
    // float3 ray_to_cam_center = wpos - pos;
    // float vox_distance = fabsf(dot(scn.dir_vec, ray_to_cam_center)) * g_res_;
    // float ray_mu_d_i = depth_potential(vox_distance, depth);
    float ray_mu_d_i = depth_potential(distance, depth, x, y);
    // float tau_i = prod_mu_o_0s * (1.0f - neg_normalized) * ray_mu_d_i;
    float tau_i = vis_i * (1.0f - neg_normalized) * ray_mu_d_i;
    accum_rev_taus -= tau_i;
    float mu_1 = accum_taus + tau_i / (1.0f - neg_normalized);
    float mu_0 = accum_taus + accum_rev_taus / neg_normalized;
    // Normalize the mu_1, and send the log odds message!
    float pos_msg =
        clamp(mu_1 / (mu_1 + mu_0 + 0.001f * F_EPS), F_EPS, 1.0f - F_EPS);
    float log_msg = logf(pos_msg) - logf(1.0f - pos_msg);

    // Atomic add to the incoming messages for this node
    // (Remember to multiply log odds by length traversed)

    float sent_msg = 0.0f;
#ifdef LEGACY
    sent_msg = pos_msg * length;
#else
    sent_msg = log_msg * length;
#endif

#ifndef FULL_INFERENCE
    if (chan == 0) {
      half2* weighted_incoming_log_odds =
          reinterpret_cast<half2*>(gvdb->atlas_dev_mem[WEIGHTED_INCOMING]) + atlas_id;
      // We add to either the empty message bank or the positive one
      half2 sent_msg_bimodal;
      if (pos_msg == F_EPS) {
        // We just accumulate the length of the empty mode messages
        sent_msg_bimodal = __floats2half2_rn(length, 0.0f);
      } else {
        sent_msg_bimodal = __floats2half2_rn(0.0f, sent_msg);
        if (hit.y == 0) {
          // This is the first voxel where we have transitioned!
          hit.y = current_voxel_counter;
        }
      }
      atomicAdd(weighted_incoming_log_odds, sent_msg_bimodal);
    }
#else
    half2* weighted_incoming_log_odds =
        reinterpret_cast<half2*>(gvdb->atlas_dev_mem[chan]) + atlas_id;
    half2 sent_msg_half2 = __floats2half2_rn(0.0f, sent_msg);
    atomicAdd(weighted_incoming_log_odds, sent_msg_half2);
#endif

#ifdef ENABLE_DEBUG
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // Pixel coordinates
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int stride = 12;
    if (x == g_selected_x_ && y == g_selected_y_) {
      // Store all these values in the scn.outbuf
      *((float*)(scn.outbuf) + current_voxel_counter * stride) = distance;
      // *((float*)(scn.outbuf) + current_voxel_counter * stride) = vox_distance;
      *((float*)(scn.outbuf) + current_voxel_counter * stride + 1) = ray_mu_d_i;
      *((float*)(scn.outbuf) + current_voxel_counter * stride + 2) = last_outgoing_msg;
      *((float*)(scn.outbuf) + current_voxel_counter * stride + 3) = pos_log_msg_sum_pre;
      *((float*)(scn.outbuf) + current_voxel_counter * stride + 4) = neg_normalized;
      *((float*)(scn.outbuf) + current_voxel_counter * stride + 5) = pos_msg;
      *((float*)(scn.outbuf) + current_voxel_counter * stride + 6) = length;
      // *((float*)(scn.outbuf) + current_voxel_counter * stride + 7) = 0.0;
      *((uint64*)(scn.outbuf) + current_voxel_counter * (stride / 2) + 4) = atlas_id;
      *((float*)(scn.outbuf) + current_voxel_counter * stride + 10) = vis_i;
    }

    if (g_selected_vox_id_ != 0 && g_selected_vox_id_ == atlas_id) {  //pos_msg > 0.85f ) {
      float* cum_len = (float*)(gvdb->atlas_dev_mem[CUM_LENS]) + atlas_id;
      float3 wpos;
      getAtlasToWorld(gvdb, vox, wpos);
      (printf(
          "\n%c x::%3d y::%3d ctr::%3d (prev_idx::%3d) depth::%.3f dist::%.3f ray_mu_d_i::%.3f cum_len::%.3f pos_log_msg_sum:: pre:%.3f last_outgoing::%.3f post:%.3f neg_norm::%.3f mu_1::%.3f mu_0::%.3f "
          "log_msg::%.3f p_occ::%.3f length::%.3f sent_msg::%.3f atlas_id::%lu",
          //  wpos: %f %f %f\n",
          (x == g_selected_x_ && y == g_selected_y_) ? '>' : ' ',
          x, y, current_voxel_counter, prev_outgoing_index, depth, distance, ray_mu_d_i, *cum_len, pos_log_msg_sum_pre, last_outgoing_msg, pos_log_msg_sum, neg_normalized, mu_1, mu_0, log_msg, pos_msg, length, sent_msg, atlas_id));
          //, wpos.x, wpos.y, wpos.z));
    }

    // if (x == g_selected_x_ && y == g_selected_y_) {
    //   DEBUG(printf(
    //       "distance at end is %f and t.x is %f accum_taus is %f prod_mu_0s is "
    //       "%f\n",
    //       distance, t.x, accum_taus, prod_mu_o_0s));
    // }
#endif

    accum_taus += tau_i;
    // prod_mu_o_0s *= neg_normalized;
    prev_vis = vis_i;
    ++current_voxel_counter;
    STEP_DDA
  }

  clr.x = accum_taus;
  // clr.y = prod_mu_o_0s;
  clr.z = accum_rev_taus;
  norm.x = cum_alphas;
  norm.y = prev_vis;
  norm.z = current_voxel_counter;

  if (distance + get_depth_bias(distance, x, y) >= depth + 3.0f * get_depth_sigma(distance, x, y)) {
    clr.w = 0.0f;
  }
}

extern "C" __global__ void set_channel_kernel(VDBInfo* gvdb, uint num_pts, float3* pts, float* alphas, int chan) {
  uint idx = blockIdx.x * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x;  // thread ID; 1D grid of 3D blocks
  if (idx >= num_pts) return;

  float3 wpos;
  wpos.x = pts[idx].x;
  wpos.y = pts[idx].y;
  wpos.z = pts[idx].z;

  //Get corresponding node to this world point
  float3 offs, vmin, vdel;
  uint64 nid;
  VDBNode* node = getNodeAtPoint(gvdb, wpos, &offs, &vmin, &vdel, &nid);
  offs += (wpos - vmin) / vdel;
  if(node == 0x0) {
    printf("[set_channel_kernel] EXCEPTION SHOULD NOT BE HERE!!!\n");
    return;
  }
  int3 vox = make_int3(offs);
  float val = alphas[idx];
  // if (idx == 0) {
  //   printf("idx::%u wpos::%f %f %f val::%f vox %d %d %d vmin %f %f %fchan::%d\n", idx, wpos.x, wpos.y, wpos.z, val, vox.x, vox.y, vox.z, vmin.x, vmin.y, vmin.z, chan);
  // }
  surf3Dwrite(val, gvdb->volOut[chan], vox.x * sizeof(float), vox.y, vox.z);
}

extern "C" __global__ void render_accuracy_kernel(VDBInfo* gvdb, uchar chan,
                                                  uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float depth = getLinearDepth(SCN_DBUF);
  if (depth <= 0) {
    *((float*)(outBuf) + y * scn.width + x) = -1;
    return;
  }
  // Overload hit.x to store max w(s)
  float3 hit = make_float3(0.0f, NOHIT, NOHIT);
  float4 clr = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
  float3 norm = make_float3(0.0f, 0.0f, 0.0f);
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));

  clr = make_float4(1.0f, .0f, .0f, 1.0f);
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          rayLikelihoodBrick);
  if (hit.x > 0 && clr.x >= 0.9){
    // i.e. if we've gone through at least one brick, but reached map boundary with most visibility intact
    // Classify as accurate if measurement is beyond distance travelled
    if(norm.x <= depth){
      clr.x = -1;
    }
    else{
      clr.x = 0;
    }
  } else if (hit.x > 0 && clr.x < 0.9) {
    // i.e. if the ray has travelled through some volume, and has a peak w
    clr.x = hit.y;
  } else {
    // We didn't go through a single brick - whoosh.
    clr.x = -1;
  }
  *((float*)(outBuf) + y * scn.width + x) = clr.x;
}

extern "C" __global__ void render_simple_accuracy_kernel(VDBInfo* gvdb, uchar chan,
                                                         uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float depth = getLinearDepth(SCN_DBUF);
  if (depth <= 0) {
    *((float*)(outBuf) + y * scn.width + x) = -1;
    return;
  }

  float3 hit = make_float3(0.0f, NOHIT, NOHIT);
  // Set x component to -1 to detect if value changed.
  float4 clr = make_float4(-1.0f, 0.0f, 0.0f, 1.0f);
  float3 norm = make_float3(0.0f, 0.0f, 0.0f);
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));

  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          raySimpleAccuracyBrick);

  if (clr.w == 1) {
    //This ray never hit anything above the threshold, send invalid
    *((float*)(outBuf) + y * scn.width + x) = -1;
  } else {
    *((float*)(outBuf) + y * scn.width + x) = clr.x;
  }
}

extern "C" __global__ void render_occ_thresh_kernel(VDBInfo* gvdb, uchar chan,
                                                    uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float depth = getLinearDepth(SCN_DBUF);
  if (depth <= 0) {
    *((float*)(outBuf) + y * scn.width + x) = -1;
    return;
  }
  float3 hit = make_float3(0.0f, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm = make_float3(0.0f, -1.0f, 0.0f);
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));

  clr = make_float4(1.0f, .0f, .0f, 1.0f);
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          rayLikelihoodBrick);
  if (norm.y != -1) {
    clr.x = norm.y;
  } else {
    clr.x = -1;
  }
  *((float*)(outBuf) + y * scn.width + x) = clr.x;
}

extern "C" __global__ void render_factors_to_node_kernel(VDBInfo* gvdb, uchar chan,
                                                         uchar* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float depth = getLinearDepth(SCN_DBUF);
  if (depth <= 0) {
    return;
  }

  float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm;
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));

  clr = make_float4(.0f, .0f, .0f, 1.0f);
  hit = make_float3(.0f, .0f, NOHIT);

  // Store the cached last outgoing index
  hit.x = outBuf[y * scn.width + x];

#ifdef SINGLE_RAY
  if (x != g_selected_x_ || y != g_selected_y_) {
    return;
  }
#endif
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          rayStartComputingMessagesFromBrickPre);
  // It could be the case that we never managed to find activated bricks
  // to exhaust the 3 sigma search for voxels beyond the measured depth
  if (clr.w != 0.0f) {
    // if (x == g_selected_x_ && y == g_selected_y_) {
    //    DEBUG(printf(
    //        "\nboo boo! We never got to see 3 sigma beyond measured depth!\n"));
    // }
    // Add inf. term
    // The argument here is that the sum over the normal=0 for inf ray, so we set it up to 1.0f for N+1
    // float ray_mu_d_i_N = depth >= Z_MAX ? 1.0f - F_EPS : 0.0f;
    // vis_infty * depth_potential(distance+bias)
    float ray_mu_d_i_N = depth > clr.z? norm.y*1.0f : 0.0f;
    clr.x += ray_mu_d_i_N;
    clr.w = 0.0f;
  }
  // Now perform post pass
  // clr should have our cached value for accum_taus
  hit.y = 0.0f;
  norm.z = 0.0f;
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          rayStartComputingMessagesFromBrickPost);
  if (!chan) {
    // Set the cached last outgoing index
    outBuf[y * scn.width + x] = static_cast<uchar>(hit.y);
  }
#ifdef ENABLE_DEBUG
  // If in debug mode just add the number of voxels used right at the end...
  if (x == g_selected_x_ && y == g_selected_y_) {
    // printf("Num voxels was %f\n", norm.z);
    *((float*)(scn.outbuf) + scn.height * scn.width - 1) = norm.z;
  }
#endif
}

extern "C" __global__ void render_length_traversed_kernel(VDBInfo* gvdb, uchar chan,
                                                          uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm;
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));

  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          rayLengthTraversedBrick);
  outBuf[y * scn.width + x] =
      make_uchar4(clr.x * 255, clr.y * 255, clr.z * 255, 255);
}

extern "C" __global__ void render_likelihood_kernel(VDBInfo* gvdb, uchar chan,
                                                    uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;
  float depth = getLinearDepth(SCN_DBUF);
  if (depth <= 0) {
    *((float*)(outBuf) + y * scn.width + x) = 0;
    return;
  }
  float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm;
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));
  clr = make_float4(1.0f, .0f, .0f, 1.0f);
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          rayLikelihoodBrick);
  float vis_inf = clr.x, sum = clr.z;
  // Add final term
  sum += vis_inf * (depth >= Z_MAX ? 1.0f - LIKE_EPS : LIKE_EPS);

  // if (x == g_selected_x_ && y == g_selected_y_) {
  // DEBUG(printf(" cumsum::%f sum::%f logf::%f\n", cumsum, sum, logf(sum)));
  // }

  clr.x = logf(sum);
  *((float*)(outBuf) + y * scn.width + x) = clr.x;
}

extern "C" __global__ void render_cumulative_length_kernel(VDBInfo* gvdb, uchar chan,
                                                           uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float depth = getLinearDepth(SCN_DBUF);
  if (depth <= 0) {
    return;
  }

#ifdef SINGLE_RAY
  if (x != g_selected_x_ || y != g_selected_y_) {
    return;
  }
#endif

  float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm;
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));
  clr = make_float4(.0f, .0f, .0f, 1.0f);
  hit = make_float3(.0f, .0f, NOHIT);
  // Update cumulative lengths
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr, rayComputeCumLens);
}

extern "C" __global__ void render_subtract_cumulative_length_kernel(VDBInfo* gvdb, uchar chan,
                                                                    uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float depth = getLinearDepth(SCN_DBUF);
  if (depth <= 0) {
    return;
  }

#ifdef SINGLE_RAY
  if (x != g_selected_x_ || y != g_selected_y_) {
    return;
  }
#endif

  float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm;
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));
  clr = make_float4(.0f, .0f, .0f, 1.0f);
  hit = make_float3(-1.0f, .0f, NOHIT);
  // Update cumulative lengths
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr, rayComputeCumLens);
}

extern "C" __global__ void render_expected_depth_kernel(VDBInfo* gvdb, uchar chan,
                                                        uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm;
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));

  clr = make_float4(1.0f, .0f, .0f, 1.0f);
  rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr,
          rayExpectedDepthBrick);
  // Add the final event that none are occupied
  // float cumprod = clr.x, cumsum = clr.y, sum = clr.z;
  // float b_i = cumprod;
  // float depth = getLinearDepth(SCN_DBUF);
  // if (depth >= Z_MAX) {
  //   cumsum = Z_MAX;
  // } else {
  //   cumsum /= sum + F_EPS;
  // }
  // // Assign the colour of this pixel
  // clr.x = cumsum;
  // Because int_0_inf w(s) ds = 1 - vis_inf
  float sum = clr.z / (1.0f - clr.x + F_EPS);
  clr.x = sum;

  *((float*)(outBuf) + y * scn.width + x) = clr.x;
}

extern "C" __global__ void render_diagnostics_xy_kernel(VDBInfo* gvdb, uchar chan,
                                                        uchar4* outBuf) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= scn.width || y >= scn.height) return;

  float3 hit = make_float3(NOHIT, NOHIT, NOHIT);
  float4 clr = make_float4(1, 1, 1, 1);
  float3 norm;
  float3 rdir = normalize(
      getViewRay((float(x) + 0.5f) / scn.width, (float(y) + 0.5f) / scn.height));

  if (x == g_selected_x_ && y == g_selected_y_) {
    // Pull out diagnostic information from chosen selected point
    clr = make_float4(0.0f, .0f, .0f, 1.0f);
    rayCast(gvdb, chan, scn.campos, rdir, hit, norm, clr, rayDiagnostics);
    // Just add the number of voxels used right at the end...
    *((float*)(scn.outbuf) + scn.height * scn.width - 1) = clr.y;
  }
}