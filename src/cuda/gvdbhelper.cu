#include <thrust/adjacent_difference.h>
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sort.h>
#include "gvdb.h"

__host__ float sum_likelihood(CUdeviceptr &ptr, uint num_pixels) {
  thrust::device_ptr<float> likelihoods_ptr(reinterpret_cast<float *>(ptr));
  float sum = thrust::reduce(likelihoods_ptr, likelihoods_ptr + num_pixels);
  return sum;
}

__host__ float compute_accuracy(CUdeviceptr &ptr, uint num_pixels) {
  thrust::device_ptr<float> accuracy_img_ptr(reinterpret_cast<float *>(ptr));
  // We have, -1 == invalid pixel (NaN depth), 0 == inaccurate peak, 1 == accurate peak
  // First, sort the pointer
  thrust::sort(accuracy_img_ptr, accuracy_img_ptr + num_pixels);
  // And now find the ends of each bin
  thrust::counting_iterator<int> search_begin(-1);
  int num_bins = 3;  // For -1, 0, and 1
  //static_cast<int>(*(accuracy_img_ptr + num_pixels - 1)) - static_cast<int>(*(accuracy_img_ptr)) + 1;

  thrust::device_vector<int> histogram(num_bins);
  // Find the end of each bin of values
  thrust::upper_bound(accuracy_img_ptr, accuracy_img_ptr + num_pixels, search_begin, search_begin + num_bins, histogram.begin());

  // Find individual bin counts
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());
  thrust::host_vector<int> h_bins = histogram;
  // std::cout << h_bins[0] <<" "<< h_bins[1] <<" "<<h_bins[2]<<"\n" ;
  // Accuracy = number of correctly classified pixels / (total number of pixels - number of invalid pixels)
  return 1.0f * h_bins[2] / (num_pixels - h_bins[0]);
}

__host__ float compute_tp_fp_tn_fn(CUdeviceptr &ptr, uint num_pixels, float &tp, float &fp, float &tn, float &fn) {
  thrust::device_ptr<float> accuracy_img_ptr(reinterpret_cast<float *>(ptr));
  // We have, -1 == invalid pixel (out of Z_MAX range), 0 == false negative, 1 == true pos, 2 == false pos
  // First, sort the pointer
  thrust::sort(accuracy_img_ptr, accuracy_img_ptr + num_pixels);
  // And now find the ends of each bin
  thrust::counting_iterator<int> search_begin(-1);
  int num_bins = 4;  //static_cast<int>(*(accuracy_img_ptr + num_pixels - 1)) - static_cast<int>(*(accuracy_img_ptr)) + 1;

  thrust::device_vector<int> histogram(num_bins);
  // Find the end of each bin of values
  thrust::upper_bound(accuracy_img_ptr, accuracy_img_ptr + num_pixels, search_begin, search_begin + num_bins, histogram.begin());

  // Find individual bin counts
  thrust::adjacent_difference(histogram.begin(), histogram.end(),
                              histogram.begin());
  thrust::host_vector<int> h_bins = histogram;

  tp = h_bins[2];
  fp = h_bins[3];
  fn = h_bins[1];
  tn = h_bins[0];

  return (tp + tn) / (num_pixels);
}