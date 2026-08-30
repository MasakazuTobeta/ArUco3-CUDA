// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TOOLS_EVALUATE_ACCURACY_HPP
#define ARUCO3CUDA_TOOLS_EVALUATE_ACCURACY_HPP

#include <cstddef>
#include <string>
#include <vector>

#include "geometry.hpp"

/// Matches detections against the ground truth and computes accuracy metrics.
///
/// Purpose:
///     The [difference report](../report/report_diff.hpp) takes the CPU baseline
///     implementation as its reference. That baseline, however, is an oracle for
///     compatibility, not ground truth. A marker the baseline itself missed never
///     shows up as a difference, so "100% agreement" can still mean the marker was
///     not detected. Matching against the true values known at generation time
///     measures precision and recall separately.
namespace aruco3cuda::evaluate {

using aruco3cuda::report::Quad;

/// One true value known at generation time.
///
/// Ownership: value type. Holds no references or pointers.
/// Synchronization: none.
///
/// Example input: a marker with ID 7 drawn at center (100, 100) with a 40-pixel side
/// Example output: id_ = 7, side_px_ = 40.0, fully_inside_ = true
struct TruthMarker {
    int id_ = -1;
    /// The four corners, corresponding to (0,0), (S,0), (S,S), (0,S) in marker
    /// coordinates.
    Quad corners_{};
    /// Length of one side, in pixels.
    double side_px_ = 0.0;
    /// Whether all four corners fall inside the image.
    bool fully_inside_ = true;
    /// Area ratio of the occlusion covering the marker.
    double occlusion_ratio_ = 0.0;
};

/// Outcome for one true value.
enum class TruthOutcome {
    /// Matched by position, with the ID also agreeing.
    kDetected,
    /// Matched by position, but the ID was read incorrectly.
    kIdMismatched,
    /// No detection matched it.
    kMissed,
};

/// One detection returned by the route under evaluation.
///
/// Ownership: value type.
/// Synchronization: none.
///
/// Example input: an ID of 7 and the four corners returned by the detector
/// Example output: id_ = 7, corners_ holding 8 elements
struct Observation {
    int id_ = -1;
    Quad corners_{};
};

/// Settings for the matching.
struct MatchConfig {
    /// Upper bound on the centroid distance for treating a true value and a detection
    /// as the same marker, as a ratio of the true value's side length.
    double match_radius_ratio_ = 0.5;
};

/// Matching result for one image.
///
/// Each of the following two items applies to all public member functions.
/// Ownership: value type. Holds no references or pointers.
/// Synchronization: none. The const member functions may be called concurrently.
///
/// Example input: an image where 3 of 4 true values were detected with the correct ID
///                and 1 was missed
/// Example output: truth_count_ = 4, true_positive_ = 3, false_negative_ = 1
struct ImageAccuracy {
    std::string image_name_;
    std::size_t truth_count_ = 0;
    std::size_t observed_count_ = 0;
    /// Number matched by position with the ID also agreeing.
    std::size_t true_positive_ = 0;
    /// Detections with no match, plus detections whose ID was wrong.
    std::size_t false_positive_ = 0;
    /// True values with no match, plus true values whose ID was read incorrectly.
    std::size_t false_negative_ = 0;
    /// How many of true_positive_ also agree on the corner ordering.
    std::size_t rotation_agreed_ = 0;
    /// Sum of squared corner distances over true_positive_, in pixel^2.
    double corner_squared_sum_px2_ = 0.0;
    /// Number of corners entered into the sum of squares. Four times true_positive_.
    std::size_t corner_sample_count_ = 0;
    /// Largest corner distance over true_positive_, in pixels.
    double corner_max_px_ = 0.0;
    /// Outcome per true value. The order corresponds to the truth passed to
    /// compare_to_truth.
    std::vector<TruthOutcome> truth_outcomes_;
    /// Sum of squared corner distances per true value, in pixel^2. 0 unless kDetected.
    std::vector<double> truth_squared_error_px2_;
    /// Largest corner distance per true value, in pixels. 0 unless kDetected.
    std::vector<double> truth_corner_max_px_;
    /// Whether the corner ordering agrees, per true value. false unless kDetected.
    std::vector<bool> truth_rotation_agreed_;
};

/// Aggregate over several images.
///
/// Ownership: value type.
/// Synchronization: none.
///
/// Example input: two images accumulated, each detecting 3 of 4 true values correctly
/// Example output: image_count_ = 2, truth_count_ = 8, true_positive_ = 6
struct AccuracySummary {
    std::size_t image_count_ = 0;
    std::size_t truth_count_ = 0;
    std::size_t observed_count_ = 0;
    std::size_t true_positive_ = 0;
    std::size_t false_positive_ = 0;
    std::size_t false_negative_ = 0;
    std::size_t rotation_agreed_ = 0;
    double corner_squared_sum_px2_ = 0.0;
    std::size_t corner_sample_count_ = 0;
    double corner_max_px_ = 0.0;
};

/// Matches detections against the true values and computes the metrics for one image.
///
/// The pairing uses centroid proximity rather than the ID. Pairing by ID would, for a
/// misread ID, raise one "missed" and one "extra", which makes it impossible to tell
/// a wrong ID on the same marker from a false detection somewhere else.
///
/// @param image_name Image name to record in the report. Only read, never retained.
/// @param truth The true values from generation time. Only read, never retained.
/// @param observed The detections of the route under evaluation. Only read, never
///        retained.
/// @param config Settings for the pairing. Only read, never retained.
/// @return The metrics for one image.
///
/// Ownership: none of the argument storage is retained. The result is returned by
/// value.
/// Synchronization: none. The arguments are not modified, so the function may be
/// called concurrently on the same input.
///
/// Example input: one true value and one detection at the same position with the same
///                ID
/// Example output: true_positive_ = 1, false_positive_ = 0, false_negative_ = 0
ImageAccuracy compare_to_truth(const std::string& image_name, const std::vector<TruthMarker>& truth,
                               const std::vector<Observation>& observed, const MatchConfig& config);

/// Accumulates the metrics of one image into an aggregate.
///
/// @param image The per-image metrics to accumulate. Only read, never retained.
/// @param out_summary The aggregate to accumulate into. The storage is owned by the
///        caller. Must not be nullptr.
/// @return Nothing. Only the contents of out_summary change.
///
/// Ownership: none of the argument storage is retained.
/// Synchronization: none. Concurrent accumulation into the same out_summary must be
/// serialized by the caller.
///
/// Example input: one image with true_positive_ = 1 added to an empty aggregate
/// Example output: image_count_ = 1, true_positive_ = 1
void accumulate(const ImageAccuracy& image, AccuracySummary* out_summary);

/// Accumulates only a subset of the true values into an aggregate.
///
/// Used to produce recall broken down by condition. ArUco3 cannot, in principle,
/// detect a marker whose side after downscaling falls below
/// `min_side_length_canonical_img_px`. Computing recall while still including true
/// values of an undetectable size mixes the implementation's misses with the
/// strategy's inherent lower bound, leaving it unclear which the number reflects.
///
/// Detections with no match (false positives) belong to no true value and are not
/// accumulated. Precision therefore cannot be defined from the resulting aggregate;
/// read only recall and the corner metrics from it.
///
/// @param image The per-image metrics to accumulate. Only read, never retained.
/// @param selected Selection flags corresponding to the order of the true values.
///        Does nothing if the element count differs from truth_outcomes_.
/// @param out_summary The aggregate to accumulate into. The storage is owned by the
///        caller. Must not be nullptr.
/// @return Nothing. Only the contents of out_summary change.
///
/// Ownership: none of the argument storage is retained.
/// Synchronization: none. Concurrent accumulation into the same out_summary must be
/// serialized by the caller.
///
/// Example input: selection flags choosing only the first of two true values
/// Example output: truth_count_ = 1; observed_count_ and false_positive_ do not grow
void accumulate_selected(const ImageAccuracy& image, const std::vector<bool>& selected,
                         AccuracySummary* out_summary);

/// Computes precision.
///
/// It is undefined when there is not a single detection. Treating the division by
/// zero as 1.0 would mean "the less it detects, the higher its precision", which
/// misleads the judgement.
///
/// @param summary The aggregate in question. Only read, never retained.
/// @param out_value Receives the value when it is defined. The storage is owned by
///        the caller. Must not be nullptr.
/// @return true when the value is defined.
///
/// Ownership: none of the argument storage is retained.
/// Synchronization: none. Reentrant.
///
/// Example input: an aggregate with true_positive_ = 3 and false_positive_ = 1
/// Example output: true, out_value = 0.75
bool precision(const AccuracySummary& summary, double* out_value);

/// Computes recall.
///
/// It is undefined when there is not a single true value. This does occur in
/// practice, because the corpus contains scenes with zero markers.
///
/// @param summary The aggregate in question. Only read, never retained.
/// @param out_value Receives the value when it is defined. The storage is owned by
///        the caller. Must not be nullptr.
/// @return true when the value is defined.
///
/// Ownership: none of the argument storage is retained.
/// Synchronization: none. Reentrant.
///
/// Example input: an aggregate with true_positive_ = 3 and false_negative_ = 1
/// Example output: true, out_value = 0.75
bool recall(const AccuracySummary& summary, double* out_value);

/// Computes the corner RMSE.
///
/// Kept separate from the maximum, so that one badly displaced corner can be told
/// apart from four corners displaced uniformly.
///
/// @param summary The aggregate in question. Only read, never retained.
/// @param out_value Receives the value when it is defined. The storage is owned by
///        the caller. Must not be nullptr.
/// @return true when there is at least one match.
///
/// Ownership: none of the argument storage is retained.
/// Synchronization: none. Reentrant.
///
/// Example input: an aggregate with corner_squared_sum_px2_ = 4.0 and
///                corner_sample_count_ = 4
/// Example output: true, out_value = 1.0
bool corner_rmse_px(const AccuracySummary& summary, double* out_value);

}  // namespace aruco3cuda::evaluate

#endif  // ARUCO3CUDA_TOOLS_EVALUATE_ACCURACY_HPP
