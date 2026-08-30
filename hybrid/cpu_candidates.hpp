// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_HYBRID_CPU_CANDIDATES_HPP
#define ARUCO3CUDA_HYBRID_CPU_CANDIDATES_HPP

#include <opencv2/core.hpp>

#include <vector>

#include "aruco3cuda/config.hpp"

/// Quad candidate extraction for the CPU route.
///
/// Purpose:
///     Finds candidates in a binarized image with the same procedure as
///     OpenCV's ArUco detector. It is used by the hybrid route and also as the
///     reference for measuring how the GPU route differs. Calling the same code
///     from both sides keeps the comparison focused on the implementation
///     difference alone.
namespace aruco3cuda::hybrid {

/// The corners, contour and perimeter of a single candidate.
///
/// The perimeter is not the contour length but the sum of the edge lengths of
/// the quad through the four corners. This matches OpenCV's MarkerCandidate and
/// is used to pick which candidate of a group is kept.
///
/// Ownership: value type. Owns the vectors it holds.
/// Synchronization: none. It is a plain aggregate.
///
/// Example input: the contour of a square with side 60
/// Example output: perimeter_ = 240
struct MarkerCandidate {
    std::vector<cv::Point2f> corners_;
    std::vector<cv::Point> contour_;
    float perimeter_ = 0.0F;
};

/// Holds the candidates as a containment tree, like OpenCV's
/// MarkerCandidateTree.
///
/// parent_ is the index of the candidate that encloses this one, and depth_ is
/// the largest number of levels nested inside this one. Identification runs in
/// increasing order of depth_, and once a marker is confirmed its parent is
/// dropped from the identification set.
///
/// Ownership: value type. Owns the vectors it holds.
/// Synchronization: none. It is a plain aggregate.
///
/// Example input: both the outer and the inner edge of a black border became
/// candidates
/// Example output: the inner one's parent_ is the outer one's index, and the
/// outer one's depth_ is 1
struct CandidateNode : MarkerCandidate {
    int parent_ = -1;
    int depth_ = 0;
    std::vector<MarkerCandidate> close_contours_;
};

/// Sum of the edge lengths of the quad through the four corners.
///
/// @param corners The four corners. Exactly 4 elements.
/// @return The sum of the edge lengths.
///
/// Ownership: does not retain the argument.
/// Synchronization: none. Reentrant.
///
/// Example input: a square with side 10
/// Example output: 40
float quad_perimeter(const std::vector<cv::Point2f>& corners);

/// Mean corner-to-corner distance between two candidates, taking the smallest
/// of the four possible starting-vertex correspondences.
///
/// Same as OpenCV's getAverageDistance. All correspondences are tried so that
/// two candidates whose vertex order is rotated by one still register as the
/// same marker.
///
/// @param first One set of four corners. Exactly 4 elements.
/// @param second The other set of four corners. Exactly 4 elements.
/// @return The mean distance.
///
/// Ownership: does not retain the arguments.
/// Synchronization: none. Reentrant.
///
/// Example input: two squares at the same position
/// Example output: 0
float average_quad_distance(const std::vector<cv::Point2f>& first,
                            const std::vector<cv::Point2f>& second);

/// Mean number of pixels along one cell edge, derived from the four corners.
/// Same as OpenCV's getAverageModuleSize.
///
/// @param corners The four corners. Exactly 4 elements.
/// @param marker_size Number of cells in the dictionary.
/// @param border_bits Number of cells in the outer border.
/// @return The mean number of pixels along one cell edge.
///
/// Ownership: does not retain the argument.
/// Synchronization: none. Reentrant.
///
/// Example input: a square with side 80, marker_size 6, border_bits 1
/// Example output: 10
float average_module_size(const std::vector<cv::Point2f>& corners, int marker_size,
                          int border_bits);

/// Whether all four corners of first lie inside second. Same as OpenCV's
/// checkMarker1InMarker2.
///
/// @param first The corners tested for being inside. Exactly 4 elements.
/// @param second The enclosing corners. Exactly 4 elements.
/// @return true if every point is inside or on the boundary.
///
/// Ownership: does not retain the arguments.
/// Synchronization: none. Reentrant.
///
/// Example input: a small square and a larger square surrounding it
/// Example output: true
bool quad_inside_quad(const std::vector<cv::Point2f>& first,
                      const std::vector<cv::Point2f>& second);

/// Extracts quad candidates from a binarized image.
///
/// Applies the same tests as OpenCV's _findMarkerContours. When ArUco3 is
/// enabled, the lower perimeter bound is replaced by
/// minSideLengthCanonicalImg * 4.
///
/// @param binary Input binarized image. 0 is background, anything else is
///               foreground.
/// @param config Detector configuration.
/// @param candidates Receives the corners that were found, appended. Existing
///                   elements are kept.
/// @param contours_out Receives the matching contours, appended. It grows by
///                     the same count as candidates.
/// @return Nothing. Results are appended to the vectors passed in.
///
/// Ownership: the caller owns the vectors passed in.
/// Synchronization: none. Reentrant.
///
/// Example input: a binarized image containing three squares
/// Example output: candidates grows by three elements
void find_quad_candidates(const cv::Mat& binary, const DetectorConfig& config,
                          std::vector<std::vector<cv::Point2f>>& candidates,
                          std::vector<std::vector<cv::Point>>& contours_out);

/// Puts the four corners in clockwise order. Same as OpenCV's
/// _reorderCandidatesCorners.
///
/// @param candidate The corners to reorder. Exactly 4 elements. Modified in
///                  place.
/// @return Nothing. The argument is modified.
///
/// Ownership: the caller owns the argument.
/// Synchronization: none. Reentrant.
///
/// Example input: four corners in counter-clockwise order
/// Example output: the second and the fourth corner are swapped
void reorder_corners(std::vector<cv::Point2f>& candidate);

/// Groups nearby candidates, picks a representative per group and builds the
/// containment tree.
///
/// Changing the binarization window yields slightly different candidates for
/// the same marker. As in OpenCV's filterTooCloseCandidates, the candidates are
/// sorted by decreasing perimeter, nearby ones form a group, and the candidate
/// with the largest perimeter in the group becomes its representative. The
/// non-representative candidates that are far from the representative are kept
/// in close_contours_ as fallbacks for when identifying the representative
/// fails.
///
/// If the representative is too close to the image border, the whole group is
/// discarded. OpenCV does the same, because the corners of a marker that runs
/// off the edge cannot be trusted.
///
/// @param image_size Size of the binarized image. Used to test the distance to
///                   the border.
/// @param candidates The candidate corners. The contents are moved out, so the
///                   vector is empty on return.
/// @param contours The matching contours. Also moved out.
/// @param config Detector configuration.
/// @param marker_size Number of cells in the dictionary.
/// @return The representative candidates, ordered by decreasing perimeter.
///
/// Ownership: the elements of the vectors passed in are moved into the return
///            value.
/// Synchronization: none. Reentrant.
///
/// Example input: three candidates of differing perimeter from the same marker
/// Example output: one element. The candidate with the largest perimeter
/// remains
std::vector<CandidateNode> filter_too_close_candidates(
        const cv::Size& image_size, std::vector<std::vector<cv::Point2f>>& candidates,
        std::vector<std::vector<cv::Point>>& contours, const DetectorConfig& config,
        int marker_size);

}  // namespace aruco3cuda::hybrid

#endif  // ARUCO3CUDA_HYBRID_CPU_CANDIDATES_HPP
