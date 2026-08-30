// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_TOOLS_REPORT_REPORT_DIFF_HPP
#define ARUCO3CUDA_TOOLS_REPORT_REPORT_DIFF_HPP

#include <array>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

/// Classifies the differences between detection results.
///
/// Purpose:
///     Compares the detections returned by the CPU baseline implementation against
///     those of the route under evaluation and counts the differences by kind. A
///     single number such as "98% agreement" cannot distinguish a slight corner
///     displacement from a missed marker. The kinds are recorded separately so the
///     result keeps the granularity a judgement needs.
namespace aruco3cuda::report {

/// A single detection used for the comparison.
///
/// The internal representation differs between routes, so this is a common form
/// holding only the fields the comparison needs.
struct Detection {
    int id_ = -1;
    /// Holds the four corners in the order x0, y0, x1, y1, x2, y2, x3, y3.
    std::array<double, 8> corners_{};
};

/// Kind of difference.
enum class DiffKind {
    /// Present in the baseline, absent in the target.
    kMissed,
    /// Present in the target, absent in the baseline.
    kExtra,
    /// At the same position, but with a different ID.
    kIdMismatch,
    /// Same ID, but the corner ordering is rotated.
    kRotationMismatch,
    /// Same ID and same ordering, but the corner positions exceed the tolerance.
    kCornerShift,
};

/// A single difference.
///
/// Which fields carry meaning depends on the kind. For kMissed, target_id_ is -1;
/// for kExtra, baseline_id_ is -1; rotation_steps_ is non-zero only for
/// kRotationMismatch.
///
/// Ownership: value type. Holds no references or pointers.
/// Synchronization: none. This is a plain aggregate; synchronization is the
/// caller's responsibility.
///
/// Example input: a target with id=8 sits at the position of baseline id=7
/// Example output: kind_ = kIdMismatch, baseline_id_ = 7, target_id_ = 8
struct Diff {
    DiffKind kind_ = DiffKind::kMissed;
    /// ID on the baseline side. -1 for kExtra.
    int baseline_id_ = -1;
    /// ID on the target side. -1 for kMissed.
    int target_id_ = -1;
    /// How many positions the target corners are rotated relative to the baseline.
    /// Meaningful only for kRotationMismatch.
    int rotation_steps_ = 0;
    /// Largest corner-to-corner distance in pixels. 0 when no pairing was made.
    double corner_error_px_ = 0.0;
    /// Centroid on the baseline side in pixels. Kept to locate the difference.
    double center_x_px_ = 0.0;
    double center_y_px_ = 0.0;
};

/// Comparison result for one image.
///
/// Each of the following two items applies to all public member functions.
/// Ownership: value type. Owns its internal vector and string.
/// Synchronization: none. The const member functions may be called concurrently.
///
/// Example input: an image with 4 baseline detections and 3 target detections,
///                one of which was missed
/// Example output: baseline_count_ = 4, target_count_ = 3, agreed_count_ = 3,
///                 diffs_ holding one kMissed
struct ImageComparison {
    std::string image_path_;
    std::size_t baseline_count_ = 0;
    std::size_t target_count_ = 0;
    /// Number of detections with no difference of any kind.
    std::size_t agreed_count_ = 0;
    /// Largest corner-to-corner distance in pixels among the paired detections.
    double worst_corner_error_px_ = 0.0;
    std::vector<Diff> diffs_;

    /// Whether there is no difference at all.
    ///
    /// @return true when there is not a single difference.
    ///
    /// Example input: diffs_ is empty
    /// Example output: true
    bool agrees() const { return this->diffs_.empty(); }
};

/// Settings for the comparison.
struct CompareConfig {
    /// Corners within this distance are treated as agreeing. The unit is pixels.
    double corner_tolerance_px_ = 1.0;
    /// Upper bound on the centroid distance for treating two detections as the same
    /// marker, as a ratio of one marker side.
    ///
    /// It is a ratio rather than an absolute value because the reasonable distance
    /// changes with the image resolution and the size of the marker.
    double match_radius_ratio_ = 0.5;
};

/// Returns the display name of a kind. Used when presenting a report.
///
/// @param kind The kind of difference whose name is wanted.
/// @return A string in static storage. Returns "unknown" for a value outside the
///         enumeration.
///
/// Ownership: ownership of the returned string is not transferred; it must not be
/// freed.
/// Synchronization: none. Reentrant.
///
/// Example input: DiffKind::kMissed
/// Example output: "missed"
const char* diff_kind_name(DiffKind kind);

/// Compares the baseline against the target and classifies the differences.
///
/// The pairing uses centroid proximity rather than the ID. Pairing by ID would, for
/// a misread ID, record one "missed" and one "extra", which hides what actually
/// happened: the ID of the same marker was read incorrectly.
///
/// Example input: 4 baseline detections and 4 target detections, one with a
///                differing ID
/// Example output: diffs_ holding one kIdMismatch, agreed_count_ = 3
ImageComparison compare_detections(const std::string& image_path,
                                   const std::vector<Detection>& baseline,
                                   const std::vector<Detection>& target,
                                   const CompareConfig& config);

/// Aggregate of the comparison results over several images.
///
/// Ownership: value type. Holds no references or pointers.
/// Synchronization: none.
///
/// Example input: 3 images, one of which agrees, while the other two hold one
///                missed and one extra detection respectively
/// Example output: image_count_ = 3, agreed_image_count_ = 1, and kMissed and
///                 kExtra in kind_counts_ are each 1
struct Summary {
    std::size_t image_count_ = 0;
    std::size_t agreed_image_count_ = 0;
    std::size_t baseline_detection_count_ = 0;
    std::size_t target_detection_count_ = 0;
    std::size_t agreed_detection_count_ = 0;
    /// Counts by kind, indexed in the order of DiffKind.
    std::array<std::size_t, 5> kind_counts_{};
    double worst_corner_error_px_ = 0.0;
};

/// Aggregates the per-image comparison results.
///
/// @param comparisons The results to aggregate. Their order does not affect the
///        outcome.
/// @return An aggregate holding the counts by kind and the worst difference.
///
/// Ownership: the argument is not retained. The result is returned by value.
/// Synchronization: none. The argument is not modified, so the function may be
/// called concurrently on the same input.
///
/// Example input: one agreeing image and one image holding a single missed
///                detection
/// Example output: image_count_ = 2, agreed_image_count_ = 1
Summary summarize(const std::vector<ImageComparison>& comparisons);

/// Writes the report in a human-readable form.
///
/// Every image with a difference is listed, so that no one can present only the
/// favorable results.
///
/// @param out The output stream. Used only during the call.
/// @param comparisons The per-image comparison results.
/// @param summary The aggregate of comparisons.
/// @return Nothing. Only the state of out changes.
///
/// Ownership: ownership of out is not transferred.
/// Synchronization: none. Concurrent writes to the same out must be serialized by
/// the caller.
///
/// Example input: a single comparison result with no differences
/// Example output: several lines including "No differences."
void write_text_report(std::ostream& out, const std::vector<ImageComparison>& comparisons,
                       const Summary& summary);

/// Writes the report in a machine-readable form.
///
/// @param out The output stream. Used only during the call.
/// @param comparisons The per-image comparison results.
/// @param summary The aggregate of comparisons.
/// @param config The settings used for the comparison. Included in the output as
///        the basis for the judgement.
/// @return Nothing. Only the state of out changes.
///
/// Ownership: ownership of out is not transferred.
/// Synchronization: none. Concurrent writes to the same out must be serialized by
/// the caller.
///
/// Example input: a single comparison result holding one missed detection
/// Example output: JSON in which summary.kindCounts.missed is 1
void write_json_report(std::ostream& out, const std::vector<ImageComparison>& comparisons,
                       const Summary& summary, const CompareConfig& config);

}  // namespace aruco3cuda::report

#endif  // ARUCO3CUDA_TOOLS_REPORT_REPORT_DIFF_HPP
