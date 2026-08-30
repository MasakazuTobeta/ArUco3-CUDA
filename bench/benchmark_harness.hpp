// SPDX-License-Identifier: Apache-2.0
#ifndef ARUCO3CUDA_BENCH_BENCHMARK_HARNESS_HPP
#define ARUCO3CUDA_BENCH_BENCHMARK_HARNESS_HPP

#include <cstddef>
#include <ostream>
#include <string>
#include <vector>

#include "aruco3cuda/config.hpp"
#include "aruco3cuda/util/statistics.hpp"
#include "reference_runner.hpp"

namespace aruco3cuda::bench {

/// The comparison routes defined by the evaluation plan.
///
/// All four routes can be measured. So that the requested route and the
/// processing actually performed never disagree, an unsupported combination
/// fails instead of being substituted with CPU. Substituting would make the
/// recorded route disagree with what was measured, and the results could no
/// longer be interpreted afterwards.
enum class Route : int {
    kCpu = 0,        ///< OpenCV ArUco3. From a cv::Mat input to the results
    /// CUDA with host input. Includes upload, detection, download and synchronization
    kCudaEndToEnd,
    kCudaResident,   ///< CUDA with device input. From a GPU-resident image to the device results
    kHybrid,         ///< A combination of CUDA and CPU
};

/// Memory type of the input buffer.
///
/// DGX Spark and Jetson Orin are both integrated GPUs, where the cost of an
/// explicit copy differs greatly from a discrete GPU. It is recorded as a
/// measurement axis independent of the route.
enum class MemoryMode : int {
    kNotApplicable = 0, ///< CPU route
    /// Ordinary host memory. The measured interval includes the transfer to the device.
    kHostPageable,
    kHostPinned,
    kManaged,
    /// Device resident. The transfer happens once, outside the measured interval.
    kDevice,
};

/// Convert a Route to the notation of the evaluation plan. Used as the
/// identifier in the result JSONL.
///
/// @param route The value to convert. Never returns nullptr, even for a value
///              outside the enumeration.
/// @return A string with static storage duration.
///
/// Ownership: the return value points to static storage. The caller neither
///            frees nor modifies it.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: Route::kCudaEndToEnd
/// Example output: "CUDA-E2E"
const char* to_string(Route route);

/// Convert a MemoryMode to the notation of the evaluation plan.
///
/// @param mode The value to convert. Never returns nullptr, even for a value
///             outside the enumeration.
/// @return A string with static storage duration.
///
/// Ownership: the return value points to static storage. The caller neither
///            frees nor modifies it.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: MemoryMode::kHostPinned
/// Example output: "M-Pinned"
const char* to_string(MemoryMode mode);

/// Measurement conditions.
struct BenchmarkConfig {
    Route route_ = Route::kCpu;
    MemoryMode memory_mode_ = MemoryMode::kNotApplicable;

    /// Number of preparatory runs excluded from the measured interval.
    int warmup_iterations_ = 20;
    /// Number of single-frame latency measurements.
    int latency_iterations_ = 200;
    /// Number of frames processed back to back for the throughput measurement.
    /// 0 skips the measurement.
    int throughput_frames_ = 100;
    /// Whether to include every sample in the results. Used when the
    /// distribution itself has to be saved.
    bool save_all_samples_ = false;

    /// List of CPU numbers to measure on. Empty leaves the assignment to the OS.
    ///
    /// On a machine that mixes performance and efficiency cores, such as the DGX
    /// Spark GB10, the core type a run lands on changes the CPU baseline by a
    /// factor of 1.6. Without pinning the core, the measured values become
    /// bimodal from run to run and the crossover point is judged wrongly.
    std::vector<int> cpu_affinity_;

    aruco3cuda::reference::ReferenceConfig detector_;

    /// Detection settings for the CUDA routes. Unused on the CPU route.
    ///
    /// It is kept separate from detector_ because the CUDA side has items the
    /// CPU baseline does not have (the candidate limit, the lower bound on
    /// squareness, the block dimensions). The shared items are copied from
    /// detector_ by cuda_config_from_reference().
    aruco3cuda::DetectorConfig cuda_detector_;
};

/// Copy the shared items from the CPU baseline settings into the CUDA route settings.
///
/// Used to keep the conditions aligned across the two routes. Leaving the
/// default-constructed value in place would let use_aruco3_detection_ and the
/// downscale ratio diverge, so that two different conditions were being
/// compared.
///
/// @param config The CPU baseline settings.
/// @return The CUDA route settings with the shared items copied in. All other
///         items keep their default values.
///
/// Ownership: does not retain the arguments.
/// Synchronization: host only, with no synchronization point.
///
/// Example input: use_aruco3_detection_ = true, min_side_length_canonical_img_px_ = 32
/// Example output: a DetectorConfig holding the same values
aruco3cuda::DetectorConfig cuda_config_from_reference(
        const aruco3cuda::reference::ReferenceConfig& config);

/// Measurement result for a single input.
struct MeasurementRecord {
    std::string image_path_;
    std::string image_sha256_;
    int width_px_ = 0;
    int height_px_ = 0;
    std::size_t detection_count_ = 0;

    /// Effective ArUco3 downscale ratio. Always recorded as a measurement condition.
    double fxfy_effective_ = 1.0;

    /// Wall-clock from preparing the input to obtaining the host results, in ms.
    aruco3cuda::util::SampleStatistics end_to_end_ms_;
    /// Detection time measured with CUDA events. Currently unmeasured on every route.
    ///
    /// The evaluation plan asks for it to be separated from wall-clock, but that
    /// is not implemented yet. Filling this in with stage times would make it
    /// impossible to tell afterwards whether a recorded value came from CUDA
    /// events, so it is left empty.
    bool kernel_time_available_ = false;
    aruco3cuda::util::SampleStatistics kernel_ms_;

    /// Per-stage wall-clock within the route. Unmeasured on the CPU route.
    ///
    /// For hybrid, the GPU side (preprocessing, thresholding, transfer to the
    /// host) and the CPU side (candidate extraction through decoding) are
    /// recorded separately. Their sum is smaller than end_to_end_ms_; the
    /// difference is the cost of transferring the input and of the calls.
    bool stage_times_available_ = false;
    aruco3cuda::util::SampleStatistics gpu_stage_ms_;
    aruco3cuda::util::SampleStatistics cpu_stage_ms_;

    /// From the moment the image is ready until the first detection result is
    /// available, in ms.
    ///
    /// Includes preparing the route and detecting the first image. On the CUDA
    /// routes, context creation and kernel loading fall in here and make it
    /// hundreds of times the steady state. Looking only at the percentiles after
    /// warm-up hides this cost from the results. For a one-shot detection or a
    /// short burst, the startup cost dominates the steady-state difference.
    double time_to_first_result_ms_ = 0.0;
    /// Time spent on the detection of the first image alone, in ms.
    ///
    /// The difference from time_to_first_result_ms_ is the preparation of the route.
    double first_frame_ms_ = 0.0;

    /// Frames per second under continuous processing. Unmeasured when
    /// throughput_frames_ is 0.
    bool throughput_available_ = false;
    double throughput_fps_ = 0.0;

    /// Stored only when save_all_samples_ is true.
    std::vector<double> end_to_end_samples_ms_;
};

/// Record of the execution environment.
struct EnvironmentRecord {
    std::string hostname_;
    std::string os_;
    std::string kernel_;
    std::string architecture_;
    std::string opencv_version_;
    int opencv_threads_ = 0;
    std::string cuda_toolkit_version_;
    std::string gpu_name_;
    std::string gpu_compute_capability_;
    std::string driver_version_;
    /// L4T release of a Jetson. A Jetson has no nvidia-smi and therefore no
    /// driver version, so this is recorded as the corresponding information.
    std::string platform_release_;
    /// Board name. Obtained from the device tree.
    std::string platform_model_;
    /// Power mode. The evaluation plan requires it to be recorded as a
    /// measurement condition.
    std::string power_mode_;
    /// CPU core configuration. On a machine that mixes performance and
    /// efficiency cores, measured values cannot be compared without knowing
    /// which type they were taken on.
    std::string cpu_topology_;
    /// The CPU numbers actually used. "unpinned" when no pinning was applied.
    std::string cpu_affinity_;
    /// State of address space layout randomization (ASLR).
    ///
    /// On the CPU route at every resolution, differences in memory layout caused
    /// by ASLR alone move p50 by 9%. Disabling it makes runs match exactly.
    /// Measured values cannot be compared without knowing which of the two they
    /// were taken under.
    std::string address_randomization_;
    /// Maximum GPU clock (MHz). When it cannot be obtained it is left unset
    /// rather than 0.
    bool gpu_clock_available_ = false;
    int gpu_max_clock_mhz_ = 0;
    int gpu_current_clock_mhz_ = 0;
    bool gpu_integrated_ = false;
    /// Time spent creating the CUDA context, in ms. 0 when there is no CUDA device.
    ///
    /// It happens once per process. Because it is triggered implicitly by the
    /// first CUDA API call, it does not show up in the per-route measurements.
    /// For a one-shot detection this cost is hundreds of times the detection itself.
    double cuda_context_ms_ = 0.0;
    int cpu_online_cores_ = 0;
    /// Reason the GPU information could not be obtained. Empty when it succeeded.
    /// Prevents items from going missing silently and keeps the cause traceable
    /// afterwards.
    std::string gpu_probe_error_;
};

/// Collect information about the execution environment.
///
/// The GPU information is filled in only when a CUDA device is available. Items
/// that cannot be obtained are left as empty strings and are never filled in by
/// guesswork.
///
/// @param config Detection settings. Only used for the OpenCV thread count.
/// @return The collected environment information. Items that could not be
///         obtained hold an empty string or a flag indicating "unset".
///
/// Ownership: the return value is a value and references no internal resource.
/// Synchronization: calls the CUDA runtime to query the device properties, but
///                  performs no device synchronization. Launches external
///                  commands to obtain the power mode and the driver version.
///
/// Side effects: when config.detector_.num_threads_ is 1 or more, calls
///               cv::setNumThreads() and thereby changes the thread count of all
///               subsequent OpenCV processing. This is a deliberate side effect
///               that keeps the measurement conditions fixed.
///
/// Example input: a default BenchmarkConfig
/// Example output: opencv_version_ = "4.14.0", gpu_name_ = "Orin", power_mode_ = "MAXN (0)"
EnvironmentRecord collect_environment(const BenchmarkConfig& config);

/// Measure a single image.
///
/// @param image_path The image to measure.
/// @param config Measurement conditions.
/// @param out_record Receives the result on success.
/// @param out_error Receives the reason on failure.
/// @return true on success.
///
/// Notes:
///   Only the CPU route is implemented at present. Specifying any other route
///   fails, stating explicitly that it is unimplemented. It is never silently
///   substituted with the CPU route.
bool measure_image(const std::string& image_path, const BenchmarkConfig& config,
                   MeasurementRecord* out_record, std::string* out_error);

/// Write the environment information as one JSONL line.
///
/// Writes a single trailing newline. JSONL holds one record per line, and this
/// line becomes the first line of the result file.
///
/// @param out Destination. Ownership stays with the caller; this function
///            neither closes nor flushes it. Checking for write failure is the
///            caller's responsibility.
/// @param environment The environment information to write. A clock that could
///                    not be obtained becomes null rather than 0.
/// @return Nothing.
///
/// Synchronization: host only, with no synchronization point.
///
/// Example input: the return value of collect_environment()
/// Example output: {"type":"environment","schema_version":3,...}
void write_environment_line(std::ostream& out, const EnvironmentRecord& environment);

/// Write a measurement result as one JSONL line.
///
/// Writes a single trailing newline. The CPU route has no kernel time, so when
/// kernel_time_available_ is false the output is null rather than a 0 fill.
/// Writing 0 would be misread during aggregation as "a very fast kernel".
///
/// @param out Destination. Ownership stays with the caller; this function
///            neither closes nor flushes it. Checking for write failure is the
///            caller's responsibility.
/// @param config Measurement conditions. The route, the memory type and the
///               detection settings are recorded as conditions.
/// @param record The measurement result.
/// @return Nothing.
///
/// Synchronization: host only, with no synchronization point.
///
/// Example input: the return value of measure_image() and the BenchmarkConfig used for it
/// Example output: {"type":"measurement","route":"CPU",...,"kernel":null,...}
void write_measurement_line(std::ostream& out, const BenchmarkConfig& config,
                            const MeasurementRecord& record);

}  // namespace aruco3cuda::bench

#endif  // ARUCO3CUDA_BENCH_BENCHMARK_HARNESS_HPP
