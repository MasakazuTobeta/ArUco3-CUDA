// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "count_first_contours.hpp"
int main() {
    cv::setNumThreads(1);
    aruco3cuda::cpu::experiment::ContourWorkspace workspace;
    for (auto name : {"cpu_1280x720_n4_clean.png", "cpu_1280x720_n4_noise.png",
                      "cpu_3840x2160_n4_clean.png", "cpu_3840x2160_n4_noise.png"}) {
        cv::Mat gray = cv::imread(std::string("data/cpu-experiment/") + name, cv::IMREAD_GRAYSCALE);
        if (gray.empty()) {
            std::cerr << "Missing corpus image: " << name << '\n';
            return 1;
        }
        for (bool aruco : {true, false}) {
            aruco3cuda::DetectorConfig config;
            config.use_aruco3_detection_ = aruco;
            cv::Mat image = gray;
            if (aruco) {
                float s = 32.F / (32.F + std::max(gray.cols, gray.rows) * .05F);
                cv::resize(gray, image, cv::Size(cvRound(gray.cols * s), cvRound(gray.rows * s)));
            }
            std::vector<cv::Mat> binaries(3);
            for (int i = 0; i < 3; ++i)
                cv::adaptiveThreshold(image, binaries[i], 255, cv::ADAPTIVE_THRESH_MEAN_C,
                                      cv::THRESH_BINARY_INV, 3 + 10 * i, 7.);
            std::vector<std::vector<cv::Point2f>> a, b;
            std::vector<std::vector<cv::Point>> ca, cb;
            for (auto& im : binaries) {
                a.clear();
                b.clear();
                ca.clear();
                cb.clear();
                workspace.append(im, config, a, ca, false);
                workspace.append(im, config, b, cb, true);
                if (a != b || ca != cb) {
                    std::cerr << "difference";
                    return 1;
                }
            }
            std::vector<double> times[2];
            for (int n = -3; n < 20; ++n)
                for (int k = 0; k < 2; ++k) {
                    int v = (n + 3 + k) % 2;
                    auto start = std::chrono::steady_clock::now();
                    for (auto& im : binaries) {
                        a.clear();
                        ca.clear();
                        workspace.append(im, config, a, ca, v == 1);
                    }
                    double ms = std::chrono::duration<double, std::milli>(
                                        std::chrono::steady_clock::now() - start)
                                        .count();
                    if (n >= 0)
                        times[v].push_back(ms);
                }
            for (int v = 0; v < 2; ++v) {
                std::sort(times[v].begin(), times[v].end());
                std::cout << name << " aruco=" << aruco << " count_first=" << v
                          << " p50=" << times[v][9] << " p95=" << times[v][18]
                          << " p99=" << times[v][19] << std::endl;
            }
        }
    }
}
