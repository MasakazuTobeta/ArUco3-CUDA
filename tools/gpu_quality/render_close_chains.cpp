// SPDX-License-Identifier: Apache-2.0
// Deterministic public OpenCV marker rendering; no detector implementation copied.
#include <array>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_dictionary.hpp>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: aruco3cuda_render_close_chains OUTPUT_DIRECTORY\n";
            return 1;
        }
        const std::filesystem::path directory(argv[1]);
        if (std::filesystem::exists(directory)) {
            throw std::runtime_error("output directory already exists");
        }
        std::filesystem::create_directories(directory);
        std::ofstream truth_file(directory / "truth.jsonl");
        if (!truth_file) {
            throw std::runtime_error("cannot create truth file");
        }
        const cv::aruco::Dictionary dictionary =
                cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        int sequence = 0;
        for (const int count : {2, 3}) {
            for (const int gap : {-8, -2, 0, 1, 3, 7, 16, 32}) {
                for (int pattern = 0; pattern < 4; ++pattern) {
                    cv::Mat image(360, 640, CV_8UC1, cv::Scalar(235));
                    constexpr int kSidePx = 80;
                    constexpr int kTopPx = 140;
                    const int start_px = (640 - count * kSidePx - (count - 1) * gap) / 2;
                    std::vector<std::array<cv::Point2f, 4>> corners;
                    std::vector<int> ids;
                    for (int i = 0; i < count; ++i) {
                        const int x = start_px + i * (kSidePx + gap);
                        const int id = 17 + i * 7;
                        const int rotation = (sequence + i) % 4;
                        cv::Mat marker;
                        dictionary.generateImageMarker(id, kSidePx, marker);
                        for (int k = 0; k < rotation; ++k) {
                            cv::rotate(marker, marker, cv::ROTATE_90_CLOCKWISE);
                        }
                        marker.copyTo(image(cv::Rect(x, kTopPx, kSidePx, kSidePx)));
                        const std::array<cv::Point2f, 4> physical = {
                                cv::Point2f(x, kTopPx), cv::Point2f(x + kSidePx - 1, kTopPx),
                                cv::Point2f(x + kSidePx - 1, kTopPx + kSidePx - 1),
                                cv::Point2f(x, kTopPx + kSidePx - 1)};
                        std::array<cv::Point2f, 4> oriented{};
                        for (int k = 0; k < 4; ++k) {
                            oriented[k] = physical[(k + rotation) % 4];
                        }
                        corners.push_back(oriented);
                        ids.push_back(id);
                    }
                    if (pattern == 1 || pattern == 2) {
                        const int value = pattern == 1 ? 0 : 135;
                        cv::line(image, {start_px, kTopPx + 1},
                                 {start_px + count * kSidePx + (count - 1) * gap - 1, kTopPx + 1},
                                 cv::Scalar(value), 2);
                    }
                    if (pattern == 3) {
                        for (int i = 0; i < count; ++i) {
                            for (const int pad : {6, 12}) {
                                cv::rectangle(image,
                                              {start_px + i * (kSidePx + gap) - pad, kTopPx - pad,
                                               kSidePx + 2 * pad, kSidePx + 2 * pad},
                                              cv::Scalar(120 + pad * 4), 1);
                            }
                        }
                    }
                    const std::string name = "chain_" + std::to_string(sequence++);
                    if (!cv::imwrite((directory / (name + ".png")).string(), image)) {
                        throw std::runtime_error("cannot write image");
                    }
                    truth_file << "{\"name\":\"" << name << "\",\"count\":" << count
                               << ",\"gap\":" << gap << ",\"pattern\":" << pattern
                               << ",\"truth\":[";
                    for (int i = 0; i < count; ++i) {
                        if (i != 0) {
                            truth_file << ",";
                        }
                        truth_file << "{\"id\":" << ids[i] << ",\"corners\":[";
                        for (int j = 0; j < 4; ++j) {
                            if (j != 0) {
                                truth_file << ",";
                            }
                            truth_file << corners[i][j].x << "," << corners[i][j].y;
                        }
                        truth_file << "]}";
                    }
                    truth_file << "]}\n";
                }
            }
        }
        truth_file.close();
        if (!truth_file) {
            throw std::runtime_error("cannot finish truth file");
        }
        std::cout << "rendered " << sequence << " scenes\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
