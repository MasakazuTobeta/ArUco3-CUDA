# Glossary

A glossary for keeping wording consistent across documents, code comments, Doxygen, PR bodies, and reviews. Everything in this repository is written in English.

The "Avoid" column lists the loose expressions that keep coming back. The point of each row is not the wording itself but the discipline behind it: do not assume a speedup, state what was measured, and say which baseline is meant.

| Avoid | Preferred | Notes |
| --- | --- | --- |
| Moving it to the GPU makes it fast | evaluate the conditions under which the CUDA implementation is effective | Do not assume a speedup. |
| no transfer | no explicit copy | Even with unified memory, the cost of synchronization and caching remains. |
| processing time | kernel time / end-to-end time | State the measurement scope. |
| the correct answer | CPU baseline result / ground truth | State which of the two is meant. |
| ArUco3 marker | ArUco3 detection strategy | ArUco3 refers to a fast detection method, not a new Dictionary. |
| GPU machine | CUDA-capable environment | Make the hardware conditions concrete. |
| zero copy | zero-copy | State the API and the memory type used. |
| CPU fallback | CPU fallback | Refers to reduced functionality or a choice intended for small inputs. |
| crossover | crossover point | The condition at which the advantage switches between CPU and CUDA. |
| artifact | the concrete output (logs, measurement results, visualization images) | Logs, measurement results, visualization images, and the like. |

## Technical Names

Use the official or commonly accepted spelling for CUDA, OpenCV, ArUco, ArUco3, DGX Spark, Jetson Orin, Compute Capability, CMake, and C++.
