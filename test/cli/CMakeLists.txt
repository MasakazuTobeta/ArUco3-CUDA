# SPDX-License-Identifier: Apache-2.0
#
# CLI の自動テスト。
#
# 目的:
#   CLI は利用者が最初に触れる境界であり、引数解析の誤りは実行時まで表面化しない。
#   正常系、異常系、境界値を ctest から実行し、終了コードと出力で判定する。
#
# 備考:
#   実行 file を別 process として起動するが、gcov は process 終了時に計測結果を
#   書き出すため、この経路もカバレッジへ反映される。
set(kCliCorpusDir "${CMAKE_CURRENT_BINARY_DIR}/cli-corpus")

# 画像を必要とする test のための fixture。corpusgen 自体の正常系も兼ねる。
add_test(NAME cli.corpusgen.setup
  COMMAND $<TARGET_FILE:aruco3cuda_corpusgen_cli>
          --preset smoke --seed 20260827
          --output-dir "${kCliCorpusDir}"
          --manifest "${kCliCorpusDir}/manifest.json")
set_tests_properties(cli.corpusgen.setup PROPERTIES FIXTURES_SETUP cli_corpus TIMEOUT 300)

# 引数を取らない、または入力を必要としない検査。
# 形式: <test 名> <target> <期待する成功可否> <引数...>
function(aruco3cuda_add_cli_test test_name target expect_success)
  add_test(NAME "${test_name}" COMMAND $<TARGET_FILE:${target}> ${ARGN})
  set_tests_properties("${test_name}" PROPERTIES TIMEOUT 300)
  if(NOT expect_success)
    set_tests_properties("${test_name}" PROPERTIES WILL_FAIL TRUE)
  endif()
endfunction()

# --- reference runner ---
aruco3cuda_add_cli_test(cli.reference.help aruco3cuda_reference_runner TRUE --help)
aruco3cuda_add_cli_test(cli.reference.list_dictionaries aruco3cuda_reference_runner TRUE --list-dictionaries)
aruco3cuda_add_cli_test(cli.reference.no_input aruco3cuda_reference_runner FALSE --omit-timing)
aruco3cuda_add_cli_test(cli.reference.unknown_option aruco3cuda_reference_runner FALSE --nonexistent-option)
aruco3cuda_add_cli_test(cli.reference.missing_value aruco3cuda_reference_runner FALSE --input)
aruco3cuda_add_cli_test(cli.reference.unknown_dictionary aruco3cuda_reference_runner FALSE
  --dictionary DICT_NOPE --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.invalid_integer aruco3cuda_reference_runner FALSE
  --threads abc --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.invalid_real aruco3cuda_reference_runner FALSE
  --min-marker-length-ratio xyz --input /nonexistent.png)
# 境界値: ArUco3 有効で縮小の指定が両方 0 の組み合わせは OpenCV 側が拒否する。
aruco3cuda_add_cli_test(cli.reference.aruco3_zero_parameters aruco3cuda_reference_runner FALSE
  --use-aruco3 1 --min-side-length-canonical 0 --min-marker-length-ratio 0
  --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.missing_image aruco3cuda_reference_runner FALSE
  --input /nonexistent/image.png)
aruco3cuda_add_cli_test(cli.reference.unwritable_output aruco3cuda_reference_runner FALSE
  --input /nonexistent/image.png --output /nonexistent-dir/out.json)

aruco3cuda_add_cli_test(cli.reference.invalid_use_aruco3 aruco3cuda_reference_runner FALSE
  --use-aruco3 yes --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.invalid_error_correction aruco3cuda_reference_runner FALSE
  --error-correction-rate abc --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.reference.detector_options aruco3cuda_reference_runner FALSE
  --dictionary DICT_6X6_250 --threads 0 --use-aruco3 0
  --error-correction-rate 0.5 --adaptive-thresh-win-min 5
  --adaptive-thresh-win-max 21 --adaptive-thresh-win-step 8
  --input /nonexistent/image.png)

# 出力先を指定しない場合は標準出力へ書く。
add_test(NAME cli.reference.detect_to_stdout
  COMMAND $<TARGET_FILE:aruco3cuda_reference_runner>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png" --omit-timing)
set_tests_properties(cli.reference.detect_to_stdout PROPERTIES
  FIXTURES_REQUIRED cli_corpus TIMEOUT 300
  PASS_REGULAR_EXPRESSION "schema_version")

add_test(NAME cli.reference.detect
  COMMAND $<TARGET_FILE:aruco3cuda_reference_runner>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
          --omit-timing
          --output "${CMAKE_CURRENT_BINARY_DIR}/cli-reference.json")
set_tests_properties(cli.reference.detect PROPERTIES
  FIXTURES_REQUIRED cli_corpus TIMEOUT 300
  PASS_REGULAR_EXPRESSION "^$")

# --- corpusgen ---
aruco3cuda_add_cli_test(cli.corpusgen.help aruco3cuda_corpusgen_cli TRUE --help)
aruco3cuda_add_cli_test(cli.corpusgen.list_presets aruco3cuda_corpusgen_cli TRUE --list-presets)
aruco3cuda_add_cli_test(cli.corpusgen.unknown_preset aruco3cuda_corpusgen_cli FALSE
  --preset does_not_exist --output-dir "${CMAKE_CURRENT_BINARY_DIR}/cli-bad")
aruco3cuda_add_cli_test(cli.corpusgen.unknown_option aruco3cuda_corpusgen_cli FALSE --nonexistent-option x)
aruco3cuda_add_cli_test(cli.corpusgen.missing_value aruco3cuda_corpusgen_cli FALSE --seed)
aruco3cuda_add_cli_test(cli.corpusgen.invalid_seed aruco3cuda_corpusgen_cli FALSE --seed not_a_number)
aruco3cuda_add_cli_test(cli.corpusgen.unknown_dictionary aruco3cuda_corpusgen_cli FALSE
  --dictionary DICT_NOPE --output-dir "${CMAKE_CURRENT_BINARY_DIR}/cli-bad")

# --- dictgen ---
aruco3cuda_add_cli_test(cli.dictgen.help aruco3cuda_dictgen TRUE --help)
aruco3cuda_add_cli_test(cli.dictgen.list aruco3cuda_dictgen TRUE --list)
aruco3cuda_add_cli_test(cli.dictgen.no_action aruco3cuda_dictgen FALSE --dictionary DICT_ARUCO_MIP_36h12)
aruco3cuda_add_cli_test(cli.dictgen.unknown_option aruco3cuda_dictgen FALSE --nonexistent-option x)
aruco3cuda_add_cli_test(cli.dictgen.missing_value aruco3cuda_dictgen FALSE --dictionary)
aruco3cuda_add_cli_test(cli.dictgen.unknown_dictionary aruco3cuda_dictgen FALSE
  --dictionary DICT_NOPE --output "${CMAKE_CURRENT_BINARY_DIR}/cli-bad.cpp")
aruco3cuda_add_cli_test(cli.dictgen.check_missing_file aruco3cuda_dictgen FALSE
  --dictionary DICT_ARUCO_MIP_36h12 --check /nonexistent/table.cpp)
# 生成物が OpenCV と整合していることを継続的に確認する。
aruco3cuda_add_cli_test(cli.dictgen.check_generated aruco3cuda_dictgen TRUE
  --dictionary DICT_ARUCO_MIP_36h12
  --check "${PROJECT_SOURCE_DIR}/src/dictionary/generated/dict_aruco_mip_36h12.cpp")
# 生成した内容が commit 済みの内容と一致することも確認する。
add_test(NAME cli.dictgen.generate_matches_committed
  COMMAND "${CMAKE_COMMAND}"
          -DDICTGEN=$<TARGET_FILE:aruco3cuda_dictgen>
          -DGENERATED=${CMAKE_CURRENT_BINARY_DIR}/cli-dictgen.cpp
          -DCOMMITTED=${PROJECT_SOURCE_DIR}/src/dictionary/generated/dict_aruco_mip_36h12.cpp
          -P "${CMAKE_CURRENT_SOURCE_DIR}/compare_generated.cmake")
set_tests_properties(cli.dictgen.generate_matches_committed PROPERTIES TIMEOUT 300)

# --- benchmark ---
aruco3cuda_add_cli_test(cli.bench.help aruco3cuda_bench_cli TRUE --help)
aruco3cuda_add_cli_test(cli.bench.no_input aruco3cuda_bench_cli FALSE --warmup 1)
aruco3cuda_add_cli_test(cli.bench.unknown_option aruco3cuda_bench_cli FALSE --nonexistent-option x)
aruco3cuda_add_cli_test(cli.bench.missing_value aruco3cuda_bench_cli FALSE --warmup)
aruco3cuda_add_cli_test(cli.bench.invalid_number aruco3cuda_bench_cli FALSE --warmup abc --input /x.png)
aruco3cuda_add_cli_test(cli.bench.unknown_route aruco3cuda_bench_cli FALSE --route CUDA-Nope --input /x.png)
# 経路と memory 種別の組合せが不正なら失敗する。CUDA-E2E は host 入力の経路であり
# device 常駐入力 (M-Device) を受け付けない。この検査を欠くと、指定と異なる条件で
# 測ったまま気付かない。
aruco3cuda_add_cli_test(cli.bench.route_memory_mismatch aruco3cuda_bench_cli FALSE
  --route CUDA-E2E --memory-mode M-Device --warmup 0 --latency-iterations 1
  --throughput-frames 0 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.missing_image aruco3cuda_bench_cli FALSE
  --warmup 0 --latency-iterations 1 --input /nonexistent/image.png)
# 境界値: 遅延測定の回数が 0 は不正。
aruco3cuda_add_cli_test(cli.bench.zero_iterations aruco3cuda_bench_cli FALSE
  --warmup 0 --latency-iterations 0 --input /x.png)

# 経路名の解析。実装済みの CPU 以外も名前としては受理される必要がある。
aruco3cuda_add_cli_test(cli.bench.route_cpu aruco3cuda_bench_cli FALSE
  --route CPU --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.route_resident aruco3cuda_bench_cli FALSE
  --route CUDA-Resident --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.route_hybrid aruco3cuda_bench_cli FALSE
  --route Hybrid --warmup 0 --latency-iterations 1 --input /x.png)
# 検出設定を渡す経路。値の解析に誤りがあれば測定条件が記録と食い違う。
aruco3cuda_add_cli_test(cli.bench.detector_options aruco3cuda_bench_cli FALSE
  --dictionary DICT_6X6_250 --threads 2 --use-aruco3 0
  --min-marker-length-ratio 0.1 --min-side-length-canonical 48
  --warmup 0 --latency-iterations 1 --input /nonexistent/image.png)

# CPU 親和性。性能 core と効率 core が混在する機では固定しないと値が二極化する。
aruco3cuda_add_cli_test(cli.bench.invalid_cpu_list aruco3cuda_bench_cli FALSE
  --cpu-list 0,abc --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.empty_cpu_list aruco3cuda_bench_cli FALSE
  --cpu-list "" --warmup 0 --latency-iterations 1 --input /x.png)
aruco3cuda_add_cli_test(cli.bench.negative_cpu_list aruco3cuda_bench_cli FALSE
  --cpu-list -1 --warmup 0 --latency-iterations 1 --input /x.png)

add_test(NAME cli.bench.measure
  COMMAND $<TARGET_FILE:aruco3cuda_bench_cli>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
          --warmup 1 --latency-iterations 3 --throughput-frames 2 --save-samples --cpu-list 0
          --output "${CMAKE_CURRENT_BINARY_DIR}/cli-bench.jsonl")
set_tests_properties(cli.bench.measure PROPERTIES FIXTURES_REQUIRED cli_corpus TIMEOUT 300)

# --- 差分レポート tool ---
aruco3cuda_add_cli_test(cli.report.help aruco3cuda_report_cli TRUE --help)
aruco3cuda_add_cli_test(cli.report.no_input aruco3cuda_report_cli FALSE --fail-on-diff)
aruco3cuda_add_cli_test(cli.report.unknown_option aruco3cuda_report_cli FALSE --nonexistent-option)
aruco3cuda_add_cli_test(cli.report.missing_value aruco3cuda_report_cli FALSE --input)
aruco3cuda_add_cli_test(cli.report.invalid_tolerance aruco3cuda_report_cli FALSE
  --corner-tolerance-px abc --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.negative_tolerance aruco3cuda_report_cli FALSE
  --corner-tolerance-px -1 --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.invalid_match_radius aruco3cuda_report_cli FALSE
  --match-radius-ratio 0 --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.invalid_use_aruco3 aruco3cuda_report_cli FALSE
  --use-aruco3 yes --input /nonexistent.png)
aruco3cuda_add_cli_test(cli.report.missing_image aruco3cuda_report_cli FALSE
  --input /nonexistent/image.png)

# 正常系: 合成 corpus に対して CPU 基準と一致することを ctest から確認する。
# --fail-on-diff により、差異があれば test が失敗する。
add_test(NAME cli.report.matches_reference
  COMMAND $<TARGET_FILE:aruco3cuda_report_cli>
          --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
          --input "${kCliCorpusDir}/clean_1280x720_n4_s128.png"
          --input "${kCliCorpusDir}/degraded_1280x720.png"
          --output "${CMAKE_CURRENT_BINARY_DIR}/cli-report.json"
          --fail-on-diff)
set_tests_properties(cli.report.matches_reference PROPERTIES
  FIXTURES_REQUIRED cli_corpus TIMEOUT 300
  PASS_REGULAR_EXPRESSION "差異は無い")

aruco3cuda_add_cli_test(cli.report.unwritable_output aruco3cuda_report_cli FALSE
  --input "${kCliCorpusDir}/clean_1280x720_n1_s128.png"
  --output /nonexistent-dir/report.json)
set_tests_properties(cli.report.unwritable_output PROPERTIES FIXTURES_REQUIRED cli_corpus)


# 公開 API の Doxygen 要件を継続的に検査する。
# 宣言を追加したときに記載漏れへ気付けるようにする。
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME doc.doxygen_completeness
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/tools/check_doxygen.py"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set_tests_properties(doc.doxygen_completeness PROPERTIES TIMEOUT 120)
endif()

# 集計 script も CLI の一部として検査する。
find_package(Python3 COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_test(NAME cli.aggregate.summarize
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/bench/aggregate.py"
            "${CMAKE_CURRENT_BINARY_DIR}/cli-bench.jsonl")
  set_tests_properties(cli.aggregate.summarize PROPERTIES
    DEPENDS cli.bench.measure TIMEOUT 300
    PASS_REGULAR_EXPRESSION "測定結果")
  add_test(NAME cli.aggregate.rejects_broken_input
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/bench/aggregate.py"
            "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
  set_tests_properties(cli.aggregate.rejects_broken_input PROPERTIES WILL_FAIL TRUE TIMEOUT 300)
endif()
