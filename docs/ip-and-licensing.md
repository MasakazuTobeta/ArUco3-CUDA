# Intellectual Property and Licensing Policy

## Purpose

This document defines the policy for handling code and intellectual property from the official ArUco, OpenCV, the papers, and external implementations in this CUDA implementation of ArUco3, so that future commercial use and contributions to OpenCV are not obstructed.

> [!CAUTION]
> This document is a development risk-management policy, not legal advice. Before commercial release, product integration, or a large contribution to OpenCV, determine the countries involved and consult an intellectual property professional.

## Scope

- GPLv3 code of the official ArUco library
- Algorithms described in the ArUco3 paper
- OpenCV's ArUco implementation
- The ArUco Dictionary and evaluation data
- Patents, trademarks, papers, and figures

## Current state

- The wording on the official ArUco page of the Universidad de Córdoba was checked on 2026-08-29. The original text is as follows (the rights holder's personal mail address is withheld).

  > This software is licensed under GPLv3 license for personal, research and educational purposes. For a commercial license please contact [rights holder's contact]

- GPLv3 itself is not a license that categorically prohibits commercial use. Whether the wording above is intended as an additional condition on GPLv3, or merely points to a separate commercial license, is not settled by that wording alone. **This point remains unresolved.**
- However, **this project does not depend on that interpretation,** because it neither obtains, references, nor uses the official ArUco distribution. The implementation basis is limited to the 2018 paper and OpenCV 4.x, and the Dictionary also comes from OpenCV. Even if the wording above carries the stricter meaning, it does not reach the artifacts of this project.
- OpenCV is distributed under Apache-2.0, and OpenCV's ArUco3 support was merged after confirming through the contribution checklist that it is not based on GPL-incompatible code.
- However, **OpenCV 4.x uses different license headers on a per-file basis.** The project-wide LICENSE is Apache-2.0, but older files in `imgproc` retain 3-clause BSD headers and list Intel Corporation and others as copyright holders. The files whose behavior this project copied — `cornersubpix.cpp`, `samplers.cpp`, `thresh.cpp`, `imgwarp.cpp`, and `geometry.cpp` — are all 3-clause BSD. The per-file breakdown is in PR-005 of the [Code Provenance record](code-provenance.md).
- Current OpenCV 4.x offers not only the conventional method: `DetectorParameters::useAruco3Detection` enables the fast detection strategy of the 2018 paper.
- OpenCV 4.x includes `DICT_ARUCO_MIP_36h12` as a predefined Dictionary with 6x6 cells, 250 codes, and a minimum Hamming distance of 12.
- No file in this repository contains pasted source code from the official ArUco or from OpenCV. However, the `oracle` namespace in `test/reference/test_corner_refine.cpp` copies the arithmetic and evaluation order of `cv::cornerSubPix` and `cv::getRectSubPix` verbatim. This is so that whether the GPU implementation copied them incorrectly can be measured separately from machine differences. The source of the copy is 3-clause BSD, and **as a conservative measure the copyright notice and license text were placed in the `NOTICE` at the repository root.** Judging whether the copy legally constitutes a derivative work requires confirmation by a professional.
- The license of this repository is the Apache License 2.0. Contributions are also accepted under the same license unless there is an explicit exception.

## Distinguishing copyright from algorithms

In general, copyright protects concrete expression such as source code, object code, documents, and figures. Ideas, procedures, operation methods, mathematical concepts, and algorithms themselves are normally distinguished from what copyright can monopolize.

Therefore, if an algorithm published in a paper is understood and implemented with its own structure and code, without copying or adapting the expression of GPL code, that implementation is not normally considered to become a GPL derivative work. However, the following are avoided.

- Verbatim or substantial copying of GPL source code.
- Porting that only renames variables.
- Adaptation that preserves function decomposition, control structures, comments, and data representation with a high degree of similarity.
- Build or link configurations that distribute GPL code.
- Redistribution of copyrightable tables, images, or test data extracted from GPL code.

## Patents

Patents are separate from copyright and from open-source licenses. Even an independent implementation of an algorithm can be a problem if it falls within a patent claim that is in force in the country concerned.

This project is published under the Apache License 2.0. Under Section 3 of that license, contributors grant users a license to their own essential patents. This project holds no patents.

Before integration into a commercial product, determine the countries of intended sale and carry out at least the following.

1. Search patent families around the inventor names, university name, paper title, and priority date.
2. Confirm which claims are in force in the countries concerned.
3. Examine the marker Dictionary generation method and the detection method separately.
4. Decide whether a freedom-to-operate review by a professional is required.

**The results of the above investigation and any freedom-to-operate deliberation are not published in this repository.** Patent evaluation is the domain of legal professionals and is not the kind of thing to handle in a public document.

As a note to users, **the patent-law legitimacy of a product that uses this project must be confirmed by the user.** As stated in Sections 7 and 8 of the Apache License 2.0, this project is provided without warranty.

## Is it acceptable to publish this as OSS?

This is a separate question from whether it can be practiced as a commercial product (freedom-to-operate). The discussion here is limited to **publishing the source under Apache-2.0.**

### Copyright and licensing: no obstacles found

| Item checked | Status | Basis |
| --- | --- | --- |
| Contamination by official ArUco GPLv3 code | **None** | Neither referenced nor obtained. The implementation basis is the 2018 paper and OpenCV 4.x. Recorded per file and per hash in the [Code Provenance record](code-provenance.md) |
| Bundled third-party code (vendoring) | **None** | No `third_party` / `vendor` or similar directory exists. All dependencies are resolved from the environment via `find_package`. **This repository distributes not one byte of third-party code** |
| Licenses of dependencies | **All permissive** | OpenCV (a mix of Apache-2.0 and 3-clause BSD), GoogleTest (3-clause BSD), CUDA Toolkit (NVIDIA EULA. **Not redistributed**; built in the user's own environment) |
| Copyleft dependencies | **None** | As above |
| Origin of the Dictionary data | **OpenCV** | Converted from the output of the Apache-2.0 `getPredefinedDictionary()`. Not extracted from a GPLv3 distribution. Byte-level agreement with OpenCV is continuously verified by tests |
| Attribution for copied code | **Recorded** | The copyright notices and license text for the 5 3-clause BSD files are kept in `NOTICE` |
| SPDX markings | **Present in all source files** | Applying them to new files is required by the coding rules |

**Within this scope, nothing was found that prevents publishing the source under Apache-2.0.**

### Patents

**"It's OSS, so patents don't apply" does not hold.** Article 2(3) of the Japanese Patent Act includes the transfer and provision of a program within "working" the invention, and in the United States distribution likewise falls under 35 U.S.C. §271. Being free of charge does not determine whether it is working the invention.

This project holds no patents, and under Section 3 of the Apache License 2.0 the patent license from contributors extends to users. For the relationship with third-party patents, see "Patents" above.

### Trademarks: investigated. Class 9 does not designate computer programs in general

On 2026-08-29, TMview (the worldwide cross-jurisdiction database of EUIPO/TMDN) and the Toreru trademark search were opened in a headless browser, and all 209 worldwide records for `aruco` were retrieved.

**There is no `ARUCO` trademark in the United States.** There are 6 US registrations and applications, but each of them is a different mark: `JARUCO`, `DARUCOT`, `HARUCO`, `DRARUCO`, or `SUBARUCONNECT`. Zero marks matched `ARUCO` itself.

**In Japan there are 8 trademarks identical to `ARUCO` / `aruco`.** The 2 below, which have Class 9, are the ones that could relate to this project.

| Registration number | Mark | Rights holder | Classes | Expiration date |
| --- | --- | --- | --- | --- |
| No. 5509717 | ＡＲＵＣＯ | Gakken Holdings Co., Ltd. | 09, 16 | 2032-07-27 |
| No. 5711433 | ａｒｕｃｏ | Gakken Holdings Co., Ltd. | 09, 16, 39, 41, 43 | 2034-10-17 |

Both are trademarks in the "Chikyu no Arukikata" travel guide family, and have been transferred from Diamond-Big Co., Ltd. to Gakken Holdings (for No. 5711433, the transfer was applied for on 2020-12-10 and registered on 2021-01-22). A renewal was applied for on 2024-09-27, and **they are currently maintained.**

**The designated goods in Class 9 are as follows, and do not include computer programs in general.**

- No. 5509717, Class 9: 映写フィルム、スライドフィルム、スライドフィルム用マウント (cinematographic films, slide films, slide film mounts)
- No. 5711433, Class 9: 家庭用テレビゲーム機用プログラム、携帯用液晶画面ゲーム機用のプログラムを記憶させた電子回路及び CD-ROM、レコード、インターネットを利用して受信し及び保存することができる音楽ファイル、同じく画像ファイル、録画済みビデオディスク及びビデオテープ、映写フィルム、スライドフィルム、スライドフィルム用マウント、ダウンロード可能な電子書籍、電子出版物 (programs for home video game machines; electronic circuits and CD-ROMs storing programs for handheld games with liquid crystal displays; phonograph records; music files that can be received and stored via the Internet, and likewise image files; pre-recorded video discs and video tapes; cinematographic films, slide films, slide film mounts; downloadable electronic books and electronic publications)

**"電子計算機用プログラム" (computer programs) is designated in neither of them.** The programs that are designated are limited to those for home video game machines and handheld game machines.

The other identical trademarks in Japan belong to Rohto Pharmaceutical (No. 6805531, Class 35, issuance and management of points exchangeable for prizes for the purpose of health promotion), Nishigaki Socks (No. 6109871, Class 25, socks), and Gakken Holdings (No. 5326799, Classes 39, 41, and 43). Gakken Holdings filed 3 new applications in Classes 18, 24, and 25 on 2026-04-17, which are under examination.

**No trademark held by the Universidad de Córdoba or by the authors of ArUco was found among any of the 209 worldwide records.**

### Conclusion on the Gakken Holdings registrations

**On 2026-08-29 it was judged that the Gakken Holdings registrations are not an obstacle to this project,** because the designated goods in Class 9 are limited to programs for home and handheld game machines and to publications and video/music files, and do not include computer programs in general. **This is this project's own judgment, not a professional opinion.**

### The remaining issue is an unregistered trademark on the originators' side

Since registrations pose no obstacle, what remains is **protection as an unregistered trademark** (Article 2(1)(i) and (ii) of the Unfair Competition Prevention Act). **The party that could be at issue is not Gakken Holdings but the originators of `ArUco`.** In the field of computer vision, `ArUco` is widely understood as the name of a marker scheme, and there is room for it to be treated as a well-known indication even without registration.

The material bearing on this is as follows.

- **The Universidad de Córdoba and the 5 authors hold no trademark among any of the 209 worldwide records.** There is no registration-based foundation for asserting rights.
- **OpenCV itself has long used `aruco` as a module name.** The fact that a major Apache-2.0 library continues to publish a module of that name shows that in this field the name functions as a description of a technical scheme.
- As stated in policy 9, this project uses the name only to indicate the compatibility target and the technical scheme, and does not suggest affiliation or endorsement. The README and `NOTICE` state this.

**Even so, no evaluation under the Unfair Competition Prevention Act has been performed.** The above is material, not a conclusion.

### Limits of the trademark investigation

- The 209 records are those held by TMview. Registrations from offices that do not participate in TMview are not included.
- The sources for this section are TMview and Toreru; **no primary confirmation was made on J-PlatPat.** At the time of the investigation, J-PlatPat could not be operated due to a maintenance outage and a failure of the SPA to render.
- The similarity of designated goods was not compared using similar group codes.

### Conclusions

**In terms of copyright and licensing, the project is in a publishable state. As for trademarks, no registration was found that prevents use of the name, and the registrations that were found (Gakken Holdings) were judged not to be an obstacle to this project.**

Two things remain.

1. **The part of the patent question that does not disappear even with OSS publication.** As stated in "Patents" above, the relationship with third-party patents is not handled in this repository.
2. **An unregistered trademark on the originators' side.** No registration exists, but no evaluation under the Unfair Competition Prevention Act has been performed.

## Handling of the Dictionary

Regarding "generation" of `DICT_ARUCO_MIP_36h12`, the following operations are distinguished.

| Operation | Possible in OpenCV 4.x? | Handling in this project |
| --- | --- | --- |
| Obtain the predefined `DICT_ARUCO_MIP_36h12` | Yes. Use `getPredefinedDictionary()` | Treated as the authoritative source for the CPU baseline and compatibility data |
| Generate a marker image for a given ID | Yes. Use `Dictionary::generateImageMarker()` | Can be used to generate test fixtures |
| Generate a new custom Dictionary | Yes. `extendDictionary()` can be used | Treated as distinct from the predefined MIP Dictionary |
| Solve the MILP and regenerate the same set as the predefined MIP Dictionary | Implementable in principle from the paper, but the OpenCV API does not guarantee reproduction of the same set | Out of the initial scope. Even with the solver, constraints, seed, and tie-break fixed, byte-level agreement is verified separately |

When converting a predefined Dictionary into CUDA constant memory or the like, a test that compares byte by byte against OpenCV's `bytesList` for all 4 rotations is mandatory. See the [Dictionary policy](dictionaries.md) for details.

### Information sources that may be used

- Algorithms, formulas, and evaluation conditions described in the ArUco3 paper.
- OpenCV's public API and observable inputs and outputs.
- Apache-2.0 OpenCV source code, provided that the parts used or modified and the notices are recorded.
- Third-party implementations under permissive licenses, with the license and code provenance recorded.
- General image processing algorithms and CUDA programming techniques.

### Information sources that are not used

- Referring to implementation details of the official ArUco GPLv3 code for the purpose of porting.
- Gists, forum attachments, and generated code whose license is unknown.
- Proprietary code whose commercial license terms are unknown.

### Compatibility evaluation

Running the official ArUco or OpenCV as an executable and comparing outputs for the same input is treated separately from copying source code. During evaluation, the version, license, execution command, and settings are recorded.

## Code provenance record

For each implementation unit or PR, the following is recorded.

| Item | Content |
| --- | --- |
| Implementation | Target module / file |
| Basis | Paper, specification, original design, reference implementation |
| Source version | DOI, URL, repository commit |
| License | Apache-2.0, MIT, BSD, etc. |
| Reused expression | Whether any code / table was copied or modified |
| Patent review | Not performed, quick search done, confirmed by a professional |

## Operating under the Apache License 2.0

- The `LICENSE` at the repository root is authoritative.
- New source files carry `SPDX-License-Identifier: Apache-2.0`.
- When incorporating Apache-2.0 code, preserve the original copyright, patent, trademark, and attribution notices.
- When distributing third-party code that has a `NOTICE` upstream, add the required attribution to this repository's `NOTICE`.
- When behavior is copied from a file carrying a 3-clause BSD header, the copyright notice, conditions, and disclaimer are likewise kept in `NOTICE`. Assuming Apache-2.0 alone would drop this preservation obligation.
- Do not use the names of copyright holders to suggest promotion or endorsement of this project. This follows from clause 3 of the 3-clause BSD.
- Contributions follow Section 5 of the Apache License 2.0.

## Goals

- Make the entire repository publishable under the Apache License 2.0.
- Include no official ArUco GPLv3 code, and be able to explain through code provenance that this is not a GPL derivative work.
- Ensure that code contributed to OpenCV satisfies the Apache-2.0 contribution checklist.
- Complete the patent clearance required before commercial use.

## Open questions

- The relationship between the commercial-use wording on the official ArUco page and the text of GPLv3. The wording itself was confirmed on 2026-08-29. The interpretation remains unresolved, but this project does not depend on it because it does not use the official distribution.
- Whether attribution for the predefined Dictionary obtained from OpenCV should be stated in `NOTICE` or in the source headers. At present it is stated in `NOTICE`.
- Evaluation of **protection as an unregistered trademark on the originators' side** (Unfair Competition Prevention Act) for `ArUco`. It was confirmed on 2026-08-29 that no registration exists anywhere in the world. On the same day, the Gakken Holdings registrations were judged not to be an obstacle to this project.

## See also

- [Code Provenance record](code-provenance.md)
- [Dictionary policy](dictionaries.md)

## References

- [Universidad de Córdoba: ArUco](https://www.uco.es/investiga/grupos/ava/portfolio/aruco/)
- [WIPO: Copyright](https://www.wipo.int/en/web/copyright)
- [U.S. Copyright Office: Computer Programs](https://www.copyright.gov/register/tx-programs.html)
- [OpenCV Issue: ArUco is now GPLv3](https://github.com/opencv/opencv_contrib/issues/2242)
- [OpenCV Issue #27118](https://github.com/opencv/opencv/issues/27118)
- [OpenCV: DetectorParameters](https://docs.opencv.org/4.x/d1/dcd/structcv_1_1aruco_1_1DetectorParameters.html)
- [OpenCV: Dictionary](https://docs.opencv.org/4.x/d5/d0b/classcv_1_1aruco_1_1Dictionary.html)
- [OpenCV PR #3151: ArUco3 speedup](https://github.com/opencv/opencv_contrib/pull/3151)
- [OpenCV: predefined_dictionaries.hpp](https://github.com/opencv/opencv/blob/4.x/modules/objdetect/src/aruco/predefined_dictionaries.hpp)
