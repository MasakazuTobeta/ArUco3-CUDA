// SPDX-License-Identifier: Apache-2.0
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "apriltag.h"
#include "common/image_u8.h"
#include "tag36h11.h"

int main(int argc, char** argv) {
    const int kalibr_border = argc > 1 && strcmp(argv[1], "--kalibr-border") == 0;
    const int first_image = kalibr_border ? 2 : 1;
    if (argc <= first_image) {
        fprintf(stderr, "usage: video_reference [--kalibr-border] image.pgm [...]\n");
        return EXIT_FAILURE;
    }

    apriltag_family_t* family = tag36h11_create();
    if (kalibr_border) {
        // The Kalibr target adds one black cell on each side of the 36-bit payload.
        family->width_at_border += 2;
        family->total_width += 2;
        for (uint32_t bit = 0; bit < family->nbits; ++bit) {
            ++family->bit_x[bit];
            ++family->bit_y[bit];
        }
    }
    apriltag_detector_t* detector = apriltag_detector_create();
    apriltag_detector_add_family_bits(detector, family, 2);
    detector->nthreads = 1;
    detector->quad_decimate = 1.0f;
    detector->quad_sigma = 0.8f;
    detector->qtp.min_white_black_diff = 5;
    detector->refine_edges = 1;

    int status = EXIT_SUCCESS;
    for (int i = first_image; i < argc; ++i) {
        image_u8_t* image = image_u8_create_from_pnm(argv[i]);
        if (!image) {
            fprintf(stderr, "could not load PGM %s\n", argv[i]);
            status = EXIT_FAILURE;
            break;
        }
        zarray_t* detections = apriltag_detector_detect(detector, image);
        int count = zarray_size(detections);
        printf("frame %d %d\n", i - first_image, count);
        for (int j = 0; j < count; ++j) {
            apriltag_detection_t* tag;
            zarray_get(detections, j, &tag);
            printf("tag %d %d %d %.9g", i - first_image, tag->id, tag->hamming, tag->decision_margin);
            for (int corner = 0; corner < 4; ++corner) {
                printf(" %.17g %.17g", tag->p[corner][0], tag->p[corner][1]);
            }
            putchar('\n');
        }
        apriltag_detections_destroy(detections);
        image_u8_destroy(image);
    }

    apriltag_detector_destroy(detector);
    tag36h11_destroy(family);
    return status;
}
