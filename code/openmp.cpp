#include "change_detection.h"
#include <omp.h>

// ===== OpenMP Execution =====
double runOpenMP(const string& path1, const string& path2, const string& output_prefix) {

    Mat img1 = imread(path1);
    Mat img2 = imread(path2);

    if(img1.empty() || img2.empty()) {
        cout << "Error loading images\n";
        return -1;
    }

    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    Mat diff(gray1.size(), CV_8UC1);

    double start = omp_get_wtime();

    #pragma omp parallel for
    for(int i = 0; i < gray1.rows; i++) {
        for(int j = 0; j < gray1.cols; j++) {
            diff.at<uchar>(i,j) = abs(gray1.at<uchar>(i,j) - gray2.at<uchar>(i,j));
        }
    }

    Mat mask;
    threshold(diff, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);

    double end = omp_get_wtime();
    double time_taken = end - start;

    Mat overlay = createOverlay(img2, mask);
    Mat heatmap = createHeatmap(diff);
    Mat boxes = drawBoundingBoxes(img2, mask);

    Mat collage = createCollage(img1, img2, heatmap, overlay, boxes);
    imwrite("output/" + output_prefix + "_omp.png", collage);

    return time_taken;
}