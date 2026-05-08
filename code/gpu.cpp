#include "change_detection.h"
#include <opencv2/core/ocl.hpp>

// ===== GPU Execution =====
double runGPU(const string& path1, const string& path2, const string& output_prefix) {

    cv::ocl::setUseOpenCL(true);

    Mat temp1 = imread(path1);
    Mat temp2 = imread(path2);

    if(temp1.empty() || temp2.empty()) {
        cout << "Error loading images\n";
        return -1;
    }

    UMat img1, img2;
    temp1.copyTo(img1);
    temp2.copyTo(img2);

    UMat gray1, gray2, diff, mask;

    double start = (double)getTickCount();

    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    absdiff(gray1, gray2, diff);

    threshold(diff, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);

    double end = (double)getTickCount();
    double time_taken = (end - start) / getTickFrequency();

    Mat diff_cpu, mask_cpu;
    diff.copyTo(diff_cpu);
    mask.copyTo(mask_cpu);

    Mat overlay = createOverlay(temp2, mask_cpu);
    Mat heatmap = createHeatmap(diff_cpu);
    Mat boxes = drawBoundingBoxes(temp2, mask_cpu);

    Mat collage = createCollage(temp1, temp2, heatmap, overlay, boxes);
    imwrite("output/" + output_prefix + "_gpu.png", collage);

    return time_taken;
}