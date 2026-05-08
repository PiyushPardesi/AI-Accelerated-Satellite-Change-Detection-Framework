#include "change_detection.h"

// ===== Sequential Execution =====
double runSequential(const string& path1, const string& path2, const string& output_prefix) {

    Mat img1 = imread(path1);
    Mat img2 = imread(path2);

    if(img1.empty() || img2.empty()) {
        cout << "Error loading images\n";
        return -1;
    }

    double start = (double)getTickCount();

    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    Mat diff;
    absdiff(gray1, gray2, diff);

    Mat mask;
    threshold(diff, mask, 0, 255, THRESH_BINARY | THRESH_OTSU);

    double end = (double)getTickCount();
    double time_taken = (end - start) / getTickFrequency();

    Mat overlay = createOverlay(img2, mask);
    Mat heatmap = createHeatmap(diff);
    Mat boxes = drawBoundingBoxes(img2, mask);

    Mat collage = createCollage(img1, img2, heatmap, overlay, boxes);
    imwrite("output/" + output_prefix + "_seq.png", collage);

    return time_taken;
}