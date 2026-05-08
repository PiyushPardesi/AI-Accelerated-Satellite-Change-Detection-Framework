#include "change_detection.h"

// ===== Absolute Difference =====
Mat computeAbsDiff(const Mat& img1, const Mat& img2) {
    Mat gray1, gray2, diff;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);
    absdiff(gray1, gray2, diff);
    return diff;
}

// ===== SSIM (simplified version) =====
Mat computeSSIMDiff(const Mat& img1, const Mat& img2) {
    Mat gray1, gray2;
    cvtColor(img1, gray1, COLOR_BGR2GRAY);
    cvtColor(img2, gray2, COLOR_BGR2GRAY);

    Mat diff;
    absdiff(gray1, gray2, diff);

    // normalize (simulate SSIM-like effect)
    normalize(diff, diff, 0, 255, NORM_MINMAX);

    return diff;
}

// ===== Otsu Threshold =====
Mat applyOtsuThreshold(const Mat& diff) {
    Mat binary;
    threshold(diff, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);
    return binary;
}

// ===== Overlay (Red Highlight) =====
Mat createOverlay(const Mat& original, const Mat& mask) {
    Mat output = original.clone();

    for(int i = 0; i < mask.rows; i++) {
        for(int j = 0; j < mask.cols; j++) {
            if(mask.at<uchar>(i,j) == 255) {
                output.at<Vec3b>(i,j) = Vec3b(0,0,255); // RED
            }
        }
    }
    return output;
}

// ===== Heatmap =====
Mat createHeatmap(const Mat& diff) {
    Mat heatmap;
    applyColorMap(diff, heatmap, COLORMAP_JET);
    return heatmap;
}

// ===== Bounding Boxes =====
Mat drawBoundingBoxes(const Mat& original, const Mat& mask) {
    Mat output = original.clone();

    vector<vector<Point>> contours;
    findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    for(auto& contour : contours) {
        if(contourArea(contour) > 100) { // ignore noise
            Rect box = boundingRect(contour);
            rectangle(output, box, Scalar(0,255,0), 2); // GREEN box
        }
    }
    return output;
}

Mat createCollage(const Mat& A, const Mat& B,
                  Mat heatmap, Mat overlay, Mat boxes) {

    Mat a = A.clone();
    Mat b = B.clone();

    resize(a, a, Size(300,300));
    resize(b, b, Size(300,300));
    resize(heatmap, heatmap, Size(300,300));
    resize(overlay, overlay, Size(300,300));
    resize(boxes, boxes, Size(300,300));

    if (heatmap.channels() == 1)
        cvtColor(heatmap, heatmap, COLOR_GRAY2BGR);

    Mat top, middle, bottom;

    hconcat(a, b, top);
    hconcat(heatmap, overlay, middle);

    Mat blank = Mat::zeros(boxes.size(), boxes.type());
    hconcat(boxes, blank, bottom);

    Mat collage;
    vconcat(top, middle, collage);
    vconcat(collage, bottom, collage);

    return collage;
}