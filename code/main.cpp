#include <opencv2/opencv.hpp>
#include <iostream>
#include <omp.h>
#include <filesystem>
#include <vector>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

const int TILE_SIZE = 128; // Used for CPU methods

int main() {
    // Update these paths to match your D: drive structure
string beforePath = "D:/Mtech/Projects/2nd Sem/Parallel computing/project/data/A/";
string afterPath  = "D:/Mtech/Projects/2nd Sem/Parallel computing/project/data/B/";
string outputPath = "D:/Mtech/Projects/2nd Sem/Parallel computing/project/data/output/";

    fs::create_directories(outputPath);

    vector<string> beforeFiles, afterFiles;
    for (const auto& entry : fs::directory_iterator(beforePath))
        beforeFiles.push_back(entry.path().string());
    for (const auto& entry : fs::directory_iterator(afterPath))
        afterFiles.push_back(entry.path().string());

    sort(beforeFiles.begin(), beforeFiles.end());
    sort(afterFiles.begin(), afterFiles.end());

    int n = min(beforeFiles.size(), afterFiles.size());
    cout << "Pre-loading " << n << " images to RAM to test pure CPU/GPU speed..." << endl;

    // PRE-LOAD IMAGES TO AVOID FILE I/O BOTTLENECK
    vector<Mat> beforeImages(n), afterImages(n);
    for(int k=0; k<n; k++){
        beforeImages[k] = imread(beforeFiles[k]);
        afterImages[k] = imread(afterFiles[k]);
        if(!beforeImages[k].empty() && !afterImages[k].empty()){
            resize(afterImages[k], afterImages[k], beforeImages[k].size());
        }
    }
    cout << "Images loaded. Starting benchmarks...\n" << endl;

    // ================= 1. SEQUENTIAL =================
    double start_seq = omp_get_wtime();
    for (int k = 0; k < n; k++) {
        if (beforeImages[k].empty() || afterImages[k].empty()) continue;
        Mat img1 = beforeImages[k];
        Mat img2 = afterImages[k];
        Mat change = img2.clone();

        for (int ti = 0; ti < img1.rows; ti += TILE_SIZE) {
            for (int tj = 0; tj < img1.cols; tj += TILE_SIZE) {
                int width = min(TILE_SIZE, img1.cols - tj);
                int height = min(TILE_SIZE, img1.rows - ti);
                Rect tileRect(tj, ti, width, height);

                Mat gray1, gray2, diff, thresh;
                cvtColor(img1(tileRect), gray1, COLOR_BGR2GRAY);
                cvtColor(img2(tileRect), gray2, COLOR_BGR2GRAY);
                
                GaussianBlur(gray1, gray1, Size(5, 5), 0);
                GaussianBlur(gray2, gray2, Size(5, 5), 0);
                absdiff(gray1, gray2, diff);
                threshold(diff, thresh, 60, 255, THRESH_BINARY);
                morphologyEx(thresh, thresh, MORPH_OPEN, Mat(), Point(-1, -1), 2);
                morphologyEx(thresh, thresh, MORPH_DILATE, Mat(), Point(-1, -1), 2);

                vector<vector<Point>> contours;
                findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                for (auto& contour : contours) {
                    if (contourArea(contour) > 1500) {
                        Rect box = boundingRect(contour);
                        Rect globalBox(box.x + tj, box.y + ti, box.width, box.height);
                        rectangle(change, globalBox, Scalar(0, 255, 0), 2);
                    }
                }
            }
        }
    }
    double seq_time = omp_get_wtime() - start_seq;

    // ================= 2. OPENMP (BATCH PARALLEL) =================
    double start_par = omp_get_wtime();
    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        if (beforeImages[k].empty() || afterImages[k].empty()) continue;
        Mat img1 = beforeImages[k];
        Mat img2 = afterImages[k];
        Mat change = img2.clone();

        for (int ti = 0; ti < img1.rows; ti += TILE_SIZE) {
            for (int tj = 0; tj < img1.cols; tj += TILE_SIZE) {
                int width = min(TILE_SIZE, img1.cols - tj);
                int height = min(TILE_SIZE, img1.rows - ti);
                Rect tileRect(tj, ti, width, height);

                Mat gray1, gray2, diff, thresh;
                cvtColor(img1(tileRect), gray1, COLOR_BGR2GRAY);
                cvtColor(img2(tileRect), gray2, COLOR_BGR2GRAY);
                
                GaussianBlur(gray1, gray1, Size(5, 5), 0);
                GaussianBlur(gray2, gray2, Size(5, 5), 0);
                absdiff(gray1, gray2, diff);
                threshold(diff, thresh, 60, 255, THRESH_BINARY);
                morphologyEx(thresh, thresh, MORPH_OPEN, Mat(), Point(-1, -1), 2);
                morphologyEx(thresh, thresh, MORPH_DILATE, Mat(), Point(-1, -1), 2);

                vector<vector<Point>> contours;
                findContours(thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
                for (auto& contour : contours) {
                    if (contourArea(contour) > 1500) {
                        Rect box = boundingRect(contour);
                        Rect globalBox(box.x + tj, box.y + ti, box.width, box.height);
                        rectangle(change, globalBox, Scalar(0, 255, 0), 2);
                    }
                }
            }
        }
    }
    double par_time = omp_get_wtime() - start_par;

    // ================= 3. GPU (OPENCV UMAT) =================
    // Notice: We don't use tiles here. We process the whole image at once.
    double start_gpu = omp_get_wtime();
    for (int k = 0; k < n; k++) {
        if (beforeImages[k].empty() || afterImages[k].empty()) continue;
        
        UMat u_img1, u_img2, u_gray1, u_gray2, u_diff, u_thresh;
        beforeImages[k].copyTo(u_img1); // Host to Device
        afterImages[k].copyTo(u_img2);  // Host to Device
        
        cvtColor(u_img1, u_gray1, COLOR_BGR2GRAY);
        cvtColor(u_img2, u_gray2, COLOR_BGR2GRAY);
        
        GaussianBlur(u_gray1, u_gray1, Size(5, 5), 0);
        GaussianBlur(u_gray2, u_gray2, Size(5, 5), 0);
        
        absdiff(u_gray1, u_gray2, u_diff);
        threshold(u_diff, u_thresh, 60, 255, THRESH_BINARY);
        
        morphologyEx(u_thresh, u_thresh, MORPH_OPEN, Mat(), Point(-1, -1), 2);
        morphologyEx(u_thresh, u_thresh, MORPH_DILATE, Mat(), Point(-1, -1), 2);

        // Download mask back to CPU to draw bounding boxes
        Mat cpu_thresh, change = afterImages[k].clone();
        u_thresh.copyTo(cpu_thresh); // Device to Host

        vector<vector<Point>> contours;
        findContours(cpu_thresh, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        for (auto& contour : contours) {
            if (contourArea(contour) > 1500) {
                Rect box = boundingRect(contour);
                rectangle(change, box, Scalar(0, 255, 0), 2);
            }
        }
    }
    double gpu_time = omp_get_wtime() - start_gpu;

    // ================= RESULTS =================
    cout << "----------------------------------" << endl;
    cout << "Sequential Time: " << seq_time << " sec" << endl;
    cout << "OpenMP Time:     " << par_time << " sec (Speedup: " << seq_time/par_time << "x)" << endl;
    cout << "GPU (UMat) Time: " << gpu_time << " sec (Speedup: " << seq_time/gpu_time << "x)" << endl;
    cout << "----------------------------------" << endl;

    // ================= GRAPH GENERATION =================
    int width = 800, height = 400;
    Mat graph(height, width, CV_8UC3, Scalar(255, 255, 255)); 

    double maxTime = max({seq_time, par_time, gpu_time});

    // Draw Grid
    for (int i = 0; i <= 6; i++) {
        int x = 150 + i * 90;
        line(graph, Point(x, 80), Point(x, 300), Scalar(200, 200, 200), 1);
        double val = (maxTime / 6) * i;
        putText(graph, to_string(val).substr(0, 4), Point(x - 15, 330), FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 0, 0), 1);
    }
    line(graph, Point(150, 80), Point(150, 300), Scalar(0, 0, 0), 2);
    line(graph, Point(150, 300), Point(700, 300), Scalar(0, 0, 0), 2);

    // Bars
    int seqWidth = (seq_time / maxTime) * 500;
    int parWidth = (par_time / maxTime) * 500;
    int gpuWidth = (gpu_time / maxTime) * 500;

    rectangle(graph, Point(150, 110), Point(150 + seqWidth, 150), Scalar(139, 0, 0), FILLED);
    rectangle(graph, Point(150, 180), Point(150 + parWidth, 220), Scalar(0, 100, 0), FILLED);
    rectangle(graph, Point(150, 250), Point(150 + gpuWidth, 290), Scalar(0, 0, 139), FILLED);

    // Labels
    putText(graph, "Sequential", Point(10, 135), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
    putText(graph, "OpenMP", Point(10, 205), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);
    putText(graph, "GPU", Point(10, 275), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 1);

    putText(graph, to_string(seq_time).substr(0, 5) + "s", Point(160 + seqWidth, 135), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(graph, to_string(par_time).substr(0, 5) + "s", Point(160 + parWidth, 205), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);
    putText(graph, to_string(gpu_time).substr(0, 5) + "s", Point(160 + gpuWidth, 275), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1);

    putText(graph, "Performance Comparison: Seq vs OpenMP vs GPU", Point(150, 40), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 0), 2);
    
    imwrite(outputPath + "final_performance_graph.png", graph);

    return 0;
}