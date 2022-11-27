/**
 * @file MotionInterpolation.cpp
 * @author Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-02-11
 * @copyright Copyright (c) 2022
 */

// TODO - add if to the malloc

#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

// Set the number of arguments required
#define R_ARGS 7
// Set the precision of the motionImage
#define MOTION_PRES 0.05f

// Set the parameters for the optical flow
// Default PATCH_SIZE: 9 - SEARCH_SIZE: 17 - FILTER_FLOW: 6
#define PATCH_SIZE 9
#define SEARCH_SIZE 7
#define FILTER_FLOW 3

#define my_sizeof(type) ((char *)(&type + 1) - (char *)(&type))

using namespace cv;

struct OpticalVector
{
    int x;
    int y;
};

// Prototype to cast Mat to uchar*
void matToUchar(Mat, uchar *, int, int, int = 3);

// Prototype to cast uchar to Mat
void ucharToMat(uchar *, Mat, int, int, int = 3);

// Prototype to cast float* to Mat
void floatToMat1f(float *, Mat1f, int, int);

// Prorotype to clone a frame
void cloneFrame(uchar *, uchar *, int, int, int = 3);

// Prorotype to create an empty frame
void createFrame(uchar *, int, int, int = 3);

// Prototype to get the luminance
float *getLuminance(uchar *, int, int, int = 3);

// Prototype for the motionImage
float *getMotionImage(float *, float *, int, int);

// Prototype for the optical flow
timeval getOpticalFlow(uchar *, uchar *, OpticalVector *, int, int, int = 3, int = 0);

// Prototype for the blur effect
void blurFrame(uchar *, uchar *, OpticalVector *, int, int, int = 3, int = 0);

// Prototype for the interpolation of frame
timeval interpolateFrames(uchar *, uchar *, uchar *, OpticalVector *, int, int, int, int = 0);

// Prototype for the print the progress
void printProgressBar(int, int, timeval, timeval, int, int);

// Prototype for the write the inform
void writeInform(char *, int, int, int, int, timeval, timeval, int, int);

// Prototype to export the generated frames
void exportFrames(char *, int, Mat, Mat, OpticalVector *, int, int);

// Prototype for the interpolation of video
timeval interpolateVideo(VideoCapture, char *, char *, int = 0, bool = false, int = 0, int = 0);

// CUDA kernels

__global__ void getOpticalFlowCUDA(float *lFrame1, float *lFrame2, float *motFrame, OpticalVector *optFlow, int width, int height, int nThreads)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    // Position variables to get the optical flow
    int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
    int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1;
    // Get start position in i and j
    int i = (startPos / width), j = (startPos % width);

    // Declare the variables for the optical flow
    float fPatchDifferenceMax = INFINITY;
    int searchVectorX = 0, searchVectorY = 0;
    float fAccumDif = 0.0f;
    int patchPixelX = 0, patchPixelY = 0;
    int basePixelX = 0, basePixelY = 0;
    float fPatchPixel = 0.0f, fBasePixel = 0.0f;

    // Iterate over each pixel of the image to get the Optical Flow
    for (; startPos <= endPos; startPos++)
    {
        // Initialize the vector
        optFlow[i * width + j].x = 0;
        optFlow[i * width + j].y = 0;
        if (motFrame[i * width + j] > 0)
        {
            // Initialize the variables
            fPatchDifferenceMax = INFINITY;

            // Search over a rectangular area for a patch of the image
            //  Sx and Sy indicates the area of search
            for (int sy = 0; sy < SEARCH_SIZE; sy++)
            {
                for (int sx = 0; sx < SEARCH_SIZE; sx++)
                {
                    // vectors that iterate over the search square
                    searchVectorX = j + (sx - SEARCH_SIZE / 2);
                    searchVectorY = i + (sy - SEARCH_SIZE / 2);

                    // Variable to store the difference of the patch
                    fAccumDif = 0.0f;

                    // Iterate over the patch with px and py
                    for (int py = 0; py < PATCH_SIZE; py++)
                    {
                        for (int px = 0; px < PATCH_SIZE; px++)
                        {
                            // Iterate over the patch
                            patchPixelX = searchVectorX + (px - PATCH_SIZE / 2);
                            patchPixelY = searchVectorY + (py - PATCH_SIZE / 2);

                            // Iterate over the patch of the original pixel
                            basePixelX = j + (px - PATCH_SIZE / 2);
                            basePixelY = i + (py - PATCH_SIZE / 2);

                            // Get adjacent values for each patch checking that is inside the image
                            fPatchPixel = 0.0f;
                            fBasePixel = 0.0f;
                            if (patchPixelX >= 0 && patchPixelX < width && patchPixelY >= 0 && patchPixelY < height)
                                fPatchPixel = lFrame2[patchPixelY * width + patchPixelX];

                            if (basePixelX >= 0 && basePixelX < width && basePixelY >= 0 && basePixelY < height)
                                fBasePixel = lFrame1[basePixelY * width + basePixelX];

                            // Accumulate difference
                            fAccumDif += fabs(fPatchPixel - fBasePixel);
                        }
                    }

                    // Record the vector offset for the least different search patch
                    if (fAccumDif <= fPatchDifferenceMax)
                    {
                        fPatchDifferenceMax = fAccumDif;
                        optFlow[i * width + j].x = searchVectorX - j;
                        optFlow[i * width + j].y = searchVectorY - i;
                    }
                }
            }
        }
        j += 1;
        if (j == width)
        {
            i += 1;
            j = 0;
        }
    }
}

int main(int argc, char **argv)
{
    // Declare the variables for time measurement
    struct timeval runtime = (struct timeval){0};
    // Declare the strings of load and save video
    char *path;
    char *loadName, *saveName;
    // Declare the Matrix to store the image
    VideoCapture loadVideo, saveVideo;
    // Declare the number of blocks and threads
    int nBlocks;
    int nThreads;
    // Declare the variable to save the frames
    int framesRend = 0;
    bool expFrames = false;
    // Declare the FILE to write the times
    FILE *fp;
    // Check if the number of arguments is correct
    if (argc < R_ARGS + 1)
    {
        printf("Usage: ./MotionInterpolation <Path_to_files> <Load_video_name> <Save_video_name> <Frames to render (0 - all)> <Export_frames ( 0:false, 1:true )> <nBlocks> <nThreads>\n");
        return -1;
    }
    // Get the paths
    path = argv[1];
    loadName = argv[2];
    saveName = argv[3];
    framesRend = atoi(argv[4]);
    expFrames = (atoi(argv[5]) == 0) ? false : true;
    nBlocks = atoi(argv[6]);
    nThreads = atoi(argv[7]);

    // Load the video from the path
    loadVideo = VideoCapture(std::string(path) + loadName);

    // Check video opened successfully
    if (!loadVideo.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // Interpolate the video and get the runtime
    runtime = interpolateVideo(loadVideo, path, saveName, framesRend, expFrames, nBlocks, nThreads);

    // Imprimir informe
    printf("------------------------------------------------------------------------------\n");
    printf("Runtime: %ld.%06ld s \n", (long int)runtime.tv_sec, (long int)runtime.tv_usec);

    fp = fopen((std::string(path) + "totalTime.csv").c_str(), "a");
    if (fp == NULL)
    {
        printf("Error opening the file \n");
        exit(1);
    }
    fprintf(fp, "%d,%d,%d,%d,%ld.%06ld\n", (int)loadVideo.get(CAP_PROP_FRAME_WIDTH), (int)loadVideo.get(CAP_PROP_FRAME_HEIGHT), nBlocks, nThreads, (long int)runtime.tv_sec, (long int)runtime.tv_usec);

    fclose(fp);
    destroyAllWindows();

    return 0;
}

// Function to cast Mat to uchar*
void matToUchar(Mat frame, uchar *uFrame, int width, int height, int channels)
{
    // Make a copy of the values into the array of uchars
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                uFrame[(ch * width * height) + (i * width + j)] = frame.at<Vec3b>(i, j)[ch];
}

// Function to cast uchar to Mat
void ucharToMat(uchar *uFrame, Mat frame, int width, int height, int channels)
{
    // Create the Mat of 3 channels
    // TODO: Make it for n channels
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                frame.at<Vec3b>(i, j)[ch] = uFrame[(ch * width * height) + (i * width + j)];
}

// Function to cast float* to Mat
void floatToMat1f(float *fFrame, Mat1f frame, int width, int height)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            frame.at<float>(i, j) = fFrame[i * width + j];
}

// Function to clone a frame
void cloneFrame(uchar *originFrame, uchar *destFrame, int width, int height, int channels)
{
    // Create multidimensional array for the N three channels
    // Make a copy of the values into the array of uchars
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                destFrame[(ch * width * height) + (i * width + j)] = originFrame[(ch * width * height) + (i * width + j)];
}

// Function to create an empty frame
void createFrame(uchar *frame, int width, int height, int channels)
{
    // Create multidimensional array for the N three channels
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                frame[(ch * width * height) + (i * width + j)] = 0;
}

// Function to get the motionImage
float *getMotionImage(float *lFrame1, float *lFrame2, int width, int height)
{
    // Declare image for the result
    float *motionImage = (float *)malloc(width * height * sizeof(float));
    // Declare the variable to store the luminance differences
    float fDiff = 0;
    // Substract the luminances
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            fDiff = fabs((float)lFrame1[i * width + j] - (float)lFrame2[i * width + j]);
            motionImage[i * width + j] = (fDiff >= MOTION_PRES) ? fDiff : 0.0f;
        }
    }
    return motionImage;
}

// Function to get the luminance
float *getLuminance(uchar *frame, int width, int height, int channels)
{
    // Declare the matrix to store the luminance
    float *lFrame = (float *)malloc(width * height * sizeof(float));
    // Variables to store the results
    float fB = 0.0f;
    float fG = 0.0f;
    float fR = 0.0f;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // Get the values from each pixel
            fB = (float)frame[i * width + j] / 255.0;
            fG = (float)frame[(width * height) + (i * width + j)] / 255.0;
            fR = (float)frame[2 * (width * height) + (i * width + j)] / 255.0;

            lFrame[i * width + j] = 0.2987f * fR + 0.5870f * fG + 0.1140f * fB;
        }
    }
    return lFrame;
}

// Function to get the optical flow
timeval getOpticalFlow(uchar *frame1, uchar *frame2, OpticalVector *optFlow, int width, int height, int channels, int nBlocks, int nThreads)
{
    static int size = width * height;
    cudaError_t err = cudaSuccess;
    // Get the luminance of the two frames
    float *lFrame1 = getLuminance(frame1, width, height, channels);
    float *lFrame2 = getLuminance(frame2, width, height, channels);

    // Get the motionFrame of the frames
    float *motFrame = getMotionImage(lFrame1, lFrame2, width, height);

    // Create the opticalflow for the device
    struct OpticalVector *d_optFlow;

    // Create the luminance Matrix for each frame
    float *d_lFrame1, *d_lFrame2;

    // Create the matrix for the Motion
    float *d_motFrame;

    // Optical Flow
    // Allocate the optical flow in the device
    err = cudaMalloc((void **)&d_optFlow, size * sizeof(struct OpticalVector));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device OpticalFlow (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Luminance Frame 1
    // Allocate the luminance Frame1 in the device
    err = cudaMalloc((void **)&d_lFrame1, size * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device lumFrame1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the data from de host to the device
    err = cudaMemcpy(d_lFrame1, lFrame1, size * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy lumFrame1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Luminance Frame 2
    // Allocate the luminance Frame2 in the device
    err = cudaMalloc((void **)&d_lFrame2, size * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device lumFrame1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the data from de host to the device
    err = cudaMemcpy(d_lFrame2, lFrame2, size * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy lumFrame2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Motion Frame
    // Allocate the luminance Frame2 in the device
    err = cudaMalloc((void **)&d_motFrame, size * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device lumFrame1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the data from de host to the device
    err = cudaMemcpy(d_motFrame, motFrame, size * sizeof(float), cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy motFrame from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Declare the variables to measure time
    struct timeval tval_before = (struct timeval){0}, tval_after = (struct timeval){0}, tval_result = (struct timeval){0};

    /*Medición de tiempo de inicio*/
    gettimeofday(&tval_before, NULL);

    getOpticalFlowCUDA<<<nBlocks, nThreads>>>(d_lFrame1, d_lFrame2, d_motFrame, d_optFlow, width, height, nBlocks * nThreads);

    cudaDeviceSynchronize();

    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch getOpticalFlow kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /*Medición de tiempo de finalización*/
    gettimeofday(&tval_after, NULL);

    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);

    // printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    // Copy optFlow back to Host
    err = cudaMemcpy(optFlow, d_optFlow, size * sizeof(struct OpticalVector), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy optFlow from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free device global memory
    // Optical Flow
    err = cudaFree(d_optFlow);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device OpticalFlow (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Luminance Frame1
    err = cudaFree(d_lFrame1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device lumFrame1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Luminance Frame2
    err = cudaFree(d_lFrame2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device lumFrame2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free the memory
    free(lFrame1);
    free(lFrame2);
    free(motFrame);

    return tval_result;
}

void blurFrame(uchar *frame, uchar *resFrame, OpticalVector *opticalFlow, int width, int height, int channels, int nThreads)
{
    static float kernel[9] =
        {1 / 16.0, 1 / 8.0, 1 / 16.0,
         1 / 8.0, 1 / 4.0, 1 / 8.0,
         1 / 16.0, 1 / 8.0, 1 / 16.0};

    static int kSize = (int)sqrt(my_sizeof(kernel) / my_sizeof(kernel[0]));

    float conv[channels];
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (abs(opticalFlow[i * width + j].x) + abs(opticalFlow[i * width + j].y) > FILTER_FLOW)
            {
                if (i > kSize && i < height - kSize && j > kSize && j < width - kSize)
                {
                    for (int ch = 0; ch < channels; ch++)
                    {
                        conv[ch] = 0.0;
                        for (int i1 = 0; i1 < kSize; i1++)
                        {
                            for (int j1 = 0; j1 < kSize; j1++)
                            {
                                conv[ch] += kernel[i1 * kSize + j1] * frame[(ch * width * height) + ((i + i1 - kSize / 2) * width + (j + j1 - kSize / 2))];
                            }
                        }
                        // Check if the value is correct
                        if (conv[ch] > 255)
                            conv[ch] = 255;
                        if (conv[ch] < 0)
                            conv[ch] = 0;
                        resFrame[(ch * width * height) + (i * width + j)] = (int)conv[ch];
                    }
                }
            }
        }
    }
}

timeval interpolateFrames(uchar *frame1, uchar *frame2, uchar *resFrame, OpticalVector *optFlow, int width, int height, int channels, int nBlocks, int nThreads)
{
    // Declare the variables to measure time
    struct timeval tval_before = (struct timeval){0}, tval_after = (struct timeval){0}, tval_result = (struct timeval){0}, tval_optf = (struct timeval){0}, tval_total = (struct timeval){0};

    // Get start time
    gettimeofday(&tval_before, NULL);

    // Declare the variable for the interpolation
    int linearDiv = 2;
    // Declare the Matrix for the intermediate frame
    uchar *interFrame1 = (uchar *)malloc(width * height * channels * sizeof(uchar));
    uchar *interFrame2 = (uchar *)malloc(width * height * channels * sizeof(uchar));
    uchar *joinFrame = (uchar *)malloc(width * height * channels * sizeof(uchar));

    if (interFrame1 == NULL || interFrame2 == NULL || joinFrame == NULL)
    {
        printf("Failed to allocate the inter frames \n");
        exit(1);
    }

    cloneFrame(frame1, interFrame1, width, height, channels);
    cloneFrame(frame2, interFrame2, width, height, channels);
    createFrame(joinFrame, width, height, channels);

    // Get End time
    gettimeofday(&tval_after, NULL);

    // Calculate total time
    timersub(&tval_after, &tval_before, &tval_result);

    // Get the Optical Flow
    tval_optf = getOpticalFlow(frame1, frame2, optFlow, width, height, channels, nBlocks, nThreads);

    // Add the optflow time
    timeradd(&tval_result, &tval_optf, &tval_total);



    // Get start time
    gettimeofday(&tval_before, NULL);

    // Create the new frame interpolating the optical Flow
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            // Check if the values are inside the frame
            if (j + optFlow[i * width + j].x >= 0 && j + optFlow[i * width + j].x < width &&
                i + optFlow[i * width + j].y >= 0 && i + optFlow[i * width + j].y < height)
            {
                for (int ch = 0; ch < channels; ch++)
                {
                    int i1 = i + (int)(optFlow[i * width + j].y / linearDiv);
                    int j1 = j + (int)(optFlow[i * width + j].x / linearDiv);
                    // Interpolate using the information of the frame 1
                    interFrame1[(ch * width * height) + (i1 * width + j1)] = frame1[(ch * width * height) + (i * width + j)];
                    // Interpolate using the information of the frame 2
                    interFrame2[(ch * width * height) + (i1 * width + j1)] = frame2[(ch * width * height) + (i * width + j)];
                }
            }
    // Join frames into a result Frame

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            for (int ch = 0; ch < channels; ch++)
                joinFrame[(ch * width * height) + (i * width + j)] = (interFrame1[(ch * width * height) + (i * width + j)] + interFrame2[(ch * width * height) + (i * width + j)]) / 2;
    // Apply the blur filter over the join frame
    cloneFrame(joinFrame, resFrame, width, height, channels);
    blurFrame(joinFrame, resFrame, optFlow, width, height, channels, nThreads);

    // Get End time
    gettimeofday(&tval_after, NULL);

    // Calculate total time
    timersub(&tval_after, &tval_before, &tval_result);

    // Add the time
    timeradd(&tval_result, &tval_total, &tval_total);

    // Free the memory
    free(interFrame1);
    free(interFrame2);
    free(joinFrame);

    return tval_total;
}

void printProgressBar(int iterFrame, int frameCount, timeval tval_result, timeval runtime, int nBlocks, int nThreads)
{
    float progrss_100 = (int)((float)iterFrame / (float)frameCount * 100);
    float progress_20 = (int)((float)iterFrame / (float)frameCount * 20);
    float framesPSec = 1.0 / ((float)tval_result.tv_sec + ((float)tval_result.tv_usec / 1000000.0));

    // Clear the console
    // system("clear");

    std::cout << "Frame: " << iterFrame << " / " << frameCount << "   [" << framesPSec << " frames / s";
    std::cout << "] [";
    for (int i = 0; i < progress_20 - 1; i++)
    {
        std::cout << "=";
    }
    std::cout << ">";
    for (int i = progress_20 + 1; i < 20; i++)
    {
        std::cout << " ";
    }
    std::cout << "]   " << progrss_100 << "%" << std::endl
              << "ET: ";

    printf("%ld.%06ld s     ", (long int)runtime.tv_sec, (long int)runtime.tv_usec);
    printf("    ET/F: %ld.%06ld s   NBlocks: %d     NThreads:  %d   \n \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec, nBlocks, nThreads);
}

// Function to the write the inform
void writeInform(char *path, int width, int height, int iterFrame, int frameCount, timeval tval_result, timeval runtime, int nBlocks, int nThreads)
{
    // Declare the FILE to write the times
    FILE *fp;

    float framesPSec = 1.0 / ((float)tval_result.tv_sec + ((float)tval_result.tv_usec / 1000000.0));

    fp = fopen((std::string(path) + "frameTime.csv").c_str(), "a");
    if (fp == NULL)
    {
        printf("Error opening the file \n");
        exit(1);
    }
    fprintf(fp, "%d,%d,%d,%d,%d,%f,%ld.%06ld\n", iterFrame, width, height, nBlocks, nThreads, framesPSec, (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
    fclose(fp);
}

// Function to export the generated frames
void exportFrames(char *path, int iterFrame, Mat frame, Mat genFrame, OpticalVector *optFlow, int width, int height)
{
    // writing the image to a defined location as png
    if (imwrite((std::string(path) + "frames/frame000" + std::to_string(iterFrame) + ".jpg").c_str(), genFrame) == false)
    {
        std::cout << "Exporting the generated frame, FAILED" << std::endl;
        exit(-1);
    }
    // Drawing the optical flow into the frame
    for (int i = 0; i < height; i += 5)
    {
        for (int j = 0; j < width; j += 5)
        {
            Point p0(j, i);
            Point p1(j + optFlow[i * width + j].x, i + optFlow[i * width + j].y);
            if (abs(optFlow[i * width + j].x) + abs(optFlow[i * width + j].y) > 0)
            {
                line(frame, p0, p1, Scalar(0, 255, 0), 1, LINE_AA);
                circle(frame, p0, 1, Scalar(0, 0, 255), FILLED, LINE_8);
            }
        }
    }
    // writing the image to a defined location as png
    if (imwrite((std::string(path) + "frames/optFlow000" + std::to_string(iterFrame) + ".jpg").c_str(), frame) == false)
    {
        std::cout << "Exporting the optical flow frame, FAILED" << std::endl;
        exit(-1);
    }
}

timeval interpolateVideo(VideoCapture loadVideo, char *path, char *saveName, int framesRend, bool expFrames, int nBlocks, int nThreads)
{
    // Declare the variables for time measurement
    struct timeval tval_result = (struct timeval){0}, runtime = (struct timeval){0};

    // Declare the variables for the frames
    Mat oldFrame, newFrame, saveFrame;

    // Get the frame count, fps, width, height and number of channels
    int frameCount = framesRend == 0 ? loadVideo.get(CAP_PROP_FRAME_COUNT) - 1 : framesRend;
    int iterFrame = 0, fps = loadVideo.get(CAP_PROP_FPS);
    int width = loadVideo.get(CAP_PROP_FRAME_WIDTH), height = loadVideo.get(CAP_PROP_FRAME_HEIGHT);
    int channels = 3;

    VideoWriter saveVideo(std::string(path) + saveName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 2 * fps, Size(width, height));

    // Print the video information
    std::cout << "------------------------------------------------------------------------" << std::endl;
    std::cout << "                          Motion interpolation                          " << std::endl;
    std::cout << "------------------------------------------------------------------------" << std::endl;

    std::cout << "Video resolution: " << width << "px * " << height << "px" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Total frames: " << frameCount << std::endl;
    std::cout << "Number of blocks: " << nBlocks << std::endl;
    std::cout << "Number of Threads: " << nThreads << std::endl;

    std::cout << "------------------------------------------------------------------------" << std::endl;

    while (loadVideo.read(newFrame))
    {
        // Write the original frame

        if (iterFrame == 0)
        {
            saveVideo.write(newFrame);
            channels = newFrame.channels();
        }

        else if (iterFrame <= frameCount)
        {
            // Create uchar matrix for each frame
            uchar *uFrameOld = (uchar *)malloc(width * height * channels * sizeof(uchar));
            uchar *uFrameNew = (uchar *)malloc(width * height * channels * sizeof(uchar));
            uchar *interFrame = (uchar *)malloc(width * height * channels * sizeof(uchar));

            // Load the data from the Mat to the uchar matrix
            matToUchar(oldFrame, uFrameOld, width, height, channels);
            matToUchar(newFrame, uFrameNew, width, height, channels);
            // Create the uchar matrix of the new frame
            createFrame(interFrame, width, height, channels);
            // Create the frame Mat to export
            Mat saveFrame = Mat::zeros(Size(width, height), CV_8UC3);

            // Allocate the opticalFlow
            struct OpticalVector *optFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));

            // Interpolate the frames


            tval_result = interpolateFrames(uFrameOld, uFrameNew, interFrame, optFlow, width, height, channels, nBlocks, nThreads);


            // Convert the uchar matrix to Mat
            ucharToMat(interFrame, saveFrame, width, height, channels);


            // Update runtime
            timeradd(&runtime, &tval_result, &runtime);

            // Write the frames in the video
            saveVideo.write(saveFrame);
            saveVideo.write(newFrame);

            // Show the progress bar
            printProgressBar(iterFrame, frameCount, tval_result, runtime, nBlocks, nThreads);

            //  Write the files with the times
            writeInform(path, width, height, iterFrame, frameCount, tval_result, runtime, nBlocks, nThreads);
            // Export the frames if is required
            if (expFrames)
                exportFrames(path, iterFrame, oldFrame, saveFrame, optFlow, width, height);

            // Free memory
            free(optFlow);
            free(uFrameOld);
            free(uFrameNew);
            free(interFrame);
        }
        /*
        if (waitKey(25) >= 0)
        {
            break;
        }
        */
        iterFrame += 1;
        newFrame.copyTo(oldFrame);
    }

    saveVideo.release();
    return runtime;
}
