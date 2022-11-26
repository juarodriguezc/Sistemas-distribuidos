/**
 * @file MotionInterpolation.cpp
 * @author Juan Sebastian Rodríguez (juarodriguezc)
 * @date 2022-02-11
 * @copyright Copyright (c) 2022
 */

#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <omp.h>
#include <mpi.h>

#include <opencv2/opencv.hpp>

// Set the number of arguments required
#define R_ARGS 6
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
float *getLuminance(float *, uchar *, int, int, int = 3);

// Prototype for the motionImage
float *getMotionImage(float *, float *, float *, int, int);

// Prototype for the optical flow
void getOpticalFlow(float *, float *, float *, OpticalVector *, int, int, int = 3, int = 0, int = 0, int = 0);

// Prototype for the blur effect
void blurFrame(uchar *, uchar *, OpticalVector *, int, int, int = 3);

// Prototype for the interpolation of frame
void interpolateFrames(uchar *, uchar *, uchar *, OpticalVector *, int, int, int);

// Prototype for the print the progress
void printProgressBar(int, int, double, double, int, int);

// Prototype for the write the inform
void writeInform(char *, int, int, int, int, double, double, int, int);

// Prototype to export the generated frames
void exportFrames(char *, int, Mat, Mat, OpticalVector *, int, int);

// Prototype for the interpolation of video
void interpolateVideo(char *, char *, char *, int = 0, bool = false, int = 0);

int main(int argc, char **argv)
{
    // Declare the variables for time measurement
    struct timeval runtime;
    // Declare the strings of load and save video
    char *path;
    char *loadName, *saveName;
    // Declare the variable to save the frames
    int framesRend = 0;
    bool expFrames = false;

    // Declare the variable for the number of Threads
    int nThreads = 0;

    // Check if the number of arguments is correct
    if (argc < R_ARGS + 1)
    {
        printf("Usage: mpirun -np nProcess --hostfile mpi_hosts ./MotionInterpolation <Path_to_files> <Load_video_name> <Save_video_name> <Frames to render (0 - all)> <Export_frames ( 0:false, 1:true )> <nThreads ( 0:auto )> \n");
        return -1;
    }
    // Get the paths
    path = argv[1];
    loadName = argv[2];
    saveName = argv[3];
    framesRend = atoi(argv[4]);
    expFrames = (atoi(argv[5]) == 0) ? false : true;
    nThreads = atoi(argv[6]);
    nThreads = nThreads < 0 ? 0 : nThreads;

    // Call the interpolate function
    interpolateVideo(path, loadName, saveName, framesRend, expFrames, nThreads);

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
float *getMotionImage(float *motionImage, float *lFrame1, float *lFrame2, int width, int height)
{
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
float *getLuminance(float *lFrame, uchar *frame, int width, int height, int channels)
{
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
void getOpticalFlow(float *lFrame1, float *lFrame2, float *motFrame, OpticalVector *optFlow, int width, int height, int channels, int startPos, int endPos, int nThreads)
{
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

    #pragma omp parallel num_threads(nThreads)
    {
        // Get the id of the thread
        int thread_id = omp_get_thread_num();
        int nPixels = abs(endPos - startPos + 1);

        // Get start and end pos based on nPixels
        int sPos = (thread_id < nPixels % nThreads) ? startPos + (nPixels / nThreads) * thread_id + thread_id : startPos + (nPixels / nThreads) * thread_id + nPixels % nThreads;
        int ePos = (thread_id < nPixels % nThreads) ? sPos + (nPixels / nThreads) : sPos + (nPixels / nThreads) - 1;

        // Calculate the positions in terms of i and j
        int i = (sPos / width), j = (sPos % width);
        // Declare the variables for the optical flow
        float fPatchDifferenceMax = INFINITY;
        float fPatchDifferenceX = 0.0f, fPatchDifferenceY = 0.0f;
        int searchVectorX = 0, searchVectorY = 0;
        float fAccumDif = 0.0f;
        int patchPixelX = 0, patchPixelY = 0;
        int basePixelX = 0, basePixelY = 0;
        float fPatchPixel = 0.0f, fBasePixel = 0.0f;

        // Iterate over each pixel of the image
        for (int iter = 0; sPos <= ePos; sPos++)
        {
            // Initialize the vector
            optFlow[iter].x = 0;
            optFlow[iter].y = 0;
            if (motFrame[i * width + j] > 0)
            {
                // Initialize the variables
                fPatchDifferenceMax = INFINITY;
                fPatchDifferenceX = 0.0f;
                fPatchDifferenceY = 0.0f;

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
                            optFlow[iter].x = searchVectorX - j;
                            optFlow[iter].y = searchVectorY - i;
                        }
                    }
                }
            }
            j += 1;
            iter += 1;
            if (j == width)
            {
                i += 1;
                j = 0;
            }
        }
    }
}

// Function for the blur effect
void blurFrame(uchar *frame, uchar *resFrame, OpticalVector *opticalFlow, int width, int height, int channels)
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

void interpolateFrames(uchar *frame1, uchar *frame2, uchar *resFrame, OpticalVector *optFlow, int width, int height, int channels)
{

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
                    int j1 = j + (int)(optFlow[i * width + j].x) / linearDiv;
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
    blurFrame(joinFrame, resFrame, optFlow, width, height, channels);

    // Free the memory
    free(interFrame1);
    free(interFrame2);
    free(joinFrame);
}

void printProgressBar(int iterFrame, int frameCount, double clusterTime, double totalTime, int nProcess, int nThreads)
{
    float progrss_100 = (int)((float)iterFrame / (float)frameCount * 100);
    float progress_20 = (int)((float)iterFrame / (float)frameCount * 20);
    float framesPSec = 1.0 / (clusterTime);

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

    printf("%f s     ", totalTime);
    printf("    ET/F: %f s      NProcess:  %d       NThreads:  %d    \n \n", clusterTime, nProcess, nThreads);
}

// Function to the write the inform
void writeInform(char *path, int width, int height, int iterFrame, int frameCount, double clusterTime, double totalTime, int nProcess, int nThreads)
{
    // Declare the FILE to write the times
    FILE *fp;

    float framesPSec = 1.0 / (clusterTime);

    fp = fopen((std::string(path) + "frameTime.csv").c_str(), "a");
    if (fp == NULL)
    {
        printf("Error opening the file \n");
        exit(1);
    }
    fprintf(fp, "%d,%d,%d,%d,%d,%f,%f\n", iterFrame, width, height, nProcess, nThreads, framesPSec, clusterTime);
    fclose(fp);
}

// Function to export the generated frames
void exportFrames(char *path, int iterFrame, Mat frame, Mat genFrame, OpticalVector *optFlow, int width, int height)
{
    // writing the image to a defined location as JPG
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
    // writing the image to a defined location as JPG
    if (imwrite((std::string(path) + "frames/optFlow000" + std::to_string(iterFrame) + ".jpg").c_str(), frame) == false)
    {
        std::cout << "Exporting the optical flow frame, FAILED" << std::endl;
        exit(-1);
    }
}

void interpolateVideo(char *path, char *loadName, char *saveName, int framesRend, bool expFrames, int nThreads)
{
    // Declare the variables for time measurement
    double totalTime = 0.0, clusterTime = 0.0, processTime = 0.0, timeStart = 0.0, timeEnd = 0.0;
    // Declare the number of process and id
    int nProcess, processId;
    // Declare variables for the frame count, fps, width, height and number of channels
    int frameCount = 0, iterFrame = 0, fps = 0;
    int width = 0, height = 0, channels = 3;
    // Declare the variables for the frames
    Mat oldFrame, newFrame, saveFrame;
    // Declare the Matrix to store the image
    VideoCapture loadVideo;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcess);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    // Create a Type for  the OpticalVector Struct
    const int nitems = 2;
    int blocklengths[2] = {1, 1};
    MPI_Datatype types[2] = {MPI_INT, MPI_INT};
    MPI_Datatype MPI_OPTVECTOR_TYPE;
    MPI_Aint offsets[2];

    offsets[0] = offsetof(OpticalVector, x);
    offsets[1] = offsetof(OpticalVector, y);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_OPTVECTOR_TYPE);
    MPI_Type_commit(&MPI_OPTVECTOR_TYPE);

    // Load the video and share the parameters to the other processes
    if (processId == 0)
    {
        // Load the video from the path
        loadVideo = VideoCapture(std::string(path) + loadName);
        // Get the video properties
        frameCount = framesRend == 0 ? loadVideo.get(CAP_PROP_FRAME_COUNT) - 1 : framesRend;
        fps = loadVideo.get(CAP_PROP_FPS);
        width = loadVideo.get(CAP_PROP_FRAME_WIDTH), height = loadVideo.get(CAP_PROP_FRAME_HEIGHT);
        // Print the properties
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "                          Motion interpolation                          " << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl;

        std::cout << "Video resolution: " << width << "px * " << height << "px" << std::endl;
        std::cout << "Path: " << path << std::endl;
        std::cout << "FPS: " << fps << std::endl;
        std::cout << "Total frames: " << frameCount << std::endl;
        std::cout << "Number of Process: " << nProcess << std::endl;
        std::cout << "Number of Threads: " << nThreads << std::endl;

        std::cout << "------------------------------------------------------------------------" << std::endl;
    }
    // Put a battier to wait the video loading
    MPI_Barrier(MPI_COMM_WORLD);

    // Broadcasat the video parameters
    MPI_Bcast(&frameCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&fps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Declare the video Writer
    VideoWriter saveVideo;
    if (processId == 0)
    {
        saveVideo = VideoWriter(std::string(path) + saveName, VideoWriter::fourcc('M', 'J', 'P', 'G'), 2 * fps, Size(width, height));
    }

    // Put a battier to wait the video saving
    MPI_Barrier(MPI_COMM_WORLD);

    // Iterate over the number of frames
    while (iterFrame <= frameCount)
    {
        // Initialize the time
        processTime = 0.0;
        clusterTime = 0.0;

        // Declare boolean to check if frame exists
        int frameExists = 0;
        // Declare the variables to store the uchar frames
        uchar *uFrameOld = (uchar *)malloc(width * height * channels * sizeof(uchar));
        uchar *uFrameNew = (uchar *)malloc(width * height * channels * sizeof(uchar));
        uchar *interFrame = (uchar *)malloc(width * height * channels * sizeof(uchar));

        // Check correct allocation
        if (uFrameOld == NULL || uFrameNew == NULL || interFrame == NULL)
        {
            printf("Failed to allocate the frames \n");
            exit(1);
        }
        // Check if the new Frame exists
        if (processId == 0)
            frameExists = (loadVideo.read(newFrame)) ? 1 : 0;
        // Put a barrier to broadcast the bool of the existence
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&frameExists, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Interpolate only If the frame exists
        if (frameExists != 0)
        {
            // Write the first frame only in the first process
            if (iterFrame == 0)
            {
                if (processId == 0)
                {
                    // Write the frame
                    saveVideo.write(newFrame);
                    // Create a copy of the current frame
                    newFrame.copyTo(oldFrame);
                }
            }
            else
            {
                // Declare the float array for the luminance
                float *lFrame1 = (float *)malloc(width * height * sizeof(float));
                float *lFrame2 = (float *)malloc(width * height * sizeof(float));

                // Declare the float array for the motion
                float *motFrame = (float *)malloc(width * height * sizeof(float));

                // Check correct allocation
                if (lFrame1 == NULL || lFrame2 == NULL || motFrame == NULL)
                {
                    printf("Failed to allocate the luminance frames \n");
                    exit(1);
                }

                // Declare the positions to calculate the optical flow
                int startPos = (processId < (width * height) % nProcess) ? ((width * height) / nProcess) * processId + processId : ((width * height) / nProcess) * processId + (width * height) % nProcess;
                int endPos = (processId < (width * height) % nProcess) ? startPos + ((width * height) / nProcess) : startPos + ((width * height) / nProcess) - 1;
                int sizeMat = endPos - startPos + 1;

                // Declare the array for the result optical flow
                struct OpticalVector *resOptFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));
                // Declare the array for the individual calculation optical Flow
                struct OpticalVector *optFlow = (OpticalVector *)malloc(sizeMat * sizeof(struct OpticalVector));

                if (optFlow == NULL || resOptFlow == NULL)
                {
                    printf("Failed to allocate the opticalFlow matrix \n");
                    exit(1);
                }

                // Measure start time
                timeStart = MPI_Wtime();

                // Write the first frame or calculate the luminances in the process 0
                if (processId == 0)
                {
                    // Load the data from the Mat to the uchar matrix
                    matToUchar(oldFrame, uFrameOld, width, height, channels);
                    matToUchar(newFrame, uFrameNew, width, height, channels);

                    // Get the luminances of eeach frame
                    getLuminance(lFrame1, uFrameOld, width, height, channels);
                    getLuminance(lFrame2, uFrameNew, width, height, channels);

                    // Get the motionFrame of the frames
                    getMotionImage(motFrame, lFrame1, lFrame2, width, height);

                    // Create a copy of the current frame
                    newFrame.copyTo(oldFrame);
                }

                // Measure end time
                timeEnd = MPI_Wtime();
                processTime += fabs(timeEnd - timeStart);

                // Put a barrier to broadcast the luminances and the motion frame
                MPI_Barrier(MPI_COMM_WORLD);

                // Broadcasat the luminances and the motion frame
                MPI_Bcast(lFrame1, width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(lFrame2, width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);
                MPI_Bcast(motFrame, width * height, MPI_FLOAT, 0, MPI_COMM_WORLD);

                // Measure start time
                timeStart = MPI_Wtime();

                // Calculate the optical flow
                getOpticalFlow(lFrame1, lFrame2, motFrame, optFlow, width, height, channels, startPos, endPos, nThreads);

                // Measure end time
                timeEnd = MPI_Wtime();
                processTime += fabs(timeEnd - timeStart);

                // Barrier to wait all the results of the optical flow
                MPI_Barrier(MPI_COMM_WORLD);

                // Gather the results
                MPI_Gather(optFlow, sizeMat, MPI_OPTVECTOR_TYPE, resOptFlow, sizeMat, MPI_OPTVECTOR_TYPE, 0, MPI_COMM_WORLD);

                // Wait until the gather is done
                MPI_Barrier(MPI_COMM_WORLD);

                MPI_Reduce(&processTime, &clusterTime, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

                // Interpolate the frame using the optical flow
                if (processId == 0)
                {
                    // Measure frame generation time
                    timeStart = MPI_Wtime();

                    // Create the frame Mat to export
                    Mat saveFrame = Mat::zeros(Size(width, height), CV_8UC3);

                    // Declare the intermediate frame
                    uchar *interFrame = (uchar *)malloc(width * height * channels * sizeof(uchar));
                    if (interFrame == NULL)
                    {
                        printf("Failed to allocate the intermediate frame \n");
                        exit(1);
                    }
                    // Create the uchar matrix of the new frame
                    createFrame(interFrame, width, height, channels);

                    // Interpolate the frames
                    interpolateFrames(uFrameOld, uFrameNew, interFrame, resOptFlow, width, height, channels);

                    // Measure end time and calculate the cluster time
                    timeEnd = MPI_Wtime();
                    clusterTime += fabs(timeEnd - timeStart);
                    clusterTime = clusterTime / nProcess;

                    // Add the clusterTime to the TotalTime
                    totalTime += clusterTime;

                    // Convert the uchar matrix to Mat
                    ucharToMat(interFrame, saveFrame, width, height, channels);

                    // Write the frames in the video
                    saveVideo.write(saveFrame);
                    saveVideo.write(newFrame);

                    // Show the progress bar
                    printProgressBar(iterFrame, frameCount, clusterTime, totalTime, nProcess, nThreads);

                    // Write the files with the times
                    writeInform(path, width, height, iterFrame, frameCount, clusterTime, totalTime, nProcess, nThreads);

                    // Export the frames if is required
                    if (expFrames)
                        exportFrames(path, iterFrame, newFrame, saveFrame, resOptFlow, width, height);

                    // Release interFrame memory and Mat
                    free(interFrame);
                    saveFrame.release();
                }

                // Release the luminances and motion images memory
                free(lFrame1);
                free(lFrame2);
                free(motFrame);
                // Release the opticalFlow memory
                free(optFlow);
                free(resOptFlow);
            }
        }
        // Increase the iterator
        iterFrame++;
        // Release the frames memory
        free(uFrameOld);
        free(uFrameNew);
        free(interFrame);
    }

    // Write inform with the total time
    if (processId == 0)
    {
        // Declare the FILE to write the times
        FILE *fp;
        fp = fopen((std::string(path) + "totalTime.csv").c_str(), "a");
        if (fp == NULL)
        {
            printf("Error opening the file \n");
            exit(1);
        }
        printf("------------------------------------------------------------------------------\n");
        printf("Runtime: %f s \n", totalTime);
        fprintf(fp, "%d,%d,%d,%d,%d,%f\n", width, height, nProcess, nThreads, iterFrame - 1, totalTime);
        printf("------------------------------------------------------------------------------\n \n \n");


        fclose(fp);
    }

    MPI_Finalize();
}