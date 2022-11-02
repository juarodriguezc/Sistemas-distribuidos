#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <omp.h>

#include <opencv2/opencv.hpp>

// Set the number of arguments required
#define R_ARGS 3
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
void matToUchar(Mat, uchar **, int, int, int = 3);

// Prototype to cast uchar to Mat
void ucharToMat(uchar **, Mat, int, int, int = 3);

// Prototype to cast float* to Mat
void floatToMat1f(float *, Mat1f, int, int);

// Prorotype to clone a frame
void cloneFrame(uchar **, uchar **, int, int, int = 3);

// Prorotype to create an empty frame
void createFrame(uchar **, int, int, int = 3);

// Prototype to get the luminance
float *getLuminance(uchar **, int, int, int = 3, int = 0);

// Prototype for the motionImage
float *getMotionImage(float *, float *, int, int);

// Prototype for the optical flow
void getOpticalFlow(uchar **, uchar **, OpticalVector *, int, int, int = 3, int = 0);

// Prototype for the blur effect
void blurFrame(uchar **, uchar **, OpticalVector *, int, int, int = 3, int = 0);

// Prototype for the interpolation of frame
void interpolateFrames(uchar **, uchar **, uchar **, OpticalVector *, int, int, int, int = 0);

// Prototype for the print the progress
void printProgressBar(int, int, timeval, timeval);

// Prototype for the interpolation of video
timeval interpolateVideo(VideoCapture, char *, int = 0);

int main(int argc, char **argv)
{
    /*
    {
        // Declare the variables for time measurement
        struct timeval tval_before, tval_after, tval_result;
        // Declare the strings of load and save video
        char *loadPathFr1, *loadPathFr2, *savePath;
        // Declare the Matrix to store the image
        Mat loadIFrame1, loadIFrame2, saveImage;
        // Declare 3 Matrix for each channel
        Mat imageChF1[3], imageChF2[3];

        // Declare the size of each frame
        Size frameSize;
        int width, height, channels;
        // Declare the vector to merge the channels
        // std::vector<Mat> interFrame;

        int nThreads;
        // Check if the number of arguments is correct
        if (argc < R_ARGS + 1)
        {
            printf("Usage: ./MotionInterpolation <Load_frame1_Path> <Load_frame2_Path> <Save_frame_Path> <nThreads>\n");
            return -1;
        }
        // Update the paths
        loadPathFr1 = argv[1];
        loadPathFr2 = argv[2];
        savePath = argv[3];
        nThreads = atoi(argv[4]);

        // Get start time
        gettimeofday(&tval_before, NULL);

        VideoCapture loadVideo(loadPathFr1);

        // Check video opened successfully
        if (!loadVideo.isOpened())
        {
            std::cout << "Error opening video stream or file" << std::endl;
            return -1;
        }

        // interpolateVideo(loadVideo, tval_result);

        // Load the first frame
        loadIFrame1 = imread(loadPathFr1, IMREAD_UNCHANGED);
        if (!loadIFrame1.data)
        {
            // Show error if image not loaded correctly
            printf("No image data Frame 1\n");
            return -1;
        }
        // Load the second frame
        loadIFrame2 = imread(loadPathFr2, IMREAD_UNCHANGED);
        if (!loadIFrame2.data)
        {
            // Show error if image not loaded correctly
            printf("No image data Frame 2 \n");
            return -1;
        }

        // Get the frame dimension
        frameSize = loadIFrame1.size();
        width = frameSize.width;
        height = frameSize.height;
        channels = loadIFrame1.channels();

        uchar *uFrame1[channels];
        uchar *uFrame2[channels];
        uchar *interFrame[channels];

        matToUchar(loadIFrame1, uFrame1, width, height, channels);
        matToUchar(loadIFrame2, uFrame2, width, height, channels);

        uchar *uFrame3[channels];

        cloneFrame(uFrame1, uFrame3, width, height, channels);

        struct OpticalVector *optFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));

        // Mat resFrame = Mat::zeros(Size(width, height), CV_8UC3);

        Mat resFrame = Mat::zeros(Size(width, height), CV_8UC3);
        createFrame(interFrame, width, height, channels);

        interpolateFramesP(uFrame1, uFrame2, interFrame, optFlow, frameSize.width, frameSize.height, channels, nThreads);

        // Mat imageInter = Mat::zeros(Size(width, height), CV_8UC3);

        ucharToMat(interFrame, resFrame, width, height, channels);

        if (imwrite(savePath + std::string("ucharTest.jpg"), resFrame) == false)
        {
            std::cout << "Saving the generated frame, FAILED" << std::endl;
            return -1;
        }

        // Interpolate the frames
        // struct OpticalVector *optFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));

        /*

        // Split first frame
        split(loadIFrame1, imageChF1);
        // Split second frame
        split(loadIFrame2, imageChF2);

        // Get the frame dimension
        frameSize = loadIFrame1.size();
        width = frameSize.width;
        height = frameSize.height;

        // Interpolate the frames

        struct OpticalVector *optFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));

        imageInter = interpolateFramesP(imageChF1, imageChF2, optFlow, frameSize.width, frameSize.height, nThreads);

        Mat optFlowFram;
        loadIFrame1.copyTo(optFlowFram);

        for (int i = 0; i < height; i += 5)
        {
            for (int j = 0; j < width; j += 5)
            {
                Point p0(j, i);
                Point p1(j + optFlow[i * width + j].x, i + optFlow[i * width + j].y);
                if (abs(optFlow[i * width + j].x) + abs(optFlow[i * width + j].y) > 0)
                {
                    line(optFlowFram, p0, p1, Scalar(0, 255, 0), 1, LINE_AA);
                    circle(optFlowFram, p0, 1, Scalar(0, 0, 255), FILLED, LINE_8);
                }
            }
        }

        // Calcular los tiempos en tval_result
        //  Get end time
        gettimeofday(&tval_after, NULL);

        timersub(&tval_after, &tval_before, &tval_result);
        //Imprimir informe
        printf("------------------------------------------------------------------------------\n");
        printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

        // Save channels into the vector
        interFrame = {imageInter[0], imageInter[1], imageInter[2]};
        // Merge the channels
        merge(interFrame, saveImage);

        // writing the image to a defined location as JPEG
        if (imwrite(savePath + std::string("frame1a.jpg"), saveImage) == false)
        {
            std::cout << "Saving the generated frame, FAILED" << std::endl;
            return -1;
        }

        if (imwrite(savePath + std::string("optFlow.jpg"), optFlowFram) == false)
        {
            std::cout << "Saving the opticalFlow, FAILED" << std::endl;
            return -1;
        }
    }
    */

    // Declare the variables for time measurement
    struct timeval runtime;
    // Declare the strings of load and save video
    char *loadPathVid, *savePathVid;
    // Declare the Matrix to store the image
    VideoCapture loadVideo, saveVideo;
    // Declare the number of threads
    int nThreads;
    // Check if the number of arguments is correct
    if (argc < R_ARGS + 1)
    {
        printf("Usage: ./MotionInterpolation <Load_Video_Path> <Save_Video_Path> <nThreads>\n");
        return -1;
    }
    // Update the paths
    loadPathVid = argv[1];
    savePathVid = argv[2];
    nThreads = atoi(argv[3]);

    // Load the video from the path
    loadVideo = VideoCapture(loadPathVid);

    // Check video opened successfully
    if (!loadVideo.isOpened())
    {
        std::cout << "Error opening video stream or file" << std::endl;
        return -1;
    }

    // Interpolate the video and get the runtime
    runtime = interpolateVideo(loadVideo, savePathVid, nThreads);

    // Imprimir informe
    printf("------------------------------------------------------------------------------\n");
    printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)runtime.tv_sec, (long int)runtime.tv_usec);

    destroyAllWindows();
    return 0;
}

// Function to cast Mat to uchar*
void matToUchar(Mat frame, uchar **uFrame, int width, int height, int channels)
{
    // Create multidimensional array for the three channels
    for (int i = 0; i < channels; i++)
        uFrame[i] = (uchar *)malloc(width * height * sizeof(uchar));
    // Make a copy of the values into the array of uchars
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                uFrame[ch][i * width + j] = frame.at<Vec3b>(i, j)[ch];
}

// Function to cast uchar to Mat
void ucharToMat(uchar **uFrame, Mat frame, int width, int height, int channels)
{
    // Create the Mat of 3 channels
    // TODO: Make it for n channels
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                frame.at<Vec3b>(i, j)[ch] = uFrame[ch][i * width + j];
}

// Function to cast float* to Mat
void floatToMat1f(float *fFrame, Mat1f frame, int width, int height)
{
    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            frame.at<float>(i, j) = fFrame[i * width + j];
}

// Function to clone a frame
void cloneFrame(uchar **originFrame, uchar **destFrame, int width, int height, int channels)
{
    // Create multidimensional array for the N three channels
    for (int i = 0; i < channels; i++)
        destFrame[i] = (uchar *)malloc(width * height * sizeof(uchar));
    // Make a copy of the values into the array of uchars
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                destFrame[ch][i * width + j] = originFrame[ch][i * width + j];
}

// Function to create an empty frame
void createFrame(uchar **frame, int width, int height, int channels)
{
    // Create multidimensional array for the N three channels
    for (int i = 0; i < channels; i++)
        frame[i] = (uchar *)malloc(width * height * sizeof(uchar));
    for (int ch = 0; ch < channels; ch++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                frame[ch][i * width + j] = 0;
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

void blurFrame(uchar **frame, uchar **resFrame, OpticalVector *opticalFlow, int width, int height, int channels, int nThreads)
{
    static float kernel[9] =
        {1 / 16.0, 1 / 8.0, 1 / 16.0,
         1 / 8.0, 1 / 4.0, 1 / 8.0,
         1 / 16.0, 1 / 8.0, 1 / 16.0};

    static int kSize = (int)sqrt(my_sizeof(kernel) / my_sizeof(kernel[0]));

    // Change the value of nThreads if is zero
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

// Parallel the filter
#pragma omp parallel num_threads(nThreads)
    {
        // Get the id of the thread
        int thread_id = omp_get_thread_num();
        // Get start and end pos
        int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
        int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1;
        int i = (startPos / width), j = (startPos % width);

        // Float to store the convolution value
        float conv[channels];
        for (startPos; startPos <= endPos; startPos++)
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
                                conv[ch] += kernel[i1 * kSize + j1] * frame[ch][(i + i1 - kSize / 2) * width + (j + j1 - kSize / 2)];
                            }
                        }
                        // Check if the value is correct
                        if (conv[ch] > 255)
                            conv[ch] = 255;
                        if (conv[ch] < 0)
                            conv[ch] = 0;
                        resFrame[ch][i * width + j] = (int)conv[ch];
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
}

// Function to get the luminance
float *getLuminance(uchar **frame, int width, int height, int channels, int nThreads)
{
    // Declare the matrix to store the luminance
    float *lFrame = (float *)malloc(width * height * sizeof(float));
    // Check the value of the threads
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

#pragma omp parallel num_threads(nThreads)
    {
        // Get the id of the thread
        int thread_id = omp_get_thread_num();
        // Get start and end pos
        int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
        int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1;
        int i = (startPos / width), j = (startPos % width);

        // Variables to store the results
        float fB = 0.0f;
        float fG = 0.0f;
        float fR = 0.0f;

        for (startPos; startPos <= endPos; startPos++)
        {
            // Get the values from each pixel
            fB = (float)frame[0][i * width + j] / 255.0;
            fG = (float)frame[1][i * width + j] / 255.0;
            fR = (float)frame[2][i * width + j] / 255.0;

            lFrame[i * width + j] = 0.2987f * fR + 0.5870f * fG + 0.1140f * fB;

            j += 1;
            if (j == width)
            {
                i += 1;
                j = 0;
            }
        }
    }
    return lFrame;
}

// Function to get the optical flow
void getOpticalFlow(uchar **frame1, uchar **frame2, OpticalVector *optFlow, int width, int height, int channels, int nThreads)
{
    // Change the value of nThreads if is zero
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

    // Get the luminance of the two frames
    float *lFrame1 = getLuminance(frame1, width, height, channels, nThreads);
    float *lFrame2 = getLuminance(frame2, width, height, channels, nThreads);

    // Get the motionFrame of the frames
    float *motFrame = getMotionImage(lFrame1, lFrame2, width, height);

#pragma omp parallel num_threads(nThreads)
    {
        // Get the id of the thread
        int thread_id = omp_get_thread_num();
        // Get start and end pos
        int startPos = (thread_id < (width * height) % nThreads) ? ((width * height) / nThreads) * thread_id + thread_id : ((width * height) / nThreads) * thread_id + (width * height) % nThreads;
        int endPos = (thread_id < (width * height) % nThreads) ? startPos + ((width * height) / nThreads) : startPos + ((width * height) / nThreads) - 1;

        int i = (startPos / width), j = (startPos % width);
        // Declare the variables for the optical flow
        float fPatchDifferenceMax = INFINITY;
        float fPatchDifferenceX = 0.0f, fPatchDifferenceY = 0.0f;
        int searchVectorX = 0, searchVectorY = 0;
        float fAccumDif = 0.0f;
        int patchPixelX = 0, patchPixelY = 0;
        int basePixelX = 0, basePixelY = 0;
        float fPatchPixel = 0.0f, fBasePixel = 0.0f;

        // Iterate over each pixel of the image
        for (startPos; startPos <= endPos; startPos++)
        {
            // Initialize the vector
            optFlow[i * width + j].x = 0;
            optFlow[i * width + j].y = 0;
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
}

void interpolateFrames(uchar **frame1, uchar **frame2, uchar **resFrame, OpticalVector *optFlow, int width, int height, int channels, int nThreads)
{
    // Declare the variable for the interpolation
    int linearDiv = 2;
    // Declare the Matrix for the intermediate frame
    uchar *interFrame1[channels];
    uchar *interFrame2[channels];
    uchar *joinFrame[channels];

    cloneFrame(frame1, interFrame1, width, height, channels);
    cloneFrame(frame2, interFrame2, width, height, channels);
    createFrame(joinFrame, width, height, channels);

    //  Change the value of nThreads if is zero
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

    // Get the Optical Flow
    getOpticalFlow(frame1, frame2, optFlow, width, height, nThreads);

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
                    interFrame1[ch][i1 * width + j] = frame1[ch][i * width + j];
                    // Interpolate using the information of the frame 2
                    interFrame2[ch][i1 * width + j] = frame2[ch][i * width + j];
                }
            }
    // Join frames into a result Frame

    for (int i = 0; i < height; i++)
        for (int j = 0; j < width; j++)
            for (int ch = 0; ch < channels; ch++)
                joinFrame[ch][i * width + j] = (interFrame1[ch][i * width + j] + interFrame2[ch][i * width + j]) / 2;
    // Apply the blur filter over the join frame
    cloneFrame(joinFrame, resFrame, width, height, channels);
    blurFrame(joinFrame, resFrame, optFlow, width, height, channels, nThreads);
}

void printProgressBar(int iterFrame, int frameCount, timeval tval_result, timeval runtime)
{
    float progrss_100 = (int)((float)iterFrame / (float)frameCount * 100);
    float progress_20 = (int)((float)iterFrame / (float)frameCount * 20);
    float timeIteration = 1.0 / ((float)tval_result.tv_sec + ((float)tval_result.tv_usec / 1000000.0));

    // Clear the console
    // system("clear");

    std::cout << "Frame: " << iterFrame << " / " << frameCount << "   [" << timeIteration << " frames / s";
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
    std::cout << "]   " << progrss_100 << "%"
              << "    ET: ";

    printf("%ld.%06ld s     ", (long int)runtime.tv_sec, (long int)runtime.tv_usec);
    printf("ET/F: %ld.%06ld s \n \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);
}

timeval interpolateVideo(VideoCapture loadVideo, char *savePath, int nThreads)
{
    // Declare the variables for time measurement
    struct timeval tval_before, tval_after, tval_result, runtime = (struct timeval){0};

    // Declare the variables for the frames
    Mat oldFrame, newFrame, saveFrame;
    
    // Get the frame count, fps, width, height and number of channels
    int frameCount = loadVideo.get(CAP_PROP_FRAME_COUNT), iterFrame = 0, fps = loadVideo.get(CAP_PROP_FPS);
    int width = loadVideo.get(CAP_PROP_FRAME_WIDTH), height = loadVideo.get(CAP_PROP_FRAME_HEIGHT);
    int channels = 3;

    VideoWriter saveVideo(savePath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 2 * fps, Size(width, height));

    // Check the value of the threads
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

    // Print the video information
    std::cout << "--------------------------------------------------" << std::endl;
    std::cout << "                Motion interpolation              " << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    std::cout << "Video resolution: " << width << "px * " << height << "px" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "Total frames: " << frameCount << std::endl;

    std::cout << "--------------------------------------------------" << std::endl;

    while (loadVideo.read(newFrame))
    {
        // Write the original frame

        if (iterFrame == 0)
        {
            saveVideo.write(newFrame);
            channels = newFrame.channels();
        }

        else if (iterFrame > 0)
        {
            // Create uchar matrix for each frame
            uchar *uFrameOld[channels];
            uchar *uFrameNew[channels];
            uchar *interFrame[channels];

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

            // Get start time
            gettimeofday(&tval_before, NULL);

            interpolateFrames(uFrameOld, uFrameNew, interFrame, optFlow, width, height, channels, nThreads);

            // Get end time
            gettimeofday(&tval_after, NULL);

            // Convert the uchar matrix to Mat
            ucharToMat(interFrame, saveFrame, width, height, channels);

            // Save results
            timersub(&tval_after, &tval_before, &tval_result);

            // Update runtime
            timeradd(&runtime, &tval_result, &runtime);

            saveVideo.write(saveFrame);

            saveVideo.write(newFrame);

            printProgressBar(iterFrame, frameCount, tval_result, runtime);
        }
        if (waitKey(25) >= 0)
        {
            break;
        }
        iterFrame += 1;
        newFrame.copyTo(oldFrame);
    }

    saveVideo.release();
    return runtime;
}