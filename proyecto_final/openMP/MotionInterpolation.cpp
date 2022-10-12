#include <stdio.h>
#include <iostream>
#include <sys/time.h>
#include <omp.h>

#include <opencv2/opencv.hpp>

// Set the number of arguments required
#define R_ARGS 3
// Set the precision of the motionImage
#define MOTION_PRES 0.1f

// Set the parameters for the optical flow
// Default PATCH_SIZE: 9 - SEARCH_SIZE: 7 - FILTER_FLOW: 2
#define PATCH_SIZE 5
#define SEARCH_SIZE 5
#define FILTER_FLOW 2
#define my_sizeof(type) ((char *)(&type + 1) - (char *)(&type))

using namespace cv;

struct OpticalVector
{
    int x;
    int y;
};

// Prototype for the luminance
Mat1f getLuminanceP(Mat, Mat, Mat, int, int, int = 0);

// Prototype for the motionImage
Mat1f getMotionImage(Mat1f, Mat1f, int, int);

// Prototype for the optical flow
OpticalVector *getOpticalFlowP(Mat *, Mat *, int, int, int = 0);

// Prototype for the blur effect
void blurFrameP(Mat *, Mat *, OpticalVector *, int, int, int = 0);

// Prototype for the interpolation of frame
Mat *interpolateFramesP(Mat *, Mat *, int, int, int = 0);

// Prototype for the print the progress
void printProgressBar(int, int);

// Prototype for the interpolation of video
timeval interpolateVideo(VideoCapture, char *, int = 0);

int main(int argc, char **argv)
{
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

    runtime = interpolateVideo(loadVideo, savePathVid, nThreads);

    /*Imprimir informe*/
    printf("------------------------------------------------------------------------------\n");
    printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)runtime.tv_sec, (long int)runtime.tv_usec);

    destroyAllWindows();
    return 0;
}

// Function to get the luminance
Mat1f getLuminanceP(Mat B, Mat G, Mat R, int width, int height, int nThreads)
{
    // Declare the matrix to store the luminance
    Mat1f lMatrix(height, width);
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
            fB = (float)B.at<uchar>(i, j);
            fG = (float)G.at<uchar>(i, j);
            fR = (float)R.at<uchar>(i, j);
            lMatrix.at<float>(i, j) = 0.2987f * fR + 0.5870f * fG + 0.1140f * fB;
            j += 1;
            if (j == width)
            {
                i += 1;
                j = 0;
            }
        }
    }

    return lMatrix;
}

// Function to get the motionImage
Mat1f getMotionImage(Mat1f lFrame1, Mat1f lFrame2, int width, int height)
{
    // Declare image for the result
    Mat1f motionImage(height, width);
    // Declare the variable to store the luminance differences
    float fDiff = 0;
    // Substract the luminances
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            fDiff = fabs((float)lFrame1.at<float>(i, j) - (float)lFrame2.at<float>(i, j));
            motionImage.at<float>(i, j) = (fDiff >= MOTION_PRES) ? fDiff : 0.0f;
        }
    }
    return motionImage;
}

OpticalVector *getOpticalFlowP(Mat *frame1, Mat *frame2, int width, int height, int nThreads)
{
    // Change the value of nThreads if is zero
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

    // Get the luminance of the two frames
    Mat1f lFrame1 = getLuminanceP(frame1[0], frame1[1], frame1[2], width, height, nThreads);
    Mat1f lFrame2 = getLuminanceP(frame2[0], frame2[1], frame2[2], width, height, nThreads);
    // Get the motionFrame of the frames
    Mat1f motionFrame = getMotionImage(lFrame1, lFrame2, width, height);
    // Create array of OpticalVector for the optical flow
    struct OpticalVector *optFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));

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
        int patchPixelX = 0, PatchPixelY = 0;
        int basePixelX = 0, basePixelY = 0;
        float fPatchPixel = 0.0f, fBasePixel = 0.0f;

        // Iterate over each pixel of the image
        for (startPos; startPos <= endPos; startPos++)
        {
            // Initialize the vector
            optFlow[i * width + j].x = 0;
            optFlow[i * width + j].y = 0;
            if ((int)motionFrame.at<float>(i, j) > 0)
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
                                PatchPixelY = searchVectorY + (py - PATCH_SIZE / 2);

                                // Iterate over the patch of the original pixel
                                basePixelX = j + (px - PATCH_SIZE / 2);
                                basePixelY = i + (py - PATCH_SIZE / 2);

                                // Get adjacent values for each patch checking that is inside the image
                                fPatchPixel = 0.0f;
                                if (patchPixelX >= 0 && patchPixelX < width && PatchPixelY >= 0 && PatchPixelY < height)
                                    fPatchPixel = lFrame2.at<float>(PatchPixelY, patchPixelX);

                                fBasePixel = 0.0f;
                                if (basePixelX >= 0 && basePixelX < width && basePixelY >= 0 && basePixelY < height)
                                    fBasePixel = lFrame1.at<float>(basePixelY, basePixelX);

                                // Accumulate difference
                                fAccumDif += fabs(fPatchPixel - fBasePixel);
                            }
                        }

                        /*
                        Record the vector offset for the least different search patch
                        */
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

    return optFlow;
}

void blurFrameP(Mat *frame, Mat *resFrame, OpticalVector *opticalFlow, int width, int height, int nThreads)
{
    static float kernel[25] =
        {
            1 / 256.0, 4 / 256.0, 6 / 256.0, 4 / 256.0, 1 / 256.0,
            4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0,
            6 / 256.0, 24 / 256.0, 36 / 256.0, 24 / 256.0, 6 / 256.0,
            4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0,
            1 / 256.0, 4 / 256.0, 6 / 256.0, 4 / 256.0, 1 / 256.0};
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
        float conv[3] = {0.0, 0.0, 0.0};
        for (startPos; startPos <= endPos; startPos++)
        {
            if (abs(opticalFlow[i * width + j].x) > FILTER_FLOW || abs(opticalFlow[i * width + j].y) > FILTER_FLOW)
            {
                if (i > kSize && i < height - kSize && j > kSize && j < width - kSize)
                {
                    conv[0] = 0;
                    conv[1] = 0;
                    conv[2] = 0;
                    for (int i1 = 0; i1 < kSize; i1++)
                    {
                        for (int j1 = 0; j1 < kSize; j1++)
                        {
                            conv[0] += kernel[i1 * 5 + j1] * frame[0].at<uchar>(i + i1 - kSize / 2, j + j1 - kSize / 2);
                            conv[1] += kernel[i1 * 5 + j1] * frame[1].at<uchar>(i + i1 - kSize / 2, j + j1 - kSize / 2);
                            conv[2] += kernel[i1 * 5 + j1] * frame[2].at<uchar>(i + i1 - kSize / 2, j + j1 - kSize / 2);
                        }
                    }
                    // Check if the value is correct
                    for (int x = 0; x < 3; x++)
                    {
                        if (conv[x] > 255)
                            conv[x] = 255;
                        if (conv[x] < 0)
                            conv[x] = 0;
                        resFrame[x].at<uchar>(i, j) = (int)conv[x];
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

Mat *interpolateFramesP(Mat *frame1, Mat *frame2, int width, int height, int nThreads)
{
    // Declare the Matrix for the intermediate frame
    Mat *interFrame = new Mat[3]{frame1[0].clone(), frame1[1].clone(), frame1[2].clone()};
    Mat *resFrame;
    // Change the value of nThreads if is zero
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();
    // Declare the array for the Optical Flow
    OpticalVector *opticalFlow = getOpticalFlowP(frame1, frame2, width, height, nThreads);
    // Create the new frame interpolating the optical Flow
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            interFrame[0].at<uchar>(i + (int)(opticalFlow[i * width + j].y / 2), j + (int)(opticalFlow[i * width + j].x) / 2) = frame1[0].at<uchar>(i, j);
            interFrame[1].at<uchar>(i + (int)(opticalFlow[i * width + j].y / 2), j + (int)(opticalFlow[i * width + j].x) / 2) = frame1[1].at<uchar>(i, j);
            interFrame[2].at<uchar>(i + (int)(opticalFlow[i * width + j].y / 2), j + (int)(opticalFlow[i * width + j].x) / 2) = frame1[2].at<uchar>(i, j);
        }
    }
    resFrame = new Mat[3]{interFrame[0].clone(), interFrame[1].clone(), interFrame[2].clone()};
    // Apply the blur filter over the image
    //blurFrameP(interFrame, resFrame, opticalFlow, width, height, nThreads);
    return resFrame;
}

void printProgressBar(int iterFrame, int frameCount){
    float progress = (int)((float)iterFrame/ (float)frameCount * 100);
    std::cout<<"Frame: "<<iterFrame<< " / "<<frameCount<<"  -  Progress: "<<progress<<" % "<<std::endl;
}

timeval interpolateVideo(VideoCapture loadVideo, char *savePath, int nThreads)
{
    // Declare the variables for time measurement
    struct timeval tval_before, tval_after, tval_result, runtime = (struct timeval){0};

    // Declare the variables for the frames
    Mat oldFrame, newFrame, saveFrame;
    // Declare 3 Matrix for each channel
    Mat imageChNew[3], imageChOld[3];
    Mat *imageInter;
    std::vector<Mat> interFrame;

    // Get the frame count
    int frameCount = loadVideo.get(CAP_PROP_FRAME_COUNT), iterFrame = 0, fps = loadVideo.get(CAP_PROP_FPS);
    int width = loadVideo.get(CAP_PROP_FRAME_WIDTH), height = loadVideo.get(CAP_PROP_FRAME_HEIGHT);

    VideoWriter saveVideo(savePath, VideoWriter::fourcc('M', 'J', 'P', 'G'), 2*fps, Size(width, height));

    // Check the value of the threads
    if (nThreads <= 0)
        nThreads = omp_get_num_procs();

    std::cout << width << "-" << height << std::endl;

    while (loadVideo.read(newFrame))
    {
        //Write the original frame
        
        if (iterFrame == 0)
            saveVideo.write(newFrame);

        else if (iterFrame > 0)
        {
            // Split old frame
            split(oldFrame, imageChOld);
            // Split new frame
            split(newFrame, imageChNew);

            // Get start time
            gettimeofday(&tval_before, NULL);

            // Interpolate the frames
            imageInter = interpolateFramesP(imageChOld, imageChNew, width, height);

            //  Get end time
            gettimeofday(&tval_after, NULL);


            //Save results
            timersub(&tval_after, &tval_before, &tval_result);

            //Update runtime
            timeradd(&runtime, &tval_result, &runtime);

            interFrame = {imageInter[0], imageInter[1], imageInter[2]};

            merge(interFrame, saveFrame);

            saveVideo.write(saveFrame);
            saveVideo.write(newFrame);

            printProgressBar(iterFrame, frameCount);

            printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)runtime.tv_sec, (long int)runtime.tv_usec);
            
        }
        if (waitKey(25) >= 0)
        {
            break;
        }
        iterFrame += 1;
        oldFrame = newFrame.clone();
    }

    saveVideo.release();
    return runtime;
}