#include <stdio.h>
#include <iostream>
#include <sys/time.h>

#include <opencv2/opencv.hpp>

// Set the number of arguments required
#define R_ARGS 3
// Set the precision of the motionImage
#define MOTION_PRES 0.05f

// Set the parameters for the optical flow
// Def PATCH_SIZE: 9 - SEARCH_SIZE: 7
#define PATCH_SIZE 5
#define SEARCH_SIZE 9
#define FILTER_FLOW 2
# define my_sizeof(type) ((char *)(&type+1)-(char*)(&type))

using namespace cv;

struct OpticalVector
{
    int x;
    int y;
};

// Prototype for the luminance
Mat1f getLuminance(Mat, Mat, Mat, int, int);

// Prototype for the motionImage
Mat1f getMotionImage(Mat1f, Mat1f, int, int);

// Prototype for the optical flow
OpticalVector *getOpticalFlow(Mat *, Mat *, int, int);

Mat *interpolateFrames(Mat *, Mat *, int, int);

void blurFrame(Mat *, Mat *, OpticalVector *, int, int);

int main(int argc, char **argv)
{
    // Declare the variables for time measurement
    struct timeval tval_before, tval_after, tval_result;
    // Declare the strings of load and save video
    char *loadPathFr1, *loadPathFr2, *savePath;
    // Declare the Matrix to store the image
    Mat loadIFrame1, loadIFrame2, saveImage;
    // Declare 3 Matrix for each channel
    Mat imageChF1[3], imageChF2[3];
    Mat *imageInter;
    // Declare the size of each frame
    Size frameSize;
    int width, height;
    // Declare the vector to merge the channels
    std::vector<Mat> interFrame;

    // Check if the number of arguments is correct
    if (argc < R_ARGS + 1)
    {
        printf("Usage: ./MotionInterpolation <Load_Video_Path1> <Load_Video_Path2> <Save_Video_Path>\n");
        return -1;
    }
    // Update the paths
    loadPathFr1 = argv[1];
    loadPathFr2 = argv[2];
    savePath = argv[3];
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

    // Get start time
    gettimeofday(&tval_before, NULL);

    // Split first frame
    split(loadIFrame1, imageChF1);
    // Split second frame
    split(loadIFrame2, imageChF2);

    // Get the frame dimension
    frameSize = loadIFrame1.size();
    width = frameSize.width;
    height = frameSize.height;

    // Interpolate the frames

    imageInter = interpolateFrames(imageChF1, imageChF2, frameSize.width, frameSize.height);

    // Get end time
    gettimeofday(&tval_after, NULL);

    /*Calcular los tiempos en tval_result*/
    timersub(&tval_after, &tval_before, &tval_result);
    /*Imprimir informe*/
    printf("------------------------------------------------------------------------------\n");
    printf("Tiempo de ejecución: %ld.%06ld s \n", (long int)tval_result.tv_sec, (long int)tval_result.tv_usec);

    // Save channels into the vector
    interFrame = {imageInter[0], imageInter[1], imageInter[2]};
    // Merge the channels
    merge(interFrame, saveImage);

    // writing the image to a defined location as JPEG
    if (imwrite(savePath, saveImage) == false)
    {
        std::cout << "Saving the image, FAILED" << std::endl;
        return -1;
    }

    return 0;
}

// Function to get the luminance
Mat1f getLuminance(Mat B, Mat G, Mat R, int width, int height)
{
    // Declare the matrix to store the luminance
    Mat1f lMatrix(height, width);
    // Variables to store the results
    float fB = 0.0f;
    float fG = 0.0f;
    float fR = 0.0f;
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            fB = (float)B.at<uchar>(i, j);
            fG = (float)G.at<uchar>(i, j);
            fR = (float)R.at<uchar>(i, j);
            lMatrix.at<float>(i, j) = 0.2987f * fR + 0.5870f * fG + 0.1140f * fB;
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

OpticalVector *getOpticalFlow(Mat *frame1, Mat *frame2, int width, int height)
{
    // Get the luminance of the two frames
    Mat1f lFrame1 = getLuminance(frame1[0], frame1[1], frame1[2], width, height);
    Mat1f lFrame2 = getLuminance(frame2[0], frame2[1], frame2[2], width, height);
    // Get the motionFrame of the frames
    Mat1f motionFrame = getMotionImage(lFrame1, lFrame2, width, height);
    // Create array of OpticalVector for the optical flow
    struct OpticalVector *optFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));
    // Declare the variables for the optical flow
    float fPatchDifferenceMax = INFINITY;
    float fPatchDifferenceX = 0.0f, fPatchDifferenceY = 0.0f;
    int searchVectorX = 0, searchVectorY = 0;
    float fAccumDif = 0.0f;
    int patchPixelX = 0, PatchPixelY = 0;
    int basePixelX = 0, basePixelY = 0;
    float fPatchPixel = 0.0f, fBasePixel = 0.0f;
    // Iterate over each pixel of the image
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // Initialize the vector
            optFlow[i * width + j].x = 0;
            optFlow[i * width + j].y = 0;

            // std::cout<<(int)motionFrame.at<float>(i, j)<<std::endl;

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
        }
    }

    return optFlow;
}

void blurFrame(Mat *frame, Mat *resFrame, OpticalVector *opticalFlow, int width, int height)
{
    static float kernel[25] =
        {
            1 / 256.0, 4 / 256.0, 6 / 256.0, 4 / 256.0, 1 / 256.0,
            4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0,
            6 / 256.0, 24 / 256.0, 36 / 256.0, 24 / 256.0, 6 / 256.0,
            4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0,
            1 / 256.0, 4 / 256.0, 6 / 256.0, 4 / 256.0, 1 / 256.0};
    static int kSize = (int)sqrt(my_sizeof(kernel)/my_sizeof(kernel[0]));
    float conv[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
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
                            conv[0] += kernel[i1 * 5 + j1] * frame[0].at<uchar>(i + i1 - kSize/2, j + j1 - kSize/2);
                            conv[1] += kernel[i1 * 5 + j1] * frame[1].at<uchar>(i + i1 - kSize/2, j + j1 - kSize/2);
                            conv[2] += kernel[i1 * 5 + j1] * frame[2].at<uchar>(i + i1 - kSize/2, j + j1 - kSize/2);
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
        }
    }
}

Mat *interpolateFrames(Mat *frame1, Mat *frame2, int width, int height)
{
    // Declare the Matrix for the intermediate frame
    Mat *interFrame = new Mat[3]{frame1[0].clone(), frame1[1].clone(), frame1[2].clone()};
    Mat *resFrame;
    // Declare the array for the Optical Flow
    OpticalVector *opticalFlow = getOpticalFlow(frame1, frame2, width, height);
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
    blurFrame(interFrame, resFrame, opticalFlow, width, height);
    return resFrame;
}
