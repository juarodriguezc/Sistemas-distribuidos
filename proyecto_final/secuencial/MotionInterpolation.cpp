#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

// Set the number of arguments required
#define R_ARGS 3
// Set the precision of the motionImage
#define MOTION_PRES 0.05f

// Set the parameters for the optical flow
#define PATCH_SIZE 9
#define SEARCH_SIZE 7

using namespace cv;

struct OpticalVector
{
    float u;
    float v;
};

// Prototype for the luminance
Mat1f getLuminance(Mat, Mat, Mat, int, int);

// Prototype for the motionImage
Mat1f getMotionImage(Mat *, Mat *, int, int);

// Prototype for the optical flow
OpticalVector *getOpticalFlow(Mat *, Mat *, Mat1f, int, int);

void interpolateFrames(Mat *, Mat *, Mat *, OpticalVector *, int, int);

int main(int argc, char **argv)
{
    /*Declare the strings of load and save video*/
    char *loadPathFr1, *loadPathFr2, *savePath;
    // Declare the Matrix to store the image
    Mat loadIFrame1, loadIFrame2, saveImage;
    // Declare 3 Matrix for each channel
    Mat imageChF1[3], imageChF2[3], imageInter[3];
    // Declare the size of each frame
    Size frameSize;
    // Declare the vector to merge the channels
    std::vector<Mat> mChannels;

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
    /*
    Split the matrix into the three channels
        ImageChannel[0] = B
        ImageChannel[1] = G
        ImageChannel[2] = R
    */
    // Split first frame
    split(loadIFrame1, imageChF1);
    split(loadIFrame1, imageInter);
    // Split second frame
    split(loadIFrame2, imageChF2);

    // Get the frame dimension
    frameSize = loadIFrame1.size();

    // Apply the effect to the first frame
    Mat1f motionImage = getMotionImage(imageChF1, imageChF2, frameSize.width, frameSize.height);

    OpticalVector *opticalFlow = getOpticalFlow(imageChF1, imageChF2, motionImage, frameSize.width, frameSize.height);

    /*
    for (int i = 0; i < 10; i++)
    {
        for (int j = 0; j < 100; j++)
        {
            std::cout << "I: " << i << " J: " << j << "  ***  u: " << opticalFlow[i * frameSize.width + j].u << "  v: " << opticalFlow[i * frameSize.width + j].v << std::endl;
        }
    }
    */

    interpolateFrames(imageChF1, imageChF2, imageInter, opticalFlow, frameSize.width, frameSize.height);

    /*
    for (int i = 0; i < frameSize.height; i++)
    {
        for (int j = 0; j < frameSize.width; j++)
        {
            imageChannel[0].at<uchar>(i, j) = 255;
            imageChannel[1].at<uchar>(i, j) = 0;
        }
    }
    */

    // Save channels into the vector
    mChannels = {imageInter[0], imageInter[1], imageInter[2]};
    // Merge the channels
    merge(mChannels, saveImage);

    // writing the image to a defined location as JPEG
    if (imwrite(savePath, saveImage) == false)
    {
        std::cout << "Saving the image, FAILED" << std::endl;
        return -1;
    }

    //imshow("Blue Channel", saveImage);

    //waitKey(0);
    return 0;
}

// Function to get the luminance
Mat1f getLuminance(Mat B, Mat G, Mat R, int width, int height)
{
    Mat1f lMatrix(height, width);
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
Mat1f getMotionImage(Mat *frame1, Mat *frame2, int width, int height)
{
    // Declare image for the result
    Mat1f motionImage(height, width);
    // Calculate the luminance of each frame
    Mat1f lFrame1 = getLuminance(frame1[0], frame1[1], frame1[2], width, height);
    Mat1f lFrame2 = getLuminance(frame2[0], frame2[1], frame2[2], width, height);
    // Substract the luminances
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            float fDiff = fabs((float)lFrame1.at<float>(i, j) - (float)lFrame2.at<float>(i, j));
            motionImage.at<float>(i, j) = (fDiff >= MOTION_PRES) ? fDiff : 0.0f;
        }
    }
    return motionImage;
}

OpticalVector *getOpticalFlow(Mat *frame1, Mat *frame2, Mat1f motionFrame, int width, int height)
{
    Mat1f lFrame1 = getLuminance(frame1[0], frame1[1], frame1[2], width, height);
    Mat1f lFrame2 = getLuminance(frame2[0], frame2[1], frame2[2], width, height);
    // Create array of OpticalVector for the optical flow
    struct OpticalVector *optFlow = (OpticalVector *)malloc(width * height * sizeof(struct OpticalVector));
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            // Initialize the vector
            optFlow[i * width + j].u = 0.0f;
            optFlow[i * width + j].v = 0.0f;
            // Initialize the variables
            float fPatchDifferenceMax = INFINITY;
            float fPatchDifferenceX = 0.0f;
            float fPatchDifferenceY = 0.0f;

            /*
                Search over a rectangular area for a patch of the image
            */

            // Sx and Sy indicates the area of serch
            for (int sy = 0; sy < SEARCH_SIZE; sy++)
            {
                for (int sx = 0; sx < SEARCH_SIZE; sx++)
                {
                    // vectors that iterate over the search square
                    int searchVectorX = j + (sx - SEARCH_SIZE / 2);
                    int searchVectorY = i + (sy - SEARCH_SIZE / 2);

                    // Variable to store the difference of the patch
                    float fAccumDif = 0.0f;

                    // Iterate over the patch with px and py
                    for (int py = 0; py < PATCH_SIZE; py++)
                    {
                        for (int px = 0; px < PATCH_SIZE; px++)
                        {
                            // Iterate over the patch
                            int patchPixelX = searchVectorX + (px - PATCH_SIZE / 2);

                            int PatchPixelY = searchVectorY + (py - PATCH_SIZE / 2);

                            // Iterate over the patch of the original pixel
                            int basePixelX = j + (px - PATCH_SIZE / 2);
                            int basePixelY = i + (py - PATCH_SIZE / 2);

                            // Get adjacent values for each patch checking that is inside the image
                            float fPatchPixel = 0.0f;
                            if (patchPixelX >= 0 && patchPixelX < width && PatchPixelY >= 0 && PatchPixelY < height)
                                fPatchPixel = lFrame2.at<float>(PatchPixelY, patchPixelX);

                            float fBasePixel = 0.0f;
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
                        optFlow[i * width + j].u = (float)(searchVectorX - j);
                        optFlow[i * width + j].v = (float)(searchVectorY - i);
                    }
                }
            }
        }
    }
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            optFlow[i].u *= motionFrame.at<float>(i, j) > 0 ? 1.0f : 0.0f;
            optFlow[i].v *= motionFrame.at<float>(i, j) > 0 ? 1.0f : 0.0f;
        }
    }
    return optFlow;
}

void interpolateFrames(Mat *frame1, Mat *frame2, Mat *interFrame, OpticalVector *opticalFlow, int width, int height)
{
    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            interFrame[0].at<uchar>(i + opticalFlow[i * width + j].v, j + opticalFlow[i * width + j].u) = frame1[0].at<uchar>(i, j);
            interFrame[1].at<uchar>(i + opticalFlow[i * width + j].v, j + opticalFlow[i * width + j].u) = frame1[1].at<uchar>(i, j);
            interFrame[2].at<uchar>(i + opticalFlow[i * width + j].v, j + opticalFlow[i * width + j].u) = frame1[2].at<uchar>(i, j);
        }
    }
}
