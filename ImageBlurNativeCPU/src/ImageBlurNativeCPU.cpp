#include <iostream>
#include <string>
#include <chrono>

#include "MathHelper.h"

#include "CImg.h"
using namespace cimg_library;

static void blurImage(const std::string& imageFilename, int filterKernelSize)
{
	// There is: CImg::blur and CImg::convolve
	// do manual blur instead...

	CImg<unsigned char> tempImage1(imageFilename.c_str());
	CImg<unsigned char> tempImage2(tempImage1, false);
	std::unique_ptr<float[]> blurKernel = MathHelper::createNormalDistributionFilterKernel(filterKernelSize);
	//for(int i=0;i<filterKernelSize;++i)
	//	std::cout << blurKernel[i] << std::endl;

	std::cout << "everything loaded and prepared.. start blurring!" << std::endl;
	std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

	int halfKernelSize = filterKernelSize/2;
	int depth = std::min<int>(3, (int)tempImage1._spectrum);
	for(int channel = 0; channel <depth; ++channel)
	{
		unsigned int imageLayerOffset = tempImage1._height*tempImage1._width * channel;
		for(int y=0; y<(int)tempImage1._height; ++y)
		{
			for(int x=0; x<(int)tempImage1._width; ++x)
			{
				float dst = 0.0f;
				float* blurCoefficent = blurKernel.get();
				for(int k=-halfKernelSize; k<=halfKernelSize; ++k)
				{
					int xNew = x + k;
					if(xNew < 0)
						xNew = 0;
					else if(xNew >= (int)tempImage1._width)
						xNew = tempImage1._width-1;

					dst += (*blurCoefficent++) * (float)(*(tempImage1._data + y * tempImage1._width + xNew + imageLayerOffset));
				}
				if(dst > 255) dst = 255;
				*(tempImage2._data + y * tempImage2._width + x + imageLayerOffset) = (unsigned char)(dst);
			}
		}
	}
	for(int channel = 0; channel <depth; ++channel)
	{
		unsigned int imageLayerOffset = tempImage2._height*tempImage2._width * channel;
		for(int y=0; y<(int)tempImage2._height; ++y)
		{
			for(int x=0; x<(int)tempImage2._width; ++x)
			{
				float dst = 0.0f;
				float* blurCoefficent = blurKernel.get();
				for(int k=-halfKernelSize; k<=halfKernelSize; ++k)
				{
					int yNew = y + k;
					if(yNew < 0)
						yNew = 0;
					else if(yNew >= (int)tempImage1._height)
						yNew = tempImage1._height-1;

					dst += (*blurCoefficent++) * (float)(*(tempImage2._data + yNew * tempImage2._width + x + imageLayerOffset));
				}
				if(dst > 255) dst = 255;
				*(tempImage1._data + y * tempImage1._width + x + imageLayerOffset) = (unsigned char)(dst);
			}
		}
	}

	std::chrono::time_point<std::chrono::system_clock> end = std::chrono::system_clock::now();
	std::cout << "blurring done!" << std::endl;
	unsigned long elapsed_seconds = std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
	std::cout << "computation time (ms): " << (double)elapsed_seconds / 1000.0;


	// display images
	tempImage1.save((imageFilename.substr(0, imageFilename.find_last_of('.')) + "_convolved.bmp").c_str());
	CImgDisplay srcImageDisplay(tempImage1,"Blurred Image");
	while (!srcImageDisplay.is_closed())
	{
		srcImageDisplay.wait();
	}
}

int main(int argc, char* argv[])
{
	std::string imageFilename = "sampleimage1.bmp";
	int filterKernelSize = 11; // should be uneven!
	if (argc >= 2)
		imageFilename = argv[1];
	if (argc >= 3)
		filterKernelSize = atoi(argv[2]);

	blurImage(imageFilename, filterKernelSize);
}
