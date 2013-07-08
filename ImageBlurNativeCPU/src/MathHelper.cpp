#include "MathHelper.h"

#include <math.h>

static double PI = 3.14159265359;

namespace MathHelper
{
	std::unique_ptr<float[]> createNormalDistributionFilterKernel(int filterKernelSize)
	{
		std::unique_ptr<float[]> output(new float[filterKernelSize]);

		double my = 0.0;
		double sigma = 1.0; 
		
		// and thus the internet said: use a blur kernel sized 3*sigma in both directions
		// http://en.wikipedia.org/wiki/Gaussian_blur  
		
		double sum = 0.0f;
		for(int i=0; i<filterKernelSize; ++i)
		{
			double x = ((double)i / (filterKernelSize-1) * 2.0 - 1.0) * 3 * sigma;
			double value = (1.0 / (sigma * sqrt(2.0 * PI))) *
							exp(- (x-my)*(x-my) / (2*sigma*sigma));
			output[i] = (float)value;
			sum += value;
		}
		
		// normalize (since its not exactly summed to 1)
		for(int i=0; i<filterKernelSize; ++i)
		{
			double value = output[i] / sum;
			output[i] =  (float)value;
		}
		
		
		return output;
	}
}
