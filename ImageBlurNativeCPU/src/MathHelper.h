#pragma once

#include <memory>

namespace MathHelper
{
	/**
	 * Generates 1D gaussian normal distribution filter
	 * @param filterKernelSize number of samples computed
	 */
	std::unique_ptr<float[]> createNormalDistributionFilterKernel(int filterKernelSize);
}
