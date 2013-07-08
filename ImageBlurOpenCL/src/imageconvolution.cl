__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

__kernel void convolveX(
	read_only image2d_t inputImage,
	write_only image2d_t outputImage,
	__constant float* filterKernel,
	__private int filterSize,
	__private int2 imageSize
	)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    if(any(pos >= imageSize))
    	return;
    float4 outputColor = (float4)(0.0f,0.0f,0.0f,0.0f);

	int2 samplePos = pos;
	samplePos.y -= filterSize/2;
   	for(int i=0; i<filterSize; ++i, ++samplePos.y)
		outputColor += read_imagef(inputImage, sampler, samplePos) * filterKernel[i];

  	// write to output image
  	write_imagef(outputImage, pos, outputColor);
}

__kernel void convolveY(
	read_only image2d_t inputImage,
	write_only image2d_t outputImage,
	__constant float* filterKernel,
	__private int filterSize,
	__private int2 imageSize
	)
{
    const int2 pos = {get_global_id(0), get_global_id(1)};
    if(any(pos >= imageSize))
    	return;
    float4 outputColor = (float4)(0.0f,0.0f,0.0f,0.0f);

	int2 samplePos = pos;
	samplePos.x -= filterSize/2;
   	for(int i=0; i<filterSize; ++i, ++samplePos.x)
		outputColor += read_imagef(inputImage, sampler, samplePos) * filterKernel[i];
  	
  	// write to output image
  	write_imagef(outputImage, pos, outputColor);
}