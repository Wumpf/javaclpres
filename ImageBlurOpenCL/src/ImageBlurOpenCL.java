import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.util.*;

import org.bridj.Pointer;

public class ImageBlurOpenCL {

	public static void printDeviceInfos(CLDevice device) {
		System.out.println("Device Name: " + device.getName());
		System.out.println("OpenCL Version: " + device.getOpenCLVersion());
		System.out.println("Local Memory size: " + device.getLocalMemSize());
		System.out.println("Global Memory size: " + device.getGlobalMemSize());
		System.out.println("Global Memory Cache size: " + device.getGlobalMemCacheSize());
		System.out.println("Max Compute units: " + device.getMaxComputeUnits());
		System.out.println("Max Work Group size: " + device.getMaxWorkGroupSize());
	}

	public static BufferedImage loadImage(String filename) {
		BufferedImage image = null;
		try {
			// try resource
			image = ImageIO.read(ImageBlurOpenCL.class.getResource(filename));
		} catch (Exception e1) {
			try {
				// then try extern
				image = ImageIO.read(new File(filename));
			} catch (IOException e) {
				e.printStackTrace();
				return null;
			}
		}
		image.flush(); // this helps to keep computation time measurement error low
		
		return image;
	}
	
	public static void blurImage(String imageFilename, int filterKernelSize) {
		// create context - represents the entire OpenCL environment
		CLContext context = JavaCL.createBestContext();

		// give some informations about the default device
		System.out.println("Device specs:");
		printDeviceInfos(context.getDevices()[0]);
		System.out.println();

		// create command queue for the first device, want to do some profiling
		CLQueue queue = context.createDefaultProfilingQueue();

		// load image
		BufferedImage inputImage = loadImage(imageFilename);

		// create filter kernel
		Pointer<Float> filterKernelHost = MathHelper.createNormalDistributionFilterKernel(filterKernelSize);
		CLBuffer<Float> filterKernelDevice = context.createBuffer(Usage.Input, filterKernelHost);

		// read the program sources and compile them :
		String src = null;
		try {
			src = IOUtils.readText(ImageBlurOpenCL.class.getResource("imageconvolution.cl"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		CLProgram program = context.createProgram(src);

		// java local timer
		long javaNanoTimerStart = System.nanoTime();

		// create image on device and copy data
		CLImage2D deviceImage0 = context.createImage2D(Usage.InputOutput, inputImage, false);
		// create image on device for output
		CLImage2D deviceImage1 = context.createImage2D(Usage.InputOutput, deviceImage0.getFormat(), inputImage.getWidth(), inputImage.getHeight());

		System.out.println("Device ImageFormat: " + deviceImage0.getFormat());

		// work size
		int[] imageDimension = new int[] { inputImage.getWidth(), inputImage.getHeight() };
		int[] localWorkSize = new int[] { 16, 16 };
		int[] globalWorkSize = new int[] {
				(int) (localWorkSize[0] * Math.ceil((float) imageDimension[0] / localWorkSize[0])),
				(int) (localWorkSize[1] * Math.ceil((float) imageDimension[1] / localWorkSize[1])) };

		// convolve X
		CLKernel convolveKernelX = program.createKernel("convolveX");
		convolveKernelX.setArgs(deviceImage0, deviceImage1, filterKernelDevice, filterKernelSize, imageDimension);
		CLEvent convolutionEventX = convolveKernelX.enqueueNDRange(queue, globalWorkSize, localWorkSize); // call

		// convolve Y
		CLKernel convolveKernelY = program.createKernel("convolveY");
		convolveKernelY.setArgs(deviceImage1, deviceImage0, filterKernelDevice, filterKernelSize, imageDimension);
		CLEvent convolutionEventY = convolveKernelY.enqueueNDRange(queue, globalWorkSize, localWorkSize, convolutionEventX); // call - with wait!

		// read image (will block until computation is finished)
		BufferedImage outputImage = deviceImage0.read(queue, convolutionEventY);
		long javaNanoTimerEnd = System.nanoTime();

		// write to file
		System.out.println("Writing image to file...");
		try {
			File outputFile = new File(imageFilename.substring(0, imageFilename.lastIndexOf('.')) + "_convolved.png");
			ImageIO.write(outputImage, "png", outputFile);
		} catch (IOException e) {
			e.printStackTrace();
		}

		// output time stats
		double totalDeviceComputationTimeMS_withSync = 
				(double) (convolutionEventY.getProfilingCommandEnd() - convolutionEventX.getProfilingCommandStart()) / 1000.0 / 1000.0;
		System.out.println();
		System.out.println("Total computation time on device + sync (ms):\t " + totalDeviceComputationTimeMS_withSync);
		System.out.println("Total time including up and download to device (ms):\t " +
						 (double) (javaNanoTimerEnd - javaNanoTimerStart) / 1000.0 / 1000.);
		System.out.println();

		// no significant difference!
		// double totalDeviceComputationTimeMS =
		// ((double)(convolutionEventX.getProfilingCommandEnd() -
		// convolutionEventX.getProfilingCommandStart()) +
		// (double)(convolutionEventY.getProfilingCommandEnd() -
		// convolutionEventY.getProfilingCommandStart())) / 1000.0 / 1000.0;
		// System.out.println("Total compution time on device): " +
		// totalDeviceComputationTimeMS);

		System.out.println("all done.");
	}

	public static void main(String[] args) {
		String imageFilename = "sampleimage1.png";
		int filterKernelSize = 23; // should be uneven!
		if (args.length >= 1)
			imageFilename = args[0];
		if (args.length >= 2)
			filterKernelSize = Integer.parseInt(args[1]);

		blurImage(imageFilename, filterKernelSize);
	}
}