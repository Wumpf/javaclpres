import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.util.*;

import org.bridj.Pointer;

public class ImageBlurOpenCL {
	
	/**
	 *  Create context - represents the entire OpenCL environment.
	 */
	private CLContext context;
	
	/**
	 * OpenCL device used for OpenCL Kernels.
	 */
	private CLDevice device;
	
	/**
	 * OpenCL command queue for the choosen device.
	 */
	private CLQueue queue;
	
	
	/**
	 * Initializes CLContext and creates command queue
	 */
	public ImageBlurOpenCL(int platformIndex, int deviceIndex) {
		CLPlatform platform = JavaCL.listPlatforms()[platformIndex];
		device = platform.listAllDevices(false)[deviceIndex];
		context = platform.createContext(null, device);
		queue = device.createProfilingQueue(context);
	}
	
	/**
	 * Lists all available OpenCL platforms
	 */
	public static void listOpenCLPlatforms() {
		CLPlatform[] platforms = JavaCL.listPlatforms();
		for(int i=0; i<platforms.length; ++i) {
			System.out.println("(" + i + ") " + platforms[i].getName());
		}
	}
	
	/**
	 * Lists all available OpenCL devices for the current platform.
	 */
	public static void listOpenCLDevices(int platformIndex) {
		CLDevice[] devices = JavaCL.listPlatforms()[platformIndex].listAllDevices(true);
		for(int i=0; i<devices.length; ++i) {
			System.out.println("(" + i + ") " + devices[i].getName());
		}
	}
	
	/**
	 * Chooses an OpenCL device and creates command queue.
	 * Will create a profiling queue to make timer available.
	 * @param index		Index of the OpenCL device to use
	 */
	public void chooseOpenCLDevice(int index) {
		device = context.getDevices()[index];
		queue = device.createProfilingQueue(context);
	}
	
	
	/**
	 * Prints various informations of the current device to console.
	 */
	public void printCurrentDeviceInfos() {
		System.out.println("Device Name: " + device.getName());
		System.out.println("OpenCL Version: " + device.getOpenCLVersion());
		System.out.println("Local Memory size: " + device.getLocalMemSize());
		System.out.println("Global Memory size: " + device.getGlobalMemSize());
		System.out.println("Global Memory Cache size: " + device.getGlobalMemCacheSize());
		System.out.println("Max Compute units: " + device.getMaxComputeUnits()); 
		System.out.println("Max Work Group size: " + device.getMaxWorkGroupSize());
		System.out.println("Max Image size: " + device.getImage2DMaxWidth() + "x" + device.getImage2DMaxHeight());
	}

	
	/**
	 * loads an image from file or resource
	 * will first try to load from resource, from file if failed
	 * @param filename	filename of the image to load
	 * @return loaded image or null if not successful
	 */
	private BufferedImage loadImage(String filename) {
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
	
	public void blurImage(String imageFilename, int filterKernelSize) {
		assert(queue != null && device != null);
		
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

		// output time statistics
		double totalDeviceComputationTimeMS_withSync = 
				(double) (convolutionEventY.getProfilingCommandEnd() - convolutionEventX.getProfilingCommandStart()) / 1000.0 / 1000.0;
		System.out.println();
		double horizontalConvolutionDeviceComputationTimeMS = 
				(double) (convolutionEventX.getProfilingCommandEnd() - convolutionEventX.getProfilingCommandStart()) / 1000.0 / 1000.0;
		System.out.println("Computation time on device convolution horizontal (ms):\t " + horizontalConvolutionDeviceComputationTimeMS);
		System.out.println("Total computation time on device + intersync (ms):\t " + totalDeviceComputationTimeMS_withSync);
		System.out.println("Total time including up and download to device (ms):\t " +
						 (double) (javaNanoTimerEnd - javaNanoTimerStart) / 1000.0 / 1000.);
		System.out.println();
		
		System.out.println("all done.");
	}
	

	public static void main(String[] args) {
		// parse command arguments
		String imageFilename = "sampleimage1.png";
		int filterKernelSize = 23; // should be uneven!
		if (args.length >= 1)
			imageFilename = args[0];
		if (args.length >= 2)
			filterKernelSize = Integer.parseInt(args[1]);

		// console input
		Scanner consoleIn = new Scanner(System.in);
		
		
		// choose platform
		listOpenCLPlatforms();
		System.out.println("Please choose a platform: ");
		int platformIndex = consoleIn.nextInt();
		System.out.println();
		
		// choose device
		listOpenCLDevices(platformIndex);
		System.out.println("Please choose a device: ");
		int deviceIndex = consoleIn.nextInt();
		System.out.println();
		
		// create OpenCL context
		ImageBlurOpenCL imageBlur = new ImageBlurOpenCL(platformIndex, deviceIndex);
		System.out.println(".. created openCL context successfully..");
		
		// print some device infos
		System.out.println();
		imageBlur.printCurrentDeviceInfos();
		System.out.println();
		
		// do the job
		imageBlur.blurImage(imageFilename, filterKernelSize);
		
		
		// close console input
		consoleIn.close();
	}
}