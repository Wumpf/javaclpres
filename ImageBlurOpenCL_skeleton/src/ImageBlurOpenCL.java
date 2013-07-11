import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import javax.imageio.ImageIO;

import com.nativelibs4java.opencl.*;
import com.nativelibs4java.opencl.CLMem.Usage;
import com.nativelibs4java.util.*;

import org.bridj.Pointer;

public class ImageBlurOpenCL {
	
	/**
	 * Initializes CLContext and creates command queue
	 */
	public ImageBlurOpenCL(int platformIndex, int deviceIndex) {
	}
	
	/**
	 * Lists all available OpenCL platforms
	 */
	public static void listOpenCLPlatforms() {
	}
	
	/**
	 * Lists all available OpenCL devices for the current platform.
	 */
	public static void listOpenCLDevices(int platformIndex) {
	}
	
	/**
	 * Prints various informations of the current device to console.
	 */
	public void printCurrentDeviceInfos() {
/*		System.out.println("Device Name: " + device.getName());
		System.out.println("OpenCL Version: " + device.getOpenCLVersion());
		System.out.println("Local Memory size: " + device.getLocalMemSize());
		System.out.println("Global Memory size: " + device.getGlobalMemSize());
		System.out.println("Global Memory Cache size: " + device.getGlobalMemCacheSize());
		System.out.println("Max Compute units: " + device.getMaxComputeUnits()); 
		System.out.println("Max Work Group size: " + device.getMaxWorkGroupSize());
		System.out.println("Max Image size: " + device.getImage2DMaxWidth() + "x" + device.getImage2DMaxHeight()); */
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
		
		// load image
		BufferedImage inputImage = loadImage(imageFilename);

		// TODO: create filter kernel as constant buffer
		
		// read the program sources and compile them :
		String src = null;
		try {
			src = IOUtils.readText(ImageBlurOpenCL.class.getResource("imageconvolution.cl"));
		} catch (IOException e) {
			e.printStackTrace();
		}
		// TODO: create program

		// java local timer
		long javaNanoTimerStart = System.nanoTime();

		// TODO: create data, some computation
	
		// TODO: read image back (will block until computation is finished)
		
		long javaNanoTimerEnd = System.nanoTime();

		// write to file
/*		System.out.println("Writing image to file...");
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
		*/
		
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