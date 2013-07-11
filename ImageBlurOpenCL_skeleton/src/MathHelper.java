import static org.bridj.Pointer.allocateFloats;

import java.math.BigInteger;

import org.bridj.Pointer;


public class MathHelper {
	/**
	 * computes factorial
	 * @param n number to compute factorial of
	 * @return BigInteger with the resolution 
	 */
	public static BigInteger factorial(long n) {
		BigInteger fact = new BigInteger("1");
        for (long i = 2; i <= n; i++) {
        	fact = fact.multiply(new BigInteger(String.valueOf(i)));
        }
        return fact;
    }
	
	/**
	 * simple binomial coefficient function 
	 */
	public static BigInteger binomial(long n, long choose){
	    return factorial(n).divide(factorial(choose).multiply(factorial(n - choose)));
	}
	
	/**
	 * Generates 1D binomial filter
	 * @param filterKernelSize number of samples computed
	 */
	public static Pointer<Float> createBinomialFilterKernel(int filterKernelSize) {
		Pointer<Float> output = allocateFloats(filterKernelSize);

		double sum = 0.0f;
		for(int i=0; i<filterKernelSize; ++i) {
			double value = binomial(filterKernelSize-1, i).doubleValue();
			output.set(i, (float)value);
			sum += value;
		}
		
		double sumInv = 1.0 / sum;
		for(int i=0; i<filterKernelSize; ++i) {
			float value = output.get(i);
			output.set(i, (float)(value * sumInv));
		}
	
		return output;
	}
	
	/**
	 * Generates 1D gaussian normal distribution filter
	 * @param filterKernelSize number of samples computed
	 */
	public static Pointer<Float> createNormalDistributionFilterKernel(int filterKernelSize) {
		Pointer<Float> output = allocateFloats(filterKernelSize);

		final double my = 0.0;
		double sigma = 1.0; 
		
		// and thus the internet said: use a blur kernel sized 3*sigma in both directions
		// http://en.wikipedia.org/wiki/Gaussian_blur  
		
		double sum = 0.0f;
		for(int i=0; i<filterKernelSize; ++i) {
			double x = ((double)i / (filterKernelSize-1) * 2.0 - 1.0) * 3 * sigma;
			double value = (1.0 / (sigma * Math.sqrt(2.0 * Math.PI))) *
							Math.exp(- (x-my)*(x-my) / (2*sigma*sigma));
			output.set(i, (float)value);
			sum += value;
		}
		
		// normalize (since its not exactly summed to 1)
		for(int i=0; i<filterKernelSize; ++i) {
			double value = output.get(i) / sum;
			output.set(i, (float)value);
		}
		
		
		return output;
	}
}
