package waifu2j;

import org.jtransforms.fft.DoubleFFT_1D;
import org.jtransforms.fft.DoubleFFT_2D;
import org.junit.Test;

import java.util.Arrays;

public class JTransformTest {

    @Test
    public void testConv1d() {

        DoubleFFT_1D fft = new DoubleFFT_1D(8);

        double[] src = new double[] {1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0, 0, 0, 0};
        double[] kernel = new double[] {1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

        fft.realForwardFull(src);
        fft.realForwardFull(kernel);

        System.out.println(Arrays.toString(src));
        System.out.println(Arrays.toString(kernel));

        double[] mul = new double[16];
        for (int i = 0; i < src.length; i += 2) {
            mul[i] = src[i] * kernel[i] - src[i + 1] * kernel[i + 1];
            mul[i + 1] = src[i + 1] * kernel[i] + src[i] * kernel[i + 1];
        }

        System.out.println(Arrays.toString(mul));

        fft.complexInverse(mul, true);
        System.out.println(Arrays.toString(mul));

    }

    @Test
    public void testConv2d() {

        DoubleFFT_2D fft = new DoubleFFT_2D(4, 4);

        double[][] src = new double[][] {{1, 2, 3, 0, 0, 0, 0, 0}, {4, 5, 6, 0, 0, 0, 0, 0}, {7, 8, 9, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};
        double[][] kernel = new double[][] {{1, 1, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 0}};

        fft.realForwardFull(src);
        fft.realForwardFull(kernel);

        double[][] mul = new double[16][16];
        for (int i = 0; i < src.length; i++) {
            for (int j = 0; j < src[0].length; j += 2) {
                mul[i][j] = src[i][j] * kernel[i][j] - src[i][j + 1] * kernel[i][j + 1];
                mul[i][j + 1] = src[i][j + 1] * kernel[i][j] + src[i][j] * kernel[i][j + 1];
            }
        }

        fft.complexInverse(mul, true);
        // [ [12, 16], [24, 28] ]
        System.out.println(Arrays.toString(mul[0]));
        System.out.println(Arrays.toString(mul[1]));
        System.out.println(Arrays.toString(mul[2]));
        System.out.println(Arrays.toString(mul[3]));
    }

}
