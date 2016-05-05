package waifu2j;

import org.junit.Test;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.convolution.DefaultConvolutionInstance;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.cpu.complex.ComplexNDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.omg.PortableInterceptor.SYSTEM_EXCEPTION;

public class ND4JSample {

    @Test
    public void test2d() {
        INDArray nd = Nd4j.create(new float[] {1, 2, 3, 4}, new int[] {2, 2});
        System.out.println(nd);
        System.out.println(nd.getRow(0));
        System.out.println(nd.getColumn(0));

        System.out.println(nd.getInt(0, 0));
        System.out.println(nd.getInt(0, 1));
        System.out.println(nd.getInt(1, 0));
        System.out.println(nd.getInt(1, 1));
    }

    @Test
    public void test3d() {
        INDArray nd = Nd4j.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, new int[] {2, 2, 2});
        System.out.println(nd);
        System.out.println(nd.getFloat(new int[] {0, 0, 0})); // 1
        System.out.println(nd.getFloat(new int[] {1, 0, 0})); // 5
        System.out.println(nd.getFloat(new int[] {1, 1, 1})); // 8
    }

    @Test
    public void testSlicing() {
        INDArray nd = Nd4j.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, new int[] {2, 2, 2});
        // [[[1, 2]
        //   [3, 4]]
        //  [[5, 6]
        //   [7, 8]]]
        System.out.println(nd);
        System.out.println(nd.slice(0)); // 1, 2; 3, 4
        System.out.println(nd.slice(1)); // 5, 6; 7, 8
        System.out.println(nd.slice(0, 0)); // 1, 5
        System.out.println(nd.slice(0, 1)); // 1, 3
        System.out.println(nd.slice(0, 2)); // 1, 2

        INDArray nd2 = Nd4j.zeros(1, 2, 2);
        nd2.putSlice(0, nd.slice(0));
        System.out.println(nd2);
    }

    @Test
    public void testYCbCr() {
        INDArray image = Nd4j.create(new float[] {
                0, 50, 100, 150,
                0, 80, 160, 255,
                0, 110, 220, 255
        }, new int[] {3, 2, 2});
        System.out.println(image);

        INDArray out = Nd4j.zeros(3, 2, 2);

        INDArray r = image.slice(0);
        INDArray g = image.slice(1);
        INDArray b = image.slice(2);
        for (int h = 0 ; h < 2 ; h++) {
            for (int w = 0 ; w < 2 ; w++) {
                float rp =r.getFloat(h, w);
                float gp =g.getFloat(h, w);
                float bp =b.getFloat(h, w);

                double y = 0.299 * rp + 0.587 * gp + 0.114 * bp;
                double cb = -0.169 * rp - 0.331 * gp + 0.500 * bp;
                double cr = 0.500 * rp - 0.419 * gp - 0.081 * bp;

                System.out.println(y + ", " + cb + ", " + cr);

                out.putScalar(new int[] {0, h, w}, y);
                out.putScalar(new int[] {1, h, w}, cb);
                out.putScalar(new int[] {2, h, w}, cr);
            }
        }

        System.out.println(out);
    }

    @Test
    public void convTest() {
        INDArray nd = Nd4j.create(new float[] {100, 200, 300, 400, 500, 600, 700, 800, 900}, new int[] {3, 3});

        System.out.println(nd);
        INDArray kernel = Nd4j.create(new double[] {0.5, 0.1, 0.1, 0.1}, new int[] {2, 2});

        System.out.println(Convolution.convn(nd, kernel, Convolution.Type.VALID));
    }

}
