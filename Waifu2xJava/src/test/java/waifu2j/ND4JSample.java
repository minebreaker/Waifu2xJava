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
        // [[[1, 2]
        //   [3, 4]]
        //  [[5, 6]
        //   [7, 8]]]
        System.out.println(nd);
        System.out.println(nd.getFloat(new int[] {0, 0, 0})); // 1
        System.out.println(nd.getFloat(new int[] {1, 0, 0})); // 5
        System.out.println(nd.getFloat(new int[] {0, 1, 0})); // 3
        System.out.println(nd.getFloat(new int[] {0, 0, 1})); // 2
        System.out.println(nd.getFloat(new int[] {1, 1, 1})); // 8

        nd.putScalar(new int[] {0, 0, 0}, 100);
        nd.putScalar(new int[] {1, 0, 0}, 110);
        nd.putScalar(new int[] {0, 1, 0}, 120);
        nd.putScalar(new int[] {0, 0, 1}, 130);
        System.out.println(nd);
        // [[[100.00, 130.00]
        //   [120.00, 4.00  ]]
        //  [[110.00, 6.00  ]
        //   [7.00,   8.00  ]]]
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
    public void testSlicing2() {
        INDArray nd = Nd4j.create(
                new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
                new int[] {2, 2, 2, 2});
        System.out.println(nd);
        System.out.println(nd.slice(0));
        System.out.println(nd.slice(0).slice(0));
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
        for (int h = 0; h < 2; h++) {
            for (int w = 0; w < 2; w++) {
                float rp = r.getFloat(h, w);
                float gp = g.getFloat(h, w);
                float bp = b.getFloat(h, w);

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
        INDArray cnd = Nd4j.create(new float[] {1, 1, 1, 1, 1, 1, 1, 1, 1}, new int[] {3, 3});
        System.out.println(cnd);

        INDArray kernel = Nd4j.create(new float[] {1, 1, 1, 1}, new int[] {2, 2});

        // なんか知らんけど必ず0を返す
        System.out.println(Convolution.convn(cnd, kernel, Convolution.Type.VALID));
    }

    @Test
    public void testStack() {
        INDArray a = Nd4j.create(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, new int[] {2, 2, 2});
        INDArray b = Nd4j.create(new float[] {110, 120, 130, 140, 150, 160, 170, 180}, new int[] {2, 2, 2});
        System.out.println(Nd4j.hstack(a, b));
        System.out.println(Nd4j.vstack(a, b));
    }

    @Test
    public void testIndex() {
        INDArray nd = Nd4j.linspace(1, 18, 18).reshape(2, 3, 3);
        System.out.println(nd);
        System.out.println(nd.get(NDArrayIndex.point(0), NDArrayIndex.point(0), NDArrayIndex.point(0)));
        System.out.println(nd.get(NDArrayIndex.point(1), NDArrayIndex.point(0), NDArrayIndex.point(2)));
        System.out.println(nd.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3)));
        System.out.println(nd.get(NDArrayIndex.interval(0, 1), NDArrayIndex.interval(1, 2, true), NDArrayIndex.interval(1, 2, true)));
        System.out.println(nd.get(NDArrayIndex.interval(1, 2), NDArrayIndex.interval(1, 3), NDArrayIndex.interval(1, 3)));
    }

    @Test
    public void testCalc() {
        INDArray a = Nd4j.linspace(1, 4, 4).reshape(2, 2);
        INDArray b = Nd4j.linspace(1, 4, 4).reshape(2, 2);

        System.out.println(a.mul(2));
        System.out.println(a.mul(b));

        System.out.println(a.div(2));
        System.out.println(a.div(b));

        INDArray c = Nd4j.linspace(1, 9, 9).reshape(3, 3);
        try {
            System.out.println(c.mul(a));
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        }
        try {
            System.out.println(c.div(a));
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        }

        System.out.println(a.sumNumber());
    }

}
