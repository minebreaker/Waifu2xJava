package waifu2j;

import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.linear.ArrayFieldVector;
import org.apache.commons.math3.linear.FieldVector;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;
import org.junit.Test;

import java.util.Arrays;

public class CommonsTest {

    @Test
    public void testConv1d() {
        FastFourierTransformer fft = new FastFourierTransformer(DftNormalization.STANDARD);

        double[] src = new double[] {1, 2, 3, 4, 5, 6, 7, 8};
        double[] kernel = new double[] {1, 1, 1, 0, 0, 0, 0, 0};

        FieldVector<Complex> fSrc = new ArrayFieldVector<>(fft.transform(src, TransformType.FORWARD));
        FieldVector<Complex> fK = new ArrayFieldVector<>(fft.transform(kernel, TransformType.FORWARD));

        System.out.println(Arrays.toString(fSrc.toArray()));
        System.out.println(Arrays.toString(fK.toArray()));

        FieldVector<Complex> mul = fSrc.ebeMultiply(fK);
        System.out.println(Arrays.toString(mul.toArray()));

        Complex[] res = fft.transform(mul.toArray(), TransformType.INVERSE);
        System.out.println(Arrays.toString(res));
    }

}
