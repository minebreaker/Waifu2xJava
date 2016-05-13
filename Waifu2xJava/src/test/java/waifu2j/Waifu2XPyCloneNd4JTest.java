package waifu2j;

import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import static org.hamcrest.CoreMatchers.*;

import static org.junit.Assert.assertThat;
import static waifu2j.Waifu2xPyCloneNd4J.padWithEdge;

public class Waifu2XPyCloneNd4JTest {

    @Test
    public void testPad() {
        INDArray from = Nd4j.create(new double[] {
                1, 2, 3, 4, 5, 6, 7, 8},
                new int[] {2, 2, 2});
        INDArray res = padWithEdge(from, 1);

        assertThat(res.getInt(0, 0, 0), is(1));
        assertThat(res.getInt(0, 0, 1), is(1));
        assertThat(res.getInt(0, 0, 2), is(2));
        assertThat(res.getInt(0, 0, 3), is(2));
        assertThat(res.getInt(0, 1, 0), is(1));
        assertThat(res.getInt(0, 1, 1), is(1));
        assertThat(res.getInt(0, 1, 2), is(2));
        assertThat(res.getInt(0, 1, 3), is(2));
        assertThat(res.getInt(0, 2, 0), is(3));
        assertThat(res.getInt(0, 2, 1), is(3));
        assertThat(res.getInt(0, 2, 2), is(4));
        assertThat(res.getInt(0, 2, 3), is(4));
        assertThat(res.getInt(0, 3, 0), is(3));
        assertThat(res.getInt(0, 3, 1), is(3));
        assertThat(res.getInt(0, 3, 2), is(4));
        assertThat(res.getInt(0, 3, 3), is(4));
    }

}
