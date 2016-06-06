package waifu2j;

import org.junit.Test;

import java.util.Arrays;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.*;
import static waifu2j.Waifu2xPyClonePlain.*;

public class Waifu2xPyClonePlainTest {

    @Test
    public void testNearestPowerOf2() {
        assertThat(nearestPowerOf2(3), is(32)); // 初期化値
        assertThat(nearestPowerOf2(33), is(64));
        assertThat(nearestPowerOf2(65), is(128));
    }

    @Test
    public void testConvFft() {
        double[][] src = new double[][] {{1, 2, 3, 4}, {4, 5, 6, 7}, {7, 8, 9, 10}};
        double[][] kernel = new double[][] {{1, 1}, {1, 1}};
        double[][] dst = new double[2][2];
        convolveFft(src, kernel, dst);
        System.out.println(Arrays.toString(dst[0]));
        System.out.println(Arrays.toString(dst[1]));
    }

}