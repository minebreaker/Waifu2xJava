package waifu2j;

// waifu2x.pyのコピー実装
// Java SEのクラスのみを使用

import javax.imageio.ImageIO;
import javax.json.*;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.*;
import java.time.Instant;
import java.util.Hashtable;
import java.util.function.BiFunction;
import java.util.function.Function;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class Waifu2xPyClonePlain {

    public static void main(String[] args) throws Exception {

        Instant timestamp = Instant.now();

        String modelPath = "models/waifu2x-caffe/y/scale2.0x_model.json";
        String inPath = "in.png";

        JsonArray model; // modelはファイル全体がJSONではなく、JSON配列になっている
        try (JsonReader jsonReader = Json.createReader(ClassLoader.getSystemResourceAsStream(modelPath))) {
            model = jsonReader.readArray();
        }

        // 画像を読み込みニアレストで2倍に拡大
        BufferedImage bufferedImage = scaleUp(ImageIO.read(ClassLoader.getSystemResource(inPath)), 2.0);

        // 畳みこみで画像サイズが小さくならないよう、ネットの階層分、パディングを行う
        int padLength = model.size(); // パディングするサイズ
        double[][][] image = convertRgbToYcbcr(rasterToDouble(padWithEdge(bufferedImage.getRaster(), padLength)));

        double[][][] planes = new double[][][] {
                createArray2d(image.length, image[0].length, (q, p) -> image[q][p][0]) // yだけ取り出す
        };

        // CNNに通す
        for (JsonValue stepAsValue : model) {
            JsonObject step = (JsonObject) stepAsValue;
            System.out.println("input:  " + step.getInt("nInputPlane"));
            System.out.println("output: " + step.getInt("nOutputPlane"));

            double[][][] outputPlane = new double[step.getInt("nOutputPlane")][][];

            for (int i = 0; i < step.getJsonArray("bias").size(); i++) {

                JsonArray weight = step.getJsonArray("weight").getJsonArray(i);
                double[][] partial = new double
                        [planes[0].length - (step.getInt("kH") + 1) / 2]
                        [planes[0][0].length - (step.getInt("kW") + 1) / 2];

                for (int j = 0; j < weight.size(); j++) {
                    double[][] ip = planes[j];

                    JsonArray each = weight.getJsonArray(j); // なぜかそのままだとコンパイラーが通らない……
                    double[][] kernel = createArray2d(step.getInt("kH"), step.getInt("kW"),
                            (q, p) -> each.getJsonArray(q).getJsonNumber(p).doubleValue());

                    // 畳みこみ計算
                    for (int q = 0; q < partial.length; q++)
                        for (int p = 0; p < partial[0].length; p++)
                            for (int kH = 0; kH < kernel.length; kH++)
                                for (int kW = 0; kW < kernel[0].length; kW++)
                                    partial[q][p] += ip[q + kH][p + kW] * kernel[kH][kW];
                }

                // バイアスを加える
                double bias = step.getJsonArray("bias").getJsonNumber(i).doubleValue();
                applyAllIn(partial, val -> val + bias);

                outputPlane[i] = partial;
            }

            // 活性化関数: f(x) = { x >= 0 : x, x < 0 : 0.1 x }
            for (double[][] eachPlane : outputPlane) {
                applyAllIn(eachPlane, val -> {
                    if (val < 0.0) return val * 0.1;
                    else return val;
                });
            }

            planes = outputPlane;
        }

        // CNNを通したYと、元のままのCbCrを合成する
        double[][][] out = new double[bufferedImage.getHeight()][bufferedImage.getWidth()][];
        for (int h = 0; h < bufferedImage.getHeight(); h++) {
            for (int w = 0; w < bufferedImage.getWidth(); w++) {
                double[] sample = image[h + padLength][w + padLength];
                sample[0] = planes[0][h][w]; // yをコピー

                out[h][w] = sample;
            }
        }

        BufferedImage result = new BufferedImage(
                bufferedImage.getColorModel(), doubleToRaster(convertYcbcrToRgb(out), bufferedImage.getSampleModel()),
                bufferedImage.isAlphaPremultiplied(), new Hashtable<String, Object>());

        System.out.println("start: " + timestamp);
        System.out.println("end:   " + Instant.now());

        showImage(result);

    }

    private static void applyAllIn(double[][] src, Function<Double, Double> func) {
        for (int q = 0; q < src.length; q++)
            for (int p = 0; p < src[0].length; p++)
                src[q][p] = func.apply(src[q][p]);
    }

    private static double[][] createArray2d(int height, int width, BiFunction<Integer, Integer, Double> func) {
        double[][] out = new double[height][width];
        for (int q = 0; q < height; q++)
            for (int p = 0; p < width; p++)
                out[q][p] = func.apply(q, p);
        return out;
    }

    public static BufferedImage scaleUp(BufferedImage src, double rate) {
        BufferedImage dst = new BufferedImage(
                src.getWidth() * (int) rate, src.getHeight() * (int) rate, src.getType());
        new AffineTransformOp(AffineTransform.getScaleInstance(rate, rate), AffineTransformOp.TYPE_NEAREST_NEIGHBOR)
                .filter(src, dst);
        return dst;
    }

    private static WritableRaster padWithEdge(WritableRaster from, int len) {
        WritableRaster out = from.createCompatibleWritableRaster(from.getWidth() + len * 2, from.getHeight() + len * 2);
        for (int h = 0; h < out.getHeight(); h++) {
            for (int w = 0; w < out.getWidth(); w++) {
                if (w < len || h < len || w > out.getWidth() - len - 1 || h > out.getHeight() - len - 1) {
                    out.setPixel(w, h, from.getPixel(
                                    min(max(w, len) - len, from.getWidth() - 1),
                                    min(max(h, len) - len, from.getHeight() - 1),
                                    new double[from.getNumBands()]));
                } else {
                    out.setPixel(w, h, from.getPixel(w - len, h - len, new double[from.getNumBands()]));
                }
            }
        }
        return out;
    }

    private static double[][][] convertRgbToYcbcr(double[][][] from) {
        double[][][] out = new double[from.length][from[0].length][];
        for (int h = 0; h < from.length; h++) {
            for (int w = 0; w < from[0].length; w++) {
                double[] sampleIn = from[h][w];
                double r = sampleIn[0] / 255;
                double g = sampleIn[1] / 255;
                double b = sampleIn[2] / 255;
                double a = sampleIn[3] / 255;

                double y = 0.299 * r + 0.587 * g + 0.114 * b;
                double cb = -0.168736 * r - 0.331264 * g + 0.500 * b;
                double cr = 0.500 * r - 0.418688 * g - 0.081312 * b;

                out[h][w] = new double[] {y, cb, cr, a};
            }
        }
        return out;
    }

    private static double[][][] convertYcbcrToRgb(double[][][] from) {
        double[][][] out = new double[from.length][from[0].length][];
        for (int h = 0; h < from.length; h++) {
            for (int w = 0; w < from[0].length; w++) {
                double[] sampleIn = from[h][w];

                double y = sampleIn[0];
                double cb = sampleIn[1];
                double cr = sampleIn[2];
                double a = sampleIn[3];

                double r = y + 1.402 * cr;
                double g = y - 0.344136 * cb - 0.714136 * cr;
                double b = y + 1.772 * cb;

                r *= 255;
                g *= 255;
                b *= 255;
                a *= 255;

                r = max(min(r, 255), 0);
                g = max(min(g, 255), 0);
                b = max(min(b, 255), 0);
                a = max(min(a, 255), 0);

                out[h][w] = new double[] {r, g, b, a};
            }
        }
        return out;
    }

    private static double[][][] rasterToDouble(WritableRaster from) {
        double[][][] out = new double[from.getHeight()][from.getWidth()][];
        for (int h = 0; h < from.getHeight(); h++) {
            for (int w = 0; w < from.getWidth(); w++) {
                out[h][w] = from.getPixel(w, h, new double[from.getNumBands()]);
            }
        }
        return out;
    }

    private static WritableRaster doubleToRaster(double[][][] from, SampleModel model) {
        WritableRaster out = Raster.createWritableRaster(model, null);
        for (int h = 0; h < from.length; h++) {
            for (int w = 0; w < from[0].length; w++) {
                out.setPixel(w, h, from[h][w]);
            }
        }
        return out;
    }

    public static void showImage(BufferedImage image) {
        JFrame frame = new JFrame();

        frame.getContentPane().add(new JPanel() {
            @Override
            public void paintComponent(Graphics graphics) {
                graphics.drawImage(image, 0, 0, this);
            }
        });

        frame.setDefaultCloseOperation(WindowConstants.DISPOSE_ON_CLOSE);
        frame.getContentPane().setPreferredSize(new Dimension(image.getWidth(), image.getHeight()));
        frame.pack();
        frame.setVisible(true);
    }

}
