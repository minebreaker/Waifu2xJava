package waifu2j;

// waifu2x.pyのコピー実装

/*
 * 画像のライセンス表示
 * 「初音ミク」はクリプトン・フューチャー・メディア株式会社の著作物です。
 *  © Crypton Future Media, INC. www.piapro.net
 *
 * 参考
 * https://github.com/nagadomi/waifu2x
 * https://github.com/WL-Amigo/waifu2x-converter-cpp *モデルを借りています
 */

import javax.imageio.ImageIO;
import javax.json.*;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.*;
import java.util.Hashtable;

import static java.lang.Math.max;
import static java.lang.Math.min;

public class Waifu2xPyClonePlain {

    public static void main(String[] args) throws Exception {
        String modelPath = "model.json";
        String inPath = "in_smaller.png";
        String outPath = "out.png";

        JsonArray model; // modelはファイル全体がJSONではなく、JSON配列になっている
        try (JsonReader jsonReader = Json.createReader(ClassLoader.getSystemResourceAsStream(modelPath))) {
            model = jsonReader.readArray();
        }

        BufferedImage loadedImage = ImageIO.read(ClassLoader.getSystemResource(inPath));
//        showImage(loadedImage);
        BufferedImage jImage = new BufferedImage(loadedImage.getWidth() * 2, loadedImage.getHeight() * 2, loadedImage.getType());
        new AffineTransformOp(AffineTransform.getScaleInstance(2.0, 2.0), AffineTransformOp.TYPE_NEAREST_NEIGHBOR)
                .filter(loadedImage, jImage);
//        showImage(jImage);

        // 輝度成分を取り出す
        double[][][] image = convertRgbToYcbcr(rasterToDouble(padWithEdge(jImage.getRaster(), 7)));
        double[][][] planes = new double[1][image.length][image[0].length];
        for (int h = 0; h < image.length; h++) {
            for (int w = 0; w < image[0].length; w++) {
                planes[0][h][w] = image[h][w][0]; // yだけ取り出す
            }
        }

        for (JsonValue stepAsValue : model) {
            JsonObject step = (JsonObject) stepAsValue;
            System.out.println("input:  " + step.getInt("nInputPlane"));
            System.out.println("output: " + step.getInt("nOutputPlane"));

            double[][][] outputPlane = new double[step.getInt("nOutputPlane")][][];

            for (int i = 0; i < step.getJsonArray("bias").size(); i++) {
                JsonArray weights = step.getJsonArray("weight");
                double[][] partial = new double[planes[0].length - 2][planes[0][0].length - 2];
                for (int j = 0; j < weights.getJsonArray(i).size(); j++) {
                    double[][] ip = planes[j];
                    double[][] kernel = new double[3][3];
                    for (int kH = 0; kH < step.getInt("kH"); kH++) {
                        for (int kW = 0; kW < step.getInt("kW"); kW++) {
                            kernel[kH][kW] = step.getJsonArray("weight").getJsonArray(i).getJsonArray(j).getJsonArray(kH).getJsonNumber(kW).doubleValue();
                        }
                    }

                    // 畳みこみ計算
                    double[][] convolved = new double[ip.length - (kernel.length + 1) / 2][ip[0].length - (kernel[0].length + 1) / 2];
                    for (int q = 0; q < convolved.length; q++) {
                        for (int p = 0; p < convolved[0].length; p++) {
                            for (int kh = 0; kh < kernel.length; kh++) {
                                for (int kw = 0; kw < kernel[0].length; kw++) {
                                    partial[q][p] += ip[q + kh][p + kw] * kernel[kh][kw];
                                }
                            }
                        }
                    }
                }

                // バイアスを加える
                double bias = step.getJsonArray("bias").getJsonNumber(i).doubleValue();
                for (int q = 0; q < partial.length; q++) {
                    for (int p = 0; p < partial[0].length; p++) {
                        partial[q][p] += bias;
                    }
                }

                outputPlane[i] = partial;
            }

            for (int i = 0; i < outputPlane.length; i++) {
                for (int j = 0; j < outputPlane[0].length; j++) {
                    for (int k = 0; k < outputPlane[0][0].length; k++) {
                        if (outputPlane[i][j][k] < 0.0) {
                            outputPlane[i][j][k] *= 0.1;
                        }
                    }
                }
            }

            planes = outputPlane;
        }

        double[][][] out = new double[jImage.getHeight()][jImage.getWidth()][];
        for (int h = 0; h < jImage.getHeight(); h++) {
            for (int w = 0; w < jImage.getWidth(); w++) {
                double[] sample = image[h][w];

                sample[0] = planes[0][h][w]; // yをコピー

                out[h][w] = sample;
            }
        }

        showImage(new BufferedImage(
                jImage.getColorModel(), doubleToRaster(convertYcbcrToRgb(out), jImage.getSampleModel()),
                jImage.isAlphaPremultiplied(), new Hashtable<String, Object>()));

    }

    private static WritableRaster padWithEdge(WritableRaster from, int len) {
        WritableRaster out = from.createCompatibleWritableRaster(from.getWidth() + len * 2, from.getHeight() + len * 2);
        for (int h = 0; h < out.getHeight(); h++) {
            for (int w = 0; w < out.getWidth(); w++) {
                if (w < len || h < len || w > out.getWidth() - len - 1 || h > out.getHeight() - len - 1) {
                    out.setPixel(w, h,
                            from.getPixel(
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

    private static WritableRaster toSingleColor(WritableRaster from, int channel) {
        WritableRaster out = Raster.createWritableRaster(from.getSampleModel(), null);

        for (int h = 0; h < from.getHeight(); h++) {
            for (int w = 0; w < from.getWidth(); w++) {
                double[] temp = new double[from.getNumBands()];
                double[] in = from.getPixel(w, h, new double[from.getNumBands()]);
                temp[channel] = in[channel];
                temp[3] = in[3]; // aはそのままコピー
                out.setPixel(w, h, temp);
            }
        }

        return out;
    }

    private static BufferedImage toSingleScale(double[][][] from, int channel) {
        return null;
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
