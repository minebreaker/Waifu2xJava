package waifu2j;

// waifu2x.pyのコピー実装

/*
 * 画像のライセンス表示
 *
 * 「初音ミク」はクリプトン・フューチャー・メディア株式会社の著作物です。
 *  © Crypton Future Media, INC. www.piapro.net
 */

import org.canova.image.loader.ImageLoader;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.convolution.Convolution;
import org.nd4j.linalg.convolution.DefaultConvolutionInstance;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.cpu.complex.ComplexNDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import javax.json.*;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.function.Function;

public class Waifu2xPyClone {

    public static void main(String[] args) throws Exception {
        String modelPath = "model.json";
        String inPath = "in_smaller.png";
        String outPath = "out.png";

        JsonArray model; // modelはファイル全体がJSONではなく、JSON配列になっている
        try (JsonReader jsonReader = Json.createReader(ClassLoader.getSystemResourceAsStream(modelPath))) {
            model = jsonReader.readArray();
        }

        BufferedImage temp = ImageIO.read(ClassLoader.getSystemResource(inPath));
//        showImage(temp);
        BufferedImage jImage = new BufferedImage(temp.getWidth() * 2, temp.getHeight() * 2, temp.getType());
        new AffineTransformOp(AffineTransform.getScaleInstance(2.0, 2.0), AffineTransformOp.TYPE_NEAREST_NEIGHBOR)
                .filter(temp, jImage);
//        showImage(jImage);

        // 輝度成分を取り出す
        INDArray image = convertRgbToYcbcr(new ImageLoader().toBgr(jImage));
        image = regularize(image, val -> val / 255);
        INDArray planes = new NDArray(new int[] {1, image.slice(0, 1).length(), image.slice(0, 2).length()});
        planes.putSlice(0, image.slice(0));

        for (JsonValue stepAsValue : model) {
            JsonObject step = (JsonObject) stepAsValue;
            System.out.println("input:  " + step.getInt("nInputPlane"));
            System.out.println("output: " + step.getInt("nOutputPlane"));

            INDArray outputPlanes = Nd4j.zeros(step.getInt("nOutputPlane"), image.slice(0, 1).length(), image.slice(0, 2).length());

            for (int i = 0; i < step.getJsonArray("bias").size(); i++) {
                INDArray partial = Nd4j.zeros(3, 3); // 固定値でいいはず(kW, kH)

                for (int j = 0; j < step.getJsonArray("weight").size(); j++) {
                    INDArray ip = planes.slice(0);

                    INDArray kernel = Nd4j.zeros(step.getInt("kW"), step.getInt("kH"));
                    for (int k = 0; k < step.getInt("kW"); k++) {
                        for (int l = 0; l < step.getInt("kH"); l++) {
                            kernel.put(k, l, step.getJsonArray("weight").getJsonArray(i).getJsonArray(j).getJsonArray(k).getJsonNumber(l).doubleValue());
                        }
                    }
//                    System.out.println(ip);
//                    System.out.println(kernel);

                    // org.nd4j.linalg.api.ops.impl.transforms.convolutionに移行する模様。
                    // まだim2colとcol2imしかない。convは実装されるのだろうか。
                    INDArray p = Convolution.convn(ip, kernel, Convolution.Type.FULL);

                    partial.add(p);
                }

                partial.add(step.getJsonArray("bias").getJsonNumber(i).doubleValue());
                outputPlanes.putSlice(i, partial);
            }

            for (int d = 0; d < outputPlanes.slice(0, 0).length(); d++) {
                for (int h = 0; h < outputPlanes.slice(0, 1).length(); h++) {
                    for (int w = 0; w < outputPlanes.slice(0, 2).length(); w++) {
                        if (outputPlanes.getFloat(new int[] {d, h, w}) < 0) {
                            outputPlanes.muli(0.1);
                            break;
                        }
                    }
                }
            }
            planes = outputPlanes;
        }

        image = regularize(image, val -> {
            val = val * 255;
            if (val > 255)
                return 255.0;
            else if (val < 0)
                return 0.0;
            else
                return val;
        });
//        INDArray planes = new NDArray(new int[] {1, image.slice(0, 1).length(), image.slice(0, 2).length()});
        image.putSlice(0, image);
//        showImage(toImage(convertYcbcrToRgb(image), jImage.getType()));
//        showImage((toImage(toSingleColor(image, 1), jImage.getType())));
//        showImage(toSingleScale(imageYCbCr, 0));
    }

    private static BufferedImage toImage(INDArray from, int ImageType) {
        int height = from.slice(0, 1).length();
        int width = from.slice(0, 2).length();

        BufferedImage out = new BufferedImage(width, height, ImageType);

        INDArray r = from.slice(3);
        INDArray g = from.slice(2);
        INDArray b = from.slice(1);
        INDArray a = from.slice(0);

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                WritableRaster raster = out.getRaster();
                raster.setPixel(w, h, new int[] {r.getInt(h, w), g.getInt(h, w), b.getInt(h, w), a.getInt(h, w)});
            }
        }
        return out;
    }

    private static INDArray regularize(INDArray from, Function<Double, Double> eval) {
        INDArray out = Nd4j.zeros(from.slice(0, 0).length(), from.slice(0, 1).length(), from.slice(0, 2).length());

        for (int d = 0; d < from.slice(0, 0).length(); d++) {
            for (int h = 0; h < from.slice(0, 1).length(); h++) {
                for (int w = 0; w < from.slice(0, 2).length(); w++) {
                    out.putScalar(new int[] {d, h, w}, eval.apply(from.getDouble(d, h, w)));
                }
            }
        }

        return out;
    }

    private static BufferedImage toSingleScale(INDArray from, int channel) {
        BufferedImage out = new BufferedImage(
                from.slice(0, 2).length(), from.slice(0, 1).length(), BufferedImage.TYPE_BYTE_GRAY);

        for (int h = 0; h < from.slice(0, 1).length(); h++) {
            for (int w = 0; w < from.slice(0, 2).length(); w++) {
                WritableRaster raster = out.getRaster();
                raster.setPixel(w, h, new int[] {(int) from.getFloat(new int[] {channel, h, w})});
            }
        }
        return out;
    }

    private static INDArray toSingleColor(INDArray from, int channel) {
        INDArray out = Nd4j.zeros(from.slice(0, 0).length(), from.slice(0, 1).length(), from.slice(0, 2).length());

        for (int h = 0; h < from.slice(0, 1).length(); h++) {
            for (int w = 0; w < from.slice(0, 2).length(); w++) {
                out.putScalar(new int[] {3, h, w}, 0);
                out.putScalar(new int[] {2, h, w}, 0);
                out.putScalar(new int[] {1, h, w}, 0);
                out.putScalar(new int[] {0, h, w}, 255);
                out.putScalar(new int[] {channel, h, w}, from.getFloat(new int[] {channel, h, w}));
            }
        }

        return out;
    }

    private static INDArray convertRgbToYcbcr(INDArray from) {

        INDArray out = Nd4j.zeros(from.slice(0, 0).length(), from.slice(0, 1).length(), from.slice(0, 2).length());

        INDArray r = from.slice(3);
        INDArray g = from.slice(2);
        INDArray b = from.slice(1);
        INDArray a = from.slice(0);

        for (int h = 0; h < from.slice(0, 1).length(); h++) {
            for (int w = 0; w < from.slice(0, 2).length(); w++) {
                float rp = r.getFloat(h, w);
                float gp = g.getFloat(h, w);
                float bp = b.getFloat(h, w);
                float ap = a.getFloat(h, w);

                double y = 0.299 * rp + 0.587 * gp + 0.114 * bp;
                double cb = -0.169 * rp - 0.331 * gp + 0.500 * bp;
                double cr = 0.500 * rp - 0.419 * gp - 0.081 * bp;

//                System.out.println(y + ", " + cb + ", " + cr);

                out.putScalar(new int[] {0, h, w}, y);
                out.putScalar(new int[] {1, h, w}, cb);
                out.putScalar(new int[] {2, h, w}, cr);
                out.putScalar(new int[] {3, h, w}, ap);
            }
        }

        return out;
    }

    private static INDArray convertYcbcrToRgb(INDArray from) {

        INDArray out = Nd4j.zeros(from.slice(0, 0).length(), from.slice(0, 1).length(), from.slice(0, 2).length());

        INDArray y = from.slice(0);
        INDArray cb = from.slice(1);
        INDArray cr = from.slice(2);
        INDArray a = from.slice(3);

        for (int h = 0; h < from.slice(0, 1).length(); h++) {
            for (int w = 0; w < from.slice(0, 2).length(); w++) {
                float yp = y.getFloat(h, w);
                float cbp = cb.getFloat(h, w);
                float crp = cr.getFloat(h, w);
                float ap = a.getFloat(h, w);

                double r = yp + 1.402 * crp;
                double g = yp - 0.344 * cbp - 0.581 * crp;
                double b = yp + 2.032 * cbp;

//                System.out.println(y + ", " + cb + ", " + cr);

                out.putScalar(new int[] {0, h, w}, r);
                out.putScalar(new int[] {1, h, w}, g);
                out.putScalar(new int[] {2, h, w}, b);
                out.putScalar(new int[] {3, h, w}, ap);
            }
        }

        return out;
    }

    private static void outputArray(INDArray array, String filename) {
        try (Writer writer = Files.newBufferedWriter(Paths.get(filename))) {
            writer.write(array.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("array written.");
    }

}
