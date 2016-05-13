package waifu2j;

// waifu2x.pyのコピー実装
// ND4Jを利用

import org.canova.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import javax.json.*;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.time.Instant;
import java.util.function.Function;

import static java.lang.Math.max;
import static java.lang.Math.min;
import static waifu2j.Waifu2xPyClonePlain.scaleUp;
import static waifu2j.Waifu2xPyClonePlain.showImage;

public class Waifu2xPyCloneNd4J {

    public static void main(String[] args) throws Exception {

        Instant timestamp = Instant.now();

        String modelPath = "models/waifu2x-caffe/rgb/scale2.0x_model.json";
        String inPath = "in_smaller.png";

        JsonArray model;
        try (JsonReader jsonReader = Json.createReader(ClassLoader.getSystemResourceAsStream(modelPath))) {
            model = jsonReader.readArray();
        }

        BufferedImage bufferedImage = scaleUp(ImageIO.read(ClassLoader.getSystemResource(inPath)), 2.0);

        INDArray imageOriginal = new ImageLoader().toBgr(bufferedImage);

        // UnsupportedOperation orz
//            image = Nd4j.pad(image.slice(i), new int[] {7, 7, 7}, Nd4j.PadMode.EDGE);
        INDArray image = padWithEdge(imageOriginal, 7);
        image = normalize(image, val -> val / 255.0);

        INDArray planes = Nd4j.zeros(image.size(0) - 1, image.size(1), image.size(2));
        planes.putSlice(0, image.slice(3));
        planes.putSlice(1, image.slice(2));
        planes.putSlice(2, image.slice(1));

        for (JsonValue stepAsValue : model) {
            JsonObject step = (JsonObject) stepAsValue;
            System.out.println("input:  " + step.getInt("nInputPlane"));
            System.out.println("output: " + step.getInt("nOutputPlane"));

            INDArray outputPlanes = Nd4j.zeros(
                    step.getInt("nOutputPlane"),
                    planes.size(1) - (step.getInt("kH") + 1) / 2,
                    planes.size(2) - (step.getInt("kW") + 1) / 2);

            for (int i = 0; i < step.getJsonArray("bias").size(); i++) {

                JsonArray weight = step.getJsonArray("weight").getJsonArray(i);
                INDArray partial = Nd4j.zeros(outputPlanes.size(1), outputPlanes.size(2));

                for (int j = 0; j < weight.size(); j++) {
                    INDArray ip = planes.slice(j);

                    INDArray kernel = Nd4j.zeros(step.getInt("kW"), step.getInt("kH"));
                    for (int kH = 0; kH < step.getInt("kW"); kH++) {
                        for (int kW = 0; kW < step.getInt("kH"); kW++) {
                            kernel.putScalar(
                                    new int[] {kH, kW},
                                    weight.getJsonArray(j).getJsonArray(kH).getJsonNumber(kW).doubleValue());
                        }
                    }

                    // 実装されてないっぽい?
//                    INDArray p = Convolution.conv2d(ip, kernel, Convolution.Type.FULL);
                    int[] pos = new int[2];
                    for (int q = 0; q < partial.size(0); q++)
                        for (int p = 0; p < partial.size(1); p++)
                            for (int kH = 0; kH < kernel.size(0); kH++)
                                for (int kW = 0; kW < kernel.size(1); kW++) {
                                    pos[0] = q;
                                    pos[1] = p;
                                    partial.putScalar(
                                            pos,
                                            partial.getDouble(pos) +
                                                    ip.getDouble(q + kH, p + kW) * kernel.getDouble(kH, kW));
                                }
                }

                double bias = step.getJsonArray("bias").getJsonNumber(i).doubleValue();
                partial.addi(bias);
                outputPlanes.putSlice(i, partial);
            }

            int[] pos = new int[3];
            for (int d = 0; d < outputPlanes.slice(0, 0).length(); d++) {
                for (int h = 0; h < outputPlanes.slice(0, 1).length(); h++) {
                    for (int w = 0; w < outputPlanes.slice(0, 2).length(); w++) {
                        pos[0] = d;
                        pos[1] = h;
                        pos[2] = w;
                        double temp =outputPlanes.getDouble(pos);
                        if (temp < 0) {
                            outputPlanes.putScalar(pos, temp * 0.1);
                        }
                    }
                }
            }
            planes = outputPlanes;
        }

        planes = normalize(planes, val -> {
            val = val * 255;
            if (val > 255)
                return 255.0;
            else if (val < 0)
                return 0.0;
            else
                return val;
        });

        INDArray result = planes;
//        BufferedImage resImage = new BufferedImage(imageOriginal.size(2), imageOriginal.size(1), bufferedImage.getType());
//        for (int h = 0; h < imageOriginal.size(1); h++) {
//            for (int w = 0; w < imageOriginal.size(2); w++) {
//                resImage.getRaster().setPixel(w, h, new double[] {
//                        result.getDouble(0, h, w),
//                        result.getDouble(1, h, w),
//                        result.getDouble(2, h, w),
//                        imageOriginal.getDouble(0, h, w)
//                });
//            }
//        }
        BufferedImage resImage = toImage(result, imageOriginal, bufferedImage.getType());

        System.out.println("start: " + timestamp);
        System.out.println("end:   " + Instant.now());

        showImage(resImage);
    }

    public static INDArray padWithEdge(INDArray src, int len) {
        int channel = src.size(0);
        int height = src.size(1);
        int width = src.size(2);
        INDArray dst = Nd4j.zeros(channel, height + len * 2, width + len * 2);

        for (int h = 0; h < dst.size(1); h++) {
            for (int w = 0; w < dst.size(2); w++) {
                for (int c = 0; c < channel; c++) {
                    if (w < len || h < len || w > width - len - 1 || h > height - len - 1) {
                        dst.putScalar(new int[] {c, h, w}, src.getFloat(new int[] {
                                c,
                                min(max(h, len) - len, height - 1),
                                min(max(w, len) - len, width - 1)
                        }));
                    } else {
                        dst.putScalar(new int[] {c, h, w}, src.getFloat(new int[] {c, h - len, w - len}));
                    }
                }
            }
        }
        return dst;
    }

    private static BufferedImage toImage(INDArray from, INDArray alpha, int imageType) {
        BufferedImage resImage = new BufferedImage(from.size(2), from.size(1), imageType);
        for (int h = 0; h < from.size(1); h++) {
            for (int w = 0; w < from.size(2); w++) {
                resImage.getRaster().setPixel(w, h, new double[] {
                        from.getDouble(0, h, w),
                        from.getDouble(1, h, w),
                        from.getDouble(2, h, w),
                        alpha.getDouble(0, h, w)
                });
            }
        }
        return resImage;
    }

    private static INDArray normalize(INDArray from, Function<Double, Double> eval) {
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

}