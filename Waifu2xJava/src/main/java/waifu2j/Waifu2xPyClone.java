package waifu2j;

// waifu2x.pyのコピー実装

/*
 * 画像のライセンス表示
 *
 * 「初音ミク」はクリプトン・フューチャー・メディア株式会社の著作物です。
 *  © Crypton Future Media, INC. www.piapro.net
 */

import org.canova.image.loader.ImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.imageio.ImageIO;
import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonReader;
import javax.swing.*;
import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

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

        INDArray image = new ImageLoader().toBgr(jImage);
        // 輝度成分を取り出す
        INDArray imageYCbCr = convertRgbToYcrcb(image);
        if (image.length() != imageYCbCr.length()) {
            throw new Exception("size differs: " + image.length() + " != " + imageYCbCr.length());
        }

//        jImage = ImageLoader.toImage(image); // なんかこいつだと上手く動かない
        // マトリックスに使うらしいが…
        new ImageLoader().toBufferedImageRGB(imageYCbCr, jImage);

        showImage(jImage);
    }

    private static INDArray convertRgbToYcrcb(INDArray from) {
//        for (int i = 0 ; i < 4 ; i++) { // 色の種類: 4回 (RGBA)
//            for (int h = 0 ; h < jImage.getHeight() ; h++) {
//                for (int w = 0 ; w < jImage.getWidth() ; w++) {
//                    int[] index = new int[] {i, h, w};
//                    double r = image.getFloat(new int[] {});
//                    double g = image.getFloat(new int[] {});
//                    double b = image.getFloat(new int[] {});
//                    double value;
//
//                    switch (i) {
//                    case 0:
//                        image.getFloat(index);
//                        value = 0.299 * r;
//                        break;
//                    default:
//                        throw new IndexOutOfBoundsException("too many channels");
//                    }
//
//                    value = value / 255;
//                    imageYCbCr.putScalar(index, value);
//                }
//            }
//        }

        int originalFacts = from.slice(0, 0).length();
        int originalHeight = from.slice(0, 1).length();
        int originalWidth = from.slice(0, 2).length();

        INDArray out = Nd4j.zeros(originalFacts, originalHeight, originalWidth);

        INDArray r = from.slice(0);
        INDArray g = from.slice(1);
        INDArray b = from.slice(2);
        for (int h = 0 ; h < 2 ; h++) {
            for (int w = 0 ; w < 2 ; w++) {
                float rp =r.getFloat(h, w);
                float gp =g.getFloat(h, w);
                float bp =b.getFloat(h, w);

                double y = 0.299 * rp + 0.587 * gp + 0.114 * bp;
                double cb = -0.169 * rp - 0.331 * gp + 0.500 * bp;
                double cr = 0.500 * rp - 0.419 * gp - 0.081 * bp;

//                System.out.println(y + ", " + cb + ", " + cr);

                out.putScalar(new int[] {0, h, w}, y);
                out.putScalar(new int[] {1, h, w}, cb);
                out.putScalar(new int[] {2, h, w}, cr);
                out.putScalar(new int[] {3, h, w}, from.getFloat(new int[] {3, h, w}));
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

    private static void showImage(BufferedImage image) {
        JFrame frame = new JFrame();

        frame.getContentPane().add(new JPanel() {
            @Override
            public void paintComponent(Graphics graphics) {
                graphics.drawImage(image, 0, 0, this);
            }
        });

        frame.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        frame.getContentPane().setPreferredSize(new Dimension(image.getWidth(), image.getHeight()));
        frame.pack();
        frame.setVisible(true);
    }

}
