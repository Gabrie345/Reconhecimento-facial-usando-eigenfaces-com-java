package org.JavaCV;

import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.MatVector;
import org.bytedeco.opencv.opencv_face.EigenFaceRecognizer;
import org.bytedeco.opencv.opencv_face.FaceRecognizer;
import org.bytedeco.opencv.opencv_face.FisherFaceRecognizer;

import java.io.File;
import java.io.FilenameFilter;
import java.nio.IntBuffer;

import static org.bytedeco.opencv.global.opencv_imgproc.resize;
import static org.opencv.core.CvType.CV_32SC1;

public class Treinamento {

    public static void main(String[] args) {
        System.out.println("Treinamento");
        File diretorio = new File("src\\fotos");
        FilenameFilter filenameFilter = new FilenameFilter() {

            @Override
            public boolean accept(File dir, String name) {
                return name.endsWith(".jpg") ||
                        name.endsWith(".gif")||
                        name.endsWith(".png");
            }
        };
        File[] arquivos = diretorio.listFiles(filenameFilter);
        MatVector fotos = new MatVector(arquivos.length);
        Mat rotulos = new Mat(arquivos.length, 1, CV_32SC1);
        IntBuffer rotulosBuffer = rotulos.createBuffer();
        int contador = 0;
        for (File imagem: arquivos) {
            Mat foto = org.bytedeco.opencv.global.opencv_imgcodecs.imread(imagem.getAbsolutePath(), org.bytedeco.opencv.global.opencv_imgcodecs.IMREAD_GRAYSCALE);
            int classe = Integer.parseInt(imagem.getName().split("\\.")[1]);
            resize(foto, foto, new org.bytedeco.opencv.opencv_core.Size(160, 160));
            fotos.put(contador, foto);
            rotulosBuffer.put(contador, classe);
            contador++;
        }
        FaceRecognizer eigenFaceRecognizer = EigenFaceRecognizer.create();
        FaceRecognizer fisherFaceRecognizer = FisherFaceRecognizer.create();
        FaceRecognizer lbphFaceRecognizer = org.bytedeco.opencv.opencv_face.LBPHFaceRecognizer.create(2, 9, 9, 9, 1);
        eigenFaceRecognizer.train(fotos, rotulos);
        eigenFaceRecognizer.save("src\\recursos\\classificadorEigenFaces.yml");
        fisherFaceRecognizer.train(fotos, rotulos);
        fisherFaceRecognizer.save("src\\recursos\\classificadorFisherFaces.yml");
        lbphFaceRecognizer.train(fotos, rotulos);
        lbphFaceRecognizer.save("src\\recursos\\classificadorLBPH.yml");
    }
}
