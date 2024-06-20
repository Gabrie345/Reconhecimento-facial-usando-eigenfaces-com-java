package org.JavaCV;


import org.bytedeco.javacv.*;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;

import java.awt.event.KeyEvent;

import static org.bytedeco.opencv.global.opencv_imgcodecs.imwrite;
import static org.bytedeco.opencv.global.opencv_imgproc.*;
import static org.opencv.imgproc.Imgproc.COLOR_BGRA2GRAY;


public class Camera {
    public static void main(String arg[]) throws FrameGrabber.Exception, InterruptedException {

        KeyEvent tecla = null;
        Frame frameCapturado = null;
        Mat imagemColorida = new Mat();
        Mat imagemCinza = new Mat();
        OpenCVFrameConverter.ToMat converteMat = new OpenCVFrameConverter.ToMat();
        OpenCVFrameGrabber camera = new OpenCVFrameGrabber(0);
        CanvasFrame cFrame = new CanvasFrame("Camera", CanvasFrame.getDefaultGamma() / camera.getGamma());
        int numerodeAmostra = 30;
        int amostra = 1;

        CascadeClassifier detectorFace = createdFaceDetector();
        camera.start();

        while ((frameCapturado = camera.grab()) != null) {
            imagemColorida = converteMat.convert(frameCapturado);
            cvtColor(imagemColorida, imagemCinza, COLOR_BGRA2GRAY);
            RectVector facesDetectadas = new RectVector();
            detectorFace.detectMultiScale
                    (imagemCinza, facesDetectadas, 1.1, 1, 0, new org.bytedeco.opencv.opencv_core.Size(150, 150),
                            new org.bytedeco.opencv.opencv_core.Size(500, 500));
            for(int i = 0; i < facesDetectadas.size(); i++){

                org.bytedeco.opencv.opencv_core.Rect dadosFace = facesDetectadas.get(i);
                rectangle(imagemColorida, dadosFace, new org.bytedeco.opencv.opencv_core.Scalar(0, 0, 255, 0));
                Mat faceCapturada = new Mat(imagemCinza, dadosFace);
                resize(faceCapturada, faceCapturada, new org.bytedeco.opencv.opencv_core.Size(160, 160));

                if (tecla == null) {
                    tecla = cFrame.waitKey(5);
                }
                if(tecla != null){
                    System.out.println("Foto " + amostra + " capturada\n");
                    if (amostra <= numerodeAmostra) {
                        if (tecla.getKeyChar() == 'q') {
                            imwrite("src\\fotos\\pessoa." + 2 + "." + amostra + ".jpg", faceCapturada);
                            amostra++;
                        }
                    }
                    tecla = null;
                }
            }
            if (tecla == null) {
                tecla = cFrame.waitKey(20);
            }
            if(cFrame.isVisible()){
                cFrame.showImage(frameCapturado);
            }
            if(amostra> numerodeAmostra){
                break;
            }
        }
        cFrame.dispose();
        camera.stop();
    }

    private static CascadeClassifier createdFaceDetector() {
        CascadeClassifier detectorFace = new CascadeClassifier();
        detectorFace.load("src/recursos/haarcascade_frontalface_alt.xml");
        return detectorFace;
    }

}