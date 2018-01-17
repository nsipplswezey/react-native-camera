package com.lwansbrough.RCTCamera;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;

import com.jetpac.deepbelief.DeepBelief.JPCNNLibrary;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

import com.facebook.react.bridge.ReactContext;
import com.facebook.react.bridge.ReactApplicationContext;

/**
 * Created by nsipplswezey on 12/19/17.
 */


public class DeepBelief {

    public static final String TAG = "DeepBelief";
    static Context ctx;
    static Pointer networkHandle = null;
    static Pointer predictorHandle = null;

    // @Override
    // public void onCreate(Bundle savedInstanceState) {
    //     super.onCreate(savedInstanceState);
    //     ctx = this;
    //
    //     initDeepBelief();
    // }

    static void initDeepBelief() {

        android.util.Log.d("DeepBelief", "Init deep belief");

        ReactContext reactContext = RCTCameraModule.getReactContextSingleton();
        ctx = reactContext;

        AssetManager am = ctx.getAssets();
        String baseFileName = "jetpac.ntwk";
        String dataDir = ctx.getFilesDir().getAbsolutePath();
        String networkFile = dataDir + "/" + baseFileName;
        copyAsset(am, baseFileName, networkFile);
        android.util.Log.d("ReactNative", "networkFile: " + networkFile);
        networkHandle = JPCNNLibrary.INSTANCE.jpcnn_create_network(networkFile);

        //TODO: Load predictor and predict
        //Use the same technique to set of the predictor
        //Replace the classify image call with the predict call
        String predictorFileName = "VoltAGE_2_predictor.txt";
        String predictorFile = dataDir + "/" + predictorFileName;
        copyAsset(am, predictorFileName, predictorFile);
        //setPredictor(predictorFile);

        Bitmap lenaBitmap = getBitmapFromAsset(am,"lena.png");

        if(lenaBitmap != null){
            android.util.Log.d("ReactNative", "Classifying lena.png");
            classifyBitmap(lenaBitmap);
        }

    }

    public static void setPredictor(String predictorFilePath){
      predictorHandle = JPCNNLibrary.INSTANCE.jpcnn_load_predictor(predictorFilePath);
    }

    private static boolean copyAsset(AssetManager assetManager,
                                     String fromAssetPath, String toPath) {
        InputStream in = null;
        OutputStream out = null;
        try {
            in = assetManager.open(fromAssetPath);
            new File(toPath).createNewFile();
            out = new FileOutputStream(toPath);
            copyFile(in, out);
            in.close();
            in = null;
            out.flush();
            out.close();
            out = null;
            return true;
        } catch(Exception e) {
            e.printStackTrace();
            return false;
        }
    }

    private static void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }

    public static Bitmap getBitmapFromAsset(AssetManager mgr, String path) {
        InputStream is = null;
        Bitmap bitmap = null;
        try {
            is = mgr.open(path);
            bitmap = BitmapFactory.decodeStream(is);
        } catch (final IOException e) {
            bitmap = null;
            android.util.Log.d("ReactNative", "error in creating bitmap from asset" + e.getMessage());
            android.util.Log.d("ReactNative",  android.util.Log.getStackTraceString(e));

        } finally {
            if (is != null) {
                try {
                    is.close();
                } catch (IOException ignored) {

                }
            }
        }
        return bitmap;
    }

    public static float classifyBitmap(Bitmap bitmap) {
        final int width = bitmap.getWidth();
        final int height = bitmap.getHeight();
        final int pixelCount = (width * height);
        final int bytesPerPixel = 4;
        final int byteCount = (pixelCount * bytesPerPixel);
        ByteBuffer buffer = ByteBuffer.allocate(byteCount);
        bitmap.copyPixelsToBuffer(buffer);
        byte[] pixels = buffer.array();
        Pointer imageHandle = JPCNNLibrary.INSTANCE.jpcnn_create_image_buffer_from_uint8_data(pixels, width, height, 4, (4 * width), 0, 1);

        PointerByReference predictionsValuesRef = new PointerByReference();
        IntByReference predictionsLengthRef = new IntByReference();
        PointerByReference predictionsNamesRef = new PointerByReference();
        IntByReference predictionsNamesLengthRef = new IntByReference();
        long startT = System.currentTimeMillis();
        JPCNNLibrary.INSTANCE.jpcnn_classify_image(
                networkHandle,
                imageHandle,
                0,
                -2,
                predictionsValuesRef,
                predictionsLengthRef,
                predictionsNamesRef,
                predictionsNamesLengthRef);

        JPCNNLibrary.INSTANCE.jpcnn_destroy_image_buffer(imageHandle);

        Pointer predictionsValuesPointer = predictionsValuesRef.getValue();
        final int predictionsLength = predictionsLengthRef.getValue();


        //Start trained model prediction
        float trainedPredictionValue = JPCNNLibrary.INSTANCE.jpcnn_predict(predictorHandle, predictionsValuesPointer, predictionsLength);
        android.util.Log.d("ReactNative", "jpcnn_predict() value is " + trainedPredictionValue + ".");
        //End trained model prediction

        long stopT = System.currentTimeMillis();
        float duration = (float) (stopT - startT) / 1000.0f;
        android.util.Log.d("ReactNative", "jpcnn_classify_image() + predict() took " + duration + " seconds.");
        return trainedPredictionValue;
    }



}
