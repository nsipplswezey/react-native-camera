package com.lwansbrough.RCTCamera;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetManager;
import android.os.Bundle;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

import com.jetpac.deepbelief.DeepBelief.JPCNNLibrary;
import com.sun.jna.Pointer;

/**
 * Created by nsipplswezey on 12/19/17.
 */

public class DeepBelief extends Activity {

    public static final String TAG = "DeepBelief";
    static Context ctx;
    static Pointer networkHandle = null;
    static Pointer predictorHandle = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        ctx = this;

        initDeepBelief();
    }

    static void initDeepBelief() {

        android.util.Log.d("DeepBelief", "Init deep belief");

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
        predictorHandle = JPCNNLibrary.INSTANCE.jpcnn_load_predictor(predictorFile);

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



}
