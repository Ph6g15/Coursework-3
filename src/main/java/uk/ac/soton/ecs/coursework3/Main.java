package uk.ac.soton.ecs.coursework3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openrdf.query.algebra.Str;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Main {
    public static void main(String args[]) throws Exception {
        // Load the training and test data from URLs in the spec.
        VFSGroupDataset<FImage> trainingData = null;
        VFSListDataset<FImage> testingData = null;
        try {
            trainingData = new VFSGroupDataset<>("C:\\Users\\PiersEpsilon\\Coursework-3\\training", ImageUtilities.FIMAGE_READER);
            testingData = new VFSListDataset<>("C:\\Users\\PiersEpsilon\\Downloads\\testing", ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            System.err.println("Could not load data from URL. Maybe the website is down or has moved.");
            e.printStackTrace();
        }
        // Get group.
        // Split group into test and training sets.

        // Run 1 classification.
        //ArrayList<String> run1Predictions = Run_1.run(trainingData, testingData);
        // Output run 1  guesses.
        System.out.println("Run 1 Classification is complete");
       // writePredictions(run1Predictions, "Run1.txt");
        System.out.println("Run 1 Classification Results are saved in Run1.txt ");
        // Run 2 classification.

       // ArrayList<String> run2Predictions = Run_2.run(trainingData, testingData);
       // System.out.println("Run 2 Classification is complete");
        // Output run 2  guesses.
        //writePredictions(run2Predictions, "Run2.txt");
        System.out.println("Run 2 Classification Results are saved in Run2.txt ");

//        // Run 3 classification.
        ArrayList<String> run3Predictions = Run_3.run(trainingData, testingData);
//        // Output run 3  guesses.
        writePredictions(run3Predictions, "Run3.txt");
    }

    /**
     * Write the predictions as defined by the map to a file.
     *
     * @param predictions List of strings stating of image file names and their predicted classes.
     * @param filename    Name of file to write to.
     */
    public static void writePredictions(ArrayList<String> predictions, String filename) throws Exception {
        // Write predictions to file.
        PrintWriter writer = new PrintWriter(filename, "UTF-8");

        for (String s : predictions) {
            writer.println(s);
        }
        writer.close();
    }
}
