package uk.ac.soton.ecs.coursework3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class Main {
    public static void main(String args[]) throws Exception {
        // Load the training and test data from URLs in the spec.
        VFSGroupDataset<FImage> trainingData = null;
        VFSListDataset<FImage> testingData = null;
        try {
            trainingData = new VFSGroupDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/training.zip", ImageUtilities.FIMAGE_READER);
            testingData = new VFSListDataset<>("zip:http://comp3204.ecs.soton.ac.uk/cw/testing.zip", ImageUtilities.FIMAGE_READER);
        } catch (FileSystemException e) {
            System.err.println("Could not load data from URL. Maybe the website is down or has moved.");
            e.printStackTrace();
        }
        // Get group.
        // Split group into test and training sets.

        // Run 1 classification.
        Map<String, String> run1Predictions = Run_1.run(trainingData, testingData);
        // Output run 1  guesses.
        writePredictions(run1Predictions, "Run1.txt");

        // Run 2 classification.
        Map<String, String> run2Predictions = Run_2.run(trainingData, testingData);
        // Output run 2  guesses.
        writePredictions(run2Predictions, "Run2.txt");

//        // Run 3 classification.
//        Map<String, String> run3Predictions = Run_3.run(trainingData, testingData);
//        // Output run 3  guesses.
//        writePredictions(run3Predictions);
    }

    public static void writePredictions(Map<String, String> predictions, String filename) throws Exception {
        // Write predictions to file.
        PrintWriter writer = new PrintWriter(filename, "UTF-8");
        Iterator it = predictions.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry pair = (Map.Entry) it.next();
            writer.println(pair.getKey() + " = " + pair.getValue());
        }
        writer.close();
    }
}
