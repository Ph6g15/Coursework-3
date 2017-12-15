package uk.ac.soton.ecs.coursework3;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.global.Gist;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;

/**
 * Best Classifier.
 */
public class Run_3 {
    /**
     *
     *
     * @param trainingData Training images grouped into their correct classifiers.
     * @param testingData Ungrouped test images.
     * @return Map of image file names to the predicted class.
     */
    public static ArrayList<String> run(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) throws Exception {
        // Perform pyramid dense SIFT to apply normal dense SIFT to different sized windows.
        Gist<FImage> gist = new Gist<>();
        // Construct Gist extractor.
        FeatureExtractor<FloatFV, FImage> extractor = new GistExtractor(gist);
        // Create and train classifier.
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<>(extractor, Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        annotator.train(trainingData);

        // Convert guesses to output format.

        ArrayList<String> predictions = new ArrayList<>();
        for (int i = 0; i < testingData.size(); i++) {
            FImage testImage = testingData.get(i);
            String imageName = testingData.getID(i);
            double highestConfidence = 0;
            double confidence;
            ClassificationResult<String> prediction = annotator.classify(testImage);

            // Add prediction to map.
            String bestguessSoFar = "unknown";
            for (String imageClass : prediction.getPredictedClasses()) {
                confidence = prediction.getConfidence(imageClass);
                if (confidence > highestConfidence) {
                    highestConfidence = confidence;
                    bestguessSoFar = imageClass;
                }
            }
            predictions.add(imageName+ " " +bestguessSoFar);
        }

        return predictions;
    }

    static class GistExtractor implements FeatureExtractor<FloatFV, FImage> {
        Gist<FImage> gist;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public GistExtractor(Gist<FImage> gist)
        {
            this.gist = gist;
        }

        public FloatFV extractFeature(FImage object) {
            FImage image = object.getImage();
            gist.analyseImage(image);

            return gist.getResponse();
        }
    }
}
