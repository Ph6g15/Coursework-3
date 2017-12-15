package uk.ac.soton.ecs.coursework3;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.FloatNearestNeighbours;
import org.openimaj.knn.FloatNearestNeighboursExact;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.util.*;

/**
 * K-Nearest Neighbour Classifier.
 */
public class Run_1 {

    public static ArrayList<String> run(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) throws Exception {
        FeatureVectorClassPairArrayList featureVectorClassPairs = new FeatureVectorClassPairArrayList();

        // For each class in training set.
        for (String className : trainingData.getGroups()) {
            // For each image in group.
            TinyImageVectorExtractor tinyImageVectorExtractor = new TinyImageVectorExtractor();
            for (FImage image : trainingData.get(className)) {
                // Extract tiny image feature vector from image.
                CenterableNormalisableFloatFV tinyImageVector = tinyImageVectorExtractor.extractFeature(image);
                tinyImageVector = tinyImageVector.getNormalised();
                // Add feature and class name to array.
                featureVectorClassPairs.add(new FeatureVectorClassPair(tinyImageVector.values, className));
            }
        }
        // Get k-nearest neighbour using training feature vectors.
        float[][] trainingVectors = featureVectorClassPairs.toFeatureVectorArray();
        FloatNearestNeighboursExact kNearestNeighbours = new FloatNearestNeighboursExact(trainingVectors);
        // Perform guesses.

        ArrayList<String> predictions = new ArrayList<>();
        for (int i = 0; i < testingData.size(); i++) {
            FImage testImage = testingData.get(i);
            String imageName = testingData.getID(i);
            String prediction = imageName + " " + getBestGuess(testImage, kNearestNeighbours, featureVectorClassPairs, 38);

            predictions.add(prediction);
        }

        return predictions;
    }

    /**
     * Return guess of image class
     */
    private static String getBestGuess(FImage image, FloatNearestNeighbours neighbours, FeatureVectorClassPairArrayList key, int k) {
        // Extract feature vector.
        TinyImageVectorExtractor tinyImageVectorExtractor = new TinyImageVectorExtractor();
        CenterableNormalisableFloatFV featureVector = tinyImageVectorExtractor.extractFeature(image);
        //normalises and centres around mean for comparison
        featureVector = featureVector.getNormalised();
        // Get k-nearest neighbours.
        List<IntFloatPair> kNearestNeighbours = neighbours.searchKNN(featureVector.values, k);
        // Map containing how many neighbours there are of each class.
        Map<String, Integer> classNeighbourCount = new HashMap<String, Integer>();
        for (IntFloatPair kNearestNeighbour : kNearestNeighbours) {
            Integer index = kNearestNeighbour.getFirst();
            String neighbourClass = key.get(index).vectorClass;
            if (classNeighbourCount.containsKey(neighbourClass)) {
                Integer tempcount = classNeighbourCount.get(neighbourClass);
                tempcount++;
                classNeighbourCount.put(neighbourClass, tempcount);
            } else {
                classNeighbourCount.put(neighbourClass, 1);
            }

        }
        int maxneighboursofar = 0;
        String bestclassguess = "none";
        Iterator<Map.Entry<String, Integer>> it = classNeighbourCount.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String, Integer> pair = it.next();
            if (classNeighbourCount.get(pair.getKey()) > maxneighboursofar) {
                maxneighboursofar = classNeighbourCount.get(pair.getKey());
                bestclassguess = pair.getKey();
            }

        }
        //Return Result
        return bestclassguess;
    }

    static class TinyImageVectorExtractor implements FeatureExtractor<FloatFV, FImage> {
        public static final int TINYIMAGE_SIZE = 16;

        @Override
        public CenterableNormalisableFloatFV extractFeature(FImage image) {
            // Crop image.
            FImage tinyImage = new FImage(TINYIMAGE_SIZE, TINYIMAGE_SIZE);
            int cropSize = image.getHeight() < image.getWidth() ? image.getHeight() : image.getWidth();
            FImage croppedImage = new FImage(cropSize, cropSize);
            image.extractCentreSubPix(image.getWidth() / 2, image.getHeight() / 2, croppedImage);
            // Set tinyImage pixels to downsized version of cropped image.
            tinyImage.zero();
            tinyImage.addInplace(croppedImage.process(new ResizeProcessor(TINYIMAGE_SIZE, TINYIMAGE_SIZE)));

            // Convert tinyImage to a normalised vector
            return new CenterableNormalisableFloatFV(ArrayUtils.reshape(ArrayUtils.convertToFloat(tinyImage.pixels)));
        }
    }
}
