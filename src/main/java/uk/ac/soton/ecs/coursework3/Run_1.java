package uk.ac.soton.ecs.coursework3;

import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
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

    public static Map<String, String> run(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) throws Exception {
        FeatureVectorClassPairArrayList featureVectorClassPairs = new FeatureVectorClassPairArrayList();

        // For each class in training set.
        for (String className : trainingData.getGroups()) {
            // For each image in group.
            for (FImage image : trainingData.get(className)) {
                // Extract tiny image feature vector from image.
                TinyImageVectorExtractor tinyImageVectorExtractor = new TinyImageVectorExtractor();
                CenterableNormalisableFloatFV tinyImageVector = tinyImageVectorExtractor.extractFeature(image);
                tinyImageVector = tinyImageVector.getNormalised();
                // Add feature and class name to array.
                featureVectorClassPairs.add(new FeatureVectorClassPair(tinyImageVector.values, className));
            }
        }
        // Get k-nearest neighbour using training feature vectors.
        float [][] trainingVectors = featureVectorClassPairs.toFeatureVectorArray();
        FloatNearestNeighboursExact kNearestNeighbours = new FloatNearestNeighboursExact(trainingVectors);
        // Perform guesses.

        Map<String, String> predictions = new HashMap<>();
        for (int i = 0; i < testingData.size(); i++) {
            FImage testImage = testingData.get(i);
            String imageName = testingData.getID(i);
            double highestConfidence = 0;
            double confidence;
            String prediction = getBestGuess(testImage, kNearestNeighbours, featureVectorClassPairs);

//            // Add prediction to map.
//            for (String imageClass : prediction.getPredictedClasses()) {
//                confidence = prediction.getConfidence(imageClass);
//                if (confidence > highestConfidence) {
//                    highestConfidence = confidence;
//                }
//            }
            predictions.put(imageName, prediction);
        }

        return predictions;
    }

    /**
     * Return guess of image class
     */
    private static String  getBestGuess(FImage image, FloatNearestNeighbours neighbours,FeatureVectorClassPairArrayList key) {
        final int K = 15;
        // Extract feature vector.
        TinyImageVectorExtractor tinyImageVectorExtractor = new TinyImageVectorExtractor();
        FloatFV featureVector = tinyImageVectorExtractor.extractFeature(image);
        // Get k-nearest neighbours.
        List<IntFloatPair> kNearestNeighbours = neighbours.searchKNN(featureVector.values, K);
        // Map containing how many neighbours there are of each class.
        Map<String,Integer> classNeighbourCount = new HashMap<String,Integer>();
        for (IntFloatPair kNearestNeighbour: kNearestNeighbours) {
            Integer index = kNearestNeighbour.getFirst();
            String neighbourClass = key.get(index).vectorClass;
            if(classNeighbourCount.containsKey(neighbourClass)){
                Integer tempcount = classNeighbourCount.get(neighbourClass);
                tempcount++;
                classNeighbourCount.put(neighbourClass,tempcount);
            }
            else {
                classNeighbourCount.put(neighbourClass,1);
            }

        }
        int maxneighboursofar = 0;
        String bestclassguess = "none";
        Iterator<Map.Entry<String,Integer>> it= classNeighbourCount.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<String,Integer> pair = it.next();
            if(classNeighbourCount.get(pair.getKey())>maxneighboursofar){
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
