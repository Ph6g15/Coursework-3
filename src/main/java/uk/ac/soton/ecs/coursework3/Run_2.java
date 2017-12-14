package uk.ac.soton.ecs.coursework3;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.LocalFeatureImpl;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.BlockSpatialAggregator;
import org.openimaj.image.pixel.sampling.RectangleSampler;
import org.openimaj.math.geometry.shape.Rectangle;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * BOVW Linear Classifiers.
 */
public class Run_2 {
    /**
     * Distance that moving rectangle moves after each iteration.
     */
    private static final int STEP_SIZE = 8;
    /**
     * Size of the moving patch.
     */
    private static final int PATCH_SIZE = 12;

    /**
     * Linear classifier using bags of visual words features based on
     *
     * @param trainingData Training images grouped into their correct classifiers.
     * @param testingData Ungrouped test images.
     * @return Map of image file names to the predicted class.
     */
    public static ArrayList<String> run(VFSGroupDataset<FImage> trainingData, VFSListDataset<FImage> testingData) throws Exception {
        // Train assigner.
        HardAssigner<float[], float[], IntFloatPair> assigner = trainQuantiser(testingData, 500);
        // Train linear classifier
        FeatureExtractor<DoubleFV, FImage> featureExtractor = new ClusteredPatchFeatureExtractor(assigner);
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<>(featureExtractor, LiblinearAnnotator.Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1, 0.00001);
        annotator.train(trainingData);

        // Perform guesses.

        double highestConfidence = 0;
        double confidence;
        ArrayList<String> predictions = new ArrayList<>();
        for (int i = 0; i < testingData.size(); i++) {
            FImage testImage = testingData.get(i);
            String imageName = testingData.getID(i);

            ClassificationResult<String> prediction = annotator.classify(testImage);

            // Add prediction to map.
            for (String imageClass : prediction.getPredictedClasses()) {
                confidence = prediction.getConfidence(imageClass);
                if (confidence > highestConfidence) {
                    highestConfidence = confidence;
                }
            }
            predictions.add(imageName + " " + String.valueOf(highestConfidence));
        }

        return predictions;
    }

    /**
     * Build HardAssigner by performing K-Means clustering on patches taken from the training images.
     *
     * @param trainingData Dataset containing the images to train the assigner on.
     * @param clusters The number of clusters.
     * @return Trained assigner.
     */
    private static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> trainingData, int clusters) {
        List<float[]> featureVectors = new ArrayList<>();

        // For each image in the training data.
        for (FImage image : trainingData) {
            // Sample patches of image as features.
            List<LocalFeature<SpatialLocation, CenterableNormalisableFloatFV>> patchList = extractPatchFeatures(image, STEP_SIZE, PATCH_SIZE);
            // Add values of features to list.
            for (LocalFeature<SpatialLocation, CenterableNormalisableFloatFV> localFeature : patchList) {
                featureVectors.add(localFeature.getFeatureVector().values);
            }
        }

        // Initialise means of clusters.
        FloatKMeans kMeans = FloatKMeans.createKDTreeEnsemble(clusters);

        // Perform K-means clustering.
        FloatCentroidsResult centroidsResult = kMeans.cluster(featureVectors.toArray(new float[][]{}));

        return centroidsResult.defaultHardAssigner();
    }

    /**
     *
     *
     * @param image Image to extract patch features from.
     * @param step Distance that the patch moves on each iteration.
     * @param patchSize Size of the patch features.
     * @return List of patch features.
     */
    private static List<LocalFeature<SpatialLocation, CenterableNormalisableFloatFV>> extractPatchFeatures(FImage image, float step, float patchSize) {
        List<LocalFeature<SpatialLocation, CenterableNormalisableFloatFV>> patchFeatures = new ArrayList<>();

        // Create image patch sampler.
        RectangleSampler rectangleSampler = new RectangleSampler(image, step, step, patchSize, patchSize);

        // Iterate over patches.
        for (Rectangle rectangle : rectangleSampler) {
            // Get patch from image.
            FImage patch = image.extractROI(rectangle);
            // Convert patch to a feature vector.
            CenterableNormalisableFloatFV featureVector = new CenterableNormalisableFloatFV(ArrayUtils.reshape(patch.pixels));
            // Get feature location.
            SpatialLocation location = new SpatialLocation(rectangle.x, rectangle.y);
            // Construct feature with normalised feature vector.
            LocalFeature<SpatialLocation, CenterableNormalisableFloatFV> localFeature = new LocalFeatureImpl<>(location, featureVector.getNormalised());
            // Add feature to list.
            patchFeatures.add(localFeature);
        }

        // Return list of features.
        return patchFeatures;
    }

    /**
     * Implementation of feature extractor that clusters features using bags of visual words.
     */
    static class ClusteredPatchFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> assigner;

        /**
         * Constructor to set the extractor's assigner.
         *
         * @param assigner Assigner.
         */
        public ClusteredPatchFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner)
        {
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage image) {
            // Get bag of visual words from assigner.
            BagOfVisualWords<float[]> bovw = new BagOfVisualWords<>(assigner);
            // Group features into blocks using bag of visual words.
            BlockSpatialAggregator<float[], SparseIntFV> spatial = new BlockSpatialAggregator<>(bovw, 2, 2);

            return spatial.aggregate(extractPatchFeatures(image, STEP_SIZE, PATCH_SIZE), image.getBounds()).asDoubleFV();

        }
    }
}
