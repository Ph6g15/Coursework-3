package uk.ac.soton.ecs.coursework3;

import org.openimaj.data.dataset.Dataset;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.FloatFV;
import org.openimaj.feature.local.LocalFeature;
import org.openimaj.feature.local.SpatialLocation;
import org.openimaj.image.FImage;
import org.openimaj.ml.clustering.FloatCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.FloatKMeans;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.List;

/**
 * BOVW Linear Classifiers.
 */
public class Run_2 {
    public static void run2() throws Exception {
        // Use random splitter to get training set and train quantiser.
        // Train linear classifier
    }

    // Build HardAssigner by performing K-Means clustering on a sample of the SIFT features.
    static HardAssigner<float[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> sample) {
        List<float[]> allkeys = new ArrayList<>();

        // For each image in the sample.
        for (FImage image : sample) {
            // Sample patches of image as features.
        }


        // Initialise means of clusters.
        FloatKMeans km = FloatKMeans.createKDTreeEnsemble(300);
        float[][] datasource = allkeys.toArray(new float[][]{});

        // Perform K-means clustering.
        FloatCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    public static List<LocalFeature<SpatialLocation, FloatFV>> extractPatchFeatures(FImage image, float step, float patchSize) {
        List<LocalFeature<SpatialLocation, FloatFV>> patchFeatures = new ArrayList<>();

        // Create image patches.

        // Iterate over patches.
            // Get patch from image.
            // Convert region to a feature vector.
            // Get feature location.
            // Construct feature.
            // Add feature to list.

        // Return list of features.
        return null;
    }

    // TODO Refactor name
    static class PatchClusterFeatureExtractor implements FeatureExtractor<DoubleFV, FImage> {
        HardAssigner<float[], float[], IntFloatPair> assigner;

        public PatchClusterFeatureExtractor(HardAssigner<float[], float[], IntFloatPair> assigner)
        {
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            // Get bag of visual words from assigner.
            // Group features into blocks using bag of visual words.
            return null;
        }
    }
}
