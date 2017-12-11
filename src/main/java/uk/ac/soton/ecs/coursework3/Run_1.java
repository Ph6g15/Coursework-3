package uk.ac.soton.ecs.coursework3;

import net.didion.jwnl.data.Exc;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.experiment.evaluation.classification.BasicClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.DisplayUtilities;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;
import org.openimaj.image.annotation.evaluation.datasets.Caltech101.Record;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.extraction.FeatureVectorExtractor;
import org.openimaj.image.processing.resize.ResizeProcessor;
import org.openimaj.knn.DoubleNearestNeighbours;
import org.openimaj.knn.DoubleNearestNeighboursExact;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.util.array.ArrayUtils;
import org.openimaj.util.pair.IntDoublePair;
import org.openimaj.util.pair.IntFloatPair;

import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * K-Nearest Neighbour Classifier.
 */
public class Run_1 {
    private static FeatureVectorClassPairArrayList featureVectorClassPairs;

    public static void main( String[] args ) throws Exception {
        FImage image = ImageUtilities.readF(new URL("http://static.openimaj.org/media/tutorial/sinaface.jpg"));
        DisplayUtilities.displayName(image, "SinaFace Big");

        //
        // For each group in training set.
        // For each image in group.
        // Extract tiny image feature vector from image.
        TinyImageVectorExtractor tinyImageVectorExtractor = new TinyImageVectorExtractor();
        TinyImageDoubleFV tinyImageVector = tinyImageVectorExtractor.extractFeature(image);
        tinyImageVector.normaliseFV();
        // Add feature and class name to array.
        // Get k-nearest neighbour using training feature vectors.
        double [][] trainingVectors = featureVectorClassPairs.toFeatureVectorArray();

        DoubleNearestNeighboursExact kNearestNeighbours = new DoubleNearestNeighboursExact(trainingVectors);


    }

    public static void writePredictions(HashMap<String, String> predictions) {
        // Write predictions to file.
    }

    /**
     * Return guess of image class
     */
    public static BasicClassificationResult<String> classify(FImage image, DoubleNearestNeighbours neighbours) {
        final int K = 15;
        // Extract feature vector.
        // Convert feature vector to double array.
        // Get k-nearest neighbours.
//        List<IntDoublePair> kNearestNeighbours = neighbours.searchKNN(featureVector, K);
        // Map containing how many neighbours there are of each class.
        // For each neighbour.
        // Get class of neighbour.
        // Increment count for class.
        // Convert map to list in order to sort.
        // Sort the list.
        // Get confidence in result.
        // Guess the class that is first in the list.
        // Return result.
        return null;
    }

    // Build HardAssigner by performing K-Means clustering on a sample of the SIFT features.
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<Record<FImage>> sample, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (Record<FImage> rec : sample) {
            FImage img = rec.getImage();

            pdsift.analyseImage(img);
            allkeys.add(pdsift.getByteKeypoints(0.005f));
        }

        if (allkeys.size() > 10000)
            allkeys = allkeys.subList(0, 10000);

        ByteKMeans km = ByteKMeans.createKDTreeEnsemble(300);
        DataSource<byte[]> datasource = new LocalFeatureListDataSource<>(allkeys);
        ByteCentroidsResult result = km.cluster(datasource);

        return result.defaultHardAssigner();
    }

    static class TinyImageVectorExtractor implements FeatureExtractor<DoubleFV, FImage> {
        public static final int TINYIMAGE_SIZE = 16;

        @Override
        public TinyImageDoubleFV extractFeature(FImage image) {
            // Crop image.
            FImage tinyImage = new FImage(TINYIMAGE_SIZE, TINYIMAGE_SIZE);
            int cropSize = image.getHeight() < image.getWidth() ? image.getHeight() : image.getWidth();
            FImage croppedImage = new FImage(cropSize, cropSize);
            image.extractCentreSubPix(image.getWidth() / 2, image.getHeight() / 2, croppedImage);
            // Set tinyImage pixels to downsized version of cropped image.
            tinyImage.zero();
            tinyImage.addInplace(croppedImage.process(new ResizeProcessor(TINYIMAGE_SIZE, TINYIMAGE_SIZE)));

            // Convert tinyImage to vector and return.
            return new TinyImageDoubleFV(ArrayUtils.reshape(ArrayUtils.convertToDouble(tinyImage.pixels)));
        }
    }
}
