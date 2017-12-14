package uk.ac.soton.ecs.coursework3;

import de.bwaldvogel.liblinear.SolverType;
import org.openimaj.data.DataSource;
import org.openimaj.data.dataset.Dataset;
import org.openimaj.data.dataset.VFSGroupDataset;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.experiment.evaluation.classification.ClassificationResult;
import org.openimaj.feature.DoubleFV;
import org.openimaj.feature.FeatureExtractor;
import org.openimaj.feature.SparseIntFV;
import org.openimaj.feature.local.data.LocalFeatureListDataSource;
import org.openimaj.feature.local.list.LocalFeatureList;
import org.openimaj.image.FImage;
import org.openimaj.image.feature.dense.gradient.dsift.ByteDSIFTKeypoint;
import org.openimaj.image.feature.dense.gradient.dsift.DenseSIFT;
import org.openimaj.image.feature.dense.gradient.dsift.PyramidDenseSIFT;
import org.openimaj.image.feature.local.aggregate.BagOfVisualWords;
import org.openimaj.image.feature.local.aggregate.PyramidSpatialAggregator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator;
import org.openimaj.ml.annotation.linear.LiblinearAnnotator.Mode;
import org.openimaj.ml.clustering.ByteCentroidsResult;
import org.openimaj.ml.clustering.assignment.HardAssigner;
import org.openimaj.ml.clustering.kmeans.ByteKMeans;
import org.openimaj.ml.kernel.HomogeneousKernelMap;
import org.openimaj.util.pair.IntFloatPair;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
        DenseSIFT dsift = new DenseSIFT(3, 7);
        PyramidDenseSIFT<FImage> pdsift = new PyramidDenseSIFT<>(dsift, 6f, 4, 6, 8, 10);
        // Train quantiser with random sample of 30 images across the training set.
        HardAssigner<byte[], float[], IntFloatPair> assigner = trainQuantiser(trainingData, pdsift);
        // Construct PHOW extractor.
        HomogeneousKernelMap kernelMap = new HomogeneousKernelMap(HomogeneousKernelMap.KernelType.Chi2, HomogeneousKernelMap.WindowType.Rectangular);
        FeatureExtractor<DoubleFV, FImage> extractor = new PHOWExtractor(pdsift, assigner);
        // Create and train classifier.
        LiblinearAnnotator<FImage, String> annotator = new LiblinearAnnotator<>(kernelMap.createWrappedExtractor(extractor), Mode.MULTICLASS, SolverType.L2R_L2LOSS_SVC, 1.0, 0.00001);
        annotator.train(trainingData);

        // Convert guesses to output format.
        double highestConfidence = 0;
        double confidence;
        ArrayList<String> predictions = new ArrayList<String>();
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
            predictions.add(imageName+ " " +String.valueOf(highestConfidence));
        }

        return predictions;
    }

    /**
     * Build HardAssigner by performing K-Means clustering on a trainingData of the SIFT features.
     *
     * @param trainingData Data to train the assigner on.
     * @param pdsift Pyramid dense SIFT.
     */
    // Build HardAssigner by performing K-Means clustering on the training data SIFT features.
    static HardAssigner<byte[], float[], IntFloatPair> trainQuantiser(Dataset<FImage> trainingData, PyramidDenseSIFT<FImage> pdsift) {
        List<LocalFeatureList<ByteDSIFTKeypoint>> allkeys = new ArrayList<>();

        for (FImage rec : trainingData) {
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

    static class PHOWExtractor implements FeatureExtractor<DoubleFV, FImage> {
        PyramidDenseSIFT<FImage> pdsift;
        HardAssigner<byte[], float[], IntFloatPair> assigner;

        public PHOWExtractor(PyramidDenseSIFT<FImage> pdsift, HardAssigner<byte[], float[], IntFloatPair> assigner)
        {
            this.pdsift = pdsift;
            this.assigner = assigner;
        }

        public DoubleFV extractFeature(FImage object) {
            FImage image = object.getImage();
            pdsift.analyseImage(image);

            BagOfVisualWords<byte[]> bovw = new BagOfVisualWords<>(assigner);

            PyramidSpatialAggregator<byte[], SparseIntFV> spatial = new PyramidSpatialAggregator<>(bovw, 2, 4);

            return spatial.aggregate(pdsift.getByteKeypoints(0.015f), image.getBounds()).normaliseFV();
        }
    }
}
