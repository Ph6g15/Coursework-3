package uk.ac.soton.ecs.coursework3;

/**
 * Simple class containing a feature vector and the name of the class it belongs to.
 */
public class FeatureVectorClassPair {
    public double[] featureVector;
    public String vectorClass;

    public FeatureVectorClassPair(double[] featureVector, String vectorClass) {
        this.featureVector = featureVector;
        this.vectorClass = vectorClass;
    }
}
