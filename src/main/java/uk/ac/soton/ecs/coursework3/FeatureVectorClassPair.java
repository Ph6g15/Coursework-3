package uk.ac.soton.ecs.coursework3;

/**
 * Simple class containing a feature vector and the name of the class it belongs to.
 * Removes the need to have two arrays,
 */
public class FeatureVectorClassPair {
    /**
     * Feature vector values.
     */
    public float[] featureVector;
    /**
     * Class of feature vector.
     */
    public String vectorClass;

    /**
     *
     * @param featureVector
     * @param vectorClass
     */
    public FeatureVectorClassPair(float[] featureVector, String vectorClass) {
        this.featureVector = featureVector;
        this.vectorClass = vectorClass;
    }
}
