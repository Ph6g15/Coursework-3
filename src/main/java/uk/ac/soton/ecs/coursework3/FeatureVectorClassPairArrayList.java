package uk.ac.soton.ecs.coursework3;

import java.util.ArrayList;

/**
 * Extension of ArrayList that only takes FeatureVectorClassPairs.
 * Contains an extra method that converts the list into an array of
 * feature vectors.
 */
public class FeatureVectorClassPairArrayList extends ArrayList<FeatureVectorClassPair> {
    /**
     * Returns a list of feature vectors as a 2D array.
     *
     * @return array of feature vectors.
     */
    public float[][] toFeatureVectorArray() {
        float[][] featureVectorArray = new float[this.size()][];
        for (int i = 0; i < this.size(); i++) {
            featureVectorArray[i] = this.get(i).featureVector;
        }
        return featureVectorArray;
    }
}
