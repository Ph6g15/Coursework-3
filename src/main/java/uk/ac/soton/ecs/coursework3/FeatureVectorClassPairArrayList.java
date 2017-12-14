package uk.ac.soton.ecs.coursework3;

import java.util.ArrayList;

public class FeatureVectorClassPairArrayList extends ArrayList<FeatureVectorClassPair> {
    /**
     * Returns a list of feature vectors as a 2D array.
     *
     * @return 2D array of feature vectors
     */
    public float[][] toFeatureVectorArray() {
        float[][] featureVectorArray = new float[this.size()][];
        for (int i = 0; i < this.size(); i++) {
            featureVectorArray[i] = this.get(i).featureVector;
        }
        return featureVectorArray;
    }
}
