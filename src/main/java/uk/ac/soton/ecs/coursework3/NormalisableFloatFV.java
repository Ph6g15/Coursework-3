package uk.ac.soton.ecs.coursework3;

import org.openimaj.feature.FloatFV;

public class NormalisableFloatFV extends FloatFV {
    public NormalisableFloatFV() {
        super();
    }

    public NormalisableFloatFV(float[] floats) {
        super(floats);
    }

    public NormalisableFloatFV getNormalised() {
        // Convert to vector of doubles.
        float[] vectorArray = this.values;
        // calculate mean of values.
        double total = 0;

        for (double d : vectorArray) {
            total += d;
        }
        double mean = total / vectorArray.length;

        // Set mean to 0.
        //iterate over array deducting mean from each entry
        for (int i = 0; i < vectorArray.length; i++) {
            vectorArray[i] = (float) (vectorArray[i] - mean);
        }
        // calculate sum of squares for normalisation
        double sqrtTotal = 0;
        for (double d : vectorArray) {
            sqrtTotal += d * d;
        }
        // Get length of vector.
        double abs = Math.sqrt(sqrtTotal);
        // Set vector length to 1.
        for (int i = 0; i < vectorArray.length; i++) {
            vectorArray[i] = (float) (vectorArray[i] / abs);
        }

        // Return new vector of normalised values.
        return new NormalisableFloatFV(vectorArray);
    }

}
