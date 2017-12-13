package uk.ac.soton.ecs.coursework3;

import org.openimaj.feature.DoubleFV;

public class TinyImageDoubleFV extends DoubleFV {
    public TinyImageDoubleFV() {
        super();
    }

    public TinyImageDoubleFV(double[] doubles) {
        super(doubles);
    }

    public DoubleFV normaliseTinyImageVector(DoubleFV tinyImageVector) {
        // Convert to vector of doubles.
        double[] vectorarray = tinyImageVector.values;
        // calculate mean of values.
        double total = 0;

        for(double d: vectorarray){
            total+=d;
        }
        double mean = total/vectorarray.length;

        // Set mean to 0.
        //iterate over array deducting mean from each entry
        int i = 0;
        while (i<vectorarray.length){
            vectorarray[i] = vectorarray[i]-mean;
            i++;
        }
        // calculate sum of squares for normalisation
        double sqrtotal = 0;
        for(double d: vectorarray){

            sqrtotal +=  d*d;
        }
        // Get length of vector.
        double abs = Math.sqrt(sqrtotal);
        // Set vector length to 1.
        i = 0;
        while (i<vectorarray.length){
            vectorarray[i] = vectorarray[i]/abs;
        }

        // Return new vector of normalised values.
        return new TinyImageDoubleFV(vectorarray);
    }

}
