package uk.ac.soton.ecs.coursework3;

import org.apache.commons.vfs2.FileSystemException;
import org.openimaj.data.dataset.VFSListDataset;
import org.openimaj.image.FImage;
import org.openimaj.image.ImageUtilities;

import java.io.File;

public class Main {
    public static void main(String args[]) {
        //load in the test data using first commandline argument as path
        String testPath = args[0] ;
        VFSListDataset<FImage> testingdata = null;
        try { VFSListDataset<FImage> testing = new VFSListDataset<FImage>(testPath, ImageUtilities.FIMAGE_READER); } catch (FileSystemException e) { }

        // Get and sample data set.
        // Check that data set contains classes.

        // Get group.
        // Split group into test and training sets.

        // Run classification.
        // Output guesses.
    }
}
