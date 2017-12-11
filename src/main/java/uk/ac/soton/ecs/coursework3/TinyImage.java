package uk.ac.soton.ecs.coursework3;

import org.openimaj.image.FImage;
import org.openimaj.image.processing.resize.ResizeProcessor;

/**
 * @deprecated
 */
public class TinyImage extends FImage {
    public static final int TINYIMAGE_SIZE = 16;

    public TinyImage(FImage image) {
        super(TINYIMAGE_SIZE, TINYIMAGE_SIZE);
        // Crop image.
        int cropSize = image.getHeight() < image.getWidth() ? image.getHeight() : image.getWidth();
        FImage croppedImage = new FImage(cropSize, cropSize);
        image.extractCentreSubPix(image.getWidth() / 2, image.getHeight() / 2, croppedImage);
        // Set tinyImage pixels to downsized version of cropped image.
        this.zero();
        this.addInplace(croppedImage.process(new ResizeProcessor(TINYIMAGE_SIZE, TINYIMAGE_SIZE)));
    }
}
