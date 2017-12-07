Eric Lee (ejl966)
Guneet Dhillon (gsd352)

For this assignment, we implemented algorithms that were discussed in class.

Some time went into understand Tensorflow and how it worked.

Some time also went into implementing the featurre for increasing the accuracy.
We calculated the number of connected regions. Then added a 3 dimensional vector
to the original feature vector, with the value of the ith entry equal to 1 iff
there are i connected regions in the image. This binary vector can be used in a
better way than just adding a single feature equal to the number of connected
regions.
