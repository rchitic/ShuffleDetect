# ShuffleDetect
This code is the implementation of the following paper: https://www.mdpi.com/2076-3417/13/6/4068

Previous findings have proven that the noise introduced by some adversarial attacks is less effective when the adversarial images are shuffled.
When shuffled, adversarial images lose a lot of their adversarial properties. However, when shuffling ancestor images, the effect is small and they 
are typically still classified as their original class.

Hence, shuffling can act as a detection mechanism for adversarial images. The idea is to compare a CNN's prediction for an image before and after shuffling.
If the two predictions are different, the detector will declare the image adversarial;
If the two predictions are identical (in terms of category, not probability), the detector will declare the image clean. 

This project generates adversarial images for various attacks and runs the detector to obtain a detection rate and a false positive rate for each attack.
It contains code to generate all attacks (except the EA attack) and run the detector. The adversarial images from the EA attack are obtained separately, but the detector can still be run on them.

