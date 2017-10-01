# Activity_Recognition

Human activity recognition plays a central role in many modern applications such as surveillance systems, human computer interface, assisted living, etc. With the utilization of multi-camera data, we can improve accuracy of activity recognition due to reduced occlusion and more information from the scene. In one categorization, multi-camera methods divide into two main groups: view selection and view fusion. In this report, we use the view fusion algorithm combined with machine learning methods. We extract the silhouette of the frames and then obtain the history motion image from them. Finally, by using histogram of oriented gradients we form the feature vector to feed an MLP classifier. Finally we use to classifier for each of eight views available for each frame and decide which activity is belongs to.



# How to Run
to run the code put you test data into folder "Tests". it should be one level subdirectory. For example:

Tests\test13\view1

Note that folder test# should consist folders that contain images from different views and should be
in the format view#.

Next, simpliy run ActionRecognition.py . note that for every test folder a txt file named label.txt
will be created at the test folder after ActionRecognition.py  is finished succesfully. For example Tests\test13\label.txt. 

label.txt consist of perdictions of the activity for every frame.


# Database
[WVU MULTI-VIEW ACTION RECOGNITION DATASET](http://community.wvu.edu/~vkkulathumani/wvu-action.html)
