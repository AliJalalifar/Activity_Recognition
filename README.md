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

# Refrences

[1]	C. H. Hsieh P. S. Huang and M. D. Tang "Human Action Recognition Using Silhouette Histogram "Proceedings of the Thirty-Fourth Australasian Computer Science Conference (ACSC 2011) pp. 11-15 Perth Australia 17-20 2011.  

[2]	Bobick, F. and Davis, J.W. (2001): The recognition of human movement using temporal templates. IEEETransactions on Pattern Analysis and Machine Intelligence, 23: 257-267. 

[3]	Hsieh, J.W., Hsu, Y.T., Liao, H.Y. Mark, and Chen, C.C. (2008): Video-based human movement analysis and its application to surveillance systems,” IEEE Transactions on Multimedia, 10: 372-384 

[4]	Kavi, Rahul, and Vinod Kulathumani. "Real-time recognition of action sequences using a distributed video sensor network." Journal of Sensor and Actuator Networks 2.3 (2013): 486-508. 

[5]	Zivkovic, Zoran. "Improved adaptive Gaussian mixture model for background subtraction." Pattern Recognition, 2004. ICPR 2004. Proceedings of the 17th International Conference on. Vol. 2. IEEE, 2004.

[6]	Zivkovic, Zoran, and Ferdinand Van Der Heijden. "Efficient adaptive density estimation per image pixel for the task of background subtraction." Pattern recognition letters 27.7 (2006): 773-780.

[7]	https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

[8]	Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." Computer Vision and Pattern Recognition, 2005. CVPR 2005. IEEE Computer Society Conference on. Vol. 1. IEEE, 2005.

[9]	Rosenblatt, Frank. "The perceptron: A probabilistic model for information storage and organization in the brain." Psychological review 65.6 (1958): 386.

[10]	Hornik, Kurt. "Approximation capabilities of multilayer feedforward networks." Neural networks 4.2 (1991): 251-257.
[11]	https://www.hiit.fi/u/ahonkela/dippa/node41.html

[12]	Kulathumani, V.; Ramagiri, S.; Kavi, R. WVU multi-view activity recognition dataset. Available online: http://www.csee.wvu.edu/ vkkulathumani/wvu-action.html (accessed on 1 April 2013).

[13]	http://scikit-learn.org/

