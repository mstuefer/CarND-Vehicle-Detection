# Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

---
## Writeup / README

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.

I started by importing necessary modules, then I defined some global variables, since the IPython notebook is 'only' for
learning purposes and not production code :-), which were useful during my different tests. Once defined I set them
the values with which I tested all my different methods in the IPython notebook.

Afterwards I created the function get_hog_features, as we did during lecture, to extract the hog features. Finally
I read all images, separated them in non-vehicles and vehicles, and visualize a non-vehicle and a vehicle with the
corresponding hog images.

![alt text][image1]
![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters, starting with the setup we also used during lecture, and end up with the
current set values.

####3. Describe how you trained a classifier using your selected HOG features and color features.

The code for this step is contained in the second code cell of the IPython notebook. After importing some additional 
necessary modules, I created the functions: convert_color, bin_spatial and color_hist, which I used together
with the already explained get_hog_features to extract features within the (more general) extract_features function.

(Note I always extract the hog_features on all three color channels.)

Then I used that last extract_features function to extract the features from the vehicles (cars) and non-vehicles 
(notcars). Finally, after preparing and splitting my train and test data, I trained a linear SVM and continuously 
got a high accuracy, about 0.989...

As you can see in the commented out code, and in the final thoughts below, I also tried to improve the quality of my
classifier (also experimented using linear and rbf..), by applying GridSearchCV.

###Sliding Window Search

####1. Describe how you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

In my third code cell of the main IPython notebook you can see my find_cars function. This is basically the one we
also used during lecture, while I had some issues in applying it first, after aligning the svm it worked nicely. 
The function itself does the following 'crucial' things:

 * search only in the relevant part of the image (ystart, ystop)
 * convert_color (since I read the image via mpimg.imread, I got a RGB which I converted in what I defined in cspace, depending on the test-cycle)
 * depending on the scale, possibly resized the image
 * defined: all three hog channels, blocks and steps to use
 * extracted the hog_features
 * then for each window extracted the hog_feature for that specific patch and invoked bin_spatial and color_hist
 * scaled the features and run the prediction, in prediction was possitiv I added a rectangle to the image and added the box to my bbox container 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As already stated in point three, and in the final thoughts, I tried to improve the quality of th classifier first manually, 
then also via using the GridSearchCV. To optimize the performance instead I tried to increase the pixels per cell, but 
then stopped there since I'm still not saitisfied with my classifier on the final video.

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  
From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  
I then assumed each blob corresponded to a vehicle.  
I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most interesting part this time was building a good classifier. Initially I played around with the paramters by 
hand, where I found out that the color spaces YCrCb and YUV worked better than the others (however, I still struggle on 
the white car). Then I set different orient, pix_per_cell etc. also to increase the throughput of my pipeline, but 
couldn't really find a better setup so far. Eventually I decided to also give GridSearchCV a try, unfortunately that 
took so long on my old computer, that I had to interrupt it, will launch it again on a VM.

I should have added more images to train the classifier on.

So far we do not detect vehicles in general but just cars, on the first truck/motocycle we encounter we would already 
be in trouble. As already stated our classifier should be improved, also by training it with much more data. Last but 
not least, the pipeline must be way faster to be used in a real scenario.
