##Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./lane_lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.
In the next code cell I stored computed the camera calibration and distortion coefficients on disk in order to use them for experiments from other IPython notebooks. 

I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image, namely I used color threshold for v_channel of image and gradient over X axe for S and V channels.
```python
    binary [
        ((v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1]))
    ] = 1

    x_s_thresh = abs_sobel_thresh(s_channel, thresh=sx_thresh)
    x_v_thresh = abs_sobel_thresh(v_channel, thresh=sx_thresh)

    binary[(binary == 1) | (x_s_thresh ==1) | (x_v_thresh ==1)] = 1  
 ```
Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in the 9th code cell of the IPython notebook.
```python
#Compute direct and inverse perspective transforms
src = np.float32([[232, 678], [543, 480], [737, 480], [1048, 678]])
dst = np.float32([[280, 700], [280, 310], [1000, 310], [1000, 700]])

M = cv2.getPerspectiveTransform(src, dst) 
Minv = cv2.getPerspectiveTransform(dst, src)

#function to warp image
def warp_image(img, M):
    shape = img.shape[0:2][::-1]
    return cv2.warpPerspective(img, M, shape, flags=cv2.INTER_LINEAR)
``` 
The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points by manually selecting them on image.
Also, I used one a trick here. I intentionally used 310 for Y coordinate for top destination points.
That allowd me unwrap image far more further away compare to doing it by hand. 

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 232, 678      | 280, 700      | 
| 543, 480      | 280, 310      |
| 737, 480      | 1000, 310     |
| 1048, 678     | 1000, 700     |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used approach described by course matterials in order to identify lane line pixels.
Histogram based approach is used to identify pixels from scratch and it is implemented in find_lane_lines(binary_warped, nwindows=9, margin=100, minpix=50, debug=False) function.
Here is an example of lane lines detected by find_lane_lines function:
![alt text][image5]

And, when previous lines are known, def find_lane_lines_fast(binary_warped, left_fit_prev, right_fit_prev, debug=False) is used for faster lane lines detection.

Here is an example of lane lines detected by find_lane_lines_fast function:
![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I implemented def lane_curvature(fit_line, y_eval) which computes line curvature.
For lane lines curvature I'm using minimum value of left and right curvature:
```python
curvature = min(lane_curvature(left_fit, binary_warped.shape[0]),
                lane_curvature(right_fit, binary_warped.shape[0]))
```

Lane line position is calculted in draw_lanes_info function by the following peace of code:
```python
    vehicle_positioin = (left_fitx[-1] + right_fitx[-1]) / 2.0
    vehicle_offset = vehicle_positioin - image.shape[1] / 2.0
    vehicle_offset *= 3.7/700
 ```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function `draw_lanes_info`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.
I spent most of my time trying to identify lane lines reliably for all videos without significant progress.
So, eventually, I decided to come up with algorithm which will work only for project video and after that implement different techniques.
Even though, chosen algorithm worked reasonably well on project video without low pass filter, I decided to implement it in any case.
Then, I did several experiments with challenge_video, and make a conclusion that I took too wide image warp region. So, my alogrithm will not be able to hanlde sharp angles. Of course, it can be fixed easily.
However, I still need to improve my color and gradient thresholding pipeline, in order to better handle tricker cases.
Also, better outlier detection approach might work on complex images as well as longer lane lines history.
Will try it on my spare time.

When working on this project I though whether it is possible to apply deep learning approach for this task and think that yes it should be possible. The only problem, I would need enough training data in this case.

