# Advanced Lane Finding Project  
  
<p align="center">
    <img src="https://github.com/ablacklama/Advanced-Lane-Finding/blob/master/output_images/example.png?raw=true" width="640" alt="calib_image" /><br>    
</p> 



## Intro

This is the Advance Lane Finding Project in Udacity's Self-Driving Car Engineer Nanodegree. In this project we are asked to build a better version of the previous lane finding project using more robust algorithms.
Our goal is to find and label the lane in the video, show the radius of the curve of the lane, and show how far from the center we are.

## Files
  
[`LaneFindingP4.ipynb`](LaneFindingP4.ipynb) : Main project notebook

[`calibrate.py`](calibrate.py) : get calibration matrix and store in pickle file

[`image_class.py`](image_class.py) : Class that stores and performs operations on an image

[`line_class.py`](line_class.py) : store line information and compute curvature


---

## Camera Calibration
Most cameras have a distorting effect on images they take. This throws off our measurements so we need to correct it. 
To do this used openCV's camera calibration functions which take images of a chess board at various angles. The function finds the inside corners of the chess board in all the images and computes the distortion matrix that can be applied to images to undistort them.
I then saved the distortion matrix in a pickle file so I could quickly load it.

[`calibration code`](calibrate.py)

<p align="center">
    <img src="https://github.com/ablacklama/Advanced-Lane-Finding/blob/master/output_images/ChessBoard.png?raw=true" width="640" alt="calib_image" /><br>    
</p>
  
## Pipeline
  1. Convert to ImageClass where all the line finding funtions reside.
  2. Undistort the image. 
  3. Apply Gaussian Blur. 
  4. Convert to binary Image.
  5. Warp perspective to have aerial view of lanes.
  6. Find Lines.
  7. Compute radius of curve.
  8. Find position of car relative to lane.

  The pipeline function exists in the [lane finding notebook](LaneFindingP4.ipynb) but most of the functionality of the pipeline exists in the [`image_class.py`](image_class.py). 2-5 exist in the `binary_warp` function and 6 is also a function of the [`image class`](image_class.py).
### 1. Convert to Image to ImageClass 

<p align="center">
    <img src="test_images/test7.jpg?raw=true" width="640" alt="Starting image" /><br>
</p>  

The [`ImageClass`]('image_class.py') I created for this project was really where the majority of my work went. It's also where almost all the complecated functions are. Each function (besides the find_lane functions) operates on the image stored in the class and returns a new ImageClass with the new image stored in it.

The Functions are:
* `undistort`: undistorts an image
* `imshow` : display and image with a set title
* `GaussianBlur` : apply gaussian blur with set kernal size
* `format` : change image format (rgb, hls, hsv, gray). Can also narrow down to one image channel (i.e. l from hls)
* `sobel_thresh : produce binary image with threshold of sobel operation along specified axis
* `dir_threshold` : produce binary image of sobel direction between threshold values
* `mag_thresh` : produce binary image of sobel magnitude between threshold values
* `andImg` : preform bitwise 'and' opperation on binary image
* `orImg` : preform bitwise 'or' opperation on binary image
* `transform` : transform perspective to aerial view
* `ROI` : narrows image down to a region-of-interest
* `binary_warp` : uses previous class functions to turn rgb image into warped binary image
* `find_lanes` : finds lane line coefficients and points using windowed search
* `find_lines_with_lines` : finds lane line coefficients and points based on previus coefficients

Building a class like this (starting with the basic functions and working up) let me make more complex functions like `binary_warp` in building block style. 
  
### 2. Undistort the Image  
`ImageClass.undistort(self, mtx, dist)` handled the undistortion for me by calling cv2's undistort method. I'd put an image here but it really looks the same as the input image above because the camera has barely any distortion.

### 3. Gaussian Blur 
<p align="center">
    <img src="output_images/GaussianBlur.png?raw=true" width="640" alt="blur" /><br>
</p>  

After correcting for any camera distorion I want to make the image a little more friendly to the sobel functions. Sobel edgefinding can be sensitive to lots of tiny but still strong edges. We don't really care about tiny edges composed of a few pixels, so I used Gaussian Blur to smooth them out. [Gaussian Blur](https://en.wikipedia.org/wiki/Gaussian_blur) basically just blurs the image using a kernal of a specified size. I choose a kernal size of 5 after seeing no noticable increase in accuracy beyong that size. 

### 4. Convert to Binary Image
This is where a lot of the magic happens. I want to convert the image to a black and white image (each pixel either 0 or 1) where only the lanes (in theory) are white. I use my `ImageClass.format` method to create three grey images. One is just a grayed out version of the normal RGB image, but the other two are the 'l' and 's' channels of an 'hls' formatted version of the image. These are the 'lightness' and 'saturation' channels. Both do a good job at identifying certain types of lane lines.
(NOTE: In the binary_warp code, conversion from rgb to grey is done autimatically by dir_threshold, and mag_thresh functions)


<p align="center">
    <img src="output_images/Lchannel.png?raw=true" width="300" alt="L channel" /><br>
</p>  
<p align="center">
    <img src="output_images/Schannel.png?raw=true" width="300" alt="S channel" /><br>
</p>


I then create two new images (`dir_thresh` and `mag_thresh`) with by taking the directional magnitude and gradient magnitude of the gray image. Both of these operations use the sobel function.

<p align="center">
    <img src="output_images/DirectionalMagnitude.png?raw=true" width="300" alt="directional magnitude" /><br>
</p>  
<p align="center">
    <img src="output_images/GradientMagnitude.png?raw=true" width="300" alt="gradient magnitude" /><br>
</p>

After that I use the sobel operation to create 3 new images. The sobel function computes the gradient of change in an image along either the 'x' or 'y' axis. sobel 'x' of 'l' channel image, sobel 'x' of 's' channel image, and sobel 'y' of 's' channel image.

<p align="center">
    <img src="output_images/Sobelx_L_Channel.png?raw=true" width="300" alt="sobel along 'x' axis with 'l' channel" /><br>
</p>  
<p align="center">
    <img src="output_images/Sobelx_S_channel.png?raw=true" width="300" alt="sobel along 'x' axis with 's' channel" /><br>
</p>
<p align="center">
    <img src="output_images/Sobely_S_channel.png?raw=true" width="300" alt="sobel along 'y' axis with 's' channel" /><br>
</p>

Then I had to combine them all back into one image. I used a series of binary and/or opperations to do this by first combining `dir` and `mag` images in an 'and' operation to create `magdir`. 

<p align="center">
    <img src="output_images/GradDirMag.png?raw=true" width="300" alt="combined gradient and directional magnitude" /><br>
</p>

Then I combined `sx` and `sy` with an 'and' opperation to create `sxy`, that image and `lx` were then combine with the 'or' opperation to create `lsxy`. 

<p align="center">
    <img src="output_images/Combined_Sobel.png?raw=true" width="300" alt="sxy" /><br>
</p>
<p align="center">
    <img src="output_images/lsxy.png?raw=true" width="300" alt="lsxy" /><br>
</p>

Finally I merge `magdir` and `lsxy` to with the `or` opperation to create the `both` image which is then cropped down to a `ROI` or region of interest.

<p align="center">
    <img src="output_images/both.png?raw=true" width="300" alt="both" /><br>
</p>
<p align="center">
    <img src="output_images/ROI.png?raw=true" width="300" alt="ROI" /><br>
</p>

### 5. Perspective Transformation 
  
<p align="center">
    <img src="output_images/Warped_Perspective.png?raw=true" width="480" alt="perspective warp" /><br>
</p>  
  
Here I use my `transform` function and openCV's `getPerspectiveTransform` to change the perspective of the image to top down. This makes finding the lines and calculating the curvature much easier.

### 6. Finding Lines
To find the lines I took a histogram of the bottom half of the warped image to show which points on the x axis had the most white points along their 'y' axis.

<p align="center">
    <img src="output_images/BottomHalf.png?raw=true" width="480" alt="bottom half" /><br>
</p>  
<p align="center">
    <img src="output_images/Histogram.png?raw=true" width="480" alt="histogram" /><br>
</p>  
  
 I used the highest points on either side of the middle of the histogram as my starting points to find the lane. Finding points on the lane was done with a windowed search that moved left or right depending on where it last found the most points. This allowed it to follow the curve of the lane.
 
 <p align="center">
    <img src="output_images/Windows.png?raw=true" width="480" alt="windowed search" /><br>
</p>  
 
 Then I fit a line to the points on each lane.
 
 <p align="center">
    <img src="output_images/WindowsLines.png?raw=true" width="480" alt="fitted lines" /><br>
</p>  
 
 All of this information was saved in a [`line_class.py`](line_class.py) that I used to keep track of all line information. Knowing where the last lines were allowed me to quickly search for them again in the next frame without having to do a windowed search. This speeds up processing considerably, and if you ever don't find a line with this approach, you can always go back to a windowed search. I also have a sanity check written in here that makes sure there are enough points in the line. A line with too few points will probably have an inacurate fit.
 
### 7. Computing the Radius of the Curvature 
I assumed here that the lane I found was roughly 30 Meters in length. Then used their polynomial coefficients to calculated seperate radiuses for each lane. I also included a sanity check here to make sure the lanes were parallel within a reasonable deviation.
  
### 8. Find position of car relative to the lane
Here used the distance of the center of the car to each lane in pixels as a starting point. Then knowing that a lane is 3.7 meters accross I got the length of every pixel and used it to find how far off center the car was.


## Result  

 <p align="center">
    <img src="output_images/Output1.png?raw=true" width="480" alt="final image output" /><br>
</p>  
  
## Reflection  
  
While this architecture works well for this video, it didn't hold up well on the challenge videos. And I don't believe any architecture like this would do well in all cases. You would need to know when to use it and when not too. Some roads don't even have lanes. Additionally while the ImageClass I created was really helpful for me in writing the code and putting together the pipeline, it slows down the processing considerably. If I were to do this for a real car I would move the functions out of classes to speed up processing.

