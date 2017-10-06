# Advanced Lane Finding Project  
  
<p align="center">
    <img src="https://github.com/ablacklama/Advanced-Lane-Finding/blob/master/output_images/Output.png?raw=true" width="640" alt="calib_image" /><br>    
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
  9. Draw info on output picture.

  
### 1. Convert to ImageClass 
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
  
TODO
  
### 3. Perspective Transform  
  
 We can assume the road is a flat plane. Pick 4 points of straight lane lines and apply perspective transform to the lines look straight. It is also called `Bird's eye view`.  
  
<p align="center">
    <img src="images/warp.jpg" width="640" alt="warp" /><br>
</p>  


### 4. Sliding Window Search  
  
The code for Sliding window search is contained in the [`finding_lines.py`](finding_lines.py) or [`finding_lines_w.py`](finding_lines.py).  

In the video, we could predict the position of lane lines by checking previous frame's information. But we need an other method for a first frame.  

In my code, if the frame is first frame or lost lane position, found first window position using histogram. Just accumulated non-zero pixels along the columns in the lower 2/3 of the image.  

<p align="center">
    <img src="images/histogram.jpg" width="320" alt="hist" /><br>
</p>  
  
  
In the course, we estimated curve line by using all non-zero pixels of windows. Non-zero piexels include **color information** and **gradient information** in bird's eyes view binary image. It works well in [`project_video`](project_video.mp4).  

**But it has a problem.**  
  
<p align="center">
    <img src="images/nonzero.jpg" width="480" alt="nonzero" /><br>
</p>  
  
This is one frame of [`challenge_video`](challenge_video.mp4).  
In this image, there are some cracks and dark trails near the lane lines. Let's check the result.  

If we fit curve lines with non-zero pixels, the result is here.
<p align="center">
    <img src="images/nonzero2.jpg" width="640" alt="nonzero2" /><br>
</p> 

As you can see, we couldn't detect exact lane positions. Because our gradient information have cracks information and it occurs error of position.  

So, I used **`weighted average`** method.
I put **0.8** weight value to color information and **0.2** to gradient information. And calculated x-average by using weighted average in the window. 
This is the result.  
<p align="center">
    <img src="images/weight.jpg" width="640" alt="weight" /><br>
</p>  
  

### 5. Road information  
  
<p align="center">
    <img src="images/road_info.jpg" width="480" alt="road_info" /><br>
</p>  
  
In my output video, I included some road informations.

#### Lane Info
* estimate lane status that is a straight line, or left/right curve. To decide this, I considered a radius of curvature and a curve direction.  
  
#### Curvature
* for calculating a radius of curvature in real world, I used U.S. regulations that require a minimum lane width of 3.7 meters. And assumed the lane's length is about 30m.  
  
#### Deviation
* Estimated current vehicle position by comparing image center with center of lane line.  

#### Mini road map
* The small mini map visualizes above information.

---

## Result  

>Project Video (Click for full HD video)
  
[![Video White](images/project_gif.gif?raw=true)](https://youtu.be/z0U_zPPL9cY)  
  
  
  
>Challenge Video (Click for full HD video)
  
[![Video White](images/challenge_result.gif?raw=true)](https://youtu.be/xri9c7xW1S4)

  
---
  
## Reflection  
  
I gave my best effort to succeed in challenge video. It wasn't easy. I have to change most of the parameters of project video. It means that the parameters strongly influenced by road status(bright or dark) or weather.  
To keep the deadline, I didn't try harder challenge video yet. It looks really hard but It could be a great challenge to me.  
