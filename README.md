# Animal Tracking
### Object tracking with OpenCV in an open field behavioral test
---
## Origins
Input sample video is *rat.avi* placed in the root directory.

Output video demonstration is placed on the link below.

[![Open Field Test (Animal Tracking with OpenCV)](https://img.youtube.com/vi/GebcshN4OdE/0.jpg)](https://www.youtube.com/watch?v=GebcshN4OdE "Open Field Test (Animal Tracking with OpenCV)")

After processing the input video(s) by the script `track.py` you will get new directory `<current date>_distance` with CSV table

| Animal   | Distance     | Run Time  |
| -------- |:------------:| ---------:|
| ratN     | 56.38        |     420.1 |
| ...      | ...          |       ... |

and two subdirectories

1. **timing** with processed video(s) as above
2. **tracks** with images of animal(s) track(s) as on the left side of processed video

## Prerequisites

### Code Environment
#### Packages
Tracking is based on very simple image preprocessing and processing algorithms implemented in [OpenCV](http://opencv.org), therefore OpenCV and Numpy are the only requirement. To install it on your python environment type `pip install -r requirements.txt` in your terminal.
#### Codecs
Initially the script takes no arguments, scans its own directory and searches all AVI files encoded with [H.264](https://trac.ffmpeg.org/wiki/Encode/H.264), but it is up to you which file extension the script will search. To make a choice just replace all `*.avi` strings in the source code by your own extension.

### Camera Environment
Since the pixels of an animal contour are nested to the floor and the floor contour is nested to the box walls (animal ⊂ floor ⊂ walls), the floor have to be as contrast to its neighbors in this triplet as possible for better distinction between all of them. For example: white walls, black floor, and white animal or vice versa is the best case.

![Example of contrast](screenshots/example_of_contrast.png?raw=true "Example of contrast")

Because it's hard to fit the floor of the box within camera frame in the laboratory, the script makes some preprocessing. The script takes a certain part of each frame of the videos. It takes a right square `frame[:, w-h:w]` of a frame with sides equal to height ``h`` of a frame.

![Crop right square of the frame](screenshots/right_square_crop.png?raw=true "Crop right square of the frame")

That's why the box with an animal have to be placed in the right area of camera field of view.

(You are free to change this behavior replacing `frame[:, w-h:w]` (right square) by `frame[:, 0:w-h]` (left square), or just flip horizontally your video if you want to leave the script untouched.)

## User Interface
If requirements are complied then after starting the script you will get 1st frame of the video to be proccesed

![Good fit](screenshots/good_fit.png?raw=true "Good fit")

The right half of the window is the ordinary frame with the floor highlighted by the red quad. This highlighted area is stretched by homographic transform up to the square area on the left left half of the window.

* If you're satisfied the look of the floor you can either press any key except `ESC` or click right mouse button to continue.
* Else you are free to determine the right way of highlighting the area of the floor. To do that click left button on each corner of the floor starting with any of them.

![Fitting](screenshots/UI.gif?raw=true "Click-Move-Click")

Immediately after click on the 4th corner you can either

* press `Enter` and the processing will start
or
* repeate the loop and realign the contour of the floor again.
