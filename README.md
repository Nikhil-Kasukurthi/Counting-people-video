# Counting-people-video
Counting the number of people in a video. 

Using the Tensorflow Object detection API, we will be counting the number of people in a video. A frame is extracted every 30 seconds from the video and a forward pass of the model is performed. If a person is found in the video, then the count is increased. 

## Requirements
OpenCV - 3.3.1

Tensorflow object detection API

### Instruction to plot bounding boxes
As per the origial implementation of the tensorflow object detection API, the bounding boxes are normalised. To get the original dimensions you need to do the following. 

```
(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                              ymin * im_height, ymax * im_height)
```
