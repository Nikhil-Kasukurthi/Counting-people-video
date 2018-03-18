# Team Iris
This is under the Museum Analytics module for the [Coding Culture Hackathon](https://coding-culture.zkm.de/). 

# Counting-people-video
From the CCTV footage in each room, we get the number of people currently standing in there and generate insights about the inflow of visitors throughout the day. This way you could concentrate on the areas where visitors are not going. 

Using the Tensorflow Object detection API, we will be counting the number of people in a video. A frame is extracted every 30 seconds from the video and a forward pass of the model is performed. If a person is found in the video, then the count is increased. 

## Requirements
OpenCV - 3.3.1<br/>
Tornado<br/>
Tensorflow<br/>
Protocol buffer compiler

## Installation instructions
``` bash
# For CPU
pip install tensorflow
# For GPU
pip install tensorflow-gpu
```
For Ubuntu 
``` bash
sudo apt-get install protobuf-compiler 
```
For OSX
```
brew install protobuf
```
Other Libraries
```
pip install opencv-python
pip install tornado # For running the server 
```
Tensorflow object detection API
```
protoc utils/*.proto --python_out=.
```

## Running
If you want to test out the implementaion then you can use the object_detect.py<br/>
```
python object_detect.py --path <path to video>
```

To run the server<br/>
```
python server.py
```

### Instruction to plot bounding boxes
As per the origial implementation of the tensorflow object detection API, the bounding boxes are normalised. To get the original dimensions you need to do the following. 

```
(left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                              ymin * im_height, ymax * im_height)
```
