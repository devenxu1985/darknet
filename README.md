# darknet_demo

Modified for training with darknet

A modified from original darknet: https://github.com/pjreddie/darknet

# CHANGES

- Combine batchnorm layer and activation layer into one. (tensorrt feature)

# USAGE

`make`
`./darknet detector demo cfg/coco.data cfg/yolov3-tiny.cfg yolov3-tiny.weights -c <camera_id>`
