# darknet_extract

extract a part of coco weights

A modified from original darknet: https://github.com/pjreddie/darknet

yolov3-finetune.cfg : the configuration file to train the last layer of yolo v3 (can be modified)

# CHANGES

`./src/parser.c`
for the 3 detection layers, parser reads the full weights and replace buy the 6 weights for randomize them.
`Important rules:` Read the code

# USAGE

`make`
`./extract`
