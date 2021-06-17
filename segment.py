
# python segment.py --model enet-cityscapes/enet-model.net --classes enet-cityscapes/enet-classes.txt --colors enet-cityscapes/enet-colors.txt --image images/example_01.png

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to deep learning segmentation model")
ap.add_argument("-c", "--classes", required=True,
	help="path to .txt file containing class labels")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-l", "--colors", type=str,
	help="path to .txt file containing colors for labels")
ap.add_argument("-w", "--width", type=int, default=500,
	help="desired width (in pixels) of input image")
args = vars(ap.parse_args())

# load the class label names
CLASSES = open(args["classes"]).read().strip().split("\n")


COLORS = open(args["colors"]).read().strip().split("\n")
COLORS = [np.array(c.split(",")).astype("int") for c in COLORS]
COLORS = np.array(COLORS, dtype="uint8")


# initialize the legend visualization
legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 3), dtype="uint8")

# loop over the class names + colors
for (i, (className, color)) in enumerate(zip(CLASSES, COLORS)):
	# draw the class name + color on the legend
	color = [int(c) for c in color]
	cv2.putText(legend, className, (5, (i * 25) + 17),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25),
		tuple(color), -1)


print("[INFO] loading model...")
net = cv2.dnn.readNet(args["model"])


# ENet was trained on was 1024x512
image = cv2.imread(args["image"])
image = imutils.resize(image, width=args["width"])
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0,
	swapRB=True, crop=False)


net.setInput(blob)
start = time.time()
output = net.forward()   #output will be same dimensions as that of image but each pixel here has a corresponding class label index
end = time.time()


print("[INFO] inference took {:.4f} seconds".format(end - start))


(numClasses, height, width) = output.shape[1:4]


classMap = np.argmax(output[0], axis=0)   #here we find the class label index with largest probability for each pixels


mask = COLORS[classMap]
#and based on the classMap we index our colors provided in the args.

mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
	interpolation=cv2.INTER_NEAREST)       #we are resizing our mask n classMap such that they have exact dimensions as our image.
classMap = cv2.resize(classMap, (image.shape[1], image.shape[0]),
	interpolation=cv2.INTER_NEAREST)

# perform a weighted combination of the input image with the mask to
# form an output visualization
output = ((0.4 * image) + (0.6 * mask)).astype("uint8")

# show the input and output images
cv2.imshow("Legend", legend)
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)