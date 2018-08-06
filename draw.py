#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

with open("loss.txt", "r") as f:
	losses = f.read()
with open("IoU.txt", "r") as f:
	IoUes  = f.read()

loss = []
for i in losses.split("\n"):
	try:
		loss.append(np.log(float(i.split(" ")[1])))
	except IndexError:
		continue

IoU = []
for i in IoUes.split("\n"):
	try:
		IoU.append(1-float(i.split( )[0]))
	except IndexError:
		continue


plt.figure("Log loss")
plt.plot(range(len(loss)), loss, "r.")

#plt.figure("1-IoU")
#plt.plot(range(len(IoU)), IoU, "bo")
plt.show()

