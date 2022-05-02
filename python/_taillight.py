import numpy as np
import cv2

from PIL import Image

import numpy as np
import cv2

from PIL import Image
import timeit

def expend_cluster(crop128, src):
	extend_src = src.copy()
	marked_reg = np.zeros(src.shape, np.uint8);

	srcRect = [0, 0, crop128.shape[0], crop128.shape[1]];

	directions = [] # list of 2-tuples.
	directions.append((0, 1));
	directions.append((0, -1));
	directions.append((1, 0));
	directions.append((-1, 0));
	directions.append((1, 1));
	directions.append((-1, -1));
	directions.append((-1, 1));
	directions.append((1, -1));

	for i in range (0, src.shape[0]):
		for j in range (0, src.shape[1]):
			if (src[i][j] == 255 and marked_reg[i][j] == 0):
				marked_reg[i][j] = 1;
				p_bank = [(j, i)]
				while len(p_bank) > 0:
					cen_point = p_bank[-1];
					p_bank.pop();
					for _d in range (0, len(directions)):
						tp = cen_point + directions[_d];
						if tp[0] >= srcRect[0] and tp[0] < srcRect[1] and tp[1] >= srcRect[2] and tp[1] < srcRect[3]:
							if (crop128[tp] == crop128[i][j] and marked_reg[tp] == 0):
								extend_src[tp] = 255;
								marked_reg[tp] = 1;
								p_bank.append(tp);
							# end if
						# end if
					# end for direction
				# end while
	return extend_src;

def reduceVal128(val):
	if (val < 64): return 0;
	if (val < 128): return 64;
	if (val < 192): return 128;
	return 255;

def processColors128_old(img):
	for i in range (0, img.shape[0]):
		pixelPtr = img[i]
		for j in range (0, img.shape[1]):
			pixelPtr[j][0] = reduceVal128(pixelPtr[j][0]); # B
			pixelPtr[j][1] = reduceVal128(pixelPtr[j][1]); # G
			pixelPtr[j][2] = reduceVal128(pixelPtr[j][2]); # R
	return img

def processColors128(img):
	img = np.where(img < 64, 0, img)
	img = np.where(np.logical_and(img >= 64, img < 128), 64, img)
	img = np.where(np.logical_and(img >= 128, img < 192), 128, img)
	img = np.where(img >= 192, 255, img)
	return img

def binary_ROI(cropSrc):
	max_rgb = np.zeros((cropSrc.shape[0],cropSrc.shape[1]), dtype=np.uint8);

	for i in range (0, cropSrc.shape[0]):
		for j in range (0, cropSrc.shape[1]):
			r = int(cropSrc[i][j][2])
			g = int(cropSrc[i][j][1])
			b = int(cropSrc[i][j][0])

			diff1 = abs(r - g);
			diff2 = abs(g - b);
			diff3 = abs(r - b);
			max_rgb[i][j] = int((diff1 + diff2 + diff3) / 3.0)

	crop128 = processColors128(cropSrc);

	_, max_rgb = cv2.threshold(max_rgb, 100, 255, cv2.THRESH_BINARY);
	max_rgb = expend_cluster(crop128, max_rgb);
	max_rgb = cv2.GaussianBlur(max_rgb, (5, 5), 0.0, 0.0, 4);
	_, max_rgb = cv2.threshold(max_rgb, 100, 255, cv2.THRESH_BINARY);

	return max_rgb;

def isLightOn(bgr):
	mask1 = cv2.inRange(bgr, (0, 0, 150), (30, 30, 255));
	rr = np.sum(mask1)
	return rr

def contour(binaryROI, inputImage, x, y):
	_, contours, hierarchy = cv2.findContours(binaryROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);

	# Approximate contours to polygons + get bounding rects and circles
	acc = 0;

	coordListWithColor = []
	# Get the boundingRect and center + radius for circles
	for i in range (0, len(contours)):
		contours_poly = cv2.approxPolyDP(contours[i], 1, True);
		boundRect_x,boundRect_y,boundRect_w,boundRect_h = cv2.boundingRect(contours_poly);
		center, radius = cv2.minEnclosingCircle(contours_poly);
		center_x = int(center[0])
		center_y = int(center[1])
		roi_mt = inputImage[boundRect_y+y:boundRect_y+y+boundRect_h,boundRect_x+x:boundRect_x+x+boundRect_w]
		on = isLightOn(roi_mt);
		if (radius > binaryROI.shape[1]/25):
			coordListWithColor.append((boundRect_x+x, boundRect_y+y, boundRect_x+x+boundRect_w,  boundRect_y+y+boundRect_h, on))
			acc = acc + 1;

	return coordListWithColor

# detects tail light position within the detected vehicle.
def getTaillightPos(src, x1, y1, x2, y2):
    image_roi = src[y1:y2,x1:x2]
    binary_roi = binary_ROI(image_roi);
    coordListWithColor = contour(binary_roi, src, x1, y1);
    return coordListWithColor
