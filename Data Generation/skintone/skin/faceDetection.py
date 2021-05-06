from sklearn.cluster import KMeans
import cv2
import numpy as np
import utils
import matplotlib.pyplot as plt

def skin(color):
	temp = np.uint8([[color]])
	color = cv2.cvtColor(temp,cv2.COLOR_RGB2HSV)
	color=color[0][0]
	e8 = (color[0]<=25) and (color[0]>=0)
	e9 = (color[1]<174) and (color[1]>58)
	e10 = (color[2]<=255) and (color[2]>=50)
	return (e8 and e9 and e10)

# This function is meant to give the skin color of the person by detecting face and then
# applying k-Means Clustering.
def get_skin_color(img):

	# Load the face detector.
	face_cascade =cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	# Convert to grayscale image.
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Detect face in the image.
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	# If a face is detected.
	if(len(faces)>0):
		for (x,y,w,h) in faces:

			# Take out the face from the image.
			image=img[y:y+h,x:x+h]

			# Apply k-Means Clustering to the face to obtain most dominant color.
			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			image = image.reshape((image.shape[0] * image.shape[1], 3))
			clt = KMeans(n_clusters = 4)
			clt.fit(image)

			def centroid_histogram(clt):
				# Grab the number of different clusters and create a histogram
				# based on the number of pixels assigned to each cluster.
				numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
				(hist, _) = np.histogram(clt.labels_, bins = numLabels)
			 
				# Normalize the histogram, such that it sums to one.
				hist = hist.astype("float")
				hist /= hist.sum()
			 
				# Return the histogram.
				return hist

			def get_color(hist, centroids):

				# Obtain the color with maximum percentage of area covered.
				maxi=0
				COLOR=[0,0,0]

				# Loop over the percentage of each cluster and the color of
				# each cluster.
				for (percent, color) in zip(hist, centroids):
					if(percent>maxi):
						COLOR=color
						maxi=percent

				# Return the most dominant color.
				return COLOR

			# Obtain the color and convert it to HSV type
			hist = centroid_histogram(clt)
			skin_temp1 = get_color(hist, clt.cluster_centers_)
			# print(skin_temp1)
			skin_temp2 = np.uint8([[skin_temp1]])
			skin_color = cv2.cvtColor(skin_temp2,cv2.COLOR_RGB2HSV)
			skin_color=skin_color[0][0]
			# print(skin_color)

			# Return the color.
			return (True,skin_color)
	else:
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		image = image.reshape((image.shape[0] * image.shape[1], 3))
		clt = KMeans(n_clusters = 3)
		clt.fit(image)

		def centroid_histogram(clt):
			# Grab the number of different clusters and create a histogram
			# based on the number of pixels assigned to each cluster.
			numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
			(hist, _) = np.histogram(clt.labels_, bins = numLabels)
		 
			# Normalize the histogram, such that it sums to one.
			hist = hist.astype("float")
			hist /= hist.sum()
		 
			# Return the histogram.
			return hist

		def get_color(hist, centroids):

			# Obtain a color which satisfies skin condition.
			cnt=0
			list=[]

			# Loop over the percentage of each cluster and the color of
			# each cluster to see if there is such a color.
			for (percent, color) in zip(hist, centroids):
				if(skin(color)):
					cnt=cnt+1
					list.append([color,percent])
			if(cnt==1):
				return list[0][0]
			else:
				return [0,0,0]

		# Obtain the color and convert it to HSV type
		hist = centroid_histogram(clt)
		skin_color = get_color(hist, clt.cluster_centers_)

		if(skin_color[0]==0 and skin_color[0]==0 and skin_color[0]==0):
			return (False,skin_color)
		else:
			skin_temp2 = np.uint8([[skin_color]])
			skin_color = cv2.cvtColor(skin_temp2,cv2.COLOR_RGB2HSV)
			skin_color=skin_color[0][0]
			return skin_color

# get_skin_color('1.jpg')