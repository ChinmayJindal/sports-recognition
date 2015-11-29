import os, time
import pickle
import Util
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KDTree
from sklearn import cross_validation
import cv2
from random import shuffle
import math

# Parameters:
# src - path to the video file
# label - action type in the video
# Stores dense trajectory features for the video and loads them into RAM
#
class Video:
	def __init__(self, src, label):
		self.src = src
		self.label = label
		self.featurePath = os.path.splitext(src)[0] + '_features.txt'

		# dump features if not present
		if not os.path.exists(self.featurePath):
			Util.dumpDTF(self.src)

		self.loadFeatures()

	# load features into the memory
	def loadFeatures(self):
		self.features = np.empty([0, 426], dtype=np.float32)
		with open(self.featurePath) as f:
			for line in f:
				# consists of HOG, HOF and MBH
				rawFeature = [float(d) for d in line.split('\t')[10:-1]]
				feature = np.array(rawFeature, dtype=np.float32)
				self.features = np.vstack((self.features, feature))

	# generate bag of words given centers of visual words
	def generateBOW(self, clusterCenters):
		tree = KDTree(clusterCenters)
		indexMatch = []

		for f in self.features:
			dist, index = tree.query(f, k=1)
			indexMatch.append(index)

		hist, bin_edges = np.histogram(indexMatch, clusterCenters.shape[0])
		return hist

# Parameters:
# path - path where videos of a action are located
# name - name of the action
# Stores dense trajectory features for the video and loads them into RAM
#
class Action:
	id = 0
	def __init__(self, path, name):
		print "Starting action : " + name + " ",
		self.root = path
		self.name = name
		self.id = Action.id
		print self.id
		Action.id += 1
		self.getVideoPaths()
		self.traintestSplit()

	# explore the action directory and get paths of all the video files
	def getVideoPaths(self):
		videoDirs = filter(lambda p: os.path.isdir(self.root + os.path.sep + p), os.listdir(self.root))
		videoDirs = map(lambda p: self.root + os.path.sep + p, videoDirs)
		self.videos = []
		for v in videoDirs:
			print v
			video = filter(lambda p: p.endswith('.avi'), os.listdir(v))
			if len(video) < 1:
				continue
			video = video[0]
			self.videos.append(Video(src=v+os.path.sep+video, label=self.id))

	# split all the videos for this action into train/test set.
	def traintestSplit(self):
		shuffle(self.videos)
		numTrain = int(math.ceil(0.3*len(self.videos)))
		self.traindata = self.videos[:numTrain+1]
		self.testdata = self.videos[numTrain+1:]

#
# Parameters:
# test_size, split ratio of train and test
# k, for k-means clustering
# attempts, number of attempts for k-means
#
def main(root):
	actionDirs = filter(lambda p: os.path.isdir(root + os.path.sep + p), os.listdir(root))
	actionNames = actionDirs
	actionDirs = map(lambda p: root + os.path.sep + p, actionDirs)
	actions = map(lambda a, n: Action(a, n), actionDirs, actionNames)

	print 'Collecting cluster features'
	# collect all features for generating codebook.
	clusterFeatures = np.empty([0,426], dtype=np.float32)
	tags = []
	for action in actions:
		for video in action.traindata:
			print video.src
			start = clusterFeatures.shape[0]
			clusterFeatures = np.vstack((clusterFeatures, video.features))
			end = clusterFeatures.shape[1]
			tags.append((start, end, action.id))

	# performing k means clustering for creating dictionary of visual words
	k = 4000
	attempts = 10
	print 'Generating ' + str(k) + ' clusters'
	compactness, labels, centers = cv2.kmeans(clusterFeatures, k, criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), attempts=attempts, flags=cv2.KMEANS_RANDOM_CENTERS)


	# generating data and labels for svm training
	print 'Generating bag-of-words for each video'
	trainData = np.empty([0, k], dtype=np.float32)
	trainLabels = []
	for t in tags:
		hist, bin_edges = np.histogram(labels[t[0]:t[1]], k)
		trainData = np.vstack((trainData, hist))
		trainLabels.append(t[2])
	trainLabels = np.array(trainLabels)

	# using one v/s all svm classifier
	print 'Training SVM model with chi-squared kernel'
	model = OneVsRestClassifier(SVC(kernel=chi2_kernel, random_state=0, class_weight='auto')).fit(trainData, trainLabels)
	pickle.dump(model, open('model.p', 'w'))
	pickle.dump(centers, open('centers.p', 'w'))


	#
	#	Testing:
	#		Generate dense trajectory features for every input test video and then get bag of words.
	#		use trained svm for predicting the output
	testData = np.empty([0,k], dtype=np.float32)
	testLabels = []
	for action in actions:
		for video in action.testdata:
			hist = video.generateBOW(centers)
			testData = np.vstack((testData, hist))
			testLabels.append(action.id)
	testLabels = np.array(testLabels)

	# predicted labels compared with the true labels to get the classification accuracy
	predictedLabels = model.predict(testData)

	print "accuracy: " + str(float(np.sum(np.array(testLabels)==np.array(predictedLabels)))/predictedLabels.shape[0])


# assume data set folder in "root"
if __name__ == '__main__':
	root = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'UCFSports' + os.path.sep + 'ucf_sports_actions'
	main(root)
