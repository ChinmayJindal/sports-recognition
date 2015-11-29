import os, time
import Util
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics.pairwise import chi2_kernel
import cv2

class Action:
	id = 0
	def __init__(self, path, name):
		self.root = path
		self.name = name
		self.id = Action.id
		Action.id += 1
		self.getVideoPaths()

	def getVideoPaths(self):
		videoDirs = filter(lambda p: os.path.isdir(self.root + os.path.sep + p), os.listdir(self.root))
		videoDirs = map(lambda p: self.root + os.path.sep + p, videoDirs)
		self.videos = []
		for v in videoDirs:
			video = filter(lambda p: p.endswith('.avi'), os.listdir(v))
			if len(video) < 1:
				continue
			video = video[0]
			self.videos.append(v + os.path.sep + video)

def main(root):
	actionDirs = filter(lambda p: os.path.isdir(root + os.path.sep + p), os.listdir(root))
	actionNames = actionDirs
	actionDirs = map(lambda p: root + os.path.sep + p, actionDirs)
	actions = map(lambda a, n: Action(a, n), actionDirs, actionNames)

	# getting dense trajectory features
	print 'Getting dense trajectory features'
	for action in actions[:1]:
		print '=> ' + action.name + ':'
		for video in action.videos[:1]:
			print video,
			start = time.time()
			Util.dumpDTF(video)
			end = time.time()
			print '(' + str(end - start) + 's)'
		print

	# making the codebook
	print 'Collecting all features for codebook generation'
	featuresCombined = np.empty([0,426], dtype=np.float32)

	tags = []
	for action in actions[:1]:
		for video in action.videos[:1]:
			featurePath = os.path.splitext(video)[0] + '_features.txt'

			start = featuresCombined.shape[0]
			with open(featurePath) as f:
				for line in f:
					# consists of HOG, HOF and MBH
					rawFeature = [float(d) for d in line.split('\t')[10:-1]]
					feature = np.array(rawFeature, dtype=np.float32)

					featuresCombined = np.vstack((featuresCombined, feature))
			end = featuresCombined.shape[0]

			tags.append((start, end, action.id))

	# performing k means
	k = 4000
	attempts = 5
	print 'Generating ' + str(k) + ' clusters'
	compactness, labels, centers = cv2.kmeans(featuresCombined, k, criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), attempts=attempts, flags=cv2.KMEANS_RANDOM_CENTERS)

	print 'Generating bag-of-words for each video'
	trainData = np.empty([0, k], dtype=np.float32)
	trainLabels = []
	for t in tags:
		hist = np.histogram(labels[t[0]:t[1]], k)
		trainData = np.vstack((trainData, hist))
		trainLabels.append(t[2])

	trainLabels = np.array(trainLabels)

	print 'Training SVM model with chi-squared kernel'
	# apply kernel on all data
	K = chi2_kernel(trainData, gamma=.5)
	# create n one-vs-all classifiers
	models = []
	for actions in actions:
		y = np.zeros(trainLabels.shape)
		y[y == action.id] = 1
		svm = SVC(kernel='precomputed').fit(K, y)
		models.append(svm)

# assume data set folder in "root"
if __name__ == '__main__':
	root = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'UCFSports' + os.path.sep + 'ucf_sports_actions'
	main(root)
