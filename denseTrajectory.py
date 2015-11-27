import numpy as np
import cv2

# global variables
trackLength = 15
minDistance = 5
patchSize = 32
nxyCells = 2
ntCells = 3
scaleNum = 8
initGap = 1
scaleStride = 1.414
minFlow = 0.4

videoSrc = '/home/chinmay/Desktop/video.mp4'


class TrackInfo:
	def __init__(self, length, gap):
		self.length = length 		# length of the trajectory
		self.gap = gap 				# initialization gap for feature re-sampling

class DescInfo:
	def __init__(self, nBins, isHof, nxyCells, ntCells, size):
	    self.nBins = nBins			# number of bins for vector quantization
	    self.isHof = isHof
	    self.nxCells = nxyCells		# number of cells in x direction
	    self.nyCells = nxyCells
	    self.ntCells = ntCells

	    self.dim = self.nxCells*self.nyCells*self.nBins 	# dimension of the descriptor
	    self.height	= size			# size of the block for computing the descriptor
	    self.width = size

class Track:
	def __init__(self):
		self.point = []
		self.hog = []
		self.hof = []
		self.mbhX = []
		self.mbhY = []

class SeqInfo:
	def __init__(self, width, height, length):
		self.width = width
		self.height = height
		self.length = length

class DescMat:
	def __init__(self, height, width, nBins):
		self.height = height
		self.width = width
		self.nBins = nBins
		self.desc = [ None for i in xrange(height * width * nBins) ]

# returns a list of scales to use, using scaleStride
def InitPry(width, height):
	global scaleNum

	w,h = float(width),float(height)

	minSize = min(w,h)

	nLayers = 0
	while(minSize>=patchSize):
		minSize /= scaleStride
		nLayers+=1

	if nLayers==0:
		nLayers = 1

	scaleNum = min(scaleNum, nLayers)

	fscales = [1.0]

	curScale = 1.0
	for i in xrange(1, scaleNum):
		curScale *= scaleStride
		fscales.append(curScale)

	return fscales

def BuildDescMat(xComp, yComp, desc, descInfo):
	maxAngle = 360.0
	nDims = descInfo.nBins
	# one more bin for hof
	nBins = descInfo.nBins - 1 if descInfo.isHof else descInfo.nBins
	angleBase = float(nBins) / maxAngle

	step = (xComp.shape[1] + 1) * nDims
	index = step + nDims

	for i in xrange(xComp.shape[0]):
		xc = xComp[i]
		yc = yComp[i]

		sumArray = []
		for j in xrange(xComp.shape[1]):
			x = xc[j]
			y = yc[j]
			mag0 = np.sqrt(x * x + y * y)
			mag1 = None
			bin0, bin1 = None, None

			if descInfo.isHof && mag0 <= minFlow:
				bin0 = nBins
				mag0 = 1.0
				bin1 = 0
				mag1 = 0


		index += nDims

def HogComp(img, desc, descInfo):
	imgx = cv2.Sobel(img, cv2.CV_64F, 1, 0, 1)
	imgy = cv2.Sobel(img, cv2.CV_64F, 0, 1, 1)
	BuildDescMat(imgx, imgy, desc, descInfo)

def main():
	trackInfo = TrackInfo(length=15, gap=1)
	hogInfo = DescInfo(nBins=8, isHof=False, nxyCells=2, ntCells=3, size=32)
	hofInfo = DescInfo(nBins=9, isHof=True, nxyCells=2, ntCells=3, size=32)
	mbhInfo = DescInfo(nBins=8, isHof=False, nxyCells=2, ntCells=3, size=32)

	videoCam = cv2.VideoCapture(videoSrc)
	if not videoCam:
		raise Exception('Unable to initialise video reader')

	# fetch first frame
	flag, frame = videoCam.read()
	height, width, channels = frame.shape

	fscales = InitPry(width, height)

	# for optical flow
	image, prev_grey, grey = None, None, None
	prev_grey_pyr = [ None for i in xrange(len(fscales)) ]
	grey_pyr = [ None for i in xrange(len(fscales)) ]
	flow_pyr = [ None for i in xrange(len(fscales)) ]
	prev_poly_pyr = [ None for i in xrange(len(fscales)) ]
	poly_pyr = [ None for i in xrange(len(fscales)) ]

	xyScaleTracks = [ [] for i in xrange(len(fscales)) ]

	firstFrame = True
	initCounter = 0
	while flag:
		if firstFrame == True:
			firstFrame = False

			image = frame.copy()
			prev_grey = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)
			for i in xrange(scaleNum):
				if i==0:
					prev_grey_pyr[0] = prev_grey.copy()
				else:
					prev_grey_pyr[i] = cv2.resize(prev_grey_pyr[i-1], None, fx=fscales[i], fy=fscales[i], interpolation=cv2.INTER_LINEAR)

				# list of (x,y)
				points = []

				# @TODO do dense sampling here

				for p in points:
					xyScaleTracks[i].append(Track(p, trackInfo, hogInfo, hofInfo, mbhInfo))

			# @TODO calculate farneback polynomial here

			flag, frame = videoCam.read()
			continue

		initCounter += 1
		image = frame.copy()
		grey = cv2.cvtColor(image, cv2.cv.CV_BGR2GRAY)

		# @TODO calculate farneback polynomial here
		# @TODO calculate optical flow farneback polynomial here

		for i in xrange(scaleNum):
			if i == 0:
				grey_pyr[0] = grey.copy()
			else:
				grey_pyr[i] = cv2.resize(grey_pyr[i-1], None, fx=fscales[i], fy=fscales[i], interpolation=cv2.INTER_LINEAR)

			height, width = grey_pyr.shape

			# compute integral histograms
			hogMat = DescMat(height + 1, width + 1, hogInfo.nBins)
			BuildDescMat(hogInfo)
			hofMat = DescMat(height + 1, width + 1, hofInfo.nBins)
			mbhMatX = DescMat(height + 1, width + 1, mbhInfo.nBins)
			mbhMatY = DescMat(height + 1, width + 1, mbhInfo.nBins)


		flag, frame = videoCam.read()

main()
