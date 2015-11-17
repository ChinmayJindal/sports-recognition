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
		def __init__(self):
			self.width = width
			self.height = height
			self.length = length


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

	fscales = [1.0, width, height]

	curScale = 1.0
	for i in xrange(1,scaleNum):
		newScale = curScale*scaleStride
		newWidth, newHeight = cv2.cv.Round(w/newScale), cv2.cv.Round(h/newScale)

		fscales.append((curScale, newWidth, newHeight))

	return fscales

def main():
	trackInfo = TrackInfo(length=15, gap=1)
	hogInfo = DescInfo(nBins=8, isHof=False, nxyCells=2, ntCells=3, size=32)
	hofInfo = DescInfo(nBins=9, isHof=True, nxyCells=2, ntCells=3, size=32)
	mbhInfo = DescInfo(nBins=8, isHof=False, nxyCells=2, ntCells=3, size=32)

	image, prev_grey, grey = None,None,None

	# for optical flow
	prev_grey_pyr, grey_pyr, flow_pyr = [],[],[]
	prev_poly_pyr, poly_pyr = [],[]

	xyScaleTracks = []

	videoCam = cv2.VideoCapture(videoSrc)
	if not videoCam:
		raise Exception('Unable to initialise video reader')

	# fetch first frame
	flag, frame = videoCam.read()
	height, width, channels = frame.shape

	fscales = InitPry(width, height)
	
	firstFrame = True

	while flag:
		# process the source
		image = frame.copy()

		if firstFrame==True:
			firstFrame = False

			prev_grey = cv2.cvtColor(frame, cv2.cv.CV_BGR2GRAY)

			for i in xrange(scaleNum):
				if i==0:
					prev_grey_pyr.append(prev_grey)
				else:
					prev_grey_pyr.append(cv2.resize(prev_grey_pyr[i-1], None, fx=fscales[i][0], fy=fscales[i][0], interpolation=cv2.INTER_LINEAR))

				# list of (x,y)
				points = []
				


		flag, frame = videoCam.read()

main()