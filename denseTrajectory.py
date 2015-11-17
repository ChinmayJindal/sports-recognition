import numpy as np

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

def main():
	trackInfo = TrackInfo(length=15, gap=1)
	hogInfo = DescInfo(nBins=8, isHof=False, nxyCells=2, ntCells=3, size=32)
	hofInfo = DescInfo(nBins=9, isHof=True, nxyCells=2, ntCells=3, size=32)
	mbhInfo = DescInfo(nBins=8, isHof=False, nxyCells=2, ntCells=3, size=32)

	image, prev_grey, grey = None,None,None

	fscales = []
	sizes = []

	# for optical flow
	prev_grey_pyr, grey_pyr, flow_pyr = [],[],[]
	prev_poly_pyr, poly_pyr = [],[]

	xyScaleTracks = []

	



