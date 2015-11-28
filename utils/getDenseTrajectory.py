import os
from commands import getstatusoutput as gso

def getDenseTrajectory(videoPath):
	outputPath = os.path.splitext(videoPath)[0] + '_features.txt'
	command = './../DenseTrack ' + videoPath + ' > ' + outputPath
	out = gso(command)

	if out[0]!=0:
		print "Error while getting dense trajectory features : " + out[1]
