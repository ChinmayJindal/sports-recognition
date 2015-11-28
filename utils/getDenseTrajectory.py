import os
from commands import getstatusoutput as gso

def getDenseTrajectory(videoPath, outputPath):
	command = './../DenseTrack ' + videoPath + ' > ' + outputPath
	out = gso(command)

	if out[0]!=0:
		print "Error while getting dense trajectory features : " + out[1]

getDenseTrajectory('../data/UCFSports/ucf_sports_actions/Diving-Side/001/2538-5_70133.avi', 'testfeatures.txt')
