import os
from commands import getstatusoutput as gso

# dumps dense trajectory features at the video location
def dumpDTF(videoPath):
	outputPath = os.path.splitext(videoPath)[0] + '_features.txt'
	command = './DenseTrack ' + videoPath + ' -I 2 > ' + outputPath
	out = gso(command)

	if out[0]!=0:
		raise Exception("Error while getting dense trajectory features : " + out[1]) 
