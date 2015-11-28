import os
import Util

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
	for action in actions:
		for video in action.videos:
			Util.dumpDTF(video)

	# making the codebook
	


# assume data set folder in "root"
if __name__ == '__main__':
	root = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + 'UCFSports' + os.path.sep + 'ucf_sports_actions'
	main(root)
