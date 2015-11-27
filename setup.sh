sudo add-apt-repository -y ppa:mc3man/trusty-media
sudo apt-get -y update
sudo apt-get -y install ffmpeg libavdevice-dev libopencv-dev python-opencv

cd dense_trajectory_release_v1.2/
make clean
make
cp release/DenseTrack ../
cd ..
