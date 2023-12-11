# yolo3_tx2
light weight yolo3 tested on tx2

1. config python as python 3.6.9
sudo update-alternatives --config python3

2. install pyrealsense:
need to build from source(refer to the official website or other install txt)

3. install opencv 4.5.0:
need to build from source
cd ~/Downloads/opencv-4.5.0/release
make


4. gpu usage monitor
nvcc can't be installed, in order to check gpu usage, use this script:
cd ~/Downloads/goodyolo
python gpu_monitor.py


5.install torch: torch is pre-installed, better not modify


---------------------------------------------------------------------------
YOLOV3
6.yolov3 code:
cd ~/Downloads/goodyolo
read the readme.txt in foler
weights and config can be changed in subfolders
python main_debug.py

6.5 install pyntcloud for xyz resolve
pip install pyntcloud


6.6 yolov3 example
weights and config can be changed in subfolders
package installation
https://pypi.org/project/yolo34py/

