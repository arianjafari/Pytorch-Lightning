Overview:

The dataset consists of virtual reality (VR) videos captured from the simulator in da Vinci robotic system. There are two main folders within this dataset:

	- videos
	- annotations

Descriptions of contents of each folder are given below.

Videos:

This folder consists of further two folders:

	- fps1
	- fps30

As the name of the folder suggests, both folders consists of videos at different frame rates i.e 1fps and 30fps. The videos within these folders are identical and the only difference is the frame rate. For example, fps1/caseid_000001_fp1.mp4 is from the same original video as fps30/caseid_000001.mp4.


Annotations:

There are two types of labels that will come with the dataset:

	- Bounding boxes for the surgical tool clevis and needle (for category 1 of the challenge)
	- Objective skill based metrics (for category 2 of the challenge)

The bounding box annotations are within the "bounding_box_gt" folder. Each file within this folder contains bounding box ground truth labels corresponding to the 1fps videos available in videos/fps1 folder. In each file, the definition of the keys are:
	- obj_class: object name (needle driver or needle)
	- label_type: all labels are box
	- coordinate: coordinate of the bounding box (height, width, x coordinate of top left point, y coordinate of top left point)
	- orientation: property of the object (left or right needle drivers, needle is being grabbed or not)
	- objects: True when there are objects in view; False when there are no objects in view
	- case_id: case id of the video
	- fps: frames per second of the annotated video


The objective skill based metrics are provided in the "skill_metric_gt.csv" file. This csv consists of 3 metric labels for each video, which are needle drop counts, instrument out of view counts and economy of motion. Please check out the website for further details on descriptions of these metrics. Please note that not all metrics are available for all the videos, however most videos will have at least 2 objective skill metrics available as labels. 


Other info:
For additional info, check out the website https://www.synapse.org/simsurgskill2021 or feel free to email at aneeq.zia@intusurg.com
