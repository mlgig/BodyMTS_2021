# Fast and Robust Video-Based Exercise Classification via Body Pose Tracking and Scalable Multivariate Time Series Classifiers

This repository contains code and data for a novel proposed approach BodyMTS for human exercise performance 
classification from videos. BodyMTS uses Human Pose Estimation coupled with Multivariate Time Series Classification 
methods. The main objective is to classify exercise performance in terms of differentiating between correct
and different aberrant executions of the exercise. Incorrect execution may lead to muscoskeletal injuries and impede 
performance, therefore, automated and accurate feedback on execution is important to avoid injuries
and maximize the performance of the user. BodyMTs is a first step towards building a fully automated pipeline to 
classify the execution of physical exercises using video capture and light computation models suitable for resource 
constrained devices such as mobile phones. We evaluate BodyMTS using a real world dataset of Military Press where 53 
participants participated in the study.

![Alt text](figs/overview.jpg?raw=true)
<em>**Fig 1** Overview of the BodyMTS approach for the Military Press strength and conditioning exercise classification.
Left-to-right flow: raw video,  extracting and tracking body parts using human pose estimation, preparing the resulting
data for time series classification and interpretation.</em>

## Data Description
The data used is the video recordings of the execution of the Military Press (MP) exercise. Participants completed 10 
repetitions of the normal form and 10 repetitions with induced deviations. Time series data is obtained by repeatedly 
applying OpenPose over all the frames. The data shared in this repository here is already pre-processed i.e. 
the raw data obtained after the OpenPose step is segmented, resampled and split into training/test data. Each sample in
training/test data corresponds to one repetition of Military Press of a type. Please [email](mailto:ashish.singh@ucdconnect.ie) if you need access to the 
original videos.

The folder TrainTestData consists of data in the numpy format whereas the folder ```data/sktime_format``` contains data in the 
[sktime](https://www.sktime.org/en/latest/) format. Each sample in the data is multivariate in nature of shape 161*8,
where 161 is the length of each sample (after re-sampling) and 8 are the dimensions used (left and right of wrists
,elbows, shoulders and hips). The data is already split in the ratio of 70/30 for training and testing. There are 3
different splits for 3 different seed values 103007, 1899797 and 191099. The file ```rocket_config``` is used to pass the 
arguments for input and output paths.

### Installation
Please use the requirements.txt file to install all the dependencies. There is a configuration script for the 
classifier script which contains the relative paths to the exercise and data folders.

```python
# python rocket.py --rocket_config path_to_knn_config 
```

## Results with the latest version of OpenPose
Classifier Name | Accuracy OpenPose (v1.7) | Accuracy OpenPose (v1.4)
--------------- | -----------------------------| ---------------
FCN | 0.78 | 0.72 
ResNEt | 0.77 | 0.73
ROCKET | **0.85** | **0.81**
MINIROCKET | 0.76 | 0.75

<em>Table 1: Average accuracy on test data over 3 splits for multivariate time series classifierstrained with time series 
extracted with OpenPose version 1.4 and version 1.7.</em>


## BodyMTS Accuracy for different quality of videos
Classifier Name | Total size of videos (MB) | BodyMTS Accuracy
--------------- | -----------------------------| ---------------
Default (23) | 213 | 0.83
16 | 398 | 0.83
22 | 208 | 0.83
28 | 76 | 0.82
34 | 34 | 0.75
40 | 20 | 0.69

<em>Table 2: Accuracy of BodyMTS on test data over a single train/test split for different values of CRF. At CRF 28 we save 70% of 
data storage and maintain the same accuracy.</em>

## Comparison with SOTA deep learning classifiers
Classifier Name | Accuracy | Total execution time (mins) | Inference Time per clip (secs)
--------------- | -----------------------------| --------------- | ----------------
C2D | 0.70 | 20 | 0.40
I3D | 0.81 | 27 | 0.58
SlowFast | 0.84 | 100 | 1.69
SlowFast + NL | 0.83 | 80 | 1.29
X3D-M | 0.80 | 125 | 1.19
**BodyMTS** | 0.83 | 38 | 0.06

<em>Table 3: Accuracy, total execution time and inference time (per testing clip) for different
architectures over a single train/test split. Note: all deep learning models are pre-trained on
Kinetics-400.</em>


## Citation
Please cite this paper if it helped in your research:
```
@article{singh2021interpretable,
  title={Interpretable Classification of Human Exercise Videos through Pose Estimation and Multivariate Time Series Analysis},
  author={Singh, Ashish and Le, Binh Thanh and Le Nguyen, Thach and Whelan, Darragh and O’Reilly, Martin and Caulfield, Brian and Ifrim, Georgiana},
  year={2021},
  booktitle = {5th International Workshop on Health Intelligence(W3PHIAI-21) at AAAI21},
  publisher={Springer}
}
```

