# Fast and Robust Video-Based Exercise Classification via Body Pose Tracking and Scalable Multivariate Time Series Classifiers

This repository contains code and data for a novel proposed approach BodyMTS for human exercise performance 
classification from videos. BodyMTS uses Human Pose Estimation coupled with Multivariate Time Series Classification 
methods. The main objective is to classify exercise performance in terms of differentiating between correct
and different aberrant executions of the exercise. Incorrect execution may lead to muscoskeletal injuries and impede 
performance, therefore, automated and accurate feedback on execution is important to avoid injuries
and maximize the performance of the user. BodyMTS is a first step towards building a fully automated pipeline to 
classify the execution of physical exercises using video capture and light computation models suitable for resource 
constrained devices such as mobile phones. We evaluate BodyMTS using a real world dataset of Military Press where 53 
participants participated in the study. 

The aim of BodyMTS is to provide a scalable system that can accurately measure and evaluate end-user performance of strength and conditioning
(S&C) exercises, with a view to provide feedback in near real-time. This, in
turn can guide physiotherapists, trainers, and elite and recreational athletes
to perform exercises correctly and therefore minimise injury risk and enhance
performance.We devised the following list of application requirements based
on previous research in which we consulted with end users, clinicians, and
strength and conditioning experts on the design, implementation and evaluation of interactive feedback systems for exercise:
- Be able to accurately monitor the body parts movement, accounting for
the critical body segments involved in the exercise in question
- Detect when deviations from normal movement profile have occurred, and
which kind of deviation has occurred in each case.
– Provide clear and simple feedback to the end user, in near real-time
- Simple data capture based on ubiquitous sensor technology (e.g, single phone)
- Coverage of wide range of S&C or rehabilitation exercises.
Please refer to the paper for more details.

![Alt text](figs/overview.png?raw=true)
<em>**Fig 1**  Overview of the BodyMTS approach for the Military Press strength and conditioning exercise classification. 
Left-to-right flow: raw video, extracting and tracking body parts using human pose estimation, preparing the resulting 
data for multivariate time series classification and interpretation.</em>

## Data Description
The data used is the video recordings of the execution of the Military Press (MP) exercise. Participants completed 10 
repetitions of the normal form and 10 repetitions with induced deviations. Time series data is obtained by repeatedly 
applying OpenPose over all the frames. The data shared in this repository here is already pre-processed i.e. 
the raw data obtained after the OpenPose step is segmented, resampled and split into training/test data. Each sample in
training/test data corresponds to one repetition of Military Press of a type. Please [email](mailto:ashish.singh@ucdconnect.ie) if you need access to the 
original videos.

The folder TrainTestData consists of data in the numpy format whereas the folder ```data/sktime_format``` contains data in the 
[sktime](https://www.sktime.org/en/latest/) format. Each sample in the data is multivariate of shape 161*16,
where 161 is the length of each sample (after re-sampling) and 16 are the dimensions used (x & y coordinates for left and right wrists
,elbows, shoulders and hips). To assess the robustness of BodyMTS we have also created datasets with changing CRF and resolution.
Every file name is in the format of ```TRAIN/TEST_{file_type}_X.ts```, where ```file_type==default``` denotes the data without
any change, a value of crf28 means that the video has been tweaked to have CRF value of 28 followed by the OpenPose to 
generate the final multivariate time series data. The data is already split in the ratio of 70/30 for training and testing. 
We have also included the file containing the pid information which can be used to create further splits. The file 
```rocket_config``` is used to pass the arguments for input and output paths. 

The folder ```change_video_property``` contains the bash scripts to modify the CRF and other properties of videos.

![Alt text](figs/videoto2.png?raw=true)
<em>**Fig 1**  Extraction of time series data from video using OpenPose. Each frame in the video
is considered as a single time point in the resulting time series. Each tracked body part
results in a single time series that captures the movement of that body part. The whole
motion is captured as a multivariate time series with 50 channels, two (X,Y) channels for
each body part tracked (only 8 body parts with Y coordinate shown above). A class label
is associated with each such multivariate time series.</em>

### Installation
Please use the requirements.txt file to install all the dependencies. There is a configuration script for the 
classifier script which contains the relative paths to the exercise and data folders.

```python
# python rocket.py --rocket_config path_to_knn_config 
```

## Results with the latest version of OpenPose
Time Series Classifier | Accuracy OpenPose (v1.7) |
--------------- |--------------------------
FCN | 0.82                     |
ResNEt | 0.76                     | 
ROCKET | **0.87**                 |
MINIROCKET | 0.81                     |

<em>Table 1: Average accuracy on test data over 3 splits for multivariate time series classifiers
trained with time series extracted with OpenPose version 1.7.</em>


## Comparison with SOTA deep learning classifiers
We compare BodyMTS with state-of-the-art methods for human
activity recognition from videos. These methods employ deep learning architectures and have shown good performance on several benchmark datasets such
as UCF101, Kinetics-400 and Kinetics-600. We selected a few methods based
on their performance, execution time and resources required. 

Classifier Name | Accuracy | Total execution time (mins) | Inference Time per clip of 10 reps (minutes)
--------------- |----------|-----------------------------| ----------------
C2D | 0.67     | 42                          | 0.38
I3D | 0.79     | 55                          | 0.53
SlowFast | 0.83     | 81                          | 0.48
SlowFast + NL | 0.83     | 86                          | 0.45
X3D-M | 0.78     | 166                         | 0.68
**BodyMTS(frame-step=1)** | 0.87     | 74                          | 0.36
**BodyMTS(frame-step=3)** | 0.85     | 38                          | 0.20

<em>Table 3: Average accuracy, total execution time and time per testing clip for different architectures over three train/test splits. The average duration of all clips in training and
testing is 65 mins and 30 mins respectively. Note: all deep learning models are pre-trained
on Kinetics-400. 

## Analyzing BodyMTS robustness against different sources of noise
We analyze the robustness of BodyMTS against different sources
of noise that may occur in this application. These sources of noise can be
broadly classified into 3 categories: (1) video data capture; (2) OpenPose parameters; (3) time series data pre-processing.

We alter the video properties such as CRF, bit-rate and resolution to analyze their impact
on the BodyMTS accuracy.
We present results for different CRF values below.
We alter the CRF property in order to modify the bit rate.
CRF ranges from 0-53 and the default value of CRF is 23. We change the value
of CRF with a step size of 6 as suggested in starting from 16.
Resolution remains the same when changing the CRF.

Classifier Name | Total size of videos (MB) | BodyMTS Accuracy
--------------- | -----------------------------| ---------------
Default (23) | 213 | 0.87
16 | 398 | 0.87
22 | 208 | 0.87
28 | 76 | 0.85
34 | 34 | 0.81

<em>Table 2: Average accuracy of BodyMTS on test data over three train/test splits for different values of CRF. 
At CRF 28 we save 70% of data storage and maintain similar accuracy.</em>

Takeaway: Degrading the quality of videos by altering CRF to 28 makes it
possible to satisfy minimum accuracy requirements (e.g. above 80%) as listed
in Table 1 with 70% savings in storage space.

For other types of noise such as OpenPose's parameters and data pre-processing please refer
the paper.

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

