
# 🟠 Mars Exploration using Machine Learning ✨

This repository represents a web app with a Binary classification ML model which creates a segmented images of plain land.





## 📄 Description
* This project is developed ML model that segments plain land from input image on martian surface.

* Implementation is based on the [U-Net architecture](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png) which creates a segmented image from raw image as an input.

## 🗺️ Exploration legend : 
* 🟡 Yellow area - Danger 💀
* 🟣 Purple area - Safe 😀
## 📁 Dataset 

* [ ai4mars terrainaware autonomous driving on mars](https://www.kaggle.com/datasets/yash92328/ai4mars-terrainaware-autonomous-driving-on-mars) from Kaggle.


## 🛠 Installation

### Requirements
- Python                    3.10.9 
- Tensorflow                   2.12.3
- PIL               0.13.1 
- OpenCV 4.6.0  
- streamlit                    1.28.0 
- segmentation-models   1.0.1
- tensorboard  2.12.3

Rest of the packages are listed in packages txt file.
## 👁Download the ML Model 
- Download the "MarsSegmentationModel_20230308-131105.h5" file from following Drive [Link](https://drive.google.com/file/d/1j1dx9Rt3uuKMQNy4CobIwzwUZVnKevQk/view?usp=sharing).



## 🖥 Deployment
- Install the dependencies locally.

- To deploy this project:

```bash
  streamlit run app.py
```

- It will launch the webapp, then follow below steps :

  1. Click on Choose File.
  2. Upload any file from Input samples eg IMG1.png.
  3. Enjoy the results.


## 🧠 Hyperparameters

| Hyperparameters             | Values                                                              |
| ----------------- | ------------------------------------------------------------------ |
| Epoch  | 20  |
| Batch Size | 16|
| Learning Rate | 1e-3|
| Optimizer | Adam |
| Scheduler | ReduceLROnPlateau |
| Accuracy | IoU Score|
| Loss Function | Jaccard Loss|






## 📷 Screenshot


## 😇 Feedback

If you have any feedback, please reach out to us at coder.shrirang.kanade@gmail.com

