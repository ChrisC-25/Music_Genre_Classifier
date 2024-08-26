# **Machine Learning Music Classification by Genre:**

This project processed mp3 audio data from the FMA (Free Music for All) database and used the data to train multiple machine learning models and analyze performance.

Link to Dataset: https://github.com/mdeff/fma?tab=readme-ov-file

## Data Pre-Processing
1. Feature Selection:
    - The librosa package was used to extract features from mp3 audio files. Based on our research, we selected the features that are considered the most influential
2. Create Dataset:
    - /createData.py is the file which was used to extract features from the audio mp3 files. TinyTag was used to get the genre information from the metadata, and librosa to extract the selected audio features. The resulting data is newData.csv
3. Mean Features:
    - /createData.py used numpy to record the mean value of each feature across the song into our csv.
4. Data Cleaning:
    - Not all entries were usable. Many entries had no genre listed, or niche/invalid genres we did not have enough instances to train for.
    - /final_cleaning.py is the file used to clean the data and create finalData.csv
    - /preprocess.py reads finalData.csv then assigns valid genres an integer value and prepares the features and labels set as well as splitting the data for training, testing, and validation
    - /preprocess.py also uses sklearn to scale the features
5. Principal Component Analysis:
    - /preprocess.py also performs PCA on our data up to the top 10 features (Testing showed that the 10 most impactful features accounted for nearly 100% of the variance)
    - This reduces our dimensionality from 30 features to 10

## Training Models
 - ### Neural Networks
     - /nn.py implements our Neural Network
 - ### Convolutional Neural Networks
     - /cnn.py implements our Convolutional Neural Network
 - ### Random Forest
     - /random_forest.py implements our Random Forest

## Analysis
 - /Images/ contains the images used in our analysis
 - /compare.py contains the code used to visualize and compare the performance of our models
