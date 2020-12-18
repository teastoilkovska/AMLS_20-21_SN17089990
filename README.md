# AMSL

This projects contains CNN models for gender classification, smile detection, eye-color and face-shape detection. 

The models have been been pre-trained and saved in the corresponding file.

I have also included a main.py file, which loads the saved models and computes the scores on the test datasets.

Note that the main.py imports the datasets from the following files from the Dataset folder: import_test_celeba.py, import_test_cartoon_color.py and import_test_cartoon_gray.py. 
This was done for a clearer code structure. They also save the datasets as .npy, so there is no need to re-load them everytime.
  The corresponding paths should be set correctly in these files before running main.py. The paths correspond to the following foldernames: cartoon_set_test 
and celeba_test. 

In the same way, if the models are to be trained again for different datasets, please check the paths in the files: import_cartoon.py, import_cartoon_color.py,
and import_dataset_celeb.py
