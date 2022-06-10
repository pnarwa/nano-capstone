## The Amazon Rainforest : Tracking Deforestation

### Table of Contents

1. [Objectives](#objectives)
2. [Installation](#installation)
3. [File Descriptions](#files)
4. [Execution Instructions](#exec)
5. [Results](#results)
6. [Acknowledgements](#ack)

## Objectives<a name="objectives"></a>
<b>The Amazon</b> is the world’s largest rainforest. It absorbs more greenhouse gases than any other tropical forest. The forest helps stabilize climate, provides home to a multitude of plants and wildlife species, helps maintain the water cycle, protects against flood, drought, and erosions.<BR><BR>
But in the recent years deforestation activities in the forest have accelerated. This has contributed to reduced biodiversity, habitat loss, climate change, and other devastating effects.<BR><BR>
Data about the location of deforestation and human encroachment of forests can help governments and organizations implement effective solutions towards regaining the ecological balance.<BR><BR>
In this project we analyze satellite imagery of the Amazon basin to track human footprint. We use convolutional neural networks(CNNs) to classify the images into categories of atmospheric conditions and land cover/land usage classes. We explore techniques to improve the model's performance and also illustrate how to visualize the CNN activation maps.<BR><BR>
The image data used is from the the [Kaggle site](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data) 

## Installation <a name="installation"></a>
The code has been developed using Python version 3.7.12<BR> 
The main libraries used are tensorflow, keras, pandas, numpy, sklearn, matplotlib.<BR>
The version of tensorflow used in the implementation : 2.8.0<BR>
The version of keras used in the implementation : 2.8.0 (The keras API is integrated with tensorflow)<BR>
The version of sklearn  used in the implementation : 1.0.2<BR>

The tqdm library is used to track the progress during model training using a progress or meter bar.<BR>
Command to install the tqdm library:
- pip install tqdm

The link below has further details on the installation of the tqdm library
- https://pypi.org/project/tqdm/ 

## File Descriptions<a name="files"></a>
The directories and files are organized as depicted below:<BR>
- \notebooks : notebooks used in data processing and modeling phases<BR>
      - AmazonForest-data-process.ipynb : notebook containing the code related to data pre-processing<BR>
      - AmazonForest-model-train-V1.ipynb : notebook containing the code in building the first version of the model<BR>
      - AmazonForest-model-train-V2.ipynb.py : notebook containing the code in building the final version of the model<BR>
      - AmazonForest-visualize-activations.ipynb : notebook containing the code to visualize the CNN activation maps<BR>
- \utils : common utility functions used across the notebooks<BR>
      - data_utils.py : script containing common functions used in processing the data<BR>
      - model_utils.py : script with common functions used in model training and classification<BR>
      - plot_utils.py : script with common functions used in plotting the graphs and charts<BR>
- \data : data used in the analysis, classification<BR>
      - train-jpg/train_\*.jpg : folder containing the satellite images in jpg format used in training and classification<BR>
      - train_v2.csv : csv containing the labels associated with the images<BR>
      - train_labels.csv : csv generated during the data pre-processing phase, this contains the target dataframe<BR>
      - amazon_forest_part\*npz : npz files generated during the data pre-processing phase, these contain the image data converted to numpy array format.

Please note that due to file size limits the complete set of jpg images has not been uploaded to github. As a reference only 1000 images have been uploaded. To download the complete set, please see the [Kaggle site](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data) 


## Execution Instructions <a name="exec"></a>
There are no special instructions except the order in which the notebooks are laid out:<BR>
- AmazonForest-data-process.ipynb : This is the first notebook in the series which performs the data pre-processing<BR>
- AmazonForest-model-train-V1.ipynb : The second notebook in the series which builds the first version of the CNN Model from the pre-procesed image data<BR>
- AmazonForest-model-train-V2.ipynb.py : The third notebook in the series which builds the final version of the CNN model<BR>
- AmazonForest-visualize-activations.ipynb : The final notebook in the series which visualizes the activation maps using the model built in the V2 notebook


## Results<a name="results"></a>
The link to the GitHub repository is available [here](https://github.com/pnarwa/nano-capstone)<br />
The links to the blog posts related to the implementation are available at:
- [The Magnificent Rainforests](https://medium.com/@pnarwa/the-magnificent-rainforests-331d986f2eee)
- [Tracking Deforestation with CNNs](https://medium.com/@pnarwa/tracking-deforestation-with-cnns-afc9c97e8cb2)

## Acknowledgements<a name="ack"></a>
- The image data used in the project is from the [Kaggle site](https://www.kaggle.com/competitions/planet-understanding-the-amazon-from-space/data)  
- The precision, recall and fbeta_score functions implemented by the Keras team at [Keras GitHub](https://github.com/keras-team/keras/blob/4fa7e5d454dd4f3f33f1d756a2a8659f2e789141/keras/metrics.py#L134)
- Mongabay Report on [Why are Rainforests important](https://rainforests.mongabay.com/kids/elementary/401.html#content)
- Mongabay Report on [Consequences of Deforestation](https://rainforests.mongabay.com/09-consequences-of-deforestation.html)
