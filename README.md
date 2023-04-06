# PyTorch Merge

This repository contains a script, **py_merge.py**, that can be used to merge two PyTorch model .bin files into a single model file. This can be useful when you need to combine the weights of two models that have the same architecture and are compatible. The script averages the parameter values of the models for keys that exist in both models.

## Prerequisites
Before using this script, make sure you have the following Python packages installed:

* PyTorch
* Transformers

You can install them using pip:

```
pip install torch
pip install transformers
```

## Usage

Clone this repository:

```
git clone https://github.com/donaldafeith/Pytorch_Merge.git
cd Pytorch_Merge

```

Run the **py_merge.py** script:

```
python py_merge.py
```

The script will prompt you for the paths of the following files:
* config.json: The configuration file for the model architecture you are working with.
* model1.bin: The first model .bin file you want to merge.
* model2.bin: The second model .bin file you want to merge.

For example:

```
Enter the path to the config.json file: /path/to/config.json
Enter the path to the first model .bin file (model1.bin): /path/to/model1.bin
Enter the path to the second model .bin file (model2.bin): /path/to/model2.bin
```
After providing the paths, the script will merge the models and save the result as **pytorch_model.bin** in the current directory.

```
Merged model saved as pytorch_model.bin
```
You can now use the merged **pytorch_model.bin** file with your model architecture.

**Note**: Merging models may not always produce the desired results, especially if the models have different architectures or were trained on different data. 

Use this script only when you are sure that the models are compatible.
