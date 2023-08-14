# PyTorch Merge

This repository contains a script, **py_merge.py**, that can be used to merge two PyTorch model .bin files into a single model file. This can be useful when you need to combine the weights of two models that have the same architecture and are compatible. The script averages the parameter values of the models for keys that exist in both models.

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/R6R8K4WLS)

## Prerequisites
Before using this script, make sure you have the following Python packages installed:

* PyTorch
* Transformers

You can install them using pip:

```
pip install pytorch_merge
```

This will automatically install the dependencies (`torch` and `transformers`).

## Usage

Open a terminal, and type:

```
pytorch_merge --help
```

To get the instructions on how to use it.

This tool requires 3 arguments:

* `--config config.json` -- The configuration file for the model architecture you are working with.
* `--bin model1.bin model2.bin model3.bin` -- All the model’s weights .bin files you want to merge. You can merge weights files of one multiparts model, or weights from different models, in which case weights will be averaged. You can specify as many files as you want, they will be merged one after the others in a loop.
* `--output merged_model.bin` -- Where to save the output merged model.

For example:

```
pytorch_merge -c config.json -b model1.bin model2.bin -o merged_model.bin
```

You can now use the merged **merged_model.bin** file with your model architecture.

**Note**: Merging models may not always produce the desired results, especially if the models have different architectures or were trained on different data. 

Use this script only when you are sure that the models are compatible.

## License

This tool was made by Donalda Feith and is licensed under GNU General Public License v3 or later (GPLv3+).
