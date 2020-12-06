# README

### Contents:

* `SISO.ipynb` and `Classification2D.ipynb`  visualize the training and output of a neural network, the first for single-input-single-output data, and the second for binary classification of 2D data.
* `torch_modules.ipynb` review basic and necessary features of PyTorch that help you get started in learning about it.
* `Gradients.ipynb` demonstrates how gradients of functions with respect to many variables can be automatically computed using PyTorch.
* `neural_nets.py` is needed to run `SISO.ipynb` and `Classification2D.ipynb`. See note below.



### Necessary modification on your behalf:

Before using `SISO.ipynb` or `Classification2D.ipynb`, make sure to change the content in the first cell of each notebook.

Currently, the first cell of code looks like this:

```python
import sys
sys.path.insert(0, r'C:\Users\Rayan El Helou\Documents\Projects\ML demos\PyTorch')
```

You need to replace my folder's location **such that the new one points to the file** `neural_nets.py`.