# CST Tools
This repository replaces the gists [cst_parametric_multiASCIItables.py](https://gist.github.com/scimax/fd368299408bf99c359ea5bbf693865a) and [cst_decompose_multiASCIItables.py](https://gist.github.com/scimax/762e4da187a9a38f3f69604d560ec2fa).

The functions are initially drafted in the notebook *CSTtools.ipynb* in the *dev* directory. The functions are copied over to the package.

For the moment, to use the package it has to be added manually to the current script:

```py
import sys
sys.path.append(r"/path/to/package")
from csttools.csttools import read_CST_ASCII_format
```