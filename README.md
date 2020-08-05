# FelixbleNN
A utility to use a series of different convolutional blocks.

[![License](https://img.shields.io/github/license/KiLJ4EdeN/FelixbleNN)](https://img.shields.io/github/license/KiLJ4EdeN/FelixbleNN) [![Version](https://img.shields.io/github/v/tag/KiLJ4EdeN/FelixbleNN)](https://img.shields.io/github/v/tag/KiLJ4EdeN/FelixbleNN) [![Code size](https://img.shields.io/github/languages/code-size/KiLJ4EdeN/FelixbleNN)](https://img.shields.io/github/languages/code-size/KiLJ4EdeN/FelixbleNN) [![Repo size](https://img.shields.io/github/repo-size/KiLJ4EdeN/FelixbleNN)](https://img.shields.io/github/repo-size/KiLJ4EdeN/FelixbleNN) [![Issue open](https://img.shields.io/github/issues/KiLJ4EdeN/FelixbleNN)](https://img.shields.io/github/issues/KiLJ4EdeN/FelixbleNN)
![Issue closed](https://img.shields.io/github/issues-closed/KiLJ4EdeN/FelixbleNN)


# Usage:
## Install Dependencies w pip:

1- tensorflow


## Import the Model Class and Create an Instance with Desired Settings.
```python
from FlexibleNN import FlexibleNN
flexnn = FlexibleNN(input_shape=(112, 112, 3), width=3, conv_blocks=3,
                    dense_blocks=2, classes=1,
                    filter_start=16, filter_size=(2, 2), activation='relu',
                    dense_units=512, use_bn=True,
                    use_dropout=False, dropout_rate=0.5,
                    use_constraint=False, constraint_rate=1,
                    use_pool=True, pool_size=(2, 2),
                    )
```


## Build the Model. Custom Configs Can be Achieved by Editing the build_model Function.
```python
model = flexnn.build_model()
print(model.summary())
tf.keras.utils.plot_model(model)
```
