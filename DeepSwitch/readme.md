# DeepColor

This is the development repository of the DeepCloud project.

### First Model
Model trained with three days worth of data. No data augumentation was applied while training.

### Second Model 
Model trained with four days worth of data. Trained from weights of the 1st Model. Data augumentation was applied to improve the color of the clouds and terrain.
- Random rotation (0˚ to 180˚). 
- Random Brightness (0 to 40).

### Third Model
Model trained with four days worth of data. Trained from weights of the 1st Model. Improved data augumentation was applied:
- Random rotation (0˚ to 180˚).
- Histogram Equalization (with pre-equalization for C02) (0 when <= 11).
- Random Brightness (-40 to 0).

### Forth Model
Model trained with four days worth of data. Trained from clean weights. Data augumentation was applied:
- Random rotation (0˚ to 180˚).

All images where preprocessed by the following pipeline:
1. Earth Mask applied to zero-out the background.
2. All pixels with 1 as value was changed to 0.
3. Histogram Equalization on C02, C07 and C13.

### Fifth Model
Model trained with four days worth of data. Trained from clean weights. Data augumentation was applied:
- Random rotation (0˚ to 180˚).

All images where preprocessed by the following pipeline:
1. Earth Mask applied to zero-out the background.
2. All pixels with 1 as value was changed to 0.
3. Histogram Equalization on C02, C07, C13 and C99.

### Sixth Model
Model trained with four days worth of data. Trained from clean weights. No data augumentation was applied.

All images where preprocessed by the following pipeline:
1. Earth Mask applied to zero-out the background.
2. All pixels with 1 as value was changed to 0.
3. Histogram Equalization on C02, C07, C13 and C99.

### Seventh Model
Model trained with a fresh dataset. This dataset was generated using the automatic pipeline. It downloaded four days worth of data with three days of separation starting at 2019-08-01. Trained from clean weights. No data augumentation was applied in Model A and random blur was applied to Model B (0 to 0.5). This model also fixes the following problems encoutered with previous methods:
- Fulldisks generated with lossless PNG format instead of JPEG.
- Histogram Equalization being performed in the correct way for the True-Color Images.
- Better Earth Mask based in a ellipsoid (WGS84).
- Generation process is faster and requires less disk space.
- The validation function now works as expected.

All images where preprocessed by the following pipeline:
1. Earth Mask applied to zero-out the background.
2. All pixels with 1 as value was changed to 0.
3. Histogram Equalization on C02, C07, C13 and C99.

#### Results
Both models performed almost equally. The model with data augumentation performed better with the HRIT data.

##### Model A
- Test: 0.00013464526273310184
- HRIT: 0.0005145686848075403
##### Model B
- Test: 0.00014186200041876873
- HRIT: 0.0004808394322240287

### Eighth Model
Model used the same dataset from the last model. New segmentation layer called landmask was used to better separate the land and sea features. The model without data augumentation was dropped.

Epochs:
- 30: 0.0001947123782883864, 0.000355718423687743
- 40: 0.0001748316308294306, 0.000330278318228117
- 50: 0.0002060480783256935, 0.000390144443800262

### Ninth Model
Same model as 8th but with better training function and early stopping on validation loss.
- Test: 0.000154554832079156
- HRIT: 0.000349438060605761
- Val:  0.000287075427477248

### Tenth Model
Model trained with the dataset from the 7th model suplemented with aditional four days of new data from 2019-04-01.
Data augumentation and the landmask were improved:
- Gausian Blur (0 to 0.65).
- SaltAndPepper (0.00025).

#### Results
- Test: 0.00015077425008972
- HRIT: 0.00027727436014067
- Val:  0.00022006245853845

### v0.11 and v0.12 Model
Model trained with a fresh dataset with 12 days of data collected with an interval of 30 days each. All the samples were collected during the year of 2018. Other improvements include:
- Faster dataset pre-processing functions.
- Support for variable resolutions.

### v0.14 Model
Same structure from the last model with changes to the early stopping algorithm. Now it takes into consideration the accuracy of the prediction instead of the validation loss.