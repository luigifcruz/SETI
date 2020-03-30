# DeepSwitch

Notion Page: https://www.notion.so/luigifcruz/DeepSwitch-70c38e69dd324dcd987fcd01648e11ba<br>
Kaggle URL: https://www.kaggle.com/tentotheminus9/seti-data<br>

# Introduction

This week, I had a lot of time in my hands considering this whole COVID-19 situation. To make good use of this precious time, I decided to play with a dataset I found on [Kaggle](https://www.kaggle.com/tentotheminus9/seti-data). The dataset is composed of different kinds of radio signals (e.g. bursts, narrowband, or squiggle) compiled by the SETI Project. For those who don't know this awesome project yet. It's a non-profit organization in search of intelligent extraterrestrial communications in the Universe. Yes, trying to find smart ETs!

## Goals

- Classify different types of signals with a high degree of certainty.
- Precision is important but speed is equally as important. Alf wants to go home!
- Compare the speed and precision of this method with state-of-art models.
- Don't apply any type of dataset preprocessing using costly filters (e.g. Sobel).

# Dataset

The dataset used by this experiment was downloaded from a Kaggle [Notebook](https://www.kaggle.com/tentotheminus9/seti-data) created by Rob Harrand. To the best of my knowledge, this was one of four datasets created for the ML4SETI competition in 2017. It is composed of 7000 samples distributed equally between seven classes representing different kinds of signals. The dataset was randomly divided into three parts: Training (80%), Validation (10%), and Testing (10%).

[Classes](https://www.notion.so/0ba2c70680d94bf8a781a34ea5c6910b)

# Configuration

The Conda environment used by this project is available [here](https://github.com/luigifreitas/seti/blob/master/DeepSwitch/environment.yml).

This section is relevant as a reference for the experiment numbers given below. Expect them to rise as more computing is available on the host computer. My old NVidia 1070 Ti GPU is quite slow compared to new hardware. In particular, the new NVidia RTX series with improved half-precision performance with the Tensor Cores.

Going against most Open Source projects published by SETI, I chose to use PyTorch instead of Tensorflow. I think this is a better framework for model prototyping. If the need arises, the network can be ported quite easily to Tensorflow. For example, to use on TPUs...

### Hardware

- CPU: Intel i7-7700K 4.2 GHz.
- RAM: 24 GB of DDR4 at 2666 MHz.
- GPU: NVidia GTX 1070 Ti w/ 8GB DDR5.
- Disk: 2TB Seagate HDD 7200 RPM.

### Software

- Python 3.8.1
- PyTorch 1.4.0
- NumPy 1.18.1

# Base Model

To streamline the testing, I started with a simple Convolutional Neural Network vaguely based on the VGG-16 model. The network described on the right will be used by all experiments.

Each test can modify the parameters of any layer (number of convolutional channels, size of the dense layers, or number of consecutive convolutions) to achieve the best outcome. To further simplify testing, Residual Network designs will be ignored for now. I plan to make some experimentation with them later.

- Convolutional Block
    - Convolutional Layer
    - Batch Normalization
    - ReLU
    - Maximum Pooling
- Dense Layer
- Dropout (50%)
- Dense Layer
- Dropout (50%)
- Logarithmic Softmax

# Experimentation

In this section, we will explore techniques to increase precision while maintaining an acceptable rate of performance. Keep in mind that the conclusions reached by these experiments are highly reliant on the dataset. Therefore, they won't be necessarily valid for other types of workloads (e.g. Dogs vs. Cats, Flowers Species, or [Not Hotdog](https://www.youtube.com/watch?v=pqTntG1RXSY)).

Every model used by these experiments will be trained until exhaustion. Since my resources are limited, only a single loss function was evaluated (CrossEntropyLoss). Adam is the learning rate optimizer set with an initial value of 1e-5. This value will be decreased by a factor of ten every time the Validation Loss doesn't improve after ten epochs. The training process will be automatically terminated after the network reaches the learning rate threshold of 1e-7. 

The dataset preprocessing will be handled by the `DataLoader` method configured to use up to eight workers in parallel. This will mostly ensure a CPU bound bottleneck impacting the GPU performance won't happen. The training batch size is set to eight.

## Sample dimensions impact precision?

Let's start with a model composed of four consecutive Convolutional Blocks with a maximum depth of 256 layers. The only difference between them is the input sample dimension. The resize is being done by the preprocessor block.

The idea of this experiment is to establish a point when the level of detail — here represented by the input sample size — will meaningfully impact the precision. This is an essential step to improve performance.

By looking at the results, we can see a clear curve of diminishing returns. By dividing the size by two, we get the best precision with the slowest performance. In contrast, if we divide it by four, the precision deteriorates by 1.6%, but the speed rises 3.25 times! Therefore, we can get a substantial speed improvement by sacrificing a little bit of accuracy here.

[Results Table](https://www.notion.so/f7429f0ee4704092939d0ac51acecd8a)

## More or fewer convolutions?

We all know that Convolutional Layers are a very important building block for CNNs. It's important to perfectly balance the number of convolutions and depth to produce a good result. In this test, we are going to verify which configuration works best for this type of frequency-domain signal.

By comparing the results of this test with the best model from the last experiment, we can observe a small improvement of ~1% achieved by doubling the depth of the network. Naturally, the speed is half as fast as before, rendering this option a bad deal. In contrast, the precision deteriorates by more than 9% if the network depth is reduced four times. This time, the speed increased by five times. Finally, a compromise between speed and precision was reached by adding two convolutional blocks on top of the original model. This option (V18) archived a ~3% lower accuracy with a 4.25x speed improvement.

[Results Table](https://www.notion.so/29908d3de8874ea1b2419e64b474a9ac)

## Is the dataset the bottleneck?

The speed of this particular test was negatively impacted by CPU usage. The Folding@Home client was processing a CPU based WU at this time. #StayTheFuckHome

The bottleneck of a good Neural Network is always the dataset! Here, we explore data augmentation techniques to improve the precision of this model. After evaluating multiple approaches, I ended up applying the following augmentations.

- **Random Horizontal and Vertical Flip.** This simple change improved the validation score by ~11% while maintaining the speed.
- **Plus or minus ten degrees of random rotation.** This can't be much more because the rectangular shape of the sample could cause complete occlusion of the signal. This technique improved the previous accuracy by ~1.5%.

**Do greyscale samples reduce the computation needed?** Nope, all the color channels will be flattened by the first convolutional layer anyway. This won't be ideal because we'll lose data encoded within the waterfall colorscheme. The performance hit is seen in the 

[Results Table](https://www.notion.so/4726f87670254f8d85d68cc8b97ad211)

## More convolutional layers?

The number of convolutional layers inside each convolutional block is important. Here, we see a sizable 2% precision improvement when we add a new layer inside every block. This change comes at a cost of ~30% lower speed. To achieve an accuracy gain without the performance loss, a technique seen in models like VGG-16 is applied. In this approach, the initial layers of the model have a lower number of layers. This is exactly where the convolution is computationally intensive. This hybrid approach has the same precision while being faster than the original architecture.

[Results Table](https://www.notion.so/dc867c43a5a5418a98e47b0d9732444e)

# The Final Model

Taking into consideration everything I learned in the experimentation above I reached the final network iteration (V43). **It achieved a test accuracy of 94.4% while being able to perform inference in up to 2204 images per second.** This is one order of magnitude faster than VGG-16 or two times faster than RedNet18! The complete comparison of my network with state-of-art ImageNet networks can be found in the table below. Since this is a CPU task, the inference speeds listed bellow aren't counting the time it takes to load and preprocess the batch of samples.

[How does it compare with other models?](https://www.notion.so/ea534efbaec847b4bd567604321b2390)

In the confusion matrix of the final DeepSwitch model displayed below, we can observe that all classes are behaving as expected without any major bias towards a category. Just a slightly higher error rate is recognizable between Brightpixel and Noise category. Given their resemblance, the cause of this behavior is clear.

[Confusion Matrix](https://www.notion.so/luigifcruz/DeepSwitch-70c38e69dd324dcd987fcd01648e11ba#1b92562f24ec4551a262f4b905ecd19d)

As future work, I pretend to explore a few key points:

- Experiment with different Loss Functions other than Cross Entropy.
- Test different combinations of optimizer functions.
- Apply the transfer-learning technique to DeepSwitch.
- Use [setigen](https://github.com/bbrzycki/setigen) to generate a bigger dataset with square samples.
- Run the benchmark on faster and newer GPUs.
