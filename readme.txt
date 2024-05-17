# SPECIALIZE LANGUAGE MODEL (SLM)

## Overview

SPECIALIZED LANGUAGE MODEL (SLM) is  a machine learning project that aims at assisting in the effort to
reduce the cost of inference and  training without sacrificing too much on the larger model's accuracy,
performance and  speed. In fact, it is aiming at  making the larger model in entirety capable of highly 
contextual reasoning through its coherent responses. The details of the larger model called 'SwarmLLM', 
to which it is based on a patent pending  architectural concept we called the '3 MODELS SOLUTION', that 
is our unique design to handle a federated learning through highly distributed network both on software
and hardware aspects. The details are  covered in its own README documentations, and our whitepaper, so 
this SLM README, we will focus on its function and aim.

## SLM is a 63,037,440 parameters model

# FUNCTION OF SLM

SLM is a multimodal language model designed to 'receive' outputs from the larger model as inputs and it
subsequently will inference based on its domain specific datasets to add on more details into the data,
before passing on to generate highly accurate, with contextual and coherent manner to the users.

## SO SLM IS A COMPONENT OF A MUCH LARGER ARCHITECTURE.

# Installation

To install SLM, follow these steps:

1. Clone the repository: `git clone https://github.com/decentrefy/slm.git`
2. Navigate to the project directory: `cd slm`
3. Install the required packages: `pip install -r requirements.txt`

# Dependencies
Please scrutinize our codes carefully as you need to pip install libraries like:
torch, boto, panda, torch optim, pickle, sklearn, IO, etc before you can run these codes and also your
python interpreters are correctly installed.

## Usage

To use SLM, follow these steps:

1. [Provide step-by-step instructions on how to use your project. Include code snippets where necessary.]


## Configuration

SLM uses a JSON configuration file to set various parameters. Here's the configuration:

```json
{
    "main": {
        "batch_size": 32,
        "learning_rate": 0.001,
        "num_epochs": 100,
        "patience": 5
    },

    "SLM": {
        "api_endpoint": "https://API_ENDPOINT",
        "training_dataset_filename": "dataset.csv"
    },

    "SLMOtak": {
        "input_dim": 512,
        "output_dim": 512,
        "num_heads": 8,
        "num_layers": 6,
        "dropout": 0.1
    },

    "SLMTrainLoad": {
        "early_stopping": {
            "early_stopping_patience": 7,
            "early_stopping_verbose": false,
            "early_stopping_monitor": "val_loss",
            "early_stopping_mode": "min"
        }
    } 
}

# Training

Please note that because SLM is originally designed to be a component within a larger ecosystem, it lacks
full preprocessing  scripts. Therefore, IF you planned to clone and use this for other purposes, you need
to handle the preprocessing part yourself.

Because  it  is  originally designed to take in multimodality from the larger architecture ecosystem, you 
have to use various sources to train with your specific area of  expertise. IF your domain specific field
does not include any modality, you can just train it with the datasets  modality that you need and ignore
the ones you don't need.

However,  just  to  give  you a very rough indication of how much data storage is required to train a 63+ 
million parameters model:

##1. Data Type Distribution: 
     Analyze  the  distribution  of data  types in your training data. How much text is there compared to 
     images, videos, and audio?

##2. Estimate Data Size by Type: 
     Research or estimate the average size of each data type:
     - Average text length (in characters or words)
     - Average image resolution (megapixels) and number of images
     - Average video length (minutes or seconds) and number of videos
     - Average audio duration (seconds) and bitrate

##3. Total Data Size: 
     Multiply average size of each data type by the corresponding number of  data  points (text snippets, 
     images, videos, audio clips). Sum the results to get the total estimated data  size in  Gigabytes or 
     Terabytes.
     
     For example, if your training data has:

     10  million  text  snippets  with  an  average  length  of 100 characters (assuming mostly text data)
     10,000 images with an average resolution of 2 megapixels (compressed format)
     100 videos with an average length of 1 minute (compressed format)
     10 hours of audio (compressed format)

     You would estimate the data size for each type and then add them up. 

Imagine  you're  training  using social media platform that analyzes posts containing text, images, audio 
snippets (voice messages), and short videos:

Text:   Each post has an average of 100 words. You have 1 million text posts for training.

Images: The images are compressed JPEGs with an average size of 500 kilobytes (KB) each. You have 100,000 
        images.

Audio:  The audio snippets are compressed MP3s  with an  average  size of 5 megabytes (MB) each. You have 
        50,000 audio clips.

Video:  The  videos  are short clips (think social media stories) compressed in a format like MP4 with an 
        average size of 20 megabytes (MB) each. You have 20,000 videos.

##  Data Size Estimation:

Text:   1 million posts * 100 words/post = 100 million words (assuming 4 bytes per word) = 400 million bytes 
        (MB) ≈ 0.4 GB
Images: 100,000 images * 500 KB/image = 50,000,000 KB = 50 GB
Audio:  50,000 clips * 5 MB/clip = 250,000 MB = 250 GB
Video:  20,000 videos * 20 MB/video = 400,000 MB = 400 GB

##  Total Estimated Data Size:
    Add  the  data  size  for  each  type:  0.4  GB (text) + 50 GB (images) + 250 GB (audio) + 400 GB (video) 
    ≈ 700.4 GB

With the total data size of over 700 GB it then depends on the computational resources you have, if you  have
GPU (THE RTX 3060) it would take you very roughly a day or three to train this model with multimodal data. IF 
you are just relying on your CPU, it would take much longer, probably weeks or even months. IF you use NVIDIA
it probably would take several hours but your cost of acquiring such computational resource will be much more
compared to you using RTX3060 range or equivalent. So you have to calculate your Return of Investment.

## Participating in SLM Distributed network

Once you train your SLM for specific field of expertise, you can apply to join the SLMHub, which  is  the
distributed network of many SLMs. We are currently still constructing the larger  architecture  ecosystem
so please be patient, we will update this README in due time.

When users prompt using the chatbot interface of  the larger architecture  ecosystem, it will analyse and 
predict the right SLMs from the network to broadcast the users' prompt to, the SLM owner will have a score
from their SLM model file that measures  the confidence  score and upon higher confidence they will choose 
to inference prompt, to add more details from their domain specific field  specialization. Upon successful
inference, the larger architecture ecosystem will reward the particular SLM that did the inference. 

## Documentations

There are comment out explanation, comments to help you understand how to use  the  codes  but you need to 
know that you cannot modify too much IF you still want  to join our  SLMHub distributed  network, as these 
codes are specifically arranged/programmed to enable to receive outputs from the main model to do transfer
learning, that helps make the responses generated much more accurate, and you can be rewarded for allowing
you computational resources to join this distributed llm network that aims  at generating  highly accurate
responses.

## Multi Applications

SLM and the other larger architecture, because it is multimodality,  it can be extended to a  wide range of
functions and applications. It can generate videos, image, audio, be turned into field specific application
for different industries, it can also be extended to robots and power the 'thoughts process' of the robots.
With your assistance, we can make this entire architecture an extremely  power AI system  and it is capable
of autonomous, organic growth and upgrades with little to  no  hurdles, and you will be part and partial of 
this architecture's far reaching capabilities that benefits the society.




