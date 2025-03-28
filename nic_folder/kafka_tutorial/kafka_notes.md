# Notes about kafka

Here I want to document what I did to test and understand a bit how kafka works. I decided to use confluent because it seemed easy to setup and use with VS code.

## Setup

Here the short steps I took to make it work:
1) Installed the VS code extension for Confluent
2) In the extension there is a resources tab, there I created a local container (goes all alone if you have already docker)
3) Created a conda environment where I installed confluent-kafka (pip install confluent-kafka) -> note I'm using python3.12
4) I wrote a simple producer.py and consumer.py script to see if it worked with the container (important to use the ports opened by the container)