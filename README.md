# Air-PIN

Air-PIN is a **proof-of-concept** to enter four-digit PIN numbers contactless.

It consists of a deep learning model that is trained to recognize finger signs from 0-9. To represent values higher than five singlehanded, the [chinese number gestures](https://en.wikipedia.org/wiki/Chinese_number_gestures) are used.

The single frames from a camera stream are piped through the model to recognize digits. After a digit is recognized with high confidence, the user gets a feedback and can continue with the next number.

## Video Stream
For the implementation the RTMP stream from a GoPro Hero8 was captured. It was also used to create the dataset.

## DNN
The deep neural network consists of a ResNet18 architecture that was trained for around 50 epochs on around 1500 samples.
The validation accuracy is around 90%.

## Demo
A demo video can be found at [TODO](youtube.com)