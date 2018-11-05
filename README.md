## FilterNet
FilterNet is a ensemble neural network model used for time series analysis. It is comprised of a 1D convolutional neural
network and fast.ai's ixedInputModel. An example of the network implemented in PyTorch is located in filternet.py and 
provides the model class along with its corresponding dataset class. Links to our presentation at PyDataLA 2018, slides
and video will be provided below.

This 1D convolutional neural network (CNN) was inspired by the traditional use of filters in discrete time signal 
processing. While developed independently, it closely resembles the findings described in the 
[WaveNet paper by Borovykh et al](https://arxiv.org/pdf/1703.04691.pdf). While the 1D CNN performed well on its own, 
datasets can have a lot of context associated with them (hour of day, day of week, etc.) which the 1D CNN alone is 
unable to handle. We utilized [fastai's MixedInputModel](https://github.com/fastai/fastai), which has been used 
successfully for tabular data, to include learnings on the context portion of our datasets. The two neural networks are 
combined using a final regression layer and were found to compliment each other. In testing, the resulting ensemble 
model outperformed one of our current best production time series models ([TBATS](https://robjhyndman.com/hyndsight/forecasting-weekly-data/)).

Our hope is by open sourcing our approach it will help generate further ideas on how to improve time series modeling 
using neural networks.

PyData Los Angeles 2018 presentation:
- [Abstract](https://pydata.org/la2018/schedule/presentation/14)
- [Slides](https://docs.google.com/presentation/d/e/2PACX-1vR6eea4L_Z_hyz24kgch3Lt5eEQ9PmmI2gUys_DcQrWY0EbG5CfOy4suqeLejXEql3x-nYT2NshrQRc/pub?start=false&loop=false&delayms=3000)
- Video (Coming soon)

Contributing authors:
- Jeff Roach (Data Scientist at System1)
- Nathan Janos (Chief Data Officer at System1)

For more information about System1, please visit: www.system1.com

