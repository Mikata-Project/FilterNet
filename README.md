## FilterNet
FilterNet is a ensemble Neural Network model used for time series analysis. It is comprised of a 1D Convolutional
Neural Network and fastai's MixedInputModel. An example of the network written in PyTorch is located in filternet.py and
provides the model class along with it's corresponding dataset class.

The 1D convolutional nueral network (CNN) was inspired by their traditional use in discrete time signal processing.
While developed independently, it closely resembles the findings described in the
[WaveNet paper by Borovykh et al](https://arxiv.org/pdf/1703.04691.pdf). While the 1D CNN performed well on it's own,
our datasets at System1 have a lot of context associated with them (hour of day, day of week, international/domestic,
etc.) which the 1D CNN is unable to handle. We utilized [fastai's MixedInputModel](https://github.com/fastai/fastai),
which has been used successfully for tabular data, to include learnings on the context portion of our datasets. The two
neural networks are combined using a final regression layer and were found to compliment each other. The resulting
ensemble model outperformed one of our current best production time series models 
[TBATS](https://robjhyndman.com/hyndsight/forecasting-weekly-data/) in testing.

Our hope is by open-sourcing our approach it will help generate further ideas on how to improve time series modeling
using neural networks.


Contributing authors:
- Nathan Janos (Chief Data Officer at System1)
- Jeff Roach (Data Scientist at System1)

For more information about System1, please visit: www.system1.com

