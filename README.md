# Deep Neuroevolution experiments

This project collects a set of neuroevolution experiments with/towards deep networks for reinforcement learning control problems using an unsupervised learning feature exctactor.

## *Playing Atari with Six Neurons*

The experiments for this paper are based on [this code](https://github.com/giuse/DNE/releases/tag/six_neurons).  
The algorithms themselves are coded in the [`machine_learning_workbench` library](https://github.com/giuse/machine_learning_workbench), specifically using [version 0.8.0](https://github.com/giuse/machine_learning_workbench/releases/tag/0.8.0).


## Installation

First make sure the OpenAI Gym is pip-installed on python3, [instructions here](https://github.com/openai/gym).  
You will also need the [GVGAI_GYM](https://github.com/rubenrtorrado/GVGAI_GYM) to access GVGAI environments.

Mac users will need to tell the numo-linalg library where BLAS is:

    $ gem install numo-linalg -- --with-openblas-dir=/usr/local/opt/openblas

(per @sonot's [gist](https://gist.github.com/sonots/6fadc6cbbb170b9c4a0c9396c91a88e1))

Clone this repository, then execute:

    $ bundle install

## Usage

    bundle exec ruby experiments/cartpole.rb

## Contributing

Bug reports and pull requests are welcome on GitHub at https://github.com/giuse/DNE.

## License

The gem is available as open source under the terms of the [MIT License](https://opensource.org/licenses/MIT).

## References

Please feel free to contribute to this list (see `Contributing` above).

- **UL-ELR** stands for Unsupervised Learning plus Evolutionary Reinforcement Learning, from the paper _"Intrinsically Motivated Neuroevolution for Vision-Based Reinforcement Learning" (ICDL2011)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
- **BD-NES** stands for Block Diagonal Natural Evolution Strategy, from the homonymous paper _"Block Diagonal Natural Evolution Strategies" (PPSN2012)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
- **RNES** stands for Radial Natural Evolution Strategy, from the paper _"Novelty-Based Restarts for Evolution Strategies" (CEC2011)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
- **Online VQ** stands for Online Vector Quantization, from the paper _"Intrinsically Motivated Neuroevolution for Vision-Based Reinforcement Learning" (ICDL2011)_. Check [here](https://exascale.info/members/giuseppe-cuccu/) for citation reference and pdf.
- The **OpenAI Gym** is described [here](https://gym.openai.com/) and available on [this repo](https://github.com/openai/gym/)
- **PyCall.rb** is available on [this repo](https://github.com/mrkn/pycall.rb/).
