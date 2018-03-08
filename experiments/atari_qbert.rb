require_relative '../neuroevo'

config = {
  net: {
    type: :Recurrent,
    hidden_layers: [],
    activation_function: :logistic
  },
  env: {
    type: 'Qbert-v0'
  },
  run: {
    max_nsteps: 550,
    max_ngens: 5,
    random_seed: 1
  },
  opt: {
    type: :BDNES
  }
}

exp = Experiment.new config
exp.run
print "Re-running best individual "
exp.show_best

require 'pry'; binding.pry
