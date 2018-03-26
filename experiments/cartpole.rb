require_relative '../gym_experiment'

config = {
  net: {
    type: :Recurrent,
    hidden_layers: [],
    activation_function: :logistic
  },
  env: {
    type: 'CartPole-v1'
  },
  run: {
    max_nsteps: 550,
    max_ngens: 5,
    random_seed: 1,
    fitness_type: :parallel
    # debug: true
  },
  opt: {
    type: :BDNES
  }
}

exp = DNE::GymExperiment.new config
exp.run
print "Re-running best individual "
exp.show_best until_end: true

require 'pry'; binding.pry
