require_relative '../atari_ulerl_experiment'

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
    max_nsteps: 100,
    max_ngens: 10,
    # fitness_type: :sequential_single,
    fitness_type: :parallel, # [:sequential_single, :sequential_multi, :parallel]
    # random_seed: 1,
    skip_frames: 5,
    debug: true
  },
  opt: {
    type: :BDNES
  },
  compr: {
    type: :VectorQuantization,
    encoding: :ensemble_norm, # [:most_similar, :ensemble, :ensemble_norm]
    ncentrs: 8,
    lrate: 0.3, # actual number or `:vlr` for the variable learning rate
    downsample: [30,20] #[3, 2] # [vertical divisor, horizontal divisor]
  }
}
exp = DNE::AtariUlerlExperiment.new config
exp.run
print "Re-running best individual "
exp.show_best until_end: true

require 'pry'; binding.pry
