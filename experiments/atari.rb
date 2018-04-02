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
    max_nsteps: 200,
    max_ngens: 5,
    # fitness_type: :sequential_single,
    fitness_type: :parallel, # [:sequential_single, :sequential_multi, :parallel]
    # random_seed: 1,
    skip_frames: 50, #5,
    skip_type: :noop,
    debug: true
  },
  opt: {
    type: :RNES
  },
  compr: {
    # type: :OnlineVectorQuantization,
    type: :VectorQuantization,
    encoding: :ensemble_norm, # [:most_similar, :ensemble, :ensemble_norm]
    ncentrs: 50,
    lrate: 0.7,
    downsample: [3,2] #[3, 2] # [vertical divisor, horizontal divisor]
  }
}
exp = DNE::AtariUlerlExperiment.new config

# require 'memory_profiler'
# report = MemoryProfiler.report { exp.run }
# report.pretty_print

exp.run

print "Re-running best individual "
# exp.show_best until_end: true
exp.compr.show_centroids

# require 'pry'; binding.pry
