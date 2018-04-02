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
    max_ngens: 2,
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
    type: :OnlineVectorQuantization,
    # type: :VectorQuantization, lrate: 0.7,
    encoding: :ensemble_norm, # [:most_similar, :ensemble, :ensemble_norm]
    ncentrs: 8,
    downsample: [3, 2], # divisors [row, col]
    seed_proport: 0.6, # proportional seeding of initial centroids with env reset obs
    init_centr_vrange: [-0.5, 0.5],
    # TODO: remove (automate) the following
    obs_range: [0, 255],
    vrange: [-1,1],
    orig_size: [210, 160] # ALE image size [row, col]
  }
}
exp = DNE::AtariUlerlExperiment.new config

# require 'memory_profiler'
# report = MemoryProfiler.report { exp.run }
# report.pretty_print

exp.run

puts "Re-running best individual "
exp.show_best until_end: true
exp.compr.show_centroids

require 'pry'; binding.pry
