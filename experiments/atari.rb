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
    max_nsteps: 2000,
    max_ngens: 100,
    # fitness_type: :sequential_single, # pry-rescue exceptions avoiding Parallel workers
    fitness_type: :parallel, # [:sequential_single, :sequential_multi, :parallel]
    # random_seed: 1,
    skip_frames: 5,
    skip_type: :noop, # [:noop, :repeat]
    # debug: true
  },
  opt: {
    type: :BDNES,
    rescale_popsize: 2, # it's a multiplicative factor
    rescale_lrate: 0.5  # it's a multiplicative factor
  },
  compr: {
    type: :CopyVQ,
      equal_simil: 0.1,
    # type: :DecayingLearningRateVQ,
    #   lrate_min_den: 1,
    #   lrate_min: 0.001,
    #   decay_rate: 1,
    # type: :VectorQuantization,
    #   lrate: 0.7,
    # encoding: how to encode a vector based on similarity to centroids
    encoding_type: :ensemble_norm, # [:most_similar, :ensemble, :ensemble_norm]
    # simil_type: how to measure similarity between a vector and a centroid


    # preproc: whether/which pre-processing to do on the image before elaboration
    preproc: :subtr_bg, # [:none, :subtr_bg]
    # simil_type: :mse, # [:dot, :mse]
    # seed_proport: 0.5, # proportional seeding of initial centroids with env reset obs

    ncentrs: 200,
    downsample: [3, 2], # divisors [row, col]
    init_centr_vrange: [-0.5, 0.5],
    # TODO: remove (automate) the following
    obs_range: [0, 255],
    vrange: [-1,1],
    orig_size: [210, 160] # ALE image size [row, col]
  }
}

exp = DNE::AtariUlerlExperiment.new config
# exp.compr.show_centroids

# require 'memory_profiler'
# report = MemoryProfiler.report do
  exp.run
# end
# report.pretty_print

# exp.show_best until_end: true
# exp.compr.show_centroids

require 'pry'; binding.pry
