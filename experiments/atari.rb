require_relative '../atari_ulerl_experiment'

config = {
  net: {
    type: :Recurrent,
    hidden_layers: [],
    activation_function: :logistic,
    # noutputs: 6,     # lock to using 6 neurons
    # noutputs: 10,    # lock to using 10 neurons
    # steepness: 0.5   # activation function steepness
  },
  env: {
    type: 'DemonAttack-v0'
  },
  run: {
    max_nsteps: 200,
    max_ngens: 100,
    # fitness_type: :sequential_single, # pry-rescue exceptions avoiding Parallel workers
    fitness_type: :parallel, # [:sequential_single, :sequential_multi, :parallel]
    # random_seed: 1,
    skip_frames: 5,
    skip_type: :noop, # [:noop, :repeat]
    ntrials_per_ind: 5,
    # debug: true
  },
  opt: {
    type: :XNES,#:BDNES,
    # sigma_init: 1,
    rescale_popsize: 1.5, # multiplicative factor
    # popsize: 5, # 5 is minimum for automatic utilities to work
    # popsize: 1, utilities: [1].to_na, # to debug inside Parallel calls
    rescale_lrate: 0.5,  # multiplicative factor
    # lrate: 0.08, # "original" (CMA-ES) magic number calls for ~0.047 on 2000 dims
    mu_init: -0.5..0.5
  },
  compr: {
    type: :IncrDictVQ, # [:VectorQuantization, :CopyVQ, :IncrDictVQ]
      # LOWER for MORE centroids
      equal_simil: 0.005,
      # equal_simil: 0.00495, # Qbert
      # equal_simil: 0.005, # DemonAttack
      # equal_simil: 0.005, # SpaceInvaders => 37
      # equal_simil: 0.005, # FishingDerby
      # equal_simil: 0.0052, # Phoenix
      # equal_simil: 0.00495, # Frostbite
      # equal_simil: 0.0049, # Seaquest
      # equal_simil: 0.0049, # KungFuMaster
      # equal_simil: 0.0033, # Kangaroo => 288 start screen bug!
      # equal_simil: 0.0055, # TimePilot
      # equal_simil: 0.005, # NameThisGame
    # type: :DecayingLearningRateVQ,
    #   lrate_min_den: 1,
    #   lrate_min: 0.001,
    #   decay_rate: 1,
    # type: :VectorQuantization,
    #   lrate: 0.7,
    # encoding: how to encode a vector based on similarity to centroids
    encoding_type: :sparse_coding, # [:most_similar, :most_similar_ary, :ensemble, :norm_ensemble, :sparse_coding_v1, :sparse_coding]

    # preproc: whether/which pre-processing to do on the image before elaboration
    preproc: :none, # [:none, :subtr_bg]
    # simil_type: how to measure similarity between a vector and a centroid
    simil_type: :dot, # [:dot, :mse]


    # BEWARE! If using IncrDictVQ need to set the following to `1.0`!
    seed_proport: 1.0, # proportional seeding of initial centroids with env reset obs (nil => disabled)

    # NOTE: uncomment if switching to novelty-based selection
    # nobs_per_ind: 5, # how many observations to get from each ind to train the compressor

    # ncentrs: 20,#0, # => switched to dynamic now!
    downsample: [3, 2], # divisors [row, col]
    init_centr_vrange: [0.0, 1.0],
    # TODO: remove (automate) the following
    obs_range: [0, 255],
    # vrange: [-1,1], # => careful what your zero means
    vrange: [0.0, 1.0], # encode images in [0,1]
    orig_size: [210, 160] # ALE image size [row, col] (not true for all games! beware!)
  }
}

log_dir = 'log'
FileUtils.mkdir_p log_dir
log_fname = WB::Tools::Logging.split_to log_dir
exp_name = log_fname[/\/(.*)\.log$/,1]
print "config = "
puts config
exp = DNE::AtariUlerlExperiment.new config
# require 'memory_profiler'; report = MemoryProfiler.report { exp.run }; report.pretty_print
exp.run

WB::Tools::Logging.restore_streams
exp.dump "atari_#{exp_name}.bin"

# exp.show_ind :mean, until_end: true
# exp.compr.show_centroids
require 'pry'; binding.pry
