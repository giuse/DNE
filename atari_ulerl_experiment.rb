require 'forwardable'
require_relative 'gym_experiment'
require_relative 'observation_compressor'
require_relative 'atari_wrapper'
require_relative 'tools'

module DNE
  # TODO: why doesn't it work when I use UInt8? We're in [0,255]!
  NImage = Xumo::UInt32 # set a single data type for images

  # Specialized GymExperiment class for Atari environments and UL-ERL.
  class AtariUlerlExperiment < GymExperiment

    attr_reader :compr, :resize, :preproc

    def initialize config
      ## Why would I wish:
      # compr before super (current choice)
      # - net.struct => compr.ncentrs
      # super before compr
      # - config[:run][:debug] # can live without
      # - compr.dims => AtariWrapper.downsample # gonna duplicate the process, beware
      # - obs size => AtariWrapper orig_size
      puts "Initializing compressor" # if debug
      compr_opts = config.delete(:compr) # otherwise unavailable for debug
      seed_proport = compr_opts.delete :seed_proport
      @preproc = compr_opts.delete :preproc
      @compr = ObservationCompressor.new **compr_opts
      # overload ninputs for network
      config[:net][:ninputs] ||= compr.ncentrs
      puts "Loading Atari OpenAI Gym environment" # if debug
      super config
      # initialize the centroids based on the env's reset obs
      compr.reset_centrs single_env.reset_obs, proport: seed_proport if seed_proport
    end

    # Initializes the Atari environment
    # @note python environments interfaced through pycall
    # @param type [String] the type of environment as understood by OpenAI Gym
    # @param [Array<Integer,Integer>] optional downsampling for rows and columns
    # @return an initialized environment
    def init_env type:
      puts "  initializing env" if debug
      AtariWrapper.new gym.make(type), downsample: compr.downsample,
        skip_type: skip_type, preproc: preproc
    end

    # How to aggregate observations coming from a sequence of noops
    OBS_AGGR = {
      avg: -> (obs_lst) { obs_lst.reduce(:+) / obs_lst.size},
      new: -> (obs_lst) { obs_lst.first - env.reset_obs},
      first: -> (obs_lst) { obs_lst.first },
      last: -> (obs_lst) { obs_lst.last }
    }

    # Return the fitness of a single genotype
    # @param genotype the individual to be evaluated
    # @param env the environment to use for the evaluation
    # @param render [bool] whether to render the evaluation on screen
    # @param nsteps [Integer] how many interactions to run with the game. One interaction is one action choosing + enacting followed by `skip_frames` frame skips
    def fitness_one genotype, env: single_env, render: false, nsteps: max_nsteps, aggr_type: :last
      puts "Evaluating one individual" if debug
      puts "  Loading weights in network" if debug
      net.load_weights genotype # this also resets the state
      observation = env.reset
      # require 'pry'; binding.pry unless observation == env.reset_obs # => check passed
      env.render if render
      tot_reward = 0
      represent_obs = [nil, -Float::INFINITY] # observation representative of ind novelty

      puts "  Running (max_nsteps: #{max_nsteps})" if debug
      nsteps.times do |i|
        code = compr.encode observation
        selected_action = action_for code
        novelty = compr.novelty observation, code
        obs_lst, rew, done, info_lst = env.execute selected_action, skip_frames: skip_frames
        # puts "#{obs_lst}, #{rew}, #{done}, #{info_lst}" if debug
        observation = OBS_AGGR[aggr_type].call obs_lst
        tot_reward += rew
        # The same observation represents the state both for action selection and for individual novelty
        represent_obs = [observation, novelty] if novelty > represent_obs.last
        env.render if render
        break if done
      end
      compr.train_set << represent_obs.first
      puts "=> Done! fitness: #{tot_reward}" if debug
      print tot_reward, ' ' # if debug
      tot_reward
    end

    # Builds a function that return a list of fitnesses for a list of genotypes.
    # Since Parallel runs in separate fork, this overload is needed to fetch out
    # the training set before returning the fitness to the optimizer
    # @param type the type of computation
    # @return [lambda] function that evaluates the fitness of a list of genotype
    # @note returned function has param genotypes [Array<gtype>] list of genotypes, return [Array<Numeric>] list of fitnesses for each genotype
    def gen_fit_fn type
      if type.nil? || type == :parallel
        -> (genotypes) do
          fits, parall_infos = Parallel.map(0...genotypes.shape.first) do |i|
            fit = fitness_one genotypes[i,true], env: parall_envs[Parallel.worker_number]
            [fit, compr.parall_info]
          end.transpose
          parall_infos.each &compr.method(:add_from_parall_info)
          fits.to_na
        end
      else
        super
      end
    end

    # Return an action for an encoded observation
    # The neural network is activated on the code, then its output is
    # interpreted as a corresponding action
    # @param code [Array] encoding for the current observation
    # @return [Integer] action
    # TODO: alternatives like softmax and such
    def action_for code
      output = net.activate code
      nans = output.isnan
      # this is a pretty reliable bug indicator
      raise "\n\n\tNaN network output!!\n\n" if nans.any?
      action = output.max_index
    end

    def update_opt
      return if @curr_ninputs == compr.ncentrs
      puts "  ncentrs: #{compr.ncentrs}"
      diff = compr.ncentrs - @curr_ninputs
      @curr_ninputs = compr.ncentrs
      pl = net.struct.first(2).reduce(:*)
      nw = diff * net.struct[1]

      new_mu = opt.blocks.first.mu.insert pl, [0]*nw
      new_sigma = opt.blocks.first.sigma.insert [pl]*nw, 0, axis: 0
      new_sigma = new_sigma.insert [pl]*nw, 0, axis: 1
      new_sigma.diagonal[pl...(pl+nw)] = 1

      old = opt.blocks.first
      opt.blocks[0] = NES::XNES.new new_mu.size, old.obj_fn, old.opt_type,
        parallel_fit: old.parallel_fit, mu_init: new_mu, sigma_init: new_sigma,
        **opt_opt
      opt.blocks.first.instance_variable_set :@rng, old.rng # ensure rng continuity
      opt.ndims_lst[0] = new_mu.size
      puts "  new opt dims: #{opt.ndims_lst}"

      # FIXME: I need to run these before I can use automatic popsize again!
      # update popsize in bdnes and its blocks
      # if opt.kind_of? BDNES or something
      # opt.instance_variable_set :popsize, blocks.map(&:popsize).max
      # opt.blocks.each { |xnes| xnes.instance_variable_set :@popsize, opt.popsize }

      # update net, since inputs have changed
      @net = init_net netopts.merge({ninputs: compr.ncentrs})
      puts "  new net struct: #{net.struct}"
    end

    # Run the experiment
    def run ngens: max_ngens
      @curr_ninputs = compr.ncentrs
      ngens.times do |i|
        puts Time.now
        print "Gen #{i+1}/#{ngens} fits: "
        # it just makes more sense run first, even though at first gen the trainset is empty
        puts "Training compressor" if debug
        compr.train
        update_opt  # if I have more centroids, I should update opt

        opt.train
        puts # newline here because I'm `print`ing all ind fits in `opt.train`
        puts "Best fit so far: #{opt.best.first} -- " \
             "Avg fit: #{opt.last_fits.mean} -- " \
             "Conv: #{opt.convergence}"
        break if termination_criteria&.call(opt)
      end
    end

  end
end

puts "USAGE: `bundle exec ruby experiments/atari/<expname>.rb`" if __FILE__ == $0
