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
      compr_seed_proport = compr_opts.delete :seed_proport
      @preproc = compr_opts.delete :preproc
      @compr = ObservationCompressor.new **compr_opts
      puts "Loading Atari OpenAI Gym environment" # if debug
      super config
      # initialize the centroids based on the env's reset obs
      # TODO: make this parametrizable in exp config
      # compr.reset_centrs single_env.reset_obs, proport: compr_seed_proport
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

    # Initialize the controller
    # @param type [Symbol] name the class of neural network to use (from the WB)
    # @param hidden_layers [Array] list of hidden layer sizes for the networks structure
    # @param activation_function [Symbol] name one of the activation functions available
    # @return an initialized neural network
    def init_net type:, hidden_layers:, activation_function:, **act_fn_args
      netclass = NN.const_get(type)
      netstruct = [compr.ncentrs, *hidden_layers, single_env.act_size]
      netclass.new netstruct, act_fn: activation_function, **act_fn_args
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
    # @param nsteps [Integer] how many interactions to run with the game. One interaction is one action choosing + enacting and `skip_frames` repetitions
    def fitness_one genotype, env: single_env, render: false, nsteps: max_nsteps, aggr_type: :last
      puts "Evaluating one individual" if debug
      puts "  Loading weights in network" if debug
      net.load_weights genotype # this also resets the state
      observation = env.reset
      env.render if render
      tot_reward = 0
      compr_train = [nil, -Float::INFINITY]
      puts "  Running (max_nsteps: #{max_nsteps})" if debug
      nsteps.times do |i|
        code = compr.encode observation
        selected_action = action_for code
        novelty = compr.novelty observation, code
        obs_lst, rew, done, info_lst = env.execute selected_action, skip_frames: skip_frames
        # puts "#{obs_lst}, #{rew}, #{done}, #{info_lst}" if debug
        observation = OBS_AGGR[aggr_type].call obs_lst
        tot_reward += rew
        compr_train = [observation, novelty] if novelty > compr_train.last
        env.render if render
        break if done
      end
      compr.train_set << compr_train.first
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

    # Run the experiment
    def run ngens: max_ngens
      ngens.times do |i|
        puts Time.now
        print "Gen #{i+1}/#{ngens} fits: "
        opt.train
        puts # newline here because I'm `print`ing all ind fits in `opt.train`
        puts "Best fit so far: #{opt.best.first} -- " \
             "Avg fit: #{opt.last_fits.mean} -- " \
             "Conv: #{opt.convergence}"
        break if termination_criteria&.call(opt)
        puts "Training compressor" if debug
        compr.train
      end
    end

  end
end

puts "USAGE: `bundle exec ruby experiments/atari/<expname>.rb`" if __FILE__ == $0
