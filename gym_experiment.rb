require 'parallel'                      # https://github.com/grosser/parallel
ENV["PYTHON"] = `which python3`.strip   # set python3 path for PyCall
require 'pycall/import'                 # https://github.com/mrkn/pycall.rb/
# IMPORTANT: `require 'numo/narray`' should come AFTER the first `pyimport :gym`
# Don't ask why, don't know, don't care. Check `gym_test.rb` to try it out.
begin
  puts "Initializing OpenAI Gym through PyCall"
  include PyCall::Import
  pyimport :gym        # adds the OpenAI Gym environment
rescue PyCall::PythonNotFound => err
  raise "\n\nThis project requires Python 3.\n" \
        "You can edit the path in the `ENV['PYTHON']` " \
        "variable on top of the file.\n\n"
rescue PyCall::PyError => err
  raise "\n\nPlease install the OpenAI Gym from https://github.com/openai/gym\n" \
        "  $ git clone git@github.com:openai/gym.git openai_gym\n" \
        "  $ pip3 install --user -e openai_gym\n\n"
end
# IMPORTANT: `require 'numo/narray`' should come `AFTER PyCall::Import.pyimport :gym`
require 'machine_learning_workbench'    # https://github.com/giuse/machine_learning_workbench/


# Deep Neuroevolution
module DNE
  # Shorthands
  NES = WB::Optimizer::NaturalEvolutionStrategies
  NN = WB::NeuralNetwork

  # Loads an experiment description from a config hash, initialize everything and run it
  class GymExperiment
    include PyCall::Import

    attr_reader :config, :single_env, :net, :opt, :parall_envs, :max_nsteps, :max_ngens,
      :termination_criteria, :random_seed, :debug, :skip_frames, :skip_type, :fit_fn, :netopts,
      :opt_type, :opt_opt # hack away!! `AtariUlerlExperiment#update_opt`

    def initialize config
      @config = config
      # TODO: I really don't like these lines below...
      @max_nsteps = config[:run][:max_nsteps]
      @max_ngens = config[:run][:max_ngens]
      @termination_criteria = config[:run][:termination_criteria]
      @random_seed = config[:run][:random_seed]
      @debug = config[:run][:debug]
      @skip_frames = config[:run][:skip_frames] || 0
      @skip_type = config[:run][:skip_type] || :noop
      @fit_fn = gen_fit_fn config[:run][:fitness_type]
      if debug
        real_fit = @fit_fn
        @fit_fn = -> (ind) { puts "pre_fit"; real_fit.call(ind).tap { puts "post_fit" } }
      end
      pyimport :gym        # adds the OpenAI Gym environment to this class
      puts "Initializing single env" if debug
      @single_env = init_env config[:env] # one put aside for easy access
      puts "Initializing network" if debug
      config[:net][:ninputs] ||= single_env.obs_size
      config[:net][:noutputs] ||= single_env.act_size
      @netopts = config[:net]
      @net = init_net netopts
      puts "Initializing optimizer" if debug
      @opt = init_opt config[:opt]
      # puts "Initializing parallel environments" if debug
      # unless config[:run][:fitness_type] == :sequential_single
      #   # for parallel fitness computation
      #   # TODO: test if `single_env` forked is sufficient (i.e. if Python gets forked)
      #   @parall_envs = Parallel.processor_count.times.map { init_env config[:env] }
      # end
      @parall_envs = ParallEnvs.new method(:init_env), config[:env]

      puts "=> Initialization complete" if debug
    end

    # Automatic, dynamic environment initialization
    class ParallEnvs < Array
      attr_reader :init_fn, :config
      def initialize init_fn, config
        @init_fn = init_fn
        @config = config
        super()
      end
      def [] i
        super || (self[i] = init_fn.call config)
      end
    end

    # Debugging utility.
    # Visually inspect if the environment is properly initialized.
    def test_env env=nil, nsteps=100
      env ||= gym.make('CartPole-v1')
      env.reset
      env.render
      nsteps.times do |i|
        act = env.action_space.sample
        env.step(act)
        env.render
      end
      env.reset
    end

    # Initializes the environment
    # @note python environments interfaced through pycall
    # @param type [String] the type of environment as understood by OpenAI Gym
    # @return an initialized environment
    def init_env type:
      # TODO: make a wrapper around the observation to avoid switch-when
      puts "  initializing env" if debug

      # if type.match /gvgai/ # GVGAI Gym environment from NYU
      #   begin  # can't wait for Ruby 2.5 to simplify this to if/rescue/end
      #     puts "Loading GVGAI environment" if debug
      #     pyimport :gym_gvgai
      #   rescue PyCall::PyError => err
      #     raise "\n\nTo run GVGAI environments you need to install the Gym env:\n" \
      #           "  $ git clone git@github.com:rubenrtorrado/GVGAI_GYM.git gym-gvgai\n" \
      #           "  $ pip3 install --user -e gym-gvgai/gvgai-gym/\n\n"
      #   end
      # end

      gym.make(type).tap do |env|
        # Collect info about the observation space
        obs = env.reset.tolist.to_a
        raise "Unrecognized observation space" if obs.nil? || obs.empty?
        env.define_singleton_method(:obs_size) { obs.size }

        # Collect info about the action space
        act_type, act_size = env.action_space.to_s.match(/(.*)\((\d*)\)/).captures
        raise "Unrecognized action space" if act_type.nil? || act_size.nil?
        env.define_singleton_method(:act_type) { act_type.downcase.to_sym }
        env.define_singleton_method(:act_size) { Integer(act_size) }
        # TODO: continuous actions
        raise NotImplementedError, "Only 'Discrete' action types at the moment please" \
          unless env.act_type == :discrete
        puts "Space sizes: obs => #{env.obs_size}, act => #{env.act_size}" if debug
      end
    end

    # Initialize the controller
    # @param type [Symbol] name the class of neural network to use (from the WB)
    # @param hidden_layers [Array] list of hidden layer sizes for the networks structure
    # @param activation_function [Symbol] name one of the activation functions available
    # @param ninputs [Integer] number of inputs to the network
    # @param noutputs [Integer] number of outputs of the network
    # @return an initialized neural network
    def init_net type:, hidden_layers:, activation_function:, ninputs:, noutputs:, **act_fn_args
      netclass = NN.const_get(type)
      netstruct = [ninputs, *hidden_layers, noutputs]
      netclass.new netstruct, act_fn: activation_function, **act_fn_args
    end

    # Initialize the optimizer
    # @param type [Symbol] name the NES algorithm of choice
    # @return an initialized NES instance
    def init_opt type:, **opt_opt
      @opt_type = type
      @opt_opt = opt_opt
      dims = case type
      when :XNES, :SNES, :RNES, :FNES
        net.nweights
      when :BDNES
        net.nweights_per_layer
      else
        raise NotImplementedError, "Make sure to add `#{type}` to the accepted ES"
      end
      NES.const_get(type).new dims, fit_fn, :max, parallel_fit: true, rseed: random_seed, **opt_opt
    end

    # Return an action for an observation
    # @note this includes the non-banal sequence of converting the observation to network inputs,
    #   activating the network, then interpreting the network output as the corresponding action
    def action_for observation
      # TODO: continuous actions
      # raise NotImplementedError, "Only 'Discrete' action types at the moment please" \
      #   unless single_env.act_type == :discrete
      # We're checking this at gym_env creation, that's enough :)
      input = observation.tolist.to_a
      # TODO: probably a function generator here would be notably more efficient
      # TODO: the normalization range depends on the net's activation function
      output = net.activate input
      begin # Why do I get NaN output sometimes?
        action = output.max_index
      rescue ArgumentError, Parallel::UndumpableException
        puts "\n\nNaN NETWORK OUTPUT!"
        output.map! { |out| out = -Float::INFINITY if out.nan? }
        action = output.index output.max
      end
    end

    # Builds a function that return a list of fitnesses for a list of genotypes.
    # @param type the type of computation
    # @return [lambda] function that evaluates the fitness of a list of genotype
    # @note returned function has param genotypes [Array<gtype>] list of genotypes, return [Array<Numeric>] list of fitnesses for each genotype
    def gen_fit_fn type
      type ||= :parallel
      case type
      # SEQUENTIAL ON SINGLE ENVIRONMENT
      # => useful to catch problems with parallel env spawning
      when :sequential_single
        -> (genotypes) { genotypes.map &method(:fitness_one) }
      # SEQUENTIAL ON MULTIPLE ENVIRONMENTS
      # => `Parallel` can make debugging hard, this switches it off
      when :sequential_multi
        -> (genotypes) do
          genotypes.zip(parall_envs).map do |genotype, env|
            fitness_one genotype, env: env
          end.to_na
        end
      # PARALLEL ON MULTIPLE ENVIRONMENTS
      # => because why not
      when :parallel
        nprocs = Parallel.processor_count - 1 # it's actually faster this way
        -> (genotypes) do
          Parallel.map(0...genotypes.shape.first, in_processes: nprocs) do |i|
            fitness_one genotypes[i,true], env: parall_envs[i]
          end.to_na
        end
      else raise ArgumentError, "Unrecognized fit type: `#{type}`"
      end
    end

    SKIP_TYPE = {
      noop: -> (act, env) { env.step(0) },
      repeat: -> (act, env) { env.step(act) }
    }

    # Return the fitness of a single genotype
    # @param genotype the individual to be evaluated
    # @param env the environment to use for the evaluation
    # @param render [bool] whether to render the evaluation on screen
    # @param nsteps [Integer] how many interactions to run with the game. One interaction is one action choosing + enacting and `skip_frames` repetitions
    def fitness_one genotype, env: single_env, render: false, nsteps: max_nsteps
      puts "Evaluating one individual" if debug
      puts "  Loading weights in network" if debug
      net.deep_reset
      net.load_weights genotype
      observation = env.reset
      env.render if render
      tot_reward = 0
      puts "  Running (max_nsteps: #{max_nsteps})" if debug
      nsteps.times do |i|
        # TODO: refactor based on `atari_ulerl`
        selected_action, normobs, novelty = action_for observation
        # observation, reward, done, info = env.step(selected_action).to_a
        # skip_frames&.times { env.step 0 } # accelerate simulation


        raise "Update this from ulerl"
        # execute once, execute skip, unshift result, convert all
        # need to pass also some of the helper functions in this class


        observations, rewards, dones, infos = skip_frames.times.map do
          SKIP_TYPE[skip_type].call(selected_action, env).to_a
        end.transpose
        # NOTE: this blurs the observation. An alternative is to isolate what changes.
        observation = observations.reduce(:+) / observations.size
        reward = rewards.reduce :+
        done = dones.any?

        tot_reward += reward
        env.render if render
        break if done
      end
      puts "=> Done, fitness: #{tot_reward}" if debug
      tot_reward
    end

    # Run the experiment
    def run ngens: max_ngens
      ngens.times do |i|
        print "Gen #{i+1}: "
        opt.train
        puts "Best fit so far: #{opt.best.first} -- " \
             "Avg fit: #{opt.last_fits.mean} -- " \
             "Conv: #{opt.convergence}"

        break if termination_criteria&.call(opt)
      end
    end

    # Runs an individual, by default the highest scoring found so far
    # @param which [<:best, :mean, NArray>] which individual to run: either the best,
    #   the mean, or one passed directly as argument
    # @param until_end [bool] raises the `max_nsteps` to see interaction until `done`
    def show_ind which=:best, until_end: false
      ind = case which
        when :best then opt.best.last
        when :mean then opt.mu
        when NArray then which
        else raise ArgumentError, "Which should I show? `#{which}`"
      end
      nsteps = until_end ? max_nsteps*1000 : max_nsteps
      print "Re-running best individual "
      fit = fitness_one ind, render: true, nsteps: nsteps
      puts "-- fitness: #{fit}"
    end

  end

end

puts "USAGE: `bundle exec ruby experiments/<expname>.rb`" if __FILE__ == $0
