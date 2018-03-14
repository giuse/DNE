require 'machine_learning_workbench'    # https://github.com/giuse/machine_learning_workbench/
require 'parallel'                      # https://github.com/grosser/parallel
ENV["PYTHON"] = `which python3`.strip   # set python3 path for PyCall
require 'pycall/import'                 # https://github.com/mrkn/pycall.rb/

# Deep Neuroevolution
module DNE
  # Loads an experiment description from a config hash, initialize everything and run it
  class GymExperiment
    # Shorthands
    WB = MachineLearningWorkbench
    NES = WB::Optimizer::NaturalEvolutionStrategies
    NN = WB::NeuralNetwork

    include PyCall::Import

    attr_reader :config, :single_env, :net, :opt, :parall_envs, :max_nsteps, :max_ngens,
      :termination_criteria, :random_seed, :debug, :repeat_action, :fit_fn

    def initialize config
      @config = config
      # TODO: I really don't like these lines below...
      @max_nsteps = config[:run][:max_nsteps]
      @max_ngens = config[:run][:max_ngens]
      @termination_criteria = config[:run][:termination_criteria]
      @random_seed = config[:run][:random_seed]
      @debug = config[:run][:debug]
      @repeat_action = (config[:run][:skip_frames] || 0) + 1
      @fit_fn = gen_fit_fn config[:run][:fitness_type]
      if debug
        real_fit = @fit_fn
        @fit_fn = -> (ind) { puts "pre_fit"; real_fit.call(ind).tap { puts "post_fit" } }
      end
      puts "Loading OpenAI Gym through PyCall" if debug
      init_gym
      puts "Initializing single env" if debug
      @single_env = init_env config[:env] # one put aside for easy access
      puts "Initializing network" if debug
      @net = init_net config[:net]
      puts "Initializing optimizer" if debug
      @opt = init_opt config[:opt]
      puts "Initializing parallel environments" if debug
      unless config[:run][:fitness_type] == :sequential_single
        # these for parallel fitness computation
        @parall_envs = opt.popsize.times.map { init_env config[:env] }
        # TODO: parallel initialization not working
        # @parall_envs = Parallel.map(1..opt.popsize) { init_env config[:env] }
      end
      puts "=> Initialization complete" if debug
    end

    # Import the OpenAI Gym through PyCall
    def init_gym
      begin
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
    end

    # Initializes the environment
    # @note python environments interfaced through pycall
    # @param type [String] the type of environment as understood by OpenAI Gym
    # @return an initialized environment
    def init_env type:
      # TODO: make a wrapper around the observation to avoid switch-when
      puts "  initializing env" if debug

      if type.match /gvgai/ # GVGAI Gym environment from NYU
        begin  # can't wait for Ruby 2.5 to simplify this to if/rescue/end
          puts "Loading GVGAI environment" if debug
          pyimport :gym_gvgai
        rescue PyCall::PyError => err
          raise "\n\nTo run GVGAI environments you need to install the Gym env:\n" \
                "  $ git clone git@github.com:rubenrtorrado/GVGAI_GYM.git gym-gvgai\n" \
                "  $ pip3 install --user -e gym-gvgai/gvgai-gym/\n\n"
        end
      end
      gym.make(type).tap do |gym_env|
        # Collect info about the observation space
        obs = gym_env.reset.tolist.to_a
        raise "Unrecognized observation space" if obs.nil? || obs.empty?
        gym_env.define_singleton_method(:obs_size) { obs.size }

        # Collect info about the action space
        act_type, act_size = gym_env.action_space.to_s.match(/(.*)\((\d*)\)/).captures
        raise "Unrecognized action space" if act_type.nil? || act_size.nil?
        gym_env.define_singleton_method(:act_type) { act_type.downcase.to_sym }
        gym_env.define_singleton_method(:act_size) { Integer(act_size) }
        # TODO: continuous actions
        raise NotImplementedError, "Only 'Discrete' action types at the moment please" \
          unless gym_env.act_type == :discrete
        puts "Space sizes: obs => #{gym_env.obs_size}, act => #{gym_env.act_size}" if debug
      end
    end

    # Initialize the controller
    # @param type [Symbol] name the class of neural network to use (from the WB)
    # @param hidden_layers [Array] list of hidden layer sizes for the networks structure
    # @param activation_function [Symbol] name one of the activation functions available
    # @return an initialized neural network
    def init_net type:, hidden_layers:, activation_function:
      netclass = NN.const_get(type)
      netstruct = [single_env.obs_size, *hidden_layers, single_env.act_size]
      netclass.new netstruct, act_fn: activation_function
    end

    # Initialize the optimizer
    # @param type [Symbol] name the NES algorithm of choice
    # @return an initialized NES instance
    def init_opt type:
      dims = case type
      when :BDNES
        net.nweights_per_layer
      when :SNES, :XNES
        net.nweights
      else
        raise NotImplementedError, "Make sure to add `#{type}` to the accepted ES"
      end
      NES.const_get(type).new dims, fit_fn, :max, parallel_fit: true, rseed: random_seed
    end

    # Return an action for an observation
    # @note this includes the non-banal sequence of converting the observation to network inputs,
    #   activating the network, then interpreting the network output as the corresponding action
    def action_for observation
      # TODO: continuous actions
      # raise NotImplementedError "Only 'Discrete' action types at the moment please" \
      #   unless single_env.act_type == :discrete
      # We're checking this at gym_env creation, that's enough :)

      input = observation.tolist.to_a
      # TODO: probably a function generator here would be notably more efficient
      # TODO: the normalization range depends on the net's activation function
      output = net.activate input
      begin # Why do I get NaN output sometimes?
        action = output.index output.max
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
          end
        end
      # PARALLEL ON MULTIPLE ENVIRONMENTS
      # => because why not
      when :parallel
        -> (genotypes) do
          # TODO: initialize only as many parall_envs as Parallel workers
          Parallel.map(genotypes.zip(parall_envs)) do |genotype, env|
            fitness_one genotype, env: env
          end
        end
      else raise ArgumentError, "Unrecognized fit type: `#{type}`"
      end
    end

    # Return the fitness of a single genotype
    # @param genotype the individual to be evaluated
    # @param env the environment to use for the evaluation
    # @param render [bool] whether to render the evaluation on screen
    # @param nsteps [Integer] how many interactions to run with the game. One interaction is one action choosing + enacting and `skip_frames` repetitions
    def fitness_one genotype, env: single_env, render: false, nsteps: max_nsteps
      puts "Evaluating one individual" if debug
      puts "  Loading weights in network" if debug
      net.load_weights genotype # this also resets the state
      observation = env.reset
      env.render if render
      tot_reward = 0
      puts "  Running (max_nsteps: #{max_nsteps})" if debug
      nsteps.times do |i|
        selected_action, normobs, novelty = action_for observation
        # observation, reward, done, info = env.step(selected_action).to_a
        # skip_frames&.times { env.step 0 } # accelerate simulation

        observations, rewards, dones, infos = repeat_action.times.map do
          env.step(selected_action).to_a
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
             "Avg fit: #{opt.last_fits.reduce(:+)/opt.last_fits.size} -- " \
             "Conv: #{opt.convergence}"

        break if termination_criteria&.call(opt)
      end
    end

    # Re-run the highest scoring individual found so far
    # param @until_end [bool] raises the `max_nsteps` to see interaction until `done`
    def show_best until_end: false
      nsteps = until_end ? max_nsteps*1000 : max_nsteps
      fit = fitness_one opt.best.last, render: true, nsteps: nsteps
      puts "Fitness: #{fit}"
    end
  end

end

puts "USAGE: `bundle exec ruby experiments/<expname>.rb`" if __FILE__ == $0
