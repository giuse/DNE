require 'machine_learning_workbench'  # https://github.com/giuse/machine_learning_workbench/
require 'pycall/import'               # https://github.com/mrkn/pycall.rb/
require 'parallel'                    # https://github.com/grosser/parallel

# Shorthands
WB = MachineLearningWorkbench
NES = WB::Optimizer::NaturalEvolutionStrategies
NN = WB::NeuralNetwork

# Loads an experiment description from a config hash, initialize everything and run it
class GymExperiment
  include PyCall::Import

  attr_reader :config, :single_env, :net, :opt, :parall_envs, :max_nsteps, :max_ngens,
    :termination_criteria, :random_seed, :debug

  def initialize config
    @config = config
    # TODO: I really don't like these lines below...
    @max_nsteps = config[:run][:max_nsteps]
    @max_ngens = config[:run][:max_ngens]
    @termination_criteria = config[:run][:termination_criteria]
    @random_seed = config[:run][:random_seed]
    @debug = config[:run][:debug]
    puts "Initializing single env" if debug
    @single_env = init_env config[:env] # one put aside for easy access
    puts "Initializing network" if debug
    @net = init_net config[:net]
    puts "Initializing optimizer" if debug
    @opt = init_opt config[:opt]
    puts "Initializing parallel environments" if debug
    @parall_envs = opt.popsize.times.map { init_env config[:env] } # these for fitness computation
    puts "Initialization complete" if debug
  end

  # Initializes the environment
  # @note python environments interfaced through pycall
  # @param type [String] the type of environment as understood by OpenAI Gym
  # @return an initialized environment
  def init_env type:
    begin
      pyimport :gym        # adds the OpenAI Gym environment
      pyimport :gym_gvgai  # adds the GVGAI environment from NYU
    rescue PyCall::PyError => err
      puts "Error importing python environment: #{err.message}"
    end
    puts "  initializing env" if debug

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
    NES.const_get(type).new dims, method(:fitness_all), :max, parallel_fit: true, rseed: random_seed
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
    # TODO: should normalize here
    output = net.activate input
    # TODO: Sometimes the output is NaN...? Find out what's going on, then get rid of begin/rescue
    begin
      action = output.index output.max
    rescue ArgumentError, Parallel::UndumpableException
      output.map! { |out| out = -Float::INFINITY if out.nan? }
      action = output.index output.max
    end
  end

  # Return a list of fitnesses for a list of genotypes
  # @param genotypes [Array<gtype>] list of genotypes
  # @return [Array<Numeric>] list of fitnesses for each genotype
  def fitness_all genotypes
    ## SEQUENTIAL ON SINGLE ENVIRONMENT     => useful to catch problems with parallel env spawning
    # genotypes.map &method(:fitness_one)
    ## SEQUENTIAL ON MULTIPLE ENVIRONMENTS  => `Parallel` can make debugging hard
    # genotypes.zip(parall_envs).map { |genotype, env| fitness_one genotype, env: env }
    ## PARALLEL ON MULTIPLE ENVIRONMENTS    => should do that if you can
    Parallel.map(genotypes.zip(parall_envs)) { |genotype, env| fitness_one genotype, env: env }
  end

  # Return the fitness of a single genotype
  # @param genotype the individual to be evaluated
  # @param env the environment to use for the evaluation
  # @param render [bool] whether to render the evaluation on screen
  def fitness_one genotype, env: single_env, render: false
    net.load_weights genotype # this also resets the state
    observation = env.reset
    tot_reward = 0
    max_nsteps.times do
      env.render if render
      selected_action = action_for observation
      observation, reward, done, info = env.step(selected_action).to_a
      tot_reward += reward
      break if done
    end
    tot_reward #.tap &method(:puts)
  end

  # Run the experiment
  def run
    max_ngens.times do |i|
      print "Gen #{i+1}: "
      opt.train
      puts "Best fit so far: #{opt.best.first} -- " \
           "Avg fit: #{opt.last_fits.reduce(:+)/opt.last_fits.size} -- " \
           "Conv: #{opt.convergence}"

      break if termination_criteria&.call(opt)
    end
    puts "=> Run complete"
  end

  # Re-run the highest scoring individual found so far
  def show_best
    fit = fitness_one opt.best.last, render: true
    puts "Fitness: #{fit}"
  end
end

puts "USAGE: `bundle exec ruby experiments/<expname>.rb`" if __FILE__ == $0
