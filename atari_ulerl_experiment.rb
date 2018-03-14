require_relative 'gym_experiment'
require_relative 'observation_compressor'

module DNE
  # Specialized GymExperiment class for Atari environments and UL-ERL.
  class AtariUlerlExperiment < GymExperiment

    attr_reader :compr, :resize

    def initialize config
      # compr before super
      # - net.struct => compr.ncentrs
      # super before compr
      # - config[:run][:debug]
      puts "Initializing compressor" # if debug
      @compr = init_compr config.delete :compr
      puts "Loading Atari OpenAI Gym environment" # if debug
      super config
    end

    # Initializes the UL compressor
    def init_compr **compr_opts
      defaults = {
        dtype: :float64, vrange: [0,1],
        orig_size: [210,160] # ALE image size
      }
      DNE::ObservationCompressor.new **defaults.merge(compr_opts)
    end

    # Initializes the Atari environment
    # @note python environments interfaced through pycall
    # @param type [String] the type of environment as understood by OpenAI Gym
    # @return an initialized environment
    def init_env type:, resize_obs: nil
      puts "  initializing env" if debug
      gym.make(type).tap do |gym_env|
        obs_size = gym_env.reset.shape.to_a
        raise "Unrecognized Atari observation space" \
          unless obs_size == [210, 160, 3]
        gym_env.define_singleton_method(:obs_size) { obs_size }
        act_type, act_size = gym_env.action_space.to_s.match(/(.*)\((\d*)\)/).captures
        raise "Unrecognized Atari action space" \
          unless act_size == '6' && act_type == 'Discrete'
        gym_env.define_singleton_method(:act_type) { act_type.downcase.to_sym }
        gym_env.define_singleton_method(:act_size) { Integer(act_size) }
        gym_env.define_singleton_method(:resize_obs) { resize_obs }
      end
    end

    # Initialize the controller
    # @param type [Symbol] name the class of neural network to use (from the WB)
    # @param hidden_layers [Array] list of hidden layer sizes for the networks structure
    # @param activation_function [Symbol] name one of the activation functions available
    # @return an initialized neural network
    def init_net type:, hidden_layers:, activation_function:
      netclass = NN.const_get(type)
      netstruct = [compr.ncentrs, *hidden_layers, single_env.act_size]
      netclass.new netstruct, act_fn: activation_function
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
      compr_train = [nil, -Float::INFINITY]
      puts "  Running (max_nsteps: #{max_nsteps})" if debug
      nsteps.times do |i|
        selected_action, normobs, novelty = action_for observation
        compr_train = [normobs, novelty] if novelty > compr_train.last

        observations, rewards, dones, infos = repeat_action.times.map do
          env.step(selected_action).to_a
        end.transpose
        # NOTE: this blurs the observation. An alternative is to isolate what changes.
        # Actually, should I go for it? Remove the background? But then how do I
        # distinguish the games if I want a multi-games player?
        observation = observations.reduce(:+) / observations.size
        reward = rewards.reduce :+
        done = dones.any?

        tot_reward += reward
        env.render if render
        break if done
      end
      compr.train_set << compr_train.last
      puts "=> Done, fitness: #{tot_reward}" if debug
      tot_reward
    end

    # Return an action for an image observation
    # @note this includes the non-banal sequence of converting the observation to network inputs, activating the network, then interpreting the network output as the corresponding action
    # @param image [numpy array] screenshot from the Atari emulator
    def action_for observation
      # encode with the compressor
      input, novelty = compr.encode observation

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
      [action, input, novelty] # hack: method should just return action
    end
  end

end

puts "USAGE: `bundle exec ruby experiments/atari/<expname>.rb`" if __FILE__ == $0
