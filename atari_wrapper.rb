require 'numo/narray'

module DNE
  # Convenience wrapper for the Atari OpenAI Gym environments
  class AtariWrapper

    attr_reader :gym_env, :reset_obs, :reset_obs_py, :act_type, :act_size, :downsample, :row_div, :col_div

    extend Forwardable
    def_delegator :@gym_env, :render

    def initialize gym_env, downsample: nil, skip_type: nil
      @skip_type = skip_type
      @downsample = downsample
      @gym_env = gym_env
      @reset_obs = reset
      act_type, act_size = gym_env.action_space.to_s.match(/(.*)\((\d*)\)/).captures
      raise "Not Atari act space" unless act_size == '6' && act_type == 'Discrete'
      @act_type = act_type.downcase.to_sym
      @act_size = Integer(act_size)
    end

    # converts pyimg into NArray
    def to_img pyimg



      ## subtract `reset_obs` to clear background?
      ## pyimg -= reset_obs_py



      # resample to target resolution
      pyimg = pyimg[(0..-1).step(downsample[0]),
                    (0..-1).step(downsample[1])] if downsample
      # average color channels, flatten, convert to NArray
      NImage[*pyimg.mean(2).ravel.tolist.to_a]
    end

    # calls reset, converts to NArray img
    def reset
      @reset_obs_py = gym_env.reset
      to_img reset_obs_py
    end

    def gym_to_rb gym_ans
      obs, rew, done, info = gym_ans.to_a
      [to_img(obs), rew, done, info]
    end

    # executes an action, then frameskips; returns aggregation
    def execute act, skip_frames: 0, skip_type: skip_type
      act_ans = gym_env.step(act)
      skip_ans = skip_frames.times.map do
        SKIP_TYPE[skip_type].call(selected_action)
      end
      all_ans = skip_ans.unshift(act_ans)
      obs_lst, rew_lst, done_lst, info_lst = all_ans.map(&method(:gym_to_rb)).transpose
      rew = rew_lst.reduce :+
      done = done_lst.any?
      [obs_lst, rew, done, info_lst]
    end
  end
end
