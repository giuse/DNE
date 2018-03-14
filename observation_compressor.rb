require 'forwardable'
require 'machine_learning_workbench'

module DNE
  # Wrap a WB compressor for usage on observation in a UL-ERL + PyCall context
  class ObservationCompressor
    # Shorthands
    WB = MachineLearningWorkbench

    extend Forwardable
    def_delegators :@compr, :ncentrs, :dtype

    attr_reader :encoding, :downsample, :compr, :train_set

    def initialize type:, encoding:, orig_size:, downsample: nil, **compr_args
      @encoding = encoding
      @downsample = downsample || [1] * orig_size.size
      raise ArgumentError, "Only downward scaling" if @downsample.any? { |v| v < 1 }
      centr_size = orig_size.zip(@downsample).map { |s,r| s/r }.reduce(:*)
      compr_class = begin
        WB::Compressor.const_get(type)
      rescue NameError => err
        raise ArgumentError, "Unrecognized compressor type `#{type}`"
      end
      @compr = compr_class.new **compr_args.merge({dims: centr_size})
      @train_set = []
    end

    # Massage an observation into a form suitable for the WB compressor
    # TODO: normalization range depends on `act_fn` slope!!
    # TODO: this should be heavily optimized, doing everything in Python would be
    #   oh-so-much quicker :(
    # @param observation [python ndarray] the observation as coming from the
    #   environmnet through PyCall
    # @return [NMatrix] the observation in a form ready for processing with the WB
    def normalize observation
      # resample to target resolution
      v_divisor, h_divisor = downsample
      resized = observation[(0..-1).step(v_divisor), (0..-1).step(h_divisor)]
      # avg channels, flatten, normalize => pylist => Array => NMatrix
      (resized.mean(2).ravel / 255).tolist.to_a.to_nm(nil, compr.dtype)
    end

    # Provide encoding (+ novelty) for a raw observation
    # @param observation [python ndarray] the observation as coming from the
    #   environmnet through PyCall
    # @return [Array<NMatrix, Float>] the pair of encoded observation and its novelty
    def encode raw_obs
      obs = normalize raw_obs
      code = compr.encode(obs, type: encoding)
      # novelty as reconstruction error
      novelty = (obs - compr.reconstruction(code, type: encoding)).absolute_sum
      [code, novelty]
    end

    def train
      compr.train train_set
      @train_set = []
    end
  end
end
