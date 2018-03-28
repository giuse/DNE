require 'forwardable'
require 'machine_learning_workbench'

module DNE
  # Wrap a WB compressor for usage on observation in a UL-ERL + PyCall context
  class ObservationCompressor
    # Shorthands
    WB = MachineLearningWorkbench

    extend Forwardable
    def_delegators :@compr, :ncentrs, :centrs, :ntrains

    attr_reader :encoding, :downsample, :downsampled_size, :compr, :train_set
    attr_writer :train_set

    def initialize type:, encoding:, orig_size:, downsample: nil, **compr_args
      @encoding = encoding
      @downsample = downsample || [1] * orig_size.size
      raise ArgumentError, "Only downward scaling" if @downsample.any? { |v| v < 1 }
      @downsampled_size = orig_size.zip(@downsample).map { |s,r| s/r }.reverse
      centr_size = downsampled_size.reduce(:*)
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
    # @return [NArray] the observation in a form ready for processing with the WB
    def normalize observation
      # resample to target resolution
      v_divisor, h_divisor = downsample
      resized = observation[(0..-1).step(v_divisor), (0..-1).step(h_divisor)]
      # avg channels, flatten, normalize => pylist => Array => NArray
      (resized.mean(2).ravel / 255).tolist.to_a.to_na
    end

    # Provide encoding (+ novelty) for a raw observation
    # @param observation [python ndarray] the observation as coming from the
    #   environmnet through PyCall
    # @return [Array<Array<Float>, NArray,  Float>] the encoded observation,
    #   followed by its normalized version and corresponding novelty
    def encode raw_obs
      obs = normalize raw_obs
      code = compr.encode(obs, type: encoding)
      # novelty as reconstruction error
      novelty = (obs - compr.reconstruction(code, type: encoding)).abs.sum
      [code, obs, novelty]
    end

    # Train the compressor on the observations collected so far
    def train

      # TODO: I can save the most similar centroid and corresponding simil in the
      # training set, then the training should be just some sums! Super fast!

      compr.train train_set
      @train_set = []
    end

    # TODO: should we move some of this in VQ? I think so...

    # Show a centroid using ImageMagick
    def show_centroid idx, disp_size: [300,300]
      require 'rmagick'
      WB::Tools::Imaging.display centrs[idx], shape: downsampled_size, disp_size: disp_size
    end

    # Show all centroids using ImageMagick
    def show_centroids disp_size: [300,300]
      centrs.size.times &method(:show_centroid)
      puts "#{centrs.size} centroids displayed"
      puts "Hit return to close them"
      gets
      nil
    ensure
      WB::Tools::Execution.kill_forks
    end
  end
end
