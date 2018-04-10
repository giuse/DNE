require 'forwardable'
require 'machine_learning_workbench'

module DNE
  # Wrap a WB compressor for usage on observation in a UL-ERL + PyCall context
  class ObservationCompressor

    extend Forwardable
    def_delegators :@compr, :ncentrs, :centrs, :ntrains, :encoding

    attr_reader :downsample, :downsampled_size, :compr, :train_set, :obs_range

    def initialize type:, orig_size:, obs_range:, downsample: nil, **compr_args
      @obs_range = obs_range
      @downsample = downsample
      raise ArgumentError, "Only downward scaling" if downsample.any? { |v| v < 1 }
      @downsampled_size = orig_size.zip(downsample).map { |s,r| s/r }
      centr_size = downsampled_size.reduce(:*)
      compr_class = begin
        WB::Compressor.const_get(type)
      rescue NameError => err
        raise ArgumentError, "Unrecognized compressor type `#{type}`"
      end

      @compr = compr_class.new **compr_args.merge({dims: centr_size})
      @train_set = []
    end

    # Reset the centroids to something else than random noise
    # TODO: refactor
    def reset_centrs img, proport: nil
      img = normalize img if img.kind_of? NImage
      compr.init_centrs base: img, proport: proport
    end

    # Normalize an observation into a form preferable for the WB compressor
    # @param observation [NImage] an observation coming from the environment
    #   through `AtariWrapper` (already resampled and converted to NArray)
    # @return [NArray] the normalized observation ready for processing
    def normalize observation
      WB::Tools::Normalization.feature_scaling observation,
        from: obs_range, to: compr.vrange
        # from: [observation.class::MIN, observation.class::MAX], to: compr.vrange
    end

    # Encodes an observation using the underlying compressor
    # @param observation [NArray]
    # @return [NArray] encoded observation
    def encode obs
      obs = normalize(obs) if obs.kind_of? NImage
      compr.encode obs
    end

    # Compute the novelty of an observation as reconstruction error
    # @param observation [NArray]
    # @param code [Array]
    # @return [Float] novelty score
    def novelty obs, code
      obs = normalize(obs) if obs.kind_of? NImage
      compr.reconstr_error obs, code: code
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
      WB::Tools::Imaging.display centrs[idx], shape: downsampled_size.reverse, disp_size: disp_size
    end

    # Show all centroids using ImageMagick
    def show_centroids disp_size: [300,300], wait: true
      centrs.size.times &method(:show_centroid)
      puts "#{centrs.size} centroids displayed"
      if wait
        puts "Hit return to close them"
        gets
      end
      nil
    ensure
      WB::Tools::Execution.kill_forks
    end
  end
end
