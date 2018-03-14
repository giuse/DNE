require 'forwardable'
require 'machine_learning_workbench'

module DNE
  class Compressor
    # Shorthands
    WB = MachineLearningWorkbench
    VQ = WB::Compressor::VectorQuantization

    extend Forwardable
    def_delegator :@vq, :ncentrs

    attr_reader :vq, :encoding, :train_set, :nov_thold

    def initialize type:, encoding:, **vq_args
      raise NotImplementedError "Y U NOT VQ" unless type == :VectorQuantization
      @encoding = encoding
      @vq = VQ.new **vq_args
      @train_set = []
    end

    def train
      vq.train train_set
      @train_set = []
    end

    def encode img
      code = vq.encode(img, type: encoding)
      # novelty as reconstruction error
      novelty = (img - vq.reconstruction(code, type: encoding)).absolute_sum
      [code, novelty]
    end

  end
end
