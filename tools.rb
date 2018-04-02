
module DNE
  # Tools to support debug and implementation
  module Tools
    WB = MachineLearningWorkbench
    def self.show_image img
      shape = [80,70]
      to_disp = case img
      when Numo::NArray
        WB::Tools::Normalization.feature_scaling img,
          from: [img.class::MIN, img.class::MAX], to: [0,1]
      else # assume python image
        shape = [160, 210]
        NImage[*img.mean(2).ravel.tolist.to_a]
      end
      WB::Tools::Imaging.display to_disp,
        shape: shape, disp_size: [300,300]
    end
  end
end
