
module DNE
  # Tools to support debug and implementation.
  # I just want to be able to call `show_image` anywhere in the code and
  # be presented with something sensible.
  module Tools
    def self.show_image img
      shape = nil
      to_disp = case img
      when NArray, NImage
        img = img.cast_to NArray # in case it's NImage, need floats for the scaling
        shape = [80, 70] # BEWARE of hardcoded values
        WB::Tools::Normalization.feature_scaling img, to: [0,1]
      else # assume python image
        shape = [160, 210] # BEWARE of hardcoded values
        NImage[*img.mean(2).ravel.tolist.to_a]
      end
      WB::Tools::Imaging.display to_disp,
        shape: shape, disp_size: [300,300]
    end
  end
end
