require_relative '../gym_experiment'

config = {
  net: {
    type: :Recurrent,
    hidden_layers: [5,5],
    activation_function: :logistic
  },
  env: {
    type: 'aliens-gvgai-v0'
  },
  run: {
    max_nsteps: 500,
    max_ngens: 5,
    termination_criteria: -> (opt) { opt.best.first > -10 },
    random_seed: 1,
    debug: true
  },
  opt: {
    type: :BDNES
  }
}
