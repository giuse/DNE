
def test_pycall_gym
  ENV["PYTHON"] = `which python3`.strip   # set python3 path for PyCall
  require 'pycall/import'                 # https://github.com/mrkn/pycall.rb/

  include PyCall::Import
  pyimport :gym
  env = gym.make('CartPole-v1')

  nsteps = 100
  env.reset
  env.render
  nsteps.times do |i|
    selected_action = env.action_space.sample
    env.step(selected_action)
    env.render
  end
end

puts "Choose your test: [a,b]"
case gets.strip
when 'a'
  puts "The test works fine by itself"
  test_pycall_gym
  puts "and if I later require `numo/narray`, everything is fine"
  require 'numo/narray'
  puts "I can run the test again with no problem"
  test_pycall_gym
when 'b'
  puts "Requiring `'numo/narray'` before the first `pyimport :gym` makes PyCall crash"
  require 'numo/narray'
  test_pycall_gym
else
  puts "Please choose between 'a' and 'b' next time."
end
