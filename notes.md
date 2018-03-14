
# Atari

## Non-discrete environments
- 'Pendulum-v0' # env.action_space => Box(1,)

## Evaluation
- Individual evaluation looks to be non-deterministic, i.e. reproducing the behavior of one individual yields different outcomes on multiple runs. This seems to be linked to the random initialization of the environment, e.g. enemies spawn at random locations. `AtariEnv#reset` does not include an optional parameter for the random seed.
