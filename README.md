# Stanford CS229 Project

The `run.sh` file is a single command to run all logic.
Its contents show that we take a 5 step approach. First,
we run the ideal robot to understand its trajectories.
Then we run 6 real robots with the same wheel-frame controls
to see how significantly they deviate. Then we generate the dataset
for machine learning. Then we perform system identification using
linear regression / stochastic gradient descent. Finally, we
train a neural network as an alternative to the previous step.

The `.py` files mentioned in `run.sh` are the top-level files.
Other files provide modules to support these.

Running `run.sh` will create a `Results/` directory, with
sub-directories for each of the above steps. A huge amount of
(interim) result data is generated. Beware!
