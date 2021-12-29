local defaults = import "./defaults.libsonnet";

{
    ## this is the normal configuration
    # to set the seed for Python's random, numpy.random and torch.random
    # Set to an int to run seed_everything with this value before classes instantiation
    # (type: Optional[int], default: null)
    "seed_everything": defaults.seed_everything,
}