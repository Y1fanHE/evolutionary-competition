"""
Example Code for Evolutionary Competition
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
from evocomp.evo import de_rand_1_bin, particle_swarm_optimization
from evocomp.comp import Competition
from evocomp.comp.metrics import wasserstein_distance
from evocomp.comp.samplers import last_evaluations
from evocomp.comp.gp import simple


de1 = (
    de_rand_1_bin,                # de
    dict(n_eval = 2000,           # parameters
         n_pop  = 20,
         F      = 0.5,
         CR     = 0.5),
    "DE"                          # alias
)

de2 = (
    de_rand_1_bin,                # de
    dict(n_eval = 2000,           # parameters
         n_pop  = 20,
         F      = 0.1,
         CR     = 0.9),
    "PSO"                         # alias
)

de1_test = (
    de_rand_1_bin,                # de
    dict(n_eval = 40000,          # parameters
         n_pop  = 20,
         F      = 0.5,
         CR     = 0.5),
    "DE1"                         # alias
)

de2_test = (
    de_rand_1_bin,                # de
    dict(n_eval = 40000,          # parameters
         n_pop  = 20,
         F      = 0.1,
         CR     = 0.5),
    "DE2"                        # alias
)


if __name__ == "__main__":

    comp = Competition(
        metric      = wasserstein_distance, # distance metric
        algorithm1  = de1,                  # two eas
        algorithm2  = de2,
        sampler     = (last_evaluations, [20]),
        lower_bound = -5,                   # bounds of search space
        upper_bound = 5,
        dimension   = 2,                    # dimension of search space
        repetition  = 1,                    # repetition of ea runs
        mode        = "differ",             # mode: differ/match
        seed        = 1000
    )

    comp.evolve(                            # gp parameters
        method          = simple,
        population_size = 20,
        max_generation  = 6,
        tournament_size = 7,
        crossover_rate  = 0.8,
        mutation_rate   = 0.1,
        parallelism     = 5,
        seed            = 318,
        verbose         = True
    )

    comp.plot_space(                        # contor plot
        target     = "train.png",
        algorithm1 = de1,
        algorithm2 = de2,
        sampler    = (last_evaluations, [20]),
        seed       = 1000
    )

    comp.plot_space(                        # contor plot
        target     = "train_all.png",
        algorithm1 = de1,
        algorithm2 = de2,
        seed       = 1000
    )

    comp.plot_space(                        # contor plot
        target     = "test.png",
        algorithm1 = de1_test,
        algorithm2 = de2_test,
        sampler    = (last_evaluations, [20]),
        seed       = 1000
    )

    comp.plot_space(                        # contor plot
        target     = "test_all.png",
        algorithm1 = de1_test,
        algorithm2 = de2_test,
        seed       = 1000
    )

    comp.plot_tree(target="out_tree.png")   # tree plot
    comp.save(target="out.sol")             # text file
