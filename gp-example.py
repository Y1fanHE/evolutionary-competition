"""
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
from ea import de_rand_1_bin, particle_swarm_optimization
from comp.metrics import wasserstein_distance
from comp import Competition


de = (
    de_rand_1_bin,                # de
    dict(n_eval = 2000,           # parameters
         n_pop  = 20,
         F      = 0.5,
         CR     = 0.5),
    "DE"                          # alias
)

pso = (
    particle_swarm_optimization,  # pso
    dict(n_eval = 2000,           # parameters
         n_pop  = 20,
         w1     = 0.5,
         w2     = 0.5,
         c      = 0.1),
    "PSO"                         # alias
)


if __name__ == "__main__":

    comp = Competition(
        metric      = wasserstein_distance,
        algorithm1  = de,
        algorithm2  = pso,
        lower_bound = -5,
        upper_bound = 5,
        dimension   = 2,
        repetition  = 1
    )

    comp.evolve(
        population_size = 200,
        max_generation  = 50,
        tournament_size = 7,
        crossover_rate  = 0.8,
        mutation_rate   = 0.1,
        seed            = 318,
        verbose         = True
    )

    comp.plot_space(target="out.pdf")
    comp.plot_tree(target="out_tree.pdf")
    comp.save(target="out.sol")
