"""
created by Yifan He (heyif@outlook.com)
on Sept. 12, 2023
"""
from ea import de_rand_1_bin, particle_swarm_optimization
from comp.metrics import wasserstein_distance
from comp import EvolvedCompetition as EvoComp


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

    ec = EvoComp(
        metric      = wasserstein_distance,
        algorithm1  = de,
        algorithm2  = pso,
        lower_bound = -5,
        upper_bound = 5,
        dimension   = 2,
        repetition  = 3
    )

    ec.run(
        population_size = 200,
        max_generation  = 50,
        tournament_size = 7,
        crossover_rate  = 0.7,
        mutation_rate   = 0.1,
        seed            = 1995
    )

    ec.plot(target = "out.pdf")
    ec.save(target = "out.sol")
