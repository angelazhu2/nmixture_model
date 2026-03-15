
import numpy as np

def main() -> None:
    sites = 20
    T = 6
    lam = 5
    p = 0.25
    S = lam * 3
    EPOCHS = 10

    from method_one import run_method_one
    from method_two import run_method_two
    from method_three import run_method_three
    from method_four import run_method_four
    from method_five import run_method_five

    RANDOM_STATE = 42

    run_method_three(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)

    # t_vals = [2, 6, 12, 20]
    # p_vals = [0.1, 0.25, 0.5, 0.75]
    # lam_vals = [2, 5, 15, 30]
    # site_vals = [5, 10, 20, 40, 100]
    # for sites in site_vals:
    #     # run_method_one(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    #     # run_method_two(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    #     # run_method_three(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    #     run_method_four(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    #     # run_method_five(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
        

if __name__ == "__main__":
    main()