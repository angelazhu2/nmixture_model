
def main() -> None:
    sites = 10
    T = 6
    lam = 2
    p = 0.25
    S = lam * 3
    EPOCHS = 100000

    from method_one import run_method_one
    from method_two import run_method_two
    from method_three import run_method_three
    from method_four import run_method_four
    from method_five import run_method_five


    RANDOM_STATE = 42

    run_method_one(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    run_method_two(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    run_method_three(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    run_method_four(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    run_method_five(sites, T, lam, p, S, EPOCHS, random_state=RANDOM_STATE)
    

if __name__ == "__main__":
    main()