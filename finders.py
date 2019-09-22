
def find_ψ_p0(ψ0: float=1/2, E: float=-.01) -> float:
    """
    There should be only one bound state corresponding to a combination of ψ0 and E.
    I suspect that it's really only 1 bound state for a given E, where all ψ0 and ψ'0 combinations
    that normalize are equivalent.
    """

    # Use the shooting method to find our solution. Narrow down the range of possible values
    # iteratively.

    START = -10
    END = 10
    check_x = 6000  # Value to confirm soln is close to 0. Higher is more precise.
    ϵ = 1e-4  # Set smaller for higher precision, but isn't as sensitive as check_x.

    MAX_ATTEMPTS = 10000
    MAX_PRECISION = 1e-17  # Just return the result if upper and lower are very close

    lower = START
    upper = END
    guess = (END - START) / 2

    for i in range(MAX_ATTEMPTS):
        soln = hydrogen_static(ψ0, guess, E)

        # ψ should be near 0 for a bound state when far from the origin, represented
        # by check_val
        # check_val = soln.y[int(len(soln.y) / 2)][0]  # todo nope. Find closest to t!
        check_val = soln.y[0][800]  # todo nope. Find closest to t!

        # Debugging code
        # print(f"upper:{upper}, lower:{lower}, guess:{guess}, check: {check_val}")
        # plt.plot(soln.y[0])
        # plt.show()

        if abs(check_val) <= ϵ or abs(upper - lower) <= MAX_PRECISION:
            return guess

        # Assume negative check_val means our guess is too low.
        if check_val < 0:
            # We know the soln is bounded below by lower.
            lower = guess
            guess += (upper - lower) / 2
        else:
            upper = guess
            guess -= (upper - lower) / 2
    return guess


def find_E(ψ0, ψ_p0) -> float:
    """"""
    # Use the shooting method to find our solution. Narrow down the range of possible values
    # iteratively.

    START = -10
    END = 0  # Positive energies mean scattering states.

    CHECK_X = 20  # Value to confirm soln is close to 0. Higher is more precise.
    ϵ = 1e-4  # Set smaller for higher precision, but isn't as sensitive as check_x.

    MAX_ATTEMPTS = 10000
    MAX_PRECISION = 1e-18  # Just return the result if upper and lower are very close

    lower = START
    upper = END
    guess = -(END - START) / 2

    for i in range(MAX_ATTEMPTS):
        soln = hydrogen_static(ψ0, ψ_p0, guess)
        print(soln.y[0], "Y")

        # ψ should be near 0 for a bound state when far from the origin, represented
        # by check_val
        check_val = soln.y[0][np.where(soln.t > CHECK_X)][0]

        # Debugging code
        print(f"upper:{upper}, lower:{lower}, guess:{guess}, check: {check_val}")
        # plt.plot(soln.y[0])
        # plt.show()

        if abs(check_val) <= ϵ or abs(upper - lower) <= MAX_PRECISION:
            return guess

        # Assume negative check_val means our guessed energy is too high.
        # Low e overshoots, high E undershoots.
        if check_val < 0:
            # We know the soln is bounded below by lower.
            upper = guess
            guess -= (upper - lower) / 2
        else:
            lower = guess
            guess += (upper - lower) / 2

    return guess


# def find_ics(E: float) -> Tuple[float, float]:
#     """Return (ψ0, ψ'0) for the normalized bound state at the given energy."""
#     ψ0 = 1  # Arbitrary starting value.
#     # ψ0 = find_ψ0(E)
#     ψ_p0 = find_ψ_p0(ψ0, E)
#
#     soln = hydrogen_static(ψ0, ψ_p0, E)
#     norm = simps(soln.y[0][:500]) * 2
#
#     # Now that we know how to normalize, modify ψ0 appropriately:
#
#     return ψ0 / norm, find_ψ_p0(ψ0/norm, E)


# def plot_ics():
#     """Plot the initial conditions over bound state."""
#     SIZE = 20
#     E = np.linspace(0, -.3, SIZE)
#     ψ0 = np.empty(SIZE)
#     ψ_p0 = np.empty(SIZE)
#
#     for i in range(SIZE):
#         ψ0[i], ψ_p0[i] = find_ics(E[i])
#
#     print(ψ0)
#     print(ψ_p0)
#
#     plt.plot(E, ψ0)
#     plt.show()
#
#     plt.plot(E, ψ_p0)
#     plt.show()