"""@brief Derive the nonlinear cart-pole state equation symbolically."""

from __future__ import annotations

import sympy as sp

from .symbols import (
    TIME_TO_STATE,
    F,
    J_p,
    b_c,
    ell_p,
    g,
    m_c,
    m_p,
    q_ddot,
    q_ddot_of_t,
    q_dot,
    q_dot_of_t,
    q_of_t,
    t,
    theta_ddot,
    theta_ddot_of_t,
    theta_dot,
    theta_dot_of_t,
    theta_of_t,
)


def _balance_equations() -> tuple[tuple[sp.Expr, ...], tuple[sp.Expr, ...]]:
    """@brief Build scalar force and torque residuals and their unknowns.

    @return Pair of tuples containing the residuals and the solve unknowns.
    """
    # Pivot and normal forces
    T_x, T_y, N = sp.symbols("T_x T_y N", real=True)

    # Inertial frame unit vectors: e_x, e_y, e_z
    e_x = sp.Matrix([1, 0, 0])
    e_y = sp.Matrix([0, 1, 0])
    e_z = sp.Matrix([0, 0, 1])

    # Pole unit vector: e_p
    e_p = sp.sin(theta_of_t) * e_x - sp.cos(theta_of_t) * e_y

    # Cart and pole (COM) position vectors: p_c, p_p
    p_c = q_of_t * e_x
    p_c_ddot = p_c.diff(t, 2)
    p_p = p_c + ell_p * e_p
    p_p_ddot = p_p.diff(t, 2)

    # Pivot force
    pivot_force_on_pole = T_x * e_x + T_y * e_y
    pivot_force_on_cart = -pivot_force_on_pole

    # Force balances
    total_cart_force = (
        (F - b_c * q_dot_of_t) * e_x + (N - m_c * g) * e_y + pivot_force_on_cart
    )
    total_pole_force = (-m_p * g) * e_y + pivot_force_on_pole
    cart_force_residual = total_cart_force - m_c * p_c_ddot
    pole_force_residual = total_pole_force - m_p * p_p_ddot

    # Torque balance
    pivot_arm = p_c - p_p
    pole_torque_residual = (
        pivot_arm.cross(pivot_force_on_pole) - J_p * theta_ddot_of_t * e_z
    )

    # Setup equations to solve
    residuals = (
        cart_force_residual[0],
        cart_force_residual[1],
        pole_force_residual[0],
        pole_force_residual[1],
        pole_torque_residual[2],
    )
    unknowns = (q_ddot_of_t, theta_ddot_of_t, T_x, T_y, N)

    return residuals, unknowns


def derive_state_derivative() -> sp.Matrix:
    """@brief Derive the nonlinear state derivative from the balance equations.

    @return State derivative [q_dot, q_ddot, theta_dot, theta_ddot].
    """
    residuals, unknowns = _balance_equations()
    coefficient_matrix, forcing_vector = sp.linear_eq_to_matrix(
        residuals,
        unknowns,
    )
    solution = coefficient_matrix.LUsolve(forcing_vector)

    # Extract solution
    q_ddot_of_t_rhs = sp.simplify(solution[0])
    theta_ddot_of_t_rhs = sp.simplify(solution[1])
    state_derivative_of_t = sp.Matrix(
        [
            q_dot_of_t,
            q_ddot_of_t_rhs,
            theta_dot_of_t,
            theta_ddot_of_t_rhs,
        ]
    )

    return state_derivative_of_t.subs(TIME_TO_STATE).applyfunc(sp.simplify)


def derive_mass_forcing() -> tuple[sp.Matrix, sp.Matrix]:
    """@brief Eliminate reaction forces to obtain M * acceleration = forcing.

    @return Mass matrix and forcing vector in canonical state symbols.
    """
    residuals, unknowns = _balance_equations()
    q_accel, theta_accel, pivot_x, pivot_y, normal = unknowns

    # Pole force balances and vertical cart balance determine the reactions.
    reaction_matrix, reaction_forcing = sp.linear_eq_to_matrix(
        (residuals[1], residuals[2], residuals[3]),
        (pivot_x, pivot_y, normal),
    )
    reactions = reaction_matrix.LUsolve(reaction_forcing)
    reaction_substitutions = dict(
        zip((pivot_x, pivot_y, normal), reactions, strict=True)
    )

    # Negating both residuals gives the conventional positive mass matrix.
    reduced_residuals = [
        -residuals[0].subs(reaction_substitutions),
        -residuals[4].subs(reaction_substitutions),
    ]
    mass_matrix, forcing = sp.linear_eq_to_matrix(
        reduced_residuals, (q_accel, theta_accel)
    )
    mass_matrix = mass_matrix.subs(TIME_TO_STATE).applyfunc(sp.trigsimp)
    forcing = forcing.subs(TIME_TO_STATE).applyfunc(sp.trigsimp)
    return mass_matrix, forcing


if __name__ == "__main__":
    rhs = derive_state_derivative()
    mass_matrix, forcing = derive_mass_forcing()
    acceleration = sp.Matrix([q_ddot, theta_ddot])
    latex_names = {
        ell_p: r"\ell_{p}",
        q_dot: r"\dot{q}",
        q_ddot: r"\ddot{q}",
        theta_dot: r"\dot{\theta}",
        theta_ddot: r"\ddot{\theta}",
    }

    print("Solved accelerations:")
    for symbol, expression in ((q_ddot, rhs[1]), (theta_ddot, rhs[3])):
        equation = sp.Eq(symbol, expression)
        sp.pprint(equation)
        print("LaTeX:")
        print(sp.latex(equation, symbol_names=latex_names))
        print()

    print("Mass-matrix form, M * a = forcing:")
    print("M =")
    sp.pprint(mass_matrix)
    print("a =")
    sp.pprint(acceleration)
    print("forcing =")
    sp.pprint(forcing)
    print("LaTeX:")
    print(
        f"{sp.latex(mass_matrix, symbol_names=latex_names)} "
        f"{sp.latex(acceleration, symbol_names=latex_names)}"
        f" = {sp.latex(forcing, symbol_names=latex_names)}"
    )
