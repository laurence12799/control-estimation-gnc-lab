"""@brief Shared symbolic vocabulary for cart-pole derivation work."""

import sympy as sp

# Define time symbol
t = sp.symbols("t", real=True)

# Time-dependent coordinates used while deriving kinematics.
q_of_t = sp.Function("q", real=True)(t)
theta_of_t = sp.Function("theta", real=True)(t)
q_dot_of_t = sp.diff(q_of_t, t)
theta_dot_of_t = sp.diff(theta_of_t, t)
q_ddot_of_t = sp.diff(q_of_t, t, 2)
theta_ddot_of_t = sp.diff(theta_of_t, t, 2)

# Canonical state and input convention:
# x = [q, q_dot, theta, theta_dot]^T
# u = [F]^T
# q       : cart position [m], positive right
# q_dot   : cart velocity [m/s]
# q_ddot  : cart acceleration [m/s^2]
# theta   : pole angle [rad], zero at hanging down
# theta_dot : pole angular rate [rad/s]
# theta_ddot : pole angular acceleration [rad/s^2]
# F       : horizontal cart force [N], positive right
# SymPy recognizes dot/ddot suffixes when formatting its symbol names.
q, q_dot, q_ddot, theta, theta_dot, theta_ddot, F = sp.symbols(
    "q qdot qddot theta thetadot thetaddot F",
    real=True,
)

# Parameter convention:
# m_c: cart mass [kg]
# m_p: pole mass [kg]
# ell_p: pivot-to-pole-COM distance [m]
# J_p: pole inertia about COM [kg*m^2]
# b_c: cart damping coefficient [N*s/m]
# g: gravity [m/s^2]
m_c, m_p, ell_p, J_p, g = sp.symbols(
    "m_c m_p ℓ_p J_p g",
    positive=True,
    real=True,
)
b_c = sp.symbols("b_c", nonnegative=True, real=True)

x = sp.Matrix([q, q_dot, theta, theta_dot])

u = sp.Matrix([F])

p = sp.Matrix([m_c, m_p, ell_p, J_p, b_c, g])

# Conversion after differentiating the vector kinematics.
TIME_TO_STATE = {
    q_of_t: q,
    q_dot_of_t: q_dot,
    q_ddot_of_t: q_ddot,
    theta_of_t: theta,
    theta_dot_of_t: theta_dot,
    theta_ddot_of_t: theta_ddot,
}
