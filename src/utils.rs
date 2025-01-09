#![allow(dead_code)]
#![allow(non_snake_case)]

extern crate nalgebra as na;
use std::f64::consts::PI;

use multibody_dynamics::math_functions::skew;
use na::{
    Matrix3, Matrix6, Quaternion, SMatrix, SVector, UnitQuaternion, Vector3, Vector4, Vector6,
};

// use crate::Config;
use crate::bluerov_reach::BlueROVReachConfig;

/// Computes the added mass matrix of a slender body.
pub fn slendermasss(
    length: f64,
    radius: f64,
    fluid_density: f64,
    alpha: Option<f64>,
    coeff_added: Option<f64>, // Added mass coefficient, rescales the entire added mass matrix
) -> Matrix6<f64> {
    let mut fluid_added_mass = Matrix6::zeros();
    match alpha {
        Some(alpha) => {
            fluid_added_mass[(0, 0)] = alpha * length;
        }
        None => {
            fluid_added_mass[(0, 0)] = 0.0;
        }
    }

    fluid_added_mass[(1, 1)] = length;
    fluid_added_mass[(2, 2)] = length;
    fluid_added_mass[(4, 4)] = length.powi(3) / 3.0;
    fluid_added_mass[(5, 5)] = length.powi(3) / 3.0;
    fluid_added_mass[(2, 4)] = -length.powi(2) / 2.0;
    fluid_added_mass[(4, 2)] = -length.powi(2) / 2.0;
    fluid_added_mass[(1, 5)] = length.powi(2) / 2.0;
    fluid_added_mass[(5, 1)] = length.powi(2) / 2.0;

    // If no added mass coefficient is provided, set it to 1.0
    let coefficient = coeff_added.unwrap_or(1.0);

    fluid_added_mass * fluid_density * PI * radius.powi(2) * coefficient
}

pub fn comp_rb_inertia_cylinder(
    r_cog: &Vector3<f64>,
    radius: f64,
    length: f64,
    mass: f64,
) -> Matrix3<f64> {
    let mut I_r = Matrix3::zeros();

    let d = r_cog[2];
    let m_r = d.abs() / radius * mass;
    let m_c = mass - m_r;

    let I_c = m_c
        * Matrix3::from_diagonal(&Vector3::new(
            radius.powi(2) / 2.0,
            radius.powi(2) / 4.0 + length.powi(2) / 3.0,
            radius.powi(2) / 4.0 + length.powi(2) / 3.0,
        ));

    I_r[(0, 0)] = m_r * d.powi(2);
    I_r[(1, 1)] = m_r * (d.powi(2) + length.powi(2) / 3.0);
    I_r[(2, 2)] = m_r * (length.powi(2) / 3.0);

    I_r[(0, 2)] = -m_r * 0.5 * length * d;
    I_r[(2, 0)] = -m_r * 0.5 * length * d;

    I_r + I_c
}

pub fn comp_rb_inertia_rectangular_cuboid(
    length: &Vector3<f64>,
    pos_cog: &Vector3<f64>,
    mass: f64,
) -> Matrix3<f64> {
    let inertia_mat = 1.0 / 12.0
        * mass
        * Matrix3::from_diagonal(&Vector3::new(
            length[1].powi(2) + length[2].powi(2),
            length[0].powi(2) + length[2].powi(2),
            length[0].powi(2) + length[1].powi(2),
        ));
    inertia_mat - skew(pos_cog) * skew(pos_cog) * mass
}

/// Computes the thruster configuration matrix T: \tau = T f, where f contains thruster forces and \tau is the resulting generalized forces.
pub fn comp_tcm<const NUM_DOFS: usize, const NUM_THRUSTERS: usize>(
    cfg: &BlueROVReachConfig,
    jacs: &[SMatrix<f64, 6, NUM_DOFS>],
) -> SMatrix<f64, NUM_DOFS, NUM_THRUSTERS> {
    let num_thrusters = cfg.thruster_dirs.len();
    let mut tcm = SMatrix::<f64, NUM_DOFS, NUM_THRUSTERS>::zeros();

    // let lambda = |x: usize| -> i32 { cfg.thruster_parents[x] as i32 - 1 };
    let lambda = |_x: usize| -> i32 { 0 };

    for i in 0..num_thrusters {
        let mut B_i = Vector6::<f64>::zeros();
        B_i.fixed_view_mut::<3, 1>(0, 0)
            .copy_from(&cfg.thruster_dirs[i]);
        B_i.fixed_view_mut::<3, 1>(3, 0)
            .copy_from(&cfg.thruster_pos_offsets[i].cross(&cfg.thruster_dirs[i]));
        let col = jacs[lambda(i) as usize].transpose() * B_i;
        tcm.fixed_columns_mut(i).copy_from(&col);
    }
    tcm
}

/// Computes the drag forces and torques acting on a box-shaped rigid body, e.g. a BlueROV.
pub fn box_shaped_drag_rb(
    nu: &Vector6<f64>,
    mu: &Vector6<f64>,
    cfg: &BlueROVReachConfig,
) -> Vector6<f64> {
    Vector6::zeros()
}

pub fn discrete_quat_update(
    quat: &UnitQuaternion<f64>,
    omega: &Vector3<f64>,
) -> UnitQuaternion<f64> {
    let omg_norm = omega.norm();
    let delta_q: UnitQuaternion<f64> = if omg_norm > 1e-8 {
        let delta_epsilon = f64::sin(0.5 * omega.norm()) * omega / omega.norm();
        UnitQuaternion::from_quaternion(Quaternion::new(
            f64::cos(0.5 * omega.norm()),
            delta_epsilon[0],
            delta_epsilon[1],
            delta_epsilon[2],
        ))
    } else {
        let quat_vec = (0.5 - 1.0 / 48.0 * f64::powi(omega.norm(), 2)) * omega;
        UnitQuaternion::from_quaternion(Quaternion::new(
            1.0 - 1.0 / 8.0 * f64::powi(omega.norm(), 2) + 1.0 / 384.0 * f64::powi(omega.norm(), 4),
            quat_vec[0],
            quat_vec[1],
            quat_vec[2],
        ))
    };
    quat * delta_q
}

/// Integrate a function `f` from `a` to `b` using the [trapezoid rule](https://en.wikipedia.org/wiki/Trapezoidal_rule) with `n` partitions.
pub fn trapz<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let dx: f64 = (b - a) / (n as f64);
    dx * ((1..n).map(|k| f(a + k as f64 * dx)).sum::<f64>() + (f(b) + f(a)) / 2.)
}

pub fn trapz_vec<F>(f: F, a: f64, b: f64, n: usize) -> Vector4<f64>
where
    F: Fn(f64) -> Vector4<f64>,
{
    let mut out = Vector4::<f64>::zeros();
    let dx: f64 = (b - a) / (n as f64);
    for i in 0..4 {
        out[i] =
            dx * ((1..n).map(|k| f(a + k as f64 * dx)[i]).sum::<f64>() + (f(b) + f(a))[i] / 2.);
    }
    out
}

pub fn trans_mat_quat_dot(quat: &UnitQuaternion<f64>) -> SMatrix<f64, 4, 3> {
    let mut out = SMatrix::<f64, 4, 3>::zeros();
    let quat_vec = quat.vector();

    out.fixed_rows_mut(0).copy_from(&-quat_vec.transpose());
    out.fixed_rows_mut::<3>(1)
        .copy_from(&(quat.w * Matrix3::identity() + skew(&quat_vec.into())));

    0.5 * out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slendermasss() {
        let length = 1.0;
        let radius = 0.1;
        let fluid_density = 1000.0;
        let alpha = 0.5;
        let coeff_added = 2.0;
        let fluid_added_mass = slendermasss(
            length,
            radius,
            fluid_density,
            Some(alpha),
            Some(coeff_added),
            // None,
        );
        println!("{}", fluid_added_mass);
    }

    #[test]
    fn test_comp_rb_mass_rotational() {
        let r_cog = Vector3::new(0.0, 0.0, 0.0);
        let radius = 0.1;
        let length = 1.0;
        let mass = 1.0;
        let I_r = comp_rb_mass_rotational(r_cog, radius, length, mass);

        assert_eq!(I_r[(0, 0)], 0.005000000000000001);
        assert_eq!(I_r[(1, 1)], 0.3358333333333333);
        assert_eq!(I_r[(2, 2)], 0.3358333333333333);
        assert_eq!(I_r[(0, 2)], 0.0);
        assert_eq!(I_r[(2, 0)], 0.0);
    }

    #[test]
    fn test_comp_tcm() {
        let mut cfg = Config::default();
        cfg.thruster_dirs = vec![Vector3::new(0.0, 0.0, 1.0), Vector3::new(0.0, 1.0, 0.0)];
        cfg.thruster_pos_offsets = vec![Vector3::new(0.24, 0.0, 0.0), Vector3::new(0.35, 0.0, 0.0)];
        cfg.thruster_parents = vec![1, 1];
        let jacs = vec![SMatrix::<f64, 6, 6>::identity()];
        let tcm = comp_tcm::<6, 2>(&cfg, &jacs);
        println!("{}", tcm);
    }

    #[test]
    fn test_transmat() {
        let quat = UnitQuaternion::from_quaternion(Quaternion::new(
            0.533215448243828,
            0.592817248117098,
            0.083109566226999,
            0.597780725760344,
        ));
        let res = trans_mat_quat_dot(&quat);
        println!("res: {}", res);
    }
}
