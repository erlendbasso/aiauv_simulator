// use core::num;
use std::f64::consts::PI;

use multibody_dynamics::multibody::{Axis, JointType};
use serde::{Deserialize, Deserializer, Serialize};
use serde_yaml::{self};

use multibody_dynamics::multibody::MultiBody;

extern crate nalgebra as na;
use na::{
    stack, vector, Isometry3, Matrix3, Matrix6, Quaternion, SMatrix, SVector, Translation3,
    UnitQuaternion, Vector3, Vector4, Vector6,
};

mod utils;
use crate::utils::*;

use ode_solvers::*;

use std::{fs::File, io::BufWriter, io::Write, path::Path};

type State = SVector<f64, 43>;
type Time = f64;

#[derive(Serialize, Deserialize, Debug, Clone)]
enum SerdeAxis {
    X,
    Y,
    Z,
}

// Add conversion between SerdeAxis and multibody_dynamics::multibody::Axis
impl From<SerdeAxis> for Axis {
    fn from(axis: SerdeAxis) -> Self {
        match axis {
            SerdeAxis::X => Axis::X,
            SerdeAxis::Y => Axis::Y,
            SerdeAxis::Z => Axis::Z,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(tag = "type", content = "axis")]
enum SerdeJointType {
    Revolute(SerdeAxis),
    Prismatic(SerdeAxis),
    #[serde(rename = "SixDOF")]
    SixDOF,
}

impl From<SerdeJointType> for JointType {
    fn from(joint_type: SerdeJointType) -> Self {
        match joint_type {
            SerdeJointType::Revolute(axis) => JointType::Revolute(axis.into()),
            SerdeJointType::Prismatic(axis) => JointType::Prismatic(axis.into()),
            SerdeJointType::SixDOF => JointType::SixDOF,
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
pub struct Config {
    sim_time: f64,
    gravity: Vector3<f64>,
    dragcoeffs: Vec<Vector6<f64>>,
    #[serde(deserialize_with = "vec_joint_type")]
    joint_types: Vec<JointType>,
    #[serde(default)]
    mass: Vec<f64>,
    radius: Vec<f64>,
    length: Vec<f64>,
    fluid_density: f64,
    parents: Vec<u16>,
    /// Position of the center of gravity of each link, expressed in the link frame.
    pos_com: Vec<Vector3<f64>>,
    /// Position of the center of buoyancy of each link, expressed in the link frame.
    pos_cob: Vec<Vector3<f64>>,
    pos_offsets: Vec<Vector3<f64>>,
    roll_pitch_yaw_offsets: Vec<Vector3<f64>>,
    thruster_pos_offsets: Vec<Vector3<f64>>,
    thruster_dirs: Vec<Vector3<f64>>,
    thruster_parents: Vec<u16>,
    #[serde(default)]
    added_mass_coeffs: Vec<Option<f64>>,
    // fluid_added_mass: Vec<Matrix6<f64>>,
    added_alpha: Vec<f64>,
}

pub struct AIAUV {
    multibody: MultiBody<9, 14>,
    config: Config,
}

impl ode_solvers::System<f64, State> for AIAUV {
    fn system(&self, _t: Time, y: &State, dy: &mut State) {
        let quat = UnitQuaternion::from_quaternion(Quaternion::from_parts(
            y[3],
            Vector3::new(y[4], y[5], y[6]),
        ));
        // implement your controller here
        let pos = y.fixed_rows::<3>(0);

        let theta = y.fixed_rows::<8>(7).into(); // joint angles
        let zeta: SVector<f64, 14> = y.fixed_rows::<14>(15).into(); // joint velocities
        let z_b = y.fixed_rows::<6>(29); // integral state
        let z_j = y.fixed_rows::<8>(35); // integral theta state

        let nu_b = zeta.fixed_rows::<6>(0); // base velocities
        let theta_dot = zeta.fixed_rows::<8>(6); // joint velocities
        let lin_vel_current = Vector3::<f64>::zeros();
        let lin_accel_current = Vector3::<f64>::zeros();
        // let eta: SVector<f64, 14>;

        let pos_d = Vector3::<f64>::zeros();
        let quat_d = UnitQuaternion::identity();

        let quat_e = quat_d.inverse() * quat;

        let pos_e = quat.inverse() * (pos - pos_d);

        let k_p_b: Vector6<f64> = 1.0 * Vector6::new(70.0, 70.0, 70.0, 100.0, 500.0, 500.0);
        let k_i_b: Vector6<f64> = 0.01 * Vector6::new(10.0, 10.0, 10.0, 10.0, 20.0, 20.0);
        let k_d_b: Vector6<f64> = 0.2 * Vector6::new(100.0, 100.0, 100.0, 50.0, 150.0, 150.0);

        let k_p_j: SVector<f64, 8> =
            0.5 * vector![250.0, 250.0, 500.0, 500.0, 500.0, 500.0, 250.0, 250.0];
        let k_i_j: SVector<f64, 8> = 0.01 * vector![10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 10.0, 10.0];
        let k_d_j: SVector<f64, 8> =
            0.1 * vector![100.0, 100.0, 200.0, 200.0, 200.0, 100.0, 100.0, 100.0];

        let theta_d = SVector::<f64, 8>::from_vec(vec![
            PI / 4.0,
            0.0,
            PI / 4.0,
            0.0,
            PI / 4.0,
            0.0,
            PI / 4.0,
            0.0,
        ]);

        let theta_dotd = SVector::<f64, 8>::zeros();

        let config_err = stack![pos_e; quat_e.vector()];

        let mut f_pid_b = -k_p_b.component_mul(&config_err)
            - k_i_b.component_mul(&z_b)
            - k_d_b.component_mul(&nu_b);

        let f_pid_b_max = vector![100.0, 100.0, 100.0, 100.0, 100.0, 100.0];

        f_pid_b = f_pid_b.zip_map(&f_pid_b_max, |val, max| val.clamp(-max, max));

        let theta_e = theta - theta_d;
        let theta_e_dot = theta_dot - theta_dotd;

        let mut f_pid_joint_torque: SVector<f64, 8> = -k_p_j.component_mul(&theta_e)
            - k_i_j.component_mul(&z_j)
            - k_d_j.component_mul(&theta_e_dot);

        let f_pid_j_max = vector![80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0];

        f_pid_joint_torque =
            f_pid_joint_torque.zip_map(&f_pid_j_max, |val, max| val.clamp(-max, max));

        // let joint_torque = -kp * (theta - theta_d) - kd * (theta_dot - theta_dotd);
        // let num_thrusters = self.config.thruster_dirs.len();
        // let thrust = vec![0.0; num_thrusters];
        // let thrust = SVector::<f64, 12>::repeat(0.0);

        let configuration_base =
            Isometry3::from_parts(Translation3::new(pos[0], pos[1], pos[2]), quat);
        let conf = self
            .multibody
            .minimal_to_homogenous_configuration(&configuration_base, &theta);

        let jacs = self.multibody.compute_jacobians(&conf);
        let tcm = comp_tcm::<14, 12>(&self.config, &jacs);
        let tcm_tot = stack![
            tcm,
            stack![SMatrix::<f64, 6, 8>::zeros();
        SMatrix::<f64, 8, 8>::identity()]
        ];

        let f_pid: SVector<f64, 14> = stack![f_pid_b; f_pid_joint_torque];
        let tcm_pinv = tcm_tot.transpose() * (tcm_tot * tcm_tot.transpose()).try_inverse().unwrap();
        let u = tcm_pinv * f_pid;

        // let wrenches = compute_thruster_wrenches::<8>(&self.config, &thrust, None);
        let eta = tcm_tot * u;
        // eta = f_pid;

        let cross_flow_drag =
            &|_confs: &[Isometry3<f64>], nu: &[Vector6<f64>]| -> SMatrix<f64, 6, 9> {
                let mut out = SMatrix::<f64, 6, 9>::zeros();
                // for i in 0..9 {
                //     let drag = cross_flow_drag_rb(&nu[i], &nu[i], &self.config, i);
                //     out.column_mut(i).copy_from(&drag);
                // }
                for (i, nu_i) in nu.iter().enumerate().take(9) {
                    let drag = cross_flow_drag_rb(nu_i, nu_i, &self.config, i);
                    out.column_mut(i).copy_from(&drag);
                }
                out
            };

        // let feedforward = self.multibody.generalized_newton_euler(&conf, &zeta, mu_prime, sigma_prime, rigid_body_forces, eta)

        let accel = self.multibody.forward_dynamics_ab(
            &conf,
            &zeta,
            cross_flow_drag,
            // &wrenches,
            &vec![Vector6::<f64>::zeros(); 9],
            &eta,
            &lin_vel_current,
            &lin_accel_current,
        );

        let pos_dot = quat * zeta.fixed_rows::<3>(0);
        let quat_dot = trans_mat_quat_dot(&quat) * zeta.fixed_rows::<3>(3);

        dy.fixed_rows_mut::<3>(0).copy_from(&pos_dot);
        dy.fixed_rows_mut::<4>(3).copy_from(&quat_dot);
        dy.fixed_rows_mut::<8>(7).copy_from(&theta_dot);
        dy.fixed_rows_mut::<14>(15).copy_from(&accel);
        dy.fixed_rows_mut::<6>(29).copy_from(&config_err);
        dy.fixed_rows_mut::<8>(35).copy_from(&theta_e);
    }
}

fn vec_joint_type<'de, D>(deserializer: D) -> Result<Vec<JointType>, D::Error>
where
    D: Deserializer<'de>,
{
    let v = Vec::<SerdeJointType>::deserialize(deserializer)?;
    Ok(v.into_iter().map(|j| j.into()).collect())
}

fn setup_aiauv(cfg: &Config) -> MultiBody<9, 14> {
    let num_bodies = cfg.joint_types.len();
    let mut offset_matrices = vec![Isometry3::<f64>::identity(); num_bodies];
    let mut added_mass = vec![Matrix6::<f64>::zeros(); num_bodies];
    let mut rb_mass_rotational = vec![Matrix3::<f64>::zeros(); num_bodies];
    let mut volume = vec![0.0; num_bodies];

    let joint_types = cfg.joint_types.clone();
    let parent = cfg.parents.clone();

    let mass = if cfg.mass.is_empty() {
        println!("Mass not specified – assuming a neutrally buoyant vehicle. \nCalculating mass from length, radius and fluid density.");
        let mut mass = vec![0.0; num_bodies];
        for (i, mass_iter) in mass.iter_mut().enumerate().take(num_bodies) {
            let volume = cfg.length[i] * PI * cfg.radius[i].powi(2);
            *mass_iter = volume * cfg.fluid_density;
        }
        mass
    } else {
        cfg.mass.clone()
    };

    for i in 0..num_bodies {
        let pos_offset: Translation3<f64> = cfg.pos_offsets[i].into();
        let roll_pitch_yaw_offsets = cfg.roll_pitch_yaw_offsets[i];
        offset_matrices[i] = Isometry3::from_parts(
            pos_offset,
            UnitQuaternion::from_euler_angles(
                roll_pitch_yaw_offsets[0],
                roll_pitch_yaw_offsets[1],
                roll_pitch_yaw_offsets[2],
            ),
        );

        if !cfg.added_mass_coeffs.is_empty() {
            match cfg.added_mass_coeffs[i] {
                Some(coeff_added) => {
                    added_mass[i] = slendermasss(
                        cfg.length[i],
                        cfg.radius[i],
                        cfg.fluid_density,
                        Some(cfg.added_alpha[i]),
                        Some(coeff_added),
                    );
                }
                None => {
                    added_mass[i] = slendermasss(
                        cfg.length[i],
                        cfg.radius[i],
                        cfg.fluid_density,
                        Some(cfg.added_alpha[i]),
                        None,
                    );
                }
            }
        } else {
            added_mass[i] = slendermasss(
                cfg.length[i],
                cfg.radius[i],
                cfg.fluid_density,
                Some(cfg.added_alpha[i]),
                None,
            );
        }

        rb_mass_rotational[i] =
            comp_rb_mass_rotational(cfg.pos_com[i], cfg.radius[i], cfg.length[i], mass[i]);

        volume[i] = cfg.length[i] * PI * cfg.radius[i].powi(2);
    }

    MultiBody::new(
        offset_matrices,
        None,
        Some(added_mass),
        Some(rb_mass_rotational),
        joint_types,
        parent,
        cfg.gravity,
        Some(cfg.pos_com.clone()),
        Some(cfg.pos_cob.clone()),
        Some(mass),
        Some(volume),
        Some(cfg.fluid_density),
    )
    .unwrap()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let f = std::fs::File::open("eely_config.yml").expect("Could not open file.");
    let cfg: Config = serde_yaml::from_reader(f).expect("Could not parse file.");

    let multibody = setup_aiauv(&cfg);

    // Simulation loop
    use std::time::Instant;
    let now = Instant::now();

    let joint_angles = vector![PI / 4.0, 0.0, PI / 4.0, 0.0, PI / 4.0, 0.0, PI / 4.0, 0.0];
    let zeta = SVector::<f64, 14>::repeat(1.0);

    let system = AIAUV {
        multibody,
        config: cfg.clone(),
    };

    let mut y0 = State::zeros();
    y0.fixed_rows_mut::<4>(3).copy_from(&Vector4::x());
    y0.fixed_rows_mut::<8>(7).copy_from(&joint_angles);
    y0.fixed_rows_mut::<14>(15).copy_from(&zeta);

    // let y0 = State::zeros();

    println!("y0: {}", y0);
    // Create a stepper and run the integration.
    let mut stepper = Dopri5::new(system, 0.0, cfg.sim_time, 0.01, y0, 1.0e-4, 1.0e-4);
    // let mut stepper = Rk4::new(system, 0.0, y0, 0.01, cfg.sim_time);
    let res = stepper.integrate();

    println!("Time elapsed: {} ms", now.elapsed().as_millis());

    match res {
        Ok(stats) => {
            println!("{}", stats);
            let path = Path::new("./aiauv_dopri5.dat");
            save(stepper.x_out(), stepper.y_out(), path);
            println!("Results saved in: {:?}", path);
            println!("{}", stepper.y_out()[stepper.y_out().len() - 1]);
        }
        Err(e) => println!("An error occured: {}", e),
    }

    Ok(())
}

pub fn save(times: &[Time], states: &[State], filename: &Path) {
    // Create or open file
    let file = match File::create(filename) {
        Err(e) => {
            println!("Could not open file. Error: {:?}", e);
            return;
        }
        Ok(buf) => buf,
    };
    let mut buf = BufWriter::new(file);

    // Write time and state vector in a csv format
    for (i, state) in states.iter().enumerate() {
        buf.write_fmt(format_args!("{}", times[i])).unwrap();
        for val in state.iter() {
            buf.write_fmt(format_args!(", {}", val)).unwrap();
        }
        buf.write_fmt(format_args!("\n")).unwrap();
    }
    if let Err(e) = buf.flush() {
        println!("Could not write to file. Error: {:?}", e);
    }
}
