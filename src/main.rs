use core::num;
use std::f64::consts::PI;

use multibody_dynamics::multibody::JointType;
use serde::{Deserialize, Deserializer, Serialize};
use serde_yaml::{self};

use multibody_dynamics::multibody::MultiBody;

extern crate nalgebra as na;
use na::{
    stack, Isometry3, Matrix3, Matrix6, Quaternion, SMatrix, SVector, Translation3, UnitQuaternion,
    Vector3, Vector4, Vector6,
};

mod utils;
use crate::utils::*;

mod bluerov_reach;

use std::{fs::File, io::BufWriter, io::Write, path::Path};

use ode_solvers::*;

type State = SVector<f64, 29>;
type Time = f64;

#[derive(Serialize, Deserialize)]
#[serde(remote = "JointType")]
enum JointTypeDef {
    Revolute,
    Prismatic,
    SixDOF,
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
    fn system(&self, t: Time, y: &State, dy: &mut State) {
        let quat = UnitQuaternion::from_quaternion(Quaternion::from_parts(
            y[3],
            Vector3::new(y[4], y[5], y[6]),
        ));
        // implement your controller here
        let pos = y.fixed_rows::<3>(0);

        let theta = y.fixed_rows::<8>(7).into(); // joint angles
        let zeta: SVector<f64, 14> = y.fixed_rows::<14>(15).into(); // joint velocities
        let theta_dot = zeta.fixed_rows::<8>(6); // joint velocities
        let lin_vel_current = Vector3::<f64>::zeros();
        let lin_accel_current = Vector3::<f64>::zeros();
        let mut eta = SVector::<f64, 14>::zeros();

        // testing with a swimming gait, add your own controller here:
        // let phaseramp = 0.4 * PI;
        let phaseramp = 1.0;

        let mut theta_d = SVector::<f64, 8>::zeros();
        let mut theta_dotd = SVector::<f64, 8>::zeros();

        for i in 0..theta_d.len() {
            let phase = phaseramp * i as f64;
            let omega = 0.8;

            theta_d[i] = PI / 4.0 * f64::sin(omega * t - phase);
            theta_dotd[i] = PI / 4.0 * omega * f64::cos(omega * t - phase);
        }

        let kp = 10.0;
        let kd = 10.0;

        let joint_torque = -kp * (theta - theta_d) - kd * (theta_dot - theta_dotd);
        // let num_thrusters = self.config.thruster_dirs.len();
        // let thrust = vec![0.0; num_thrusters];
        let thrust = SVector::<f64, 12>::repeat(0.0);

        let wrenches = compute_thruster_wrenches(&self.config, &thrust, None);
        eta.fixed_rows_mut::<8>(6).copy_from(&joint_torque);

        let configuration_base =
            Isometry3::from_parts(Translation3::new(pos[0], pos[1], pos[2]), quat);
        let conf = self
            .multibody
            .minimal_to_homogenous_configuration(&configuration_base, &theta);

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

        let accel = self.multibody.forward_dynamics_ab(
            &conf,
            &zeta,
            cross_flow_drag,
            &wrenches,
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
    }
}

fn vec_joint_type<'de, D>(deserializer: D) -> Result<Vec<JointType>, D::Error>
where
    D: Deserializer<'de>,
{
    #[derive(Deserialize)]
    struct Wrapper(#[serde(with = "JointTypeDef")] JointType);

    let v = Vec::deserialize(deserializer)?;
    Ok(v.into_iter().map(|Wrapper(a)| a).collect())
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
            comp_rb_inertia_cylinder(&cfg.pos_com[i], cfg.radius[i], cfg.length[i], mass[i]);

        volume[i] = cfg.length[i] * PI * cfg.radius[i].powi(2);
    }

    // println!("Volume: {}", volume[5]);
    // println!("Added mass: {}", added_mass[5]);
    // println!("rb mass: {}", rb_mass_rotational[5]);

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

    let joint_angles = SVector::<f64, 8>::from_vec(vec![
        PI / 4.0,
        PI / 3.0,
        PI / 5.0,
        PI / 7.0,
        PI / 15.0,
        PI / 10.0,
        PI / 4.0,
        PI / 2.5,
    ]);
    // let zeta = SVector::<f64, 14>::repeat(1.0);

    // TEST
    let nu = Vector6::<f64>::new(0.1339, 0.2854, 0.0305, 0.5841, 0.1752, 0.0746);
    let theta_dot = SVector::<f64, 8>::from_column_slice(&[
        0.1166, 0.9438, 0.3364, 0.6646, 0.5577, 0.4213, 0.0738, 0.6351,
    ]);
    let zeta = stack![nu; theta_dot];

    let lin_vel_current = Vector3::<f64>::zeros();
    let lin_accel_current = Vector3::<f64>::zeros();
    let pos = Vector3::new(0.0622, 0.4052, 0.1091);
    // let quat = UnitQuaternion::from_euler_angles(0.0, 0.0, 0.0);
    let quat = UnitQuaternion::from_quaternion(Quaternion::new(0.8566, 0.4131, 0.0945, 0.2944));
    let theta = SVector::<f64, 8>::from_vec(vec![
        0.2870, 0.2197, 0.1250, 0.8382, 0.5592, 0.2592, 0.8849, 0.9270,
    ]);
    println!("theta: {}", theta);

    let configuration_base = Isometry3::from_parts(Translation3::new(pos[0], pos[1], pos[2]), quat);
    let conf = multibody.minimal_to_homogenous_configuration(&configuration_base, &theta);

    let cross_flow_drag = &|_confs: &[Isometry3<f64>], nu: &[Vector6<f64>]| -> SMatrix<f64, 6, 9> {
        let mut out = SMatrix::<f64, 6, 9>::zeros();
        for (i, nu_i) in nu.iter().enumerate().take(9) {
            let drag = cross_flow_drag_rb(nu_i, nu_i, &cfg, i);
            out.column_mut(i).copy_from(&drag);
        }
        out
    };

    // println!("drag: {}", cross_flow_drag(&conf, &[nu]));

    // let num_thrusters = cfg.thruster_dirs.len();
    // let thrust = vec![0.0; num_thrusters];
    let thrust = SVector::<f64, 12>::from_vec(vec![
        0.7456, 0.1187, 0.4123, 0.8181, 0.7854, 0.1217, 0.5034, 0.0303, 0.8609, 0.6282, 0.7324,
        0.1752,
    ]);

    // let wrenches = compute_thruster_wrenches(&cfg, &thrust, None);
    let wrenches = vec![SVector::<f64, 6>::zeros(); 9];
    // let joint_torque = SVector::<f64, 8>::from_vec(vec![
    //     0.5275, 0.0887, 0.2674, 0.6599, 0.6552, 0.4395, 0.9136, 0.7634,
    // ]);
    let joint_torque = SVector::<f64, 8>::zeros();
    let eta = stack![Vector6::zeros(); joint_torque];

    let accel = multibody.forward_dynamics_ab(
        &conf,
        &zeta,
        cross_flow_drag,
        &wrenches,
        &eta,
        &lin_vel_current,
        &lin_accel_current,
    );
    println!("Acceleration: {}", accel);

    // println!(
    //     "Hydrostatic force body 1: {}",
    //     multibody.compute_hydrostatic_force(&quat, &lin_accel_current, 0)
    // );

    // println!("quat euler: {:?}", quat.euler_angles().0 * 180.0 / PI);

    // END TEST

    // let system = AIAUV {
    //     multibody,
    //     config: cfg.clone(),
    // };

    // let mut y0 = State::zeros();
    // y0.fixed_rows_mut::<4>(3).copy_from(&Vector4::x());
    // y0.fixed_rows_mut::<8>(7).copy_from(&joint_angles);
    // y0.fixed_rows_mut::<14>(15).copy_from(&zeta);

    // // let y0 = State::zeros();

    // println!("y0: {}", y0);
    // // Create a stepper and run the integration.
    // let mut stepper = Dopri5::new(system, 0.0, cfg.sim_time, 0.01, y0, 1.0e-4, 1.0e-4);
    // let res = stepper.integrate();

    // println!("Time elapsed: {} ms", now.elapsed().as_millis());

    // match res {
    //     Ok(stats) => {
    //         println!("{}", stats);
    //         let path = Path::new("./aiauv_dop853.dat");
    //         save(stepper.x_out(), stepper.y_out(), path);
    //         println!("Results saved in: {:?}", path);
    //         println!("{}", stepper.y_out()[stepper.y_out().len() - 1]);
    //     }
    //     Err(e) => println!("An error occured: {}", e),
    // }

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
