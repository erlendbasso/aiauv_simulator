use std::f64::consts::PI;

use multibody_dynamics::multibody::MultiBody;
use multibody_dynamics::multibody::{Axis, JointType};

use serde::{Deserialize, Deserializer, Serialize};
use serde_yaml::{self};

extern crate nalgebra as na;
use na::{Isometry3, Matrix3, Matrix6, Translation3, UnitQuaternion, Vector3, Vector6};

// use crate::utils::slendermass;
use crate::utils::*;

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
pub struct BlueROVReachConfig {
    gravity: Vector3<f64>,
    dragcoeffs: Vec<Vector6<f64>>,
    #[serde(default)]
    mass: Vec<f64>,
    lengths_bluerov: Vector3<f64>,
    radius: Vec<f64>,
    link_lengths: Vec<f64>,
    fluid_density: f64,
    /// Position of the center of gravity of each link, expressed in the link frame.
    pos_com: Vec<Vector3<f64>>,
    /// Position of the center of buoyancy of each link, expressed in the link frame.
    pos_cob: Vec<Vector3<f64>>,
    pos_offsets: Vec<Vector3<f64>>,
    roll_pitch_yaw_offsets: Vec<Vector3<f64>>,
    pub thruster_pos_offsets: Vec<Vector3<f64>>,
    pub thruster_dirs: Vec<Vector3<f64>>,
    pub thruster_parents: Vec<u16>,
    #[serde(default)]
    added_mass_coeffs: Vec<Option<f64>>,
    added_alpha: Vec<f64>,
}

pub fn build_bluerov_reach(cfg: &BlueROVReachConfig) -> MultiBody<5, 10> {
    const NUM_BODIES: usize = 5;
    let mass = &cfg.mass;

    let mut offset_matrices = vec![Isometry3::<f64>::identity(); NUM_BODIES];
    let mut added_mass = vec![Matrix6::<f64>::zeros(); NUM_BODIES];
    let mut rb_mass_rotational = vec![Matrix3::<f64>::zeros(); NUM_BODIES];
    let mut volume = vec![0.0; NUM_BODIES];

    let joint_types = vec![
        JointType::SixDOF,
        JointType::Revolute(Axis::Z),
        JointType::Revolute(Axis::Z),
        JointType::Revolute(Axis::Z),
        JointType::Revolute(Axis::Z),
    ];
    let parent = vec![0, 1, 2, 3, 4];

    // Create the mass matrix for the BlueROV
    rb_mass_rotational[0] =
        comp_rb_inertia_rectangular_cuboid(&cfg.lengths_bluerov, &cfg.pos_com[0], mass[0]);
    // added_mass[0]

    // Create mass matrices for the Reach arm.
    for i in 1..5 {
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
                        cfg.link_lengths[i],
                        cfg.radius[i],
                        cfg.fluid_density,
                        Some(cfg.added_alpha[i]),
                        Some(coeff_added),
                    );
                }
                None => {
                    added_mass[i] = slendermasss(
                        cfg.link_lengths[i],
                        cfg.radius[i],
                        cfg.fluid_density,
                        Some(cfg.added_alpha[i]),
                        None,
                    );
                }
            }
        } else {
            added_mass[i] = slendermasss(
                cfg.link_lengths[i],
                cfg.radius[i],
                cfg.fluid_density,
                Some(cfg.added_alpha[i]),
                None,
            );
        }

        rb_mass_rotational[i] =
            comp_rb_inertia_cylinder(&cfg.pos_com[i], cfg.radius[i], cfg.link_lengths[i], mass[i]);

        volume[i] = cfg.link_lengths[i] * PI * cfg.radius[i].powi(2);
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
        Some(mass.to_vec()),
        Some(volume),
        Some(cfg.fluid_density),
    )
    .unwrap()
}
