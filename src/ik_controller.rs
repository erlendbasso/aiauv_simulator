extern crate nalgebra as na;
use na::{UnitQuaternion, Vector3, Vector4, Vector6};
use nalgebra::{stack, Isometry3, Quaternion, SMatrix, Translation3, Unit};

// use multibody_dynamics::multibody::MultiBody;

// use crate::bluerov_reach::{build_bluerov_reach, BlueROVReachConfig};

pub struct IKParams {
    pub k_p_ee: Vector3<f64>,
    pub k_q: f64,
    pub dt: f64,
    pub delta: f64,
}

pub struct InverseKinematicController {
    // Desired joint angles
    pub q_d: Vector4<f64>,

    // End-effector reference
    pub pos_ee_d: Vector3<f64>,
    pub quat_ee_d: UnitQuaternion<f64>,
    pub vel_ee_d: Vector6<f64>,

    // End-effector state
    pub pos_ee: Vector3<f64>,
    pub quat_ee: UnitQuaternion<f64>,
    pub vel_ee: Vector6<f64>,

    // Quat error
    pub quat_err: UnitQuaternion<f64>,

    // Nullspace reference
    pub q_null: Vector4<f64>,

    // gains
    pub k_p_ee: Vector3<f64>,
    pub k_q: f64,

    // Parameters
    dt: f64,
    delta: f64,

    // States
    control_mode_ee: i64,
    // Robot model
    // pub bluerov_reach: MultiBody<5, 10>,
}

impl InverseKinematicController {
    pub fn new(params: &IKParams) -> InverseKinematicController {
        InverseKinematicController {
            q_d: Vector4::zeros(),
            pos_ee_d: Vector3::zeros(),
            quat_ee_d: UnitQuaternion::identity(),
            vel_ee_d: Vector6::zeros(),
            pos_ee: Vector3::zeros(),
            quat_ee: UnitQuaternion::identity(),
            vel_ee: Vector6::zeros(),
            quat_err: UnitQuaternion::identity(),
            q_null: Vector4::zeros(),
            k_p_ee: params.k_p_ee,
            k_q: params.k_q,
            dt: params.dt,
            delta: params.delta,
            control_mode_ee: 1,
            // bluerov_reach: build_bluerov_reach(&config),
        }
    }

    pub fn set_ee_references(
        &mut self,
        pos_ee_d: Vector3<f64>,
        quat_ee_d: UnitQuaternion<f64>,
        vel_ee_d: Vector6<f64>,
    ) {
        self.pos_ee_d = pos_ee_d;
        self.quat_ee_d = quat_ee_d;
        self.vel_ee_d = vel_ee_d;
    }

    pub fn set_nullspace_reference(&mut self, q_null: Vector4<f64>) {
        self.q_null = q_null;
    }

    pub fn set_gains(&mut self, k_p_ee: Vector3<f64>, k_q: f64) {
        self.k_p_ee = k_p_ee;
        self.k_q = k_q;
    }

    pub fn update(
        &mut self,
        jacobian_ee: SMatrix<f64, 6, 4>,
        dt: f64,
    ) -> (Vector4<f64>, Vector4<f64>) {
        let quat_e = self.quat_ee_d.inverse() * self.quat_ee;

        if quat_e.dot(&self.quat_err) < 0.0 {
            self.quat_err = UnitQuaternion::from_quaternion(Quaternion::new(
                -quat_e.w, -quat_e.i, -quat_e.j, -quat_e.k,
            ));
        } else {
            self.quat_err = quat_e;
        }

        let pos_err = self.quat_ee_d.inverse() * (self.pos_ee - self.pos_ee_d);

        if self.control_mode_ee as f64 * self.quat_err.w <= -self.delta {
            self.control_mode_ee = -self.control_mode_ee;
        }

        let gradient_position = self.quat_err.inverse() * self.k_p_ee.component_mul(&pos_err);
        let gradient_orientation =
            self.k_q * (self.control_mode_ee as f64) * self.quat_err.vector();

        let gradient: Vector6<f64> = stack![gradient_position; gradient_orientation];

        let h_ee = Isometry3::from_parts(Translation3::from(self.pos_ee), self.quat_ee);

        let vel_ee_r = multibody_dynamics::math_functions::Ad_inv(&h_ee) * self.vel_ee_d;

        let jac_pinv = jacobian_ee.transpose()
            * (jacobian_ee * jacobian_ee.transpose())
                .try_inverse()
                .unwrap();

        let q_d_dot = jac_pinv * (vel_ee_r - gradient);
        self.q_d += q_d_dot * dt;

        (self.q_d, q_d_dot)
    }
}
