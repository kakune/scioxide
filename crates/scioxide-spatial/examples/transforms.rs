//! Active rigid motion and continuous-acceleration attitude interpolation.
use scioxide_core::SciOxideError;
use scioxide_spatial::{RigidTransform, Rotation, RotationSpline, Slerp};
fn main() -> Result<(), SciOxideError> {
    let turn = Rotation::from_axis_angle([0.0_f64, 0.0, 1.0], core::f64::consts::FRAC_PI_2)?;
    let pose = RigidTransform::new(turn, [1.0, 2.0, 0.0])?;
    println!(
        "point after rotation and translation: {:?}",
        pose.apply([1.0, 0.0, 0.0])?
    );
    let orientations = [Rotation::identity(), turn, Rotation::identity()];
    let spline = RotationSpline::new(&[0.0, 1.0, 2.0], &orientations)?;
    let queries = [0.0, 0.5, 1.0, 1.5, 2.0];
    let slerp = Slerp::new(&[0.0, 1.0, 2.0], &orientations)?;
    println!(
        "SLERP midpoint: {:?}",
        slerp.evaluate_rotations(&queries)?[1].as_quaternion()
    );
    println!(
        "spline midpoint: {:?}",
        spline.evaluate_rotations(&queries)?[1].as_quaternion()
    );
    Ok(())
}
