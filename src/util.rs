use crate::{Vec3f, Ray, BIAS};

/// Returns the new direction the light ray would travel to
/// both vectors have to be normalize to work.
/// Returned vector is normalized.
pub fn snellius(in_direction: Vec3f, surface_normal: Vec3f, hit_point: Vec3f, refractive_index: f64) -> Option<Ray> {

    // Explanation of algorithm:
    // n1, s1, in_direction are used to construct a rectangular triangle
    // at the 'incoming' side.
    // n2, s2, result make up a rectangular triangle at the 'outgoing' side.
    // s1 and s2 are the adjacent sides of the triangles and therefore
    // relate to each other by snellius' law.


    let dot = surface_normal.dot(&in_direction);
    let mut ref_index = refractive_index;
    if dot < 0. { ref_index = 1./ref_index }

    let n1 = -1. * surface_normal * dot;
    let s1 = n1 + in_direction;
    let s1_length = s1.length();

    let s2_length = (ref_index) * s1_length;

    if s2_length >= 1. {
        return None;
    } // total reflection

    let n2_length = (1. - s2_length * s2_length).sqrt();

    let n2 = surface_normal * (dot.signum() *n2_length);
    let s2 = s1 * (s2_length / s1_length);

    let result = s2 - n2;
    Some(Ray {
        direction: result,
        origin: hit_point + surface_normal * BIAS * dot.signum() * -1.0
    })
}

pub fn create_reflection(in_direction: Vec3f, surface_normal: Vec3f, hit_point: Vec3f) -> Ray {
    let vertical_component = surface_normal.dot(&in_direction);
    //in_direction and normal have to be calculated with the small angle.
    //Therefore their sign is shifted if they are negative
    let horizontal_component = in_direction - surface_normal * vertical_component * vertical_component.signum();
    let reflection = -1. * in_direction + 2. * horizontal_component;
    Ray {
        direction: reflection,
        origin: hit_point - surface_normal * BIAS * vertical_component.signum()
    }
}

pub fn create_reflection_dir(in_direction: &Vec3f, surface_normal: &Vec3f) -> Vec3f {
    let vertical_component = surface_normal.dot(&in_direction);
    //in_direction and normal have to be calculated with the small angle.
    //Therefore their sign is shifted if they are negative
    let horizontal_component = in_direction - surface_normal * vertical_component * vertical_component.signum();
    let reflection = -1. * in_direction + 2. * horizontal_component;
    reflection
}

#[cfg(test)]
mod tests {
    use crate::util::{snellius};
    use crate::Vec3f;

    #[test]
    fn test_snellius_no_refraction() {
        let in_dir = Vec3f::new(1., 1., 0.).normalize();
        let norm_dir = Vec3f::new(0., -1., 0.);
        let n = Vec3f::new(0., 0., 0.);

        let out_dir = snellius(in_dir, norm_dir, n, 1.).unwrap();
        assert!(Vec3f::new(1., 1., 0.).normalize().equal_within_err(out_dir.direction));
    }

    #[test]
    fn test_snellius_in_out() {
        let in_dir = Vec3f::new(1., 2., 0.).normalize();
        let norm_dir = Vec3f::new(0., -1., 0.);
        let intersection_point = Vec3f::new(0., 0., 0.);

        let out_dir = snellius(in_dir, norm_dir, intersection_point, 1.2).unwrap();
        let reverse_dir = snellius(out_dir.direction, -norm_dir, intersection_point, 1.2).unwrap();
        assert_eq!(in_dir, reverse_dir.direction);
    }
}