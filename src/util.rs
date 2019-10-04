use crate::Vec3f;

/// Returns the new direction the light ray would travel to
/// both vectors have to be normalize to work.
/// Returned vector is normalized.
pub fn snellius(in_direction: &Vec3f, surface_normal: &Vec3f, refractive_index: f32) -> Option<Vec3f> {

    // Explanation of algorithm:
    // n1, s1, in_direction are used to construct a rectangular triangle
    // at the 'incoming' side.
    // n2, s2, result make up a rectangular triangle at the 'outgoing' side.
    // s1 and s2 are the adjacent sides of the triangles and therefore
    // relate to each other by snellius' law.


    let dot = surface_normal.dot(in_direction);
    let mut ref_index = refractive_index;
    if dot < 0. { ref_index = 1./ref_index }

    let n1 = -1. * surface_normal * dot;
    let s1 = (n1 + in_direction);
    let s1_length = s1.length();

    let s2_length = (ref_index) * s1_length;

    if s2_length >= 1. { return None;} // total reflection

    let n2_length = f32::sqrt(1. - s2_length * s2_length);

    let n2 = surface_normal * (f32::signum(dot) *n2_length);
    let s2 = s1 * (s2_length / s1_length);

    let result = s2 - n2;
    Some(result)
}

#[cfg(test)]
mod tests {
    use crate::util::snellius;
    use crate::Vec3f;

    #[test]
    fn test_snellius_no_refraction() {
        let in_dir = Vec3f::new(1., 1., 0.).normalize();
        let norm_dir = Vec3f::new(0., -1., 0.);

        let out_dir = snellius(&in_dir, &norm_dir, 1.).unwrap();
        assert_eq!(Vec3f::new(1., 1., 0.).normalize(), out_dir);
    }

    #[test]
    fn test_snellius_no_refraction_opposite_normal() {
        let in_dir = Vec3f::new(1., 1., 0.).normalize();
        let norm_dir = Vec3f::new(0., 1., 0.);

        let out_dir = snellius(&in_dir, &norm_dir, 1.).unwrap();
        assert_eq!(Vec3f::new(1., 1., 0.).normalize(), out_dir);
    }

//    #[test]
    fn test_snellius() {
        let in_dir = Vec3f::new(-0.242, 0.032, -0.969).normalize();
        let norm_dir = Vec3f::new(0.683, -1.05, 2.72);

        let out_dir = snellius(&in_dir, &norm_dir, 2.2).unwrap();
        assert_eq!(Vec3f::new(1., 1., 0.).normalize(), out_dir);
    }
}