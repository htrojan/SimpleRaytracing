use crate::Vec3f;

/// Returns the new direction the light ray would travel to
/// both vectors have to be normalize to work.
/// Returned vector is normalized.
pub fn snellius(in_direction: &Vec3f, surface_normal: &Vec3f, n1_index: f32, n2_index: f32) -> Vec3f{

    // Explanation of algorithm:
    // n1, s1, in_direction are used to construct a rectangular triangle
    // at the 'incoming' side.
    // n2, s2, result make up a rectangular triangle at the 'outgoing' side.
    // s1 and s2 are the adjacent sides of the triangles and therefore
    // relate to each other by snellius' law.

    let n1 = surface_normal; // Just for shorthand writing
    let n1_length = n1.dot(&in_direction.times(-1.));

    // Connection in the triangle
    let n = n1.times(n1_length);
    let s1 = n.minus(&in_direction.times(-1.));
    let s1_length= s1.length();

    let s2_length = (n1_index/n2_index) * s1_length;
    let n2_length = f32::sqrt(1. - s2_length * s2_length);

    let n2 = surface_normal.times(-1. * n2_length);
    let s2 = s1.times(s2_length / s1_length);

    let result = n2.plus(&s2);
    result
}

#[cfg(test)]
mod tests {
    use crate::Vec3f;
    use crate::util::snellius;

    #[test]
    fn test_snellius_no_refraction() {
        let in_dir = Vec3f::new(1., 1., 0.).normalize();
        let norm_dir = Vec3f::new(0., -1., 0.);

        let out_dir = snellius(&in_dir, &norm_dir, 1., 1.);
        assert_eq!(Vec3f::new(1., 1., 0.).normalize(), out_dir);
    }
}