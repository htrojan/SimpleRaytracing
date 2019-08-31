use std::f32::consts::PI;

pub const ERR: f32 = 0.000001;

pub const WIDTH: u32 = 1024;
pub const HEIGHT: u32 = 768;

#[derive(Debug, PartialOrd, PartialEq, Clone)]
pub struct Vec3f {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3f {
    pub fn new(x: f32, y: f32, z: f32) -> Vec3f {
        Vec3f { x, y, z }
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn minus(&self, other: &Self) -> Self {
        Vec3f { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    #[inline]
    pub fn plus(&self, other: Self) -> Self {
        Vec3f { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    #[inline]
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub fn times(&self, factor: f32) -> Self {
        Vec3f { x: self.x * factor, y: self.y * factor, z: self.z * factor }
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let length = f32::sqrt(self.length_squared());
        Vec3f { x: self.x / length, y: self.y / length, z: self.z / length }
    }

    #[inline]
    pub fn max_element(&self) -> f32 {
        f32::max(self.x, f32::max(self.y, self.z))
    }

    /// Returns the shortest point and a parameter for the straight line equation of the ray.
    /// If positive the point lies in direction of the ray.
    /// If negative in the opposite direction.
    ///
    /// This method assumes that the direction of the ray is normalized!
    pub fn shortest_point_to_ray(&self, ray: &Ray) -> (Self, f32) {
        let point_center = self.minus(&ray.origin);

        let plane_prod = ray.direction.dot(&point_center);

        let lambda: f32 = plane_prod;

        let shortest_point = ray.origin.plus(ray.direction.times(lambda));
        (shortest_point, lambda)
    }

    fn equal_within_err(&self, other: Self) -> bool {
        (
            self.x < other.x + ERR && self.x > other.x - ERR &&
                self.y < other.y + ERR && self.y > other.y - ERR &&
                self.z < other.z + ERR && self.z > other.z - ERR
        )
    }
}

pub struct Sphere {
    pub center: Vec3f,
    pub radius: f32,
}

pub struct Ray {
    origin: Vec3f,
    direction: Vec3f,
}

impl Ray {
    pub fn new(origin: Vec3f, direction: Vec3f) -> Self {
        Ray { origin, direction: direction.normalize() }
    }

    pub fn hit_from_params(&self, hit_params: &HitParam) -> Hit {
        let point = self.origin.plus(self.direction.times(hit_params.lambda));
        let normal = point.minus(&hit_params.sphere.center).normalize();
        Hit{ normal, point}
    }
}

pub struct Hit {
    normal: Vec3f,
    point: Vec3f,
}

pub struct HitParam<'a> {
    lambda: f32,
    sphere: &'a Sphere,
}

impl Sphere {
    //The ray has to have a normalized direction vector
    pub fn intersects_with_ray(&self, ray: &Ray) -> Option<HitParam> {
        let (shortest_point, lambda) = self.center.shortest_point_to_ray(&ray);

        let distance_vector = self.center.minus(&shortest_point);

        let l = self.radius * self.radius - distance_vector.length_squared();

//        let is_hit: bool = (l > 0.);

        if l <= 0. {
            return None;
        }

        //These are the two possible hit points
        let mut hit_1 = lambda - l;
        let hit_2 = lambda + l;

        if hit_1 < 0. { hit_1 = hit_2 }
        if hit_1 < 0. { return None; }

//        println!("Chosen: {}", hit_1);
        let hit_point = ray.origin.plus(ray.direction.times(hit_1));
        let normal = hit_point.minus(&self.center).normalize();
        Some(HitParam {lambda: hit_1, sphere: self})
    }
}

pub struct Scene {
    objects: Vec<Sphere>,
    lights: Vec<Light>,
}

impl Scene {
    pub fn new(spheres: Vec<Sphere>, lights: Vec<Light>) -> Self {
        Scene { objects: spheres, lights }
    }

    /// Returns the color of the pixel the ray originates from
    pub fn cast_ray(&self, ray: &Ray) -> Vec3f {
        let mut foreground_hit: Option<HitParam> = None;

        for o in self.objects.iter() {
            match o.intersects_with_ray(&ray) {
                Some(hit) => {
                    match &mut foreground_hit {
                        None => foreground_hit = Some(hit),
                        Some(fg) => {
                            if fg.lambda < hit.lambda {
                                foreground_hit = Some(hit);
                            }
                        },
                    }
                }
                _ => continue,
            }
        }

        match &foreground_hit {
            Some(hit) => {
                let hit = ray.hit_from_params(&hit);
                let mut diffuse_light_intensity: f32 = 0.;
                for light in self.lights.iter() {
                    let light_dir = light.position.minus(&hit.point).normalize();
                    diffuse_light_intensity +=
                        light.intensity * f32::max(0., light_dir.dot(&hit.normal));
                }
                return Vec3f::new(0.3, 0.1, 0.1).times(diffuse_light_intensity);
            },
            None => return Vec3f::new(0.4, 0.4, 0.3),
        }

    }

    pub fn render(&self) -> Vec<Vec3f> {
        let mut frame_buffer = Vec::<Vec3f>::with_capacity((WIDTH * HEIGHT) as usize);

        let fov = PI / 3.;
        for j in 0..HEIGHT {
            for i in 0..WIDTH {
                let x: f32 = (2. * (i as f32 + 0.5) / WIDTH as f32 - 1.) * f32::tan(fov / 2.);
                let y: f32 = -(2. * (j as f32 + 0.5) / WIDTH as f32 - 1.) * f32::tan(fov / 2.);
                let ray = Ray::new(
                    Vec3f::new(0., 0., 0.),
                    Vec3f::new(x, y, -1.),
                );

                frame_buffer.push(self.cast_ray(&ray));
            }
        }

        frame_buffer
    }
}

pub struct Light {
    position: Vec3f,
    intensity: f32,
}

impl Light {
    pub fn new(position: Vec3f, intensity: f32) -> Self {
        Light { position, intensity }
    }
}

#[cfg(test)]
mod tests {

    use super::*;


    #[test]
    fn test_shortest_point() {
        let p = Vec3f::new(0., 5., 6.);
        let ray = Ray::new
            (
                Vec3f::new(2., 0., 1.),
                Vec3f::new(-4., 1., 1.),
            );

        let (result, lambda) = p.shortest_point_to_ray(&ray);
        let expected = Vec3f::new(-2., 1., 2.);
        let lambda_expected = f32::sqrt(Vec3f::new(-4., 1., 1.).length_squared());
        dbg!(lambda_expected);
        dbg!(lambda);
        assert!(lambda + ERR > lambda_expected);
        assert!(expected.equal_within_err(result));
    }

    #[test]
    fn test_ray_intersects_with_sphere() {
        let sphere = Sphere {
            center: Vec3f::new(3., 3., 0.),
            radius: 1.,
        };

        let ray = Ray::new
            (
                Vec3f::new(0., 0., 0.),
                Vec3f::new(1., 1., 0.),
            );

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
    }

    #[test]
    fn test_ray_intersects_with_sphere_inside() {
        let sphere = Sphere {
            center: Vec3f::new(-0.1, -0.1, 0.),
            radius: 1.,
        };

        let ray = Ray
            {
                origin: Vec3f::new(0., 0., 0.),
                direction: Vec3f::new(1., 1., 0.),
            };

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
    }

    #[test]
    fn test_ray_intersect_normal() {
        let sphere = Sphere {
            center: Vec3f::new(3., 3., 0.),
            radius: 1.,
        };

        let ray = Ray::new
            (
                Vec3f::new(0., 0., 0.),
                Vec3f::new(1., 1., 0.),
            );

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
        let hit = result.unwrap();
        let hit = ray.hit_from_params(&hit);
        let normal = hit.normal;
        assert!(normal.equal_within_err(Vec3f::new(-1., -1., 0.).normalize()));
    }

    #[test]
    fn test_ray_intersect_normal_under() {
        let sphere = Sphere {
            center: Vec3f::new(3., 0., 0.),
            radius: 1.,
        };

        let ray = Ray::new
            (
                Vec3f::new(0., 0., 0.),
                Vec3f::new(1., 0., 0.),
            );

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
        let hit = result.unwrap();
        let hit = ray.hit_from_params(&hit);
        let normal = hit.normal;
        assert_eq!(normal, Vec3f::new(-1., 0., 0.).normalize());
    }

    #[test]
    fn test_ray_intersect_normal_2() {
        let sphere = Sphere {
            center: Vec3f::new(2., 0., 0.),
            radius: 1.,
        };

        let ray = Ray::new
            (
                Vec3f::new(0., 0., 0.),
                Vec3f::new(1., -0.5, 0.),
            );

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
        let hit = result.unwrap();
        let hit = ray.hit_from_params(&hit);
        let normal = hit.normal;
        assert_eq!(normal, Vec3f::new(-1., 0., 0.).normalize());
    }

    #[test]
    fn test_normalize() {
        let vec = Vec3f::new(2., 2., 0.);
        assert_eq!(vec.normalize(), Vec3f::new(2., 2., 0.).times(1. / f32::sqrt(8.)))
    }

}
