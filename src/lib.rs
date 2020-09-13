use std::f64::consts::PI;
use std::ops::{Add, Mul, Neg, Sub};

use rayon::prelude::*;

use crate::util::{snellius, create_reflection_dir};
use crate::HitResult::{NoHit, HitDetected, OutsideSphere};

mod util;
#[macro_use]
mod op_help;

#[allow(dead_code)]
const ETA: f64 = 1.0e-8;
// const ETA: f64 = 0.;
// const BIAS: f64 = 1.0e-5;
const BIAS: f64 = 0.0;

pub const WIDTH: usize = 1024;
pub const HEIGHT: usize = 768;

// pub const WIDTH: usize = 1920;
// pub const HEIGHT: usize = 1080;

#[derive(Debug, PartialOrd, PartialEq, Clone, Copy)]
pub struct Vec3f {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct Scene {
    objects: Vec<Sphere>,
    lights: Vec<Light>,
}

#[derive(Debug)]
pub struct HitParams<'a> {
    sphere: &'a Sphere,
    distance_to_hit: f64,
}

#[derive(Debug)]
pub struct Sphere {
    pub center: Vec3f,
    pub radius: f64,
    pub material: Material,
}

#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub color: Vec3f,
    pub ref_index: f64,
    pub translucency: f64,
    pub diffuse: f64,
    pub specular: f64,
}

#[derive(Debug)]
pub struct Hit {
    normal: Vec3f,
    point: Vec3f,
}

#[derive(Debug)]
pub struct Light {
    position: Vec3f,
    intensity: f64,
}

#[derive(Debug)]
pub struct Ray {
    origin: Vec3f,
    direction: Vec3f,
}


impl Vec3f {
    pub fn new(x: f64, y: f64, z: f64) -> Vec3f {
        Vec3f { x, y, z }
    }

    #[inline]
    pub fn dot(&self, other: &Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    pub fn minus(&self, other: &Self) -> Self {
        Vec3f { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    #[inline]
    pub fn plus(&self, other: &Self) -> Self {
        Vec3f { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    #[inline]
    pub fn length_squared(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(&self) -> f64 {
        f64::sqrt(self.length_squared())
    }

    #[inline]
    pub fn times(&self, factor: f64) -> Self {
        Vec3f { x: self.x * factor, y: self.y * factor, z: self.z * factor }
    }

    #[inline]
    pub fn normalize(&self) -> Self {
        let length = f64::sqrt(self.length_squared());
        Vec3f { x: self.x / length, y: self.y / length, z: self.z / length }
    }

    #[inline]
    pub fn max_element(&self) -> f64 {
        f64::max(self.x, f64::max(self.y, self.z))
    }

    /// Returns the shortest point and a parameter for the straight line equation of the ray.
    /// If positive the point lies in direction of the ray.
    /// If negative in the opposite direction.
    ///
    /// This method assumes that the direction of the ray is normalized!
    pub fn shortest_point_to_ray(&self, ray: &Ray) -> (Self, f64) {
        let point_center = self.minus(&ray.origin);

        let plane_prod = ray.direction.dot(&point_center);

        let lambda: f64 = plane_prod;

        let shortest_point = ray.origin.plus(&ray.direction.times(lambda));
        (shortest_point, lambda)
    }

    #[allow(dead_code)]
    fn equal_within_err(&self, other: Self) -> bool {
            self.x < other.x + ETA && self.x > other.x - ETA &&
                self.y < other.y + ETA && self.y > other.y - ETA &&
                self.z < other.z + ETA && self.z > other.z - ETA
    }
}

impl_op!(+ |a: Vec3f, b: Vec3f| -> Vec3f {
        Vec3f { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
});

impl_op!(- |a: Vec3f, b: Vec3f| -> Vec3f {
        Vec3f { x: a.x + b.x, y: a.y + b.y, z: a.z + b.z }
});

impl_op_commutative!(* |a: Vec3f, b: f64| -> Vec3f {
        Vec3f { x: a.x * b, y: a.y * b, z: a.z * b}
});



impl Neg for Vec3f {
    type Output = Vec3f;

    fn neg(self) -> Self::Output {
        Vec3f { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl Material {
    pub fn new_diffuse(color: Vec3f) -> Self {
        Material { color, translucency: 0., ref_index: 1., diffuse: 1.0, specular: 0.0}
    }

    pub fn new(color: Vec3f, specular: f64, translucency: f64, n_index: f64) -> Self {
        Material { color, translucency, ref_index: n_index, diffuse: 1.0, specular}
    }

    pub fn default() -> Self {
        Material::new_diffuse(Vec3f::new(0.3, 0.1, 0.1))
    }
}

impl Ray {
    pub fn new(origin: Vec3f, direction: Vec3f) -> Self {
        Ray { origin, direction: direction.normalize() }
    }
}

impl<'a> HitParams<'a> {
    #[inline]
    fn to_hit(&self, ray: &Ray) -> Hit {
        let point = ray.origin.plus(&ray.direction.times(self.distance_to_hit));
        let normal = point.minus(&self.sphere.center).times(1. / self.sphere.radius);
        Hit { point, normal }
    }
}

impl Sphere {
    //The ray has to have a normalized direction vector
    pub fn intersects_with_ray(&self, ray: &Ray) -> Option<HitParams> {
        let (shortest_point, lambda) = self.center.shortest_point_to_ray(&ray);

        let distance_vector = self.center.minus(&shortest_point);

        let l = self.radius * self.radius - distance_vector.length_squared();

        if l <= 0. {
            return None;
        }
        let l = l.sqrt();
        //These are the two possible Hit points
        let mut hit_1 = lambda - l;
        let hit_2 = lambda + l;

        // The ETA is important! Otherwise the same point can be collided with twice during recursion
        if hit_1 < ETA { hit_1 = hit_2 }
        if hit_1 < ETA { return None; }
        Some(HitParams { sphere: &self, distance_to_hit: hit_1 })
    }

    pub fn debug_intersection(&self, ray: &Ray) -> HitResult{
        let (shortest_point, lambda) = self.center.shortest_point_to_ray(&ray);

        let distance_vector = self.center.minus(&shortest_point);

        let l = self.radius * self.radius - distance_vector.length_squared();

        if l <= 0. {
            return OutsideSphere(distance_vector.length_squared())
        }
        let l = l.sqrt();
        //These are the two possible Hit points
        let mut hit_1 = lambda - l;
        let hit_2 = lambda + l;

        if hit_1 < 0. { hit_1 = hit_2 }
        if hit_1 < 0. { return NoHit(HitParams { sphere: &self, distance_to_hit: hit_1}); }

        HitDetected(HitParams { sphere: &self, distance_to_hit: hit_1 })
    }
}

pub enum HitResult<'a> {
    HitDetected(HitParams<'a>),
    NoHit(HitParams<'a>),
    //Save distance to sphere in f64
    OutsideSphere(f64)
}

impl Scene {
    pub fn new(spheres: Vec<Sphere>, lights: Vec<Light>) -> Self {
        Scene { objects: spheres, lights }
    }

    fn get_intersection(&self, ray: &Ray) -> Option<HitParams> {
        let mut hit_params: Option<HitParams> = None;

        for o in self.objects.iter() {
            match o.intersects_with_ray(&ray) {
                Some(hit_param) => {
                    match &hit_params {
                        None => hit_params = Some(hit_param),
                        Some(params) => {
                            if hit_param.distance_to_hit <= params.distance_to_hit {
                                hit_params = Some(hit_param)
                            }
                        }
                    }
                }
                _ => continue,
            }
        }
        hit_params
    }

    #[allow(dead_code)]
    fn get_intersection_debug(&self, ray: &Ray) -> HitResult {
        let mut hit_params: HitResult = OutsideSphere(-1.);

        for o in self.objects.iter() {
            match o.debug_intersection(&ray) {
                HitDetected(hit_param) => {
                    match &hit_params {
                        HitDetected(params) => {
                            if hit_param.distance_to_hit <= params.distance_to_hit {
                                hit_params = HitDetected(hit_param)
                            }
                        }
                        _ => hit_params = HitDetected(hit_param),
                    }
                }

                NoHit(hit_param) => {
                    match &hit_params {
                        HitDetected(_) => continue,
                        _ => hit_params = NoHit(hit_param)
                    }
                }

                OutsideSphere(distance) => {
                    match &hit_params {
                        OutsideSphere(d) => {
                            if distance < d.clone() {hit_params = OutsideSphere(distance)}
                        }
                        _ => continue
                    }
                }
            }
        }
        hit_params
    }

    //Takes the hitpoint and the direction of the ray that produced the hit
    #[inline]
    fn diffuse_specular_intensity(&self, hit: &Hit, ray_direction: &Vec3f) -> (f64, f64) {
        // Diffuse light
        let mut diffuse_light_intensity: f64 = 0.1;
        let mut specular_light_intensity: f64 = 0.0;
        for light in self.lights.iter() {
            let light_dir = light.position.minus(&hit.point).normalize();
            let shadow_orig = match light_dir.dot(&hit.normal) > 0. {
                true => hit.point.plus(&hit.normal.times(BIAS)),
                false => hit.point.plus(&hit.normal.times(-BIAS)),
            };

            let shadow_ray = Ray::new(shadow_orig, light_dir.times(1.));
            let is_hit = self.get_intersection(&shadow_ray).is_some();

            if is_hit { continue; }
            let reflection = create_reflection_dir(&-light_dir, &hit.normal);
            specular_light_intensity += reflection.dot(&ray_direction).powf(2.);
            // dbg!(specular_light_intensity);
            //Calculate specular highlight
            diffuse_light_intensity +=
                light.intensity * f64::max(0., light_dir.dot(&hit.normal));
        }

        (diffuse_light_intensity, specular_light_intensity)
    }


    /// Returns the color of the pixel the ray originates from
    pub fn cast_ray(&self, ray: &Ray, max_depth: u32) -> Vec3f {
        let hit_params = self.get_intersection(ray);

        return match &hit_params {
            Some(params) => {
                let material = &params.sphere.material;
                let hit = params.to_hit(ray);
                let material_color = &material.color;
                let light_color = Vec3f::new(1.0, 1.0, 1.0);
                //Diffuse light
                let (diffuse_light_intensity, specular_light_intensity) =
                    self.diffuse_specular_intensity(&hit, &ray.direction);
                let mut color =
                    material_color.times(diffuse_light_intensity) * material.diffuse +
                    light_color.times(specular_light_intensity) * material.specular;

                // Refracted Light
                if params.sphere.material.translucency > 0. && max_depth > 0 {
                    let new_ray = snellius(
                        ray.direction,
                        hit.normal,
                        hit.point,
                        material.ref_index
                    );

                    if new_ray.is_none() { // Total reflection. No reflection implemented up to now
                        println!("None:");
                        return Vec3f::new(1.0, 1., 1.0)
                    }

                    let new_ray = new_ray.unwrap();
                    let ray_color = self.cast_ray(&new_ray, max_depth - 1);
                    color = color + material.translucency * ray_color;
                }
                color
            }
            None => Vec3f::new(0.4, 0.4, 0.3), //Default background color
        }
    }

    pub fn render(&self) -> Vec<Vec3f> {
        const NUM_ELEMENTS: usize = (WIDTH * HEIGHT) as usize;
        const FOV: f64 = (PI / 3.) as f64;

        let mut frame_buffer = Vec::<Vec3f>::with_capacity(NUM_ELEMENTS);
        frame_buffer.par_extend((0..NUM_ELEMENTS).into_par_iter()
            .map(|i| {
                let j: usize = i / WIDTH;
                let i: usize = i % WIDTH;

                let x: f64 = (2. * (i as f64 + 0.5) / WIDTH as f64 - 1.) * f64::tan(FOV / 2.);
                let y: f64 = -(2. * (j as f64 + 0.5) / WIDTH as f64 - 1.) * f64::tan(FOV / 2.);
                let ray = Ray::new(
                    Vec3f::new(0., 0., 0.),
                    Vec3f::new(x, y, -1.),
                );
                // println!("{}", i);
                let result = self.cast_ray(&ray, 2);
                result
            }
            ));

        frame_buffer
    }
}

impl Light {
    pub fn new(position: Vec3f, intensity: f64) -> Self {
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
        let lambda_expected = (Vec3f::new(-4., 1., 1.).length_squared()).sqrt();
        dbg!(lambda_expected);
        dbg!(lambda);
        assert!(lambda + ETA > lambda_expected);
        assert!(expected.equal_within_err(result));
    }

    #[test]
    fn test_ray_intersects_with_sphere() {
        let sphere = Sphere {
            center: Vec3f::new(3., 3., 0.),
            radius: 1.,
            material: Material::default(),
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
    fn test_ray_intersects_with_sphere_inner() {
        let sphere = Sphere {
            center: Vec3f::new(0., 0., 0.),
            radius: 1.,
            material: Material::default(),
        };

        let ray = Ray::new
            (
                Vec3f::new(0.9999, 0., 0.),
                Vec3f::new(0., 1., 0.),
            );

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
        println!("{:?}", result.unwrap());
    }

    #[test]
    fn test_ray_intersects_with_sphere_inside() {
        let sphere = Sphere {
            center: Vec3f::new(-0.1, -0.1, 0.),
            radius: 1.,
            material: Material::default(),
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
            material: Material::default(),
        };

        let ray = Ray::new
            (
                Vec3f::new(0., 0., 0.),
                Vec3f::new(1., 1., 0.),
            );

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
        let hit = result.unwrap().to_hit(&ray);
        let normal = hit.normal;
        dbg!(&normal);
        assert!(normal.equal_within_err(Vec3f::new(-1., -1., 0.).normalize()));
    }

    #[test]
    fn test_ray_intersect_normal_under() {
        let sphere = Sphere {
            center: Vec3f::new(3., 0., 0.),
            radius: 1.,
            material: Material::default(),
        };

        let ray = Ray::new
            (
                Vec3f::new(0., 0., 0.),
                Vec3f::new(1., 0., 0.),
            );

        let result = sphere.intersects_with_ray(&ray);
        assert_eq!(result.is_some(), true);
        let hit = result.unwrap().to_hit(&ray);
        let normal = hit.normal;
        assert_eq!(normal, Vec3f::new(-1., 0., 0.).normalize());
    }

    #[test]
    fn test_normalize() {
        let vec = Vec3f::new(2., 2., 0.);
        assert_eq!(vec.normalize(), Vec3f::new(2., 2., 0.).times(1. / f64::sqrt(8.)))
    }

    #[test]
    fn test_intersections() {
        let sphere = Sphere {
            center: Vec3f::new(-3., 1.5, -16.),
            radius: 4.,
            material: Material::default(),
        };
        let normal = Vec3f::new(-0.036804415141077684, 0.93360159907275841, 0.35641757705662025);

        let ray = Ray::new
            (
                Vec3f::new(-3.147217460509538, 5.2344060423785228, -14.574328778141195),
                Vec3f::new(-0.20005477263768598, 0.35391251108935218, -0.91363232344270351),
            );
    }
}
