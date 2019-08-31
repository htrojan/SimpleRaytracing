use std::f32::consts::PI;
use std::fs::File;
use std::io::Write;

const WIDTH: u32 = 1024;
const HEIGHT: u32 = 768;

#[derive(Debug, PartialOrd, PartialEq, Clone)]
struct Vec3f {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3f {
    fn new(x: f32, y: f32, z: f32) -> Vec3f {
        Vec3f { x, y, z }
    }

    #[inline]
    fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    #[inline]
    fn minus(&self, other: &Self) -> Self {
        Vec3f { x: self.x - other.x, y: self.y - other.y, z: self.z - other.z }
    }

    #[inline]
    fn plus(&self, other: Self) -> Self {
        Vec3f { x: self.x + other.x, y: self.y + other.y, z: self.z + other.z }
    }

    #[inline]
    fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn times(&self, factor: f32) -> Self {
        Vec3f { x: self.x * factor, y: self.y * factor, z: self.z * factor }
    }

    /// Returns the shortest point and a parameter for the straight line equation of the ray.
    /// If positive the point lies in direction of the ray.
    /// If negative in the opposite direction.
    fn shortest_point_to_ray(&self, ray: &Ray) -> (Self, f32) {
        let point_center = self.minus(&ray.origin);
        let direction_length_sq2 = ray.direction.length_squared();

        let plane_prod = ray.direction.dot(&point_center);

        let lambda: f32 = plane_prod / direction_length_sq2;

        let shortest_point = ray.origin.plus(ray.direction.times(lambda));
        (shortest_point, lambda)
    }
}

struct Sphere {
    center: Vec3f,
    radius: f32,
}

struct Ray {
    origin: Vec3f,
    direction: Vec3f,
}

struct Hit {
    normal: Vec3f,
    point: Vec3f,
}

impl Sphere {
    //The ray has to have a normalized direction vector
    fn intersects_with_ray(&self, ray: &Ray) -> Option<Hit>{
        let (shortest_point, lambda) = self.center.shortest_point_to_ray(&ray);

        let distance_vector = self.center.minus(&shortest_point);

        let l = self.radius * self.radius - distance_vector.length_squared();

        let is_hit: bool = (l > 0.) ;

        //No hits
        if !is_hit {
            return None;
        }

        //These are the two possible hit points
        let mut hit_1 = lambda - l;
        let hit_2 = lambda + l;

        if hit_1 < 0. {hit_1 = hit_2}
        if hit_1 < 0. { return None }
        let hit_point = ray.origin.plus(ray.direction.times(hit_1));
        let normal = hit_point.minus(&self.center);
        Some(Hit { point: hit_point, normal })
    }
}

struct Scene {
    objects: Vec<Sphere>,
    lights: Vec<Light>,
}

impl Scene {
    fn new(spheres: Vec<Sphere>, lights: Vec<Light>) -> Self {
        Scene { objects: spheres, lights }
    }

    /// Returns the color of the pixel the ray originates from
    fn cast_ray(&self, ray: &Ray) -> Vec3f {
        for o in self.objects.iter() {
            match o.intersects_with_ray(&ray) {
                Some(hit) => return Vec3f::new(0.2, 0.7, 0.3),
                _ => continue,
            }
        }
        // Background color
        Vec3f::new(0.4, 0.4, 0.3)
    }

    fn render(&self) -> Vec<Vec3f> {
        let mut frame_buffer = Vec::<Vec3f>::with_capacity((WIDTH * HEIGHT) as usize);

        let fov = PI / 2.;
        for j in 0..HEIGHT {
            for i in 0..WIDTH {
                let x: f32 = (2. * (i as f32 + 0.5) / WIDTH as f32 - 1.) * f32::tan(fov / 2.);
                let y: f32 = -(2. * (j as f32 + 0.5) / WIDTH as f32 - 1.) * f32::tan(fov / 2.);
                let ray = Ray {
                    origin: Vec3f::new(0., 0., 0.),
                    direction: Vec3f::new(x, y, -1.),
                };

                frame_buffer.push(self.cast_ray(&ray));
            }
        }

        frame_buffer
    }
}

struct Light {
    position: Vec3f,
    intensity: f32,
}

impl Light {
    fn new(position: Vec3f, intensity: f32) -> Self {
        Light { position, intensity }
    }
}


fn main() {
    let mut image_buffer = image::ImageBuffer::new(WIDTH, HEIGHT);


    println!("Rendering...");
    let spheres = vec![
        Sphere { radius: 2.0, center: Vec3f::new(-3., 0., -16.) },
        Sphere { radius: 2.0, center: Vec3f::new(-1., -1.5, -12.) },
        Sphere { radius: 3., center: Vec3f::new(1.5, -0.5, -18.) },
        Sphere { radius: 4., center: Vec3f::new(7., 5., -12.) },
    ];
    let lights = vec![
        Light {intensity: 1., position: Vec3f::new(0., 0., 0.)}
    ];
    let scene = Scene::new(spheres, lights);
    let mut frame_buffer = scene.render();


    println!("Writing image...");
    // Converts internal Vec3f representation to the format of the image library
    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        let color = frame_buffer.get(((y % WIDTH) * WIDTH + x) as usize).unwrap();
        *pixel = image::Rgb([(255. * color.x) as u8, (color.y * 255.) as u8, (color.z * 255.) as u8]);
    }

    image_buffer.save("output.png").unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shortest_point() {
        let p = Vec3f::new(0., 5., 6.);
        let ray = Ray
            {
                origin: Vec3f::new(2., 0., 1.),
                direction: Vec3f::new(-4., 1., 1.),
            };

        let (result, lambda) = p.shortest_point_to_ray(&ray);
        assert_eq!(result, Vec3f::new(-2., 1., 2.));
        assert_eq!(lambda, 1.);
    }

    #[test]
    fn test_ray_intersects_with_sphere() {
        let sphere = Sphere {
            center: Vec3f::new(3., 3., 0.),
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
}
