use raytracing_test::*;
use raytracing_test::{HEIGHT, WIDTH};

fn main() {
    let mut image_buffer = image::ImageBuffer::new(WIDTH as u32, HEIGHT as u32);

    println!("Rendering...");
    let _materials = vec![Material::new_diffuse(Vec3f::new(0.3, 0.1, 0.1))];
    let _glass = Material {
        color: Vec3f::new(1., 1., 1.).normalize(),
        diffuse: 0.0,
        translucency: 0.8,
        ref_index: 1.05,
        specular: 0.5,
    };
    let _spec1 = Material {
        color: Vec3f::new(0.0, 0.2, 1.0).normalize(),
        diffuse: 0.5,
        translucency: 0.0,
        ref_index: 1.,
        specular: 0.4,
    };
    let _spec2 = Material {
        color: Vec3f::new(1.0, 0.0, 0.0).normalize(),
        diffuse: 0.8,
        translucency: 0.0,
        ref_index: 1.,
        specular: 0.2,
    };
    let _spheres = vec![
        Sphere {
            radius: 4.0,
            center: Vec3f::new(-3., 3., -25.),
            material: _spec2,
        },
        Sphere {
            radius: 3.0,
            center: Vec3f::new(-4.5, 1.5, -16.),
            material: _glass,
        },
        Sphere { radius: 3., center: Vec3f::new(4.5, 5.5, -18.), material: _spec1 },
        //         Sphere { radius: 4., center: Vec3f::new(7., 5., -16.), material: Material::default() },
    ];
    let _lights = vec![Light::new(Vec3f::new(-30., 20., -16.), 1.5),
                        Light::new(Vec3f::new(30., 15., 0.), 1.)];
    let scene = Scene::new(_spheres, _lights);
    let frame_buffer = scene.render();

    println!("Writing image...");
    // Converts internal Vec3f representation to the format of the image library
    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        let color = frame_buffer
            .get(((y % WIDTH as u32) * WIDTH as u32 + x) as usize)
            .unwrap();
        // Normalize colors if the renderer overshot the intensity of the lights
        let mut max = color.max_element();
        if max < 1. {
            max = 1.;
        }
        //        max = 1.;
        let color = color.times(1. / max);
        *pixel = image::Rgb([
            (color.x * 255.) as u8,
            (color.y * 255.) as u8,
            (color.z * 255.) as u8,
        ]);
    }

    image_buffer.save("output.png").unwrap();
}
