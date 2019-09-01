use std::fs::File;
use std::io::Write;
use raytacing_test::{WIDTH, HEIGHT};
use raytacing_test::*;


fn main() {
    let mut image_buffer = image::ImageBuffer::new(WIDTH, HEIGHT);


    println!("Rendering...");
    let materials = vec![
        Material::new(Vec3f::new(0.3, 0.1, 0.1)),
    ];
    let spheres = vec![
        Sphere { radius: 2.0, center: Vec3f::new(-3., 0., -16.), material: Material::default()},
        Sphere { radius: 2.0, center: Vec3f::new(-1., -1.5, -12.), material: Material::new(Vec3f::new(0.1, 0.1, 0.3))},
        Sphere { radius: 3., center: Vec3f::new(1.5, -0.5, -18.), material: Material::default() },
        Sphere { radius: 4., center: Vec3f::new(7., 5., -15.), material: Material::default() },
    ];
    let lights = vec![
        Light::new ( Vec3f::new(-30., 20., -5.), 1.5  )
    ];
    let scene = Scene::new(spheres, lights);
    let mut frame_buffer = scene.render();


    println!("Writing image...");
    // Converts internal Vec3f representation to the format of the image library
    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        let color = frame_buffer.get(((y % WIDTH) * WIDTH + x) as usize).unwrap();
        // Normalize colors if the renderer overshot the intensity of the lights
        let mut max = color.max_element();
        if max < 1. {
            max = 1.;
        }
//        max = 1.;
        let color = color.times(1. / max);
        *pixel = image::Rgb([(color.x * 255.) as u8, (color.y * 255.) as u8, (color.z * 255.) as u8]);
    }

    image_buffer.save("output.png").unwrap();
}

