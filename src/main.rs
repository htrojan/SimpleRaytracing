use raytacing_test::{WIDTH, HEIGHT};
use raytacing_test::*;


fn main() {
    let mut image_buffer = image::ImageBuffer::new(WIDTH as u32, HEIGHT as u32);


    println!("Rendering...");
    let _materials = vec![
        Material::new_diffuse(Vec3f::new(0.3, 0.1, 0.1)),
    ];
    let _spheres = vec![
        Sphere { radius: 2.0, center: Vec3f::new(3., 3., -16.), material: Material::default()},
        Sphere { radius: 3.0, center: Vec3f::new(-5., 1.5, -16.), material: Material::new_diffuse(Vec3f::new(0.1, 0.1, 0.3))},
//        Sphere { radius: 3., center: Vec3f::new(4.5, 5.5, -18.), material: Material::default() },
        Sphere { radius: 4., center: Vec3f::new(7., 5., -16.), material: Material::default() },
    ];
    let _lights = vec![
        Light::new ( Vec3f::new(-30., 20., -16.), 1.5  )
    ];
    let scene = Scene::new(_spheres, _lights);
    let frame_buffer = scene.render();


    println!("Writing image...");
    // Converts internal Vec3f representation to the format of the image library
    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        let color = frame_buffer.get(((y % WIDTH as u32) * WIDTH as u32 + x) as usize).unwrap();
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

