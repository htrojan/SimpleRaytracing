#[macro_use]
extern crate bencher;

use bencher::Bencher;
use raytacing_test::*;

fn bench_sample_scene(b: &mut Bencher) {
    let materials = vec![
        Material::new(Vec3f::new(0.3, 0.1, 0.1)),
    ];
    let spheres = vec![
        Sphere { radius: 2.0, center: Vec3f::new(-3., 0., -16.), material: Material::default()},
        Sphere { radius: 2.0, center: Vec3f::new(-1., -1.5, -12.), material: Material::default()},
        Sphere { radius: 3., center: Vec3f::new(1.5, -0.5, -18.), material: Material::default() },
        Sphere { radius: 4., center: Vec3f::new(7., 5., -15.), material: Material::default() },
    ];
    let lights = vec![
        Light::new ( Vec3f::new(-20., 20., 10.), 1.5  )
    ];
    let scene = Scene::new(spheres, lights);
    b.iter(|| scene.render())
}

benchmark_group!(benches, bench_sample_scene);
benchmark_main!(benches);
