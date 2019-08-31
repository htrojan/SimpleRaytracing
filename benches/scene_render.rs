#[macro_use]
extern crate bencher;

use bencher::Bencher;
use raytacing_test::*;

fn bench_sample_scene(b: &mut Bencher) {
    let spheres = vec![
        Sphere { radius: 2.0, center: Vec3f::new(-3., 0., -16.) },
        Sphere { radius: 2.0, center: Vec3f::new(-1., -1.5, -12.) },
        Sphere { radius: 4., center: Vec3f::new(7., 5., -15.) },
    ];
    let lights = vec![
        Light::new ( Vec3f::new(-30., 20., 10.), 2.5  )
    ];
    let scene = Scene::new(spheres, lights);
    b.iter(|| scene.render())
}

benchmark_group!(benches, bench_sample_scene);
benchmark_main!(benches);
