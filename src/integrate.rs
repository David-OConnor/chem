// Custom numerical integrators.

use std::collections::HashMap;

use types::{Cplx, Pde_3d_soln, Vec3};

pub const π: f64 = std::f64::consts::PI;

fn rk4_3d(
        f: &fn(Vec<Cplx>, &fn(Vec3) -> f64, Vec3, f64), 
        y: Vec<Cplx>,
        x: Vec3, 
        x_step: Vec3,
) -> Vec<Cplx> {

    let k1 = f(y, V, t, E) * x_step;
    let k2 = f(y + k1/2., V, t + x_step/2., E) * x_step;
    let k3 = f(y + k2/2., V, t + x_step/2., E) * x_step;
    let k4 = f(y + k3, V, t + x_step, E) * h;

    y + (k1 + 2.*(k2 + k3) + k4) / 6.;
}

#[derive(Debug)]
struct Pde_3d_soln {
    // Solns and posits have synchronized indices.
    // soln: Vec<Vec<Vec<f64>>>,
    // posits: Vec<Vec<Vec<f64>>>
    soln: HashMap<Vec3, Cplx>,
}

fn sphere_to_cart(r: f64, θ: f64, φ: f64) -> Vec3 {
    // Convert spherical coordinates to cartisian.
}

pub fn solve_pde_3d(rhs: &fn(Vec<Cplx>, &fn(Vec3) -> f64, Vec3, f64)) -> Vec<Cplx>, span: ((f64, f64), (f64, f64), (f64, f64)),
        y_0: Vec<Cplx>, V: &fn(Vec3) -> Cplx) -> Pde_3d_soln {
    // Solve a partial differential equation across three dimensions.
    // Currently somewhat specialized to our use case here.
    // For example, 3 spacial dimensions.

    // Let's attempt walks away in spherical coordinates (ISO),
    // starting at the expectation value.

    let mut r = 0.;
    let mut θ = 0.;
    let mut φ = 0.;

    let r_span = (0., 10., 10);
    let θ_span = (0., π / 2., 8);
    let φ_span = (0., π, 16);

    let mut y = s_0.clone();

    let mut soln = HashMap::new();

    for r in r_span.0 .. r_span.1 {
        for θ in θ_span.0 .. θ_span.1 {
            for φ in φ_span.0 .. φ_span.1 {
                let posit = sphere_to_cart(r, θ, φ);
                y = rk4(rhs, y, posit, E, V)
                soln.insert(posit, y_0)
            }
        }  
    }

}