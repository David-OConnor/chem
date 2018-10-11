// Computation Quantum Chemistry: A Primer, by Eric Cances

////// Assumptions

// Bohr-Oppenheimer:
//-Nuclei are classical point-like particles
//-The state of the electrons (represented by some electronic wavefunction ψ_e
//only depends on the positions in space (21 ..... XM) of the M nuclei,
//-this state is in fact the electronic ground state, i.e., the one that minimizes
//the energy of the electrons for the nuclear configuration (21 ..... 2M) under
//consideration.

// Non-relativistic. Heavier elements may need relativity to be accurate.

// Effects to include:
// -Special relativity, perterbation theory, spin, pauli exclusion
// uncertainty, magnetic fields.


//////

// Non-ascii idents and lower case globals for scientific constants.
#![feature(non_ascii_idents)]
#![allow(non_upper_case_globals)]

mod consts;
mod integrate;
mod types;

use std::cell::{Cell, RefCell};  // todo see note below.
use std::collections::HashMap;
use std::rc::Rc;  // todo For global state; may handle this in main() instead.

use consts::*;
use types::{Cplx, Electron, Nucleus, State, Vec3};

fn deriv(f: &Fn(f64) -> f64, x: f64) -> f64 {
    // Take a numerical derivative
    // May need to curry input functions to fit this format.
    let dx = 0.000001;  // smaller values are more precise
    f(x + dx) - f(x) / dx 
}

// fn nbody_rhs(nuclei: Vec<Nucleus>, electrons: Vec<Electrons>, t) -> (Vec<)
fn nbody_rhs(s: Vec<>. v: Vec<>, t: f64) -> (Vec<(f64, f64, f64), Vec<f64, f64, f64>) -> {
    let temp = elec_ode();
}

fn nbody_ode(nuclei: &Vec<Nucleus>, electrons: &Vec<Electron>, elec: &Electron) {
    // Position, and velocities, each of which is a Vec3.
    let nuc_x_0 = (
        Vec::new(), 
        Vec::new()
    );

    // todo not sure how to handle elecs here
    let elec_x_0 = (
        Vec::new(), 
        Vec::new()
    );

    let t_span = (0., 1000.);

    let field = nuc_elec_field(&nuclei, nucleus.position);
    let force = field.mul(nucleus.charge());
    let accel = force.div(nucleus.mass());

    let soln = solve_ivp(nbody_rhs, t_span, x_0);
}

fn elec_rhs(y: Vec<Cplx>, V: &fn(Vec3) -> f64, x: Vec3, E: f64) -> vec!<Cplx> {
    // RHS of ODE to solve the time-independent Schrodinger equation for the electron.
    // Broken into a system of 2 first-order ODEs.  Cannot solve analytically due
    // to the complicated potential function.

    // Non-relativistic, for now.
    
    // ψ is a function of 3d position.
    // V is already partially evaluated for the system state; it's only a function of position.

    // E is fixed for the time we're iterating at, and doesn't change as this ODE
    // evolves, since this is a time-ind eq (time is integrated over in the outer ODE).
    // E = Cplx::new(0., 1.) * ħ_UAS * (ψ[j] - ψ[j-1]) / Δt  ???

    let ψ = y[0];
    let φ = y[1];

    let ψ_p = φ;
    let φ_p = 2 * m_e_UAS / ħ_UAS.powi(2) * (V(x) - E) * ψ;

    vec![ψ_p, φ_p]
}

fn elec_ode(nuclei: &Vec<Nucleus>, electrons: &Vec<Electron>, elec: &Electron) {
    let ψ_0 = vec![1., 1.];  // ψ_0, φ_0
    let x_span = (-10., 10.);
    let y_span = (-10., 10.);
    let z_span = (-10., 10.);
    let step = .1;

    let V = |posit| nuc_elec_potential(nuclei, posit) + elec_elec_potential(electrons, posit);
      
    let solns = integrate::solve_pde_3d(elec_rhs, (x_span, y_span, z_span), ψ_0, V));
    }
}

fn elec_posit_prob(nuclei: &Vec<Nucleus>, electrons: &Vec<Electron>, elec: &Electron, x: &Vec3) -> f64 {
    // Find the probability of an electron being at a location
    let E = 1.; // todo
    let V_nuc = nuc_elec_potential(nuclei, x);
    let V_elec = elec_elec_potential(electrons, x);
    let V = V_nuc + V_elec;

    let ψ = elec_ode(nuclei, electrons, elec)

    (ψ.conj() * ψ).real
}

fn _calc_temp(nuclei: Vec<Nucleus>) -> f64 {
    // todo temp isn't just the velocity!
    nuclei.iter().fold(0., |acc, a| acc + a.vel.mag())
}

fn _energy(nuclei: Vec<Nucleus>, electrons: Vec<Electron>) -> f64 {
    // Total energy? Vib + rot + translat?
    0.  // todo
}

// fn nuc_poten_field(nuclei: Vec<Nucleus>, position: Vec3) -> f64 {
//     // Classical potential field
//     let mut result = 0.;
//     for nucleus in &nuclei {
//         let dist = (nucleus.position - position).mag();
//         result += k * nucleus.charge() / dist.powi(2);
//     }
//     result
// }

fn nuc_elec_field(nuclei: &Vec<Nucleus>, position: Vec3) -> Vec3 {
    // Classical electric field.
    // Points from positive to negative.
    let mut result = Vec3 {x: 0., y: 0., z: 0.};
    for nucleus in nuclei {
        let diff = position - nucleus.position;
        let dist = diff.mag();
        let direc = diff.div(dist);

        // A simpler, but less explicit way to write this would be 
        // k_UAS * charge / (direc.mag().powi(3))

        result = result + direc.mul(k_UAS * nucleus.charge() / dist.powi(2));
    }
    result
}

fn nuc_elec_potential(nuclei: &Vec<Nucleus>, position: &Vec3) -> f64 {
    // Classical potential field.
    // todo dry with elec field.
    let mut result = 0.;
    for nucleus in nuclei {
        let diff = position - nucleus.position;
        let dist = diff.mag();

        result += k_UAS * nucleus.charge() / dist.powi(2);
    }
    result
}

fn elec_elec_potential(electrons: &Vec<Electron>, position: &Vec3) -> f64 {
    // Electric  potential field from electrons.
    // Bold/questionable assumption: We can find a quasi-classical position
    // by adding up sample potentials from a range of positions, weighted by
    // the probability of the electron being at that position.
    let n_sample_pts = 36; // A multiple of 6, for both directions along 3 dimensions?

    let mut result = 0.;
    for electron in electrons {
        // Calculate a weighted-average position distance based on the electron's wave function.
        let mut positions = Vec::new();
        for i in 0..n_sample_pts {
            let test_posit = position;
            let elec_posit = ψ(electron.ψ);
            let prob = (elec_posit.conj() * elec_posit).real;
            positions.push((test_posit, prob));
            
        }
        let total_posit = Vec3::new(0., 0., 0.);
        for posit in positions {
            total_posit += posit.0.mul(posit.1)  // this whole mess is wrong todo
        }

        let diff = position - total_posit;
        let dist =  diff.mag();

        // todo I think we can think of this as calculating the expectation value
        // of position,
        // which is the weighted avg, and using that to calculate the potential.
        
        result += k_UAS * e_UAS / dist.powi(2);
    }
    result
}

// fn hamiltonian1(nuclei: Vec<Nucleus>, electrons: Vec<Electron>) -> f64 {
//     // From P 22 of E. Cances et al.
//     let M = nuclei.len();
//     let N = electrons.len();

//     // todo fix momentum terms    
//     let ke_nuc: f64 = nuclei.iter().fold(0., |acc, nuc| acc - nuc.p().powi(2) / (2. * nuc.mass()));

//     let ke_elec: f64 = electrons.iter().fold(0., |acc, nuc| acc - elec.p().powi(2) / 2.);

//     // Electrostatic energies
//     let mut electro_nuc_elec = 0.;
//     for nuc in &nuclei {
//         for elec in &electrons {
//             electro_nuc_elec -= nuc.charge() / (elec.position - nuc.position).abs();
//         }
//     }

//     let mut electro_elec_elec = 0.;
//     for elec1 in &electrons {
//         for elec2 in &electrons {
//             electro_nuc_elec += 1. / (elec1.position - elec2.position).abs();
//         }
//     }

//     let mut electro_nuc_nuc = 0.;
//     for nuc1 in &nuclei {
//         for nuc2 in &nuclei {
//             electro_nuc_elec += nuc1.charge() * nuc2.charge() / (elec1.position - elec2.position).abs();
//         }
//     }

//     ke_nuc + ke_elec + electro_nuc_elec + electro_elec_elec + electro_nuc_nuc
// }

fn _wavefunc_nuclei(nuclei: Vec<Nucleus>) {
    // All nuclei have one wf. Born-Oppenheimer.
}

fn _wavefunc_electrons(electrons: Vec<Electron>) {
    // All elecs have one wf. Born-Oppenheimer.
}

fn main() {
    let mut state = State {
        nuclei: vec![
            Nucleus::new(1, 1, Vec3::new(1., 1., 0.), Vec3::new(0., 0., 0.), 1),
            Nucleus::new(1, 1, Vec3::new(-1., -1., 0.), Vec3::new(0., 0., 0.), 1),
            Nucleus::new(1, 1, Vec3::new(-1., 1., 0.), Vec3::new(0., 0., 0.), 1),
            Nucleus::new(1, 1, Vec3::new(1., -1., 0.), Vec3::new(0., 0., 0.), 1),
        ],
        electrons: vec![],
    };

    // todo this vs normal mut vec.
    let state2: Rc<RefCell<State>> =
        Rc::new(RefCell::new(
            State {
                nuclei: vec![
                    Nucleus::new(1, 1, Vec3::new(1., 1., 0.), Vec3::new(0., 0., 0.), 1),
                    Nucleus::new(1, 1, Vec3::new(-1., -1., 0.), Vec3::new(0., 0., 0.), 1),
                    Nucleus::new(1, 1, Vec3::new(-1., 1., 0.), Vec3::new(0., 0., 0.), 1),
                    Nucleus::new(1, 1, Vec3::new(1., -1., 0.), Vec3::new(0., 0., 0.), 1),
                    ],
                electrons: vec![],
            }
        ));

    println!( "{:?}",
        nuc_elec_field(&state.nuclei, Vec3::new(0., 0., 0.))
    );
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
