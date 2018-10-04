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

//////

// Non-ascii idents and lower case globals for scientific constants.
#![feature(non_ascii_idents)]
#![allow(non_upper_case_globals)]

mod consts;
mod types;

use std::cell::{Cell, RefCell};  // todo see note below.
use std::rc::Rc;  // todo For global state; may handle this in main() instead.

use consts::*;
use types::{Cplx, Electron, Nucleus, State, Vec3};

fn deriv(f: &Fn(f64) -> f64, x: f64) -> f64 {
    // Take a numerical derivative
    // May need to curry input functions to fit this format.

    const dx = .00001;  // smaller values are more precise
    f(x + dx) - f(x) / dx 
}


fn Ψ(x: Vec3, t: f64) -> Cplx {

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

fn nuc_potential_field(nuclei: &Vec<Nucleus>, position: Vec3) -> f64 {
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

fn hamiltonian1(nuclei: Vec<Nucleus>, electrons: Vec<Electron>) -> f64 {
    // From P 22 of E. Cances et al.
    let M = nuclei.len();
    let N = electrons.len();

    // todo fix momentum terms    
    let ke_nuc: f64 = nuclei.iter().fold(0., |acc, nuc| acc - nuc.p().powi(2) / (2. * nuc.mass()));

    let ke_elec: f64 = electrons.iter().fold(0., |acc, nuc| acc - elec.p().powi(2) / 2.);

    // Electrostatic energies
    let mut electro_nuc_elec = 0.;
    for nuc in &nuclei {
        for elec in &electrons {
            electro_nuc_elec -= nuc.charge() / (elec.position - nuc.position).abs();
        }
    }

    let mut electro_elec_elec = 0.;
    for elec1 in &electrons {
        for elec2 in &electrons {
            electro_nuc_elec += 1. / (elec1.position - elec2.position).abs();
        }
    }

    let mut electro_nuc_nuc = 0.;
    for nuc1 in &nuclei {
        for nuc2 in &nuclei {
            electro_nuc_elec += nuc1.charge() * nuc2.charge() / (elec1.position - elec2.position).abs();
        }
    }

    ke_nuc + ke_elec + electro_nuc_elec + electro_elec_elec + electro_nuc_nuc
}

fn _wavefunc_nuclei(nuclei: Vec<Nucleus>) {
    // All nuclei have one wf. Born-Oppenheimer.
}

fn _wavefunc_electrons(electrons: Vec<Electron>) {
    // All elecs have one wf. Born-Oppenheimer.
}

fn rhs(nucleus: Nucleus, nuclei: Vec<Nucleus>, electrons: Vec<Electron>) {
    // Right-hand-side of ODE for numerical integration.

    // Nucleus classical n-body
    let field = nuc_elec_field(&nuclei, nucleus.position);
    let force = field.mul(nucleus.charge());
    let accel = force.div(nucleus.mass());
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
