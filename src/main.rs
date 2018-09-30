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

use std::ops::{Add, Sub, Mul};
use std::cell::{Cell, RefCell};  // todo see note below.
use std::rc::Rc;  // todo For global state; may handle this in main() instead.

// todo improve precision of these values.

const π = std::f64::consts::PI;

// Masses are in kg
const PROTON_MASS: f64 = 1.6726219e-27;
const NEUTRON_MASS: f64 = 1.674929e-27;
const ELECTRON_MASS: f64 = 9.109390e-31;

const ELEMEN_CHARGE: f64 = 1.60217662e-19;  // coulombs
const h: f64 = 6.2607004e-34; //m^2 * kg /s
const ħ: f64 = h / (2. * π);

const α = 0.007297352566206498  // Fine structure constant; dimensionless

const k: f64 = 8987551787.3681764;  // N * m^2 * C^-2

// Atomic unit system.
const m_e_UAS: f64 = 1.;
const e_UAS: f64 = 1.;
const ħ_UAS: f64 = 1.;
const ε_0_UAS: f64 = 1 / (4. * π)
// Speed of light in UAS is 1/α

const TIME_SCALE: f64 = 10e-12;  // Rough OOM scale.


#[derive(Debug)]  // todo Clone temp.
struct State {
    temp: f64,  // Kelvin
    nuclei: Vec<Nucleus>,
    electrons: Vec<Electron>,
}

#[derive(Copy, Clone, Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 {x, y, z}
    }

    fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn mag(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    // todo add addition/subtraction trait?
}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {x: self.x + other.x, y: self.y + other.y, z: self.z + other.z}
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {x: self.x - other.x, y: self.y - other.y, z: self.z - other.z}
    }
}

// impl Mul for Vec3 {
//     // Multiply by a scalar.
//     type Output = Vec3;

//     fn mul(self, val: f64) -> Vec3 {
//         Vec3 {x: self.x * val, y: self.y * val, z: self.z * val}
//     }
// }

fn mul(vect: Vec3, val: f64) -> Vec3 {
    // todo can't get trait working
        Vec3 {x: vect.x * val, y:vect.y * val, z: vect.z * val}
}

fn div(vect: Vec3, val: f64) -> Vec3 {
    // todo can't get trait working
        Vec3 {x: vect.x / val, y:vect.y / val, z: vect.z / val}
}

#[derive(Clone, Debug)]
struct Nucleus {
    // per Born-Oppenheimer approx, the nucleus is represented as classical.
    protons: i8,
    neutrons: i8,   
    position: Vec3,
    vel: Vec3,
}

enum Symmetry {
    Symmetric,
    Antisymmetrix,
}

impl Nucleus {
    fn new(protons: i8, neutrons: i8, position: Vec3, vel: Vec3) -> Nucleus {
        Nucleus {protons, neutrons, position, vel}
    }

    fn mass(&self) -> f64 {
        // Assume a point mass
        self.protons as f64 * PROTON_MASS + self.neutrons as f64 * NEUTRON_MASS
    }

    fn charge(&self) -> f64 {
        // Assume a point charge
        self.protons as f64 * e_UAS
    }

    fn symmetric(&self) -> Symmetry {
        if (self.protons + self.neutrons) % 2 == 0) {
            Symmetry::Symmetric
        }
        Symmetry::Antisymmetric
    }
}

impl Default for Nucleus {
    fn default() -> Nucleus {
        Nucleus {
            protons: 1,
            neutrons: 1,
            // For a nucleus composed of K nucleons, the spin variable
            // can take 1/4 * (K + 2)^2 values if K is even, and 
            // 1/4 * (K + 1)(K+3) values if K is odd.
            spin: i8, 
            position: Vec3::new(0., 0., 0.),
            vel: Vec3::new(0., 0., 0.),
        }
    }
}

enum Spin {
    Up,
    Down,
}

#[derive(Debug)]
struct Electron {
    spin: Spin,
    quantum_num: i8,
    energy: f64  // Joules

    // position as a simple vec??
    position: Vec3::new(0., 0., 0.),
}

// #[derive(Debug)]
// struct Atom {
//     // Could split up these fields into their own structs.
//     protons: i8,
//     neutrons: i8,
//     // electrons: i8,
//     electrons: Vec<Electron>,

//     posit: Vec3,
//     vel: Vec3,
// }

// impl Atom {
//     fn new(protons: i8) -> Atom {
//         Atom {
//             protons,
//             neutrons: protons,
//             electrons: Vec::new(),
//             posit: Vec3 {x: 0., y: 0., z: 0.},
//             vel: Vec3 {x: 0., y: 0., z: 0.}
//         }
//     }

//     fn mass(&self) -> f64 {
//         // Assume a point mass
//         self.protons as f64 * PROTON_MASS + self.neutrons as f64 * NEUTRON_MASS + 
//             self.electrons.len() as f64 * ELECTRON_MASS
//     }

//     fn charge(&self) -> f64 {
//         // Assume a point charge
//         self.protons as f64 * ELEMEN_CHARGE - self.electrons.len() as f64 * ELEMEN_CHARGE
//     }
// }

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

fn nuc_elec_field(nuclei: Vec<Nucleus>, position: Vec3) -> Vec3 {
    // Classical electric field.
    // Points from positive to negative.
    let mut result = Vec3 {x: 0., y: 0., z: 0.};
    for nucleus in &nuclei {
        let direc = position - nucleus.position;
        let dist = direc.mag();
        let unit_direc = div(direc, dist);

        // A simpler, but less explicit way to write this would be 
        // k * charge / (direc.mag().powi(3))

        result = result + mul(unit_direc, k * nucleus.charge() / dist.powi(2));
    }
    result
}

fn hamiltonian(nuclei: Vec<Nucleus>, electrons: Vec<Electron>) -> f64 {
    // From P 22 of E. Cances et al.
    const M = nuclei.len();
    const N = electrons.len();
    
    // what are the Δx_k and Δ_i terms? or momentums. Add them in! dist?
    let ke_nuc: f64 = nuclei.iter().fold(0., |acc, nuc| acc - 1/(2. * nuc.mass()));

    let ke_elec: f64 = electrons.iter().fold(0., |acc, nuc| acc - 1/2.));

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

// fn nuc_poten_direc(nuclei: Vec<Nucleus>, position: Vec3) -> f64 {
//     // Approximate the gradient of the field at this point.
// }

fn main() {
    let mut state = State {
        temp: 300.,
        nuclei: vec![
            Nucleus::new(1, 1, Vec3::new(1., 1., 0.), Vec3::new(0., 0., 0.)),
            Nucleus::new(1, 1, Vec3::new(-1., -1., 0.), Vec3::new(0., 0., 0.)),
            Nucleus::new(1, 1, Vec3::new(-1., 1., 0.), Vec3::new(0., 0., 0.)),
            Nucleus::new(1, 1, Vec3::new(1., -1., 0.), Vec3::new(0., 0., 0.)),
        ],
        electrons: vec![],
    };

    // todo this vs normal mut vec.
    let state2: Rc<RefCell<State>> =
        Rc::new(RefCell::new(
            State {
                temp: 300.,
                nuclei: vec![
                    Nucleus::new(1, 1, Vec3::new(1., 1., 0.), Vec3::new(0., 0., 0.)),
                    Nucleus::new(1, 1, Vec3::new(-1., -1., 0.), Vec3::new(0., 0., 0.)),
                    Nucleus::new(1, 1, Vec3::new(-1., 1., 0.), Vec3::new(0., 0., 0.)),
                    Nucleus::new(1, 1, Vec3::new(1., -1., 0.), Vec3::new(0., 0., 0.)),
                    ],
                electrons: vec![],
            }
        ));

    println!( "{:?}",
        nuc_elec_field(state.nuclei, Vec3::new(0., 0., 0.))
    );
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
