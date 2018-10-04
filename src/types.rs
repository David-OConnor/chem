use std::f64::atan2;
use std::ops::{Add, Sub, Mul, Div};

use consts::*;


#[derive(Debug)]  // todo Clone temp.
pub struct State {
    pub nuclei: Vec<Nucleus>,
    pub electrons: Vec<Electron>,
}

#[derive(Copy, Clone, Debug)]
pub struct Cplx {
    pub real: f64,
    pub im: f64,
}

impl Cplx {
    pub fn new(real: f64, im: f64) -> Self {
        Self {real, im}
    }

    pub fn mag(&self) -> f64 {
        (self.real.powi(2) + self.im.powi(2)).sqrt()
    }

    pub fn phase(&self) -> f64 {
        atan2(self.im, self.real)
    }
}

impl Add for Cplx {
    type Output = Self;

    fn add(self, other: Self) -> Self{
        Self {real: self.real + other.real, im: self.im + other.im}
    }
}

impl Sub for Cplx {
    type Output = Self;

    fn sub(self, other: Self) -> Self{
        Self {real: self.real - other.real, im: self.im - other.im}
    }
}

impl Mul for Cplx {
    type Output = Self;

    fn mul(self, other: Self) -> Self{
        Self {
            real: self.real * other.real - self.im * other.im, 
            im: self.real * other.im + self.im * other.real,
        }
    }
}

// todo impl Div for Cplx.

#[derive(Copy, Clone, Debug)]
pub struct Vec3 {
    x: f64,
    y: f64,
    z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3 {x, y, z}
    }

    pub fn mag(&self) -> f64 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    
    pub fn mul(&self, val: f64) -> Self {
    // Can't get operator overload working due to other not beign a Vec3.
        Vec3 {x: self.x * val, y: self.y * val, z: self.z * val}
    }

    pub fn div(&self, val: f64) -> Self {
            Vec3 {x: self.x / val, y: self.y / val, z: self.z / val}
    }
}

impl Add for Vec3 {
    type Output = Self;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3 {x: self.x + other.x, y: self.y + other.y, z: self.z + other.z}
    }
}

impl Sub for Vec3 {
    type Output = Self;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3 {x: self.x - other.x, y: self.y - other.y, z: self.z - other.z}
    }
}

#[derive(Clone, Debug)]
pub struct Nucleus {
    // per Born-Oppenheimer approx, the nucleus is represented as classical.
    pub protons: i8,
    pub neutrons: i8,   
    pub position: Vec3,
    pub vel: Vec3,

    pub spin: i8,
}

#[derive(Debug)]
enum Symmetry {
    Symmetric,
    Antisymmetric,
}

impl Nucleus {
    pub fn new(protons: i8, neutrons: i8, position: Vec3, vel: Vec3, spin: i8) -> Nucleus {
        Nucleus {protons, neutrons, position, vel, spin}
    }

    pub fn mass(&self) -> f64 {
        // Assume a point mass
        self.protons as f64 * PROTON_MASS + self.neutrons as f64 * NEUTRON_MASS
    }

    pub fn charge(&self) -> f64 {
        // Assume a point charge
        self.protons as f64 * e_UAS
    }

    pub fn symmetric(&self) -> Symmetry {
        if (self.protons + self.neutrons) % 2 == 0 {
            Symmetry::Symmetric
        } else {
            Symmetry::Antisymmetric
        }
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
            spin: 1, 
            position: Vec3::new(0., 0., 0.),
            vel: Vec3::new(0., 0., 0.),
        }
    }
}

#[derive(Debug)]
enum Spin {
    Up,
    Down,
}

#[derive(Debug)]
pub struct Electron {
    pub spin: Spin,
    pub quantum_num: i8,
    pub energy: f64,  // Joules

    // position as a simple vec??
    pub position: Vec3,
}

impl Electron {
    fn expectation(&self) {
        // Calculate the expectation value of this electron. Not sure
        // if this is even appropriate, and if so, how to do so.
    }
}