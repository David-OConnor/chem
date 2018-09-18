const proton_charge

// Masses are in kg
const proton_mass = 1.6726219 * 10.0.powi(-27)
const neutron_mass = 1.674929 * 10.0.powi(-27)
const electron_mass = 9.109390 * 10.0.powi(-31)

// coulombs
const elemen_charge = 1.60217662 * 10.0.powi(-19)

#[derive(Debug)]
struct State {
    temp: f32,  // Kelvin
    atoms: Vec<Atom>,
}

#[derive(Debug)]
struct Vec3 {
    x: f32,
    y: f32,
    z: f32,
}

impl Vec3 {
    fn add(&self, other: &Vec3) -> Vec3 {
        Vec3 {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }

    fn mag(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
}

#[derive(Debug)]
struct Electron {
    spin: i8,
    energy: f32  // Joules
}

#[derive(Debug)]
struct Atom {
    // Could split up these fields into their own structs.
    protons: i8,
    neutrons: i8,
    // electrons: i8,
    electrons: Vec<Electron>,

    posit: Vec3,
    vel: Vec3,
}

impl Atom {
    fn new(protons: i8) -> Atom {
        Atom {
            protons,
            neutrons: protons,
            electrons: Vec::new(),
            posit: Vec3 {x: 0., y: 0., z: 0.},
            vel: Vec3 {x: 0., y: 0., z: 0.}
        }
    }

    fn mass(&self) -> f32 {
        // Assume a point mass
        self.protons * proton_mass + self.neutrons * neutron_mass + 
            self.electrons.len() * electron_mass
    }

    fn charge(&self) -> f32 {
        // Assume a point charge
        self.protons * elemen_charge - self.electrons.len() * elemen_charge
    }
}

fn calc_temp(atoms: Vec<Atom>) -> f32 {
    atoms.iter().fold(0., |acc, a| acc + a.velocity.mag())
}

fn main() {
    let mut state = State {
        temp: 300.,
        atoms: vec![],
    };
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
