use ::std::collections::HashMap;

// member is private because I'm hoping there's some way
// (possibly insane) to get the nonzero optimization for
// Option<Element>

//// Represents a specific atomic number.
#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct Element(u16);

impl Element {
    pub fn from_atomic_number(n: u32) -> Option<Self>
    {
        if 1 <= n && n <= 999 { Some(Element(n as u16)) }
        else { None }
    }

    pub fn from_symbol(s: &str) -> Option<Self>
    {
        SHORT_TO_NUMBER.get(s).cloned()
            .map(Into::into)
            .and_then(Element::from_atomic_number)
    }

    pub fn atomic_number(&self) -> u32
    { self.0.into() }

    pub fn symbol(&self) -> &'static str
    { NUMBER_TO_SHORT[&self.0] }

    pub fn name(&self) -> &'static str
    { NUMBER_TO_AMERICAN[&self.0] }
}

// Table of elements with names that are not just systematically
// derived from the IUPAC recommendations.
const SPECIAL_NAMES: &'static [(u16, &'static str, &'static str)] = &[
    (001,  "H", "Hydrogen"),
    (002, "He", "Helium"),
    (003, "Li", "Lithium"),
    (004, "Be", "Beryllium"),
    (005,  "B", "Boron"),
    (006,  "C", "Carbon"),
    (007,  "N", "Nitrogen"),
    (008,  "O", "Oxygen"),
    (009,  "F", "Fluorine"),
    (010, "Ne", "Neon"),
    (011, "Na", "Sodium"),
    (012, "Mg", "Magnesium"),
    (013, "Al", "Aluminum"),
    (014, "Si", "Silicon"),
    (015,  "P", "Phosphorus"),
    (016,  "S", "Sulfur"),
    (017, "Cl", "Chlorine"),
    (018, "Ar", "Argon"),
    (019,  "K", "Potassium"),
    (020, "Ca", "Calcium"),
    (021, "Sc", "Scandium"),
    (022, "Ti", "Titanium"),
    (023,  "V", "Vanadium"),
    (024, "Cr", "Chromium"),
    (025, "Mn", "Manganese"),
    (026, "Fe", "Iron"),
    (027, "Co", "Cobalt"),
    (028, "Ni", "Nickel"),
    (029, "Cu", "Copper"),
    (030, "Zn", "Zinc"),
    (031, "Ga", "Gallium"),
    (032, "Ge", "Germanium"),
    (033, "As", "Arsenic"),
    (034, "Se", "Selenium"),
    (035, "Br", "Bromine"),
    (036, "Kr", "Krypton"),
    (037, "Rb", "Rubidium"),
    (038, "Sr", "Strontium"),
    (039,  "Y", "Yttrium"),
    (040, "Zr", "Zirconium"),
    (041, "Nb", "Niobium"),
    (042, "Mo", "Molybdenum"),
    (043, "Tc", "Technetium"),
    (044, "Ru", "Ruthenium"),
    (045, "Rh", "Rhodium"),
    (046, "Pd", "Palladium"),
    (047, "Ag", "Silver"),
    (048, "Cd", "Cadmium"),
    (049, "In", "Indium"),
    (050, "Sn", "Tin"),
    (051, "Sb", "Antimony"),
    (052, "Te", "Tellurium"),
    (053,  "I", "Iodine"),
    (054, "Xe", "Xenon"),
    (055, "Cs", "Caesium"),
    (056, "Ba", "Barium"),
    (057, "La", "Lanthanum"),
    (058, "Ce", "Cerium"),
    (059, "Pr", "Praseodymium"),
    (060, "Nd", "Neodymium"),
    (061, "Pm", "Promethium"),
    (062, "Sm", "Samarium"),
    (063, "Eu", "Europium"),
    (064, "Gd", "Gadolinium"),
    (065, "Tb", "Terbium"),
    (066, "Dy", "Dysprosium"),
    (067, "Ho", "Holmium"),
    (068, "Er", "Erbium"),
    (069, "Tm", "Thulium"),
    (070, "Yb", "Ytterbium"),
    (071, "Lu", "Lutetium"),
    (072, "Hf", "Hafnium"),
    (073, "Ta", "Tantalum"),
    (074,  "W", "Tungsten"),
    (075, "Re", "Rhenium"),
    (076, "Os", "Osmium"),
    (077, "Ir", "Iridium"),
    (078, "Pt", "Platinum"),
    (079, "Au", "Gold"),
    (080, "Hg", "Mercury"),
    (081, "Tl", "Thallium"),
    (082, "Pb", "Lead"),
    (083, "Bi", "Bismuth"),
    (084, "Po", "Polonium"),
    (085, "At", "Astatine"),
    (086, "Rn", "Radon"),
    (087, "Fr", "Francium"),
    (088, "Ra", "Radium"),
    (089, "Ac", "Actinium"),
    (090, "Th", "Thorium"),
    (091, "Pa", "Protactinium"),
    (092,  "U", "Uranium"),
    (093, "Np", "Neptunium"),
    (094, "Pu", "Plutonium"),
    (095, "Am", "Americium"),
    (096, "Cm", "Curium"),
    (097, "Bk", "Berkelium"),
    (098, "Cf", "Californium"),
    (099, "Es", "Einsteinium"),
    (100, "Fm", "Fermium"),
    (101, "Md", "Mendelevium"),
    (102, "No", "Nobelium"),
    (103, "Lr", "Lawrencium"),
    (104, "Rf", "Rutherfordium"),
    (105, "Db", "Dubnium"),
    (106, "Sg", "Seaborgium"),
    (107, "Bh", "Bohrium"),
    (108, "Hs", "Hassium"),
    (109, "Mt", "Meitnerium"),
    (110, "Ds", "Darmstadtium"),
    (111, "Rg", "Roentgenium"),
    (112, "Cn", "Copernicium"),
];

lazy_static!{
    static ref SHORT_TO_NUMBER: HashMap<&'static str, u16> =
    {
        // TODO: add systematic names for all elements 100-999,
        //       including those who also have official names
        SPECIAL_NAMES.iter()
            .map(|&(num, sym, _)| (sym, num))
            .collect()
    };

    static ref NUMBER_TO_SHORT: HashMap<u16, &'static str> =
    {
        // TODO: add systematic names for missing elements up to 999
        panic!("3 letter atomic symbols not currently supported");
        SPECIAL_NAMES.iter()
            .map(|&(num, sym, _)| (num, sym))
            .collect()
    };

    static ref NUMBER_TO_AMERICAN: HashMap<u16, &'static str> =
    {
        // TODO: add systematic names for others up to 999
        SPECIAL_NAMES.iter()
            .map(|&(num, _, american)| (num, american))
            .collect()
    };
}
