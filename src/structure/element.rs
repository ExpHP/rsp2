/* ************************************************************************ **
** This file is part of rsp2, and is licensed under EITHER the MIT license  **
** or the Apache 2.0 license, at your option.                               **
**                                                                          **
**     http://www.apache.org/licenses/LICENSE-2.0                           **
**     http://opensource.org/licenses/MIT                                   **
**                                                                          **
** Be aware that not all of rsp2 is provided under this permissive license, **
** and that the project as a whole is licensed under the GPL 3.0.           **
** ************************************************************************ */

// TODO I feel this belongs in a separate crate from
//      the structure types, where it can serve as a sort
//      of element database. However, currently, it's not
//      large enough to justify the effort.

use std::collections::HashMap;
use std::fmt;
use std::str;
use failure::Backtrace;

// member is private because I'm hoping there's some way
// (possibly insane) to get the nonzero optimization for
// Option<Element>

/// Represents a specific atomic number.
///
/// Only Elements up to `MAX_ATOMIC_NUMBER` are supported.
/// This limitation enables methods to return `&'static str`.
#[derive(Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct Element(u16);

const MAX_ATOMIC_NUMBER: u32 = 999;

#[derive(Debug, Fail)]
#[fail(display = "Unable to parse {}: {:?}", kind, text)]
pub struct ElementParseError {
    text: String,
    kind: &'static str, // "element", "element symbol", "element name"
    backtrace: Backtrace,
}

impl ElementParseError {
    fn new(kind: &'static str, s: &str) -> Self
    { ElementParseError {
        text: s.to_string(),
        kind: kind,
        backtrace: Backtrace::new(),
    }}
}

impl Element {
    pub fn from_atomic_number(n: u32) -> Option<Self>
    {
        if Self::is_valid_number(n) { Some(Element(n as u16)) }
        else { None }
    }

    fn is_valid_number(n: u32) -> bool
    { 1 <= n && n <= MAX_ATOMIC_NUMBER }

    pub fn from_symbol(s: &str) -> Result<Self, ElementParseError>
    {
        let &n = SHORT_TO_NUMBER.get(s).ok_or_else(|| ElementParseError::new("element symbol", s))?;
        debug_assert!(Self::is_valid_number(n.into()));
        Ok(Element(n))
    }

    pub fn get_from_symbol(s: &str) -> Self
    {
        let n = SHORT_TO_NUMBER.get(s).unwrap();
        Element(*n)
    }

    pub fn atomic_number(&self) -> u32
    { self.0.into() }

    pub fn symbol(&self) -> &'static str
    { NUMBER_TO_SHORT[&self.0] }

    pub fn name(&self) -> &'static str
    { NUMBER_TO_AMERICAN[&self.0] }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    { fmt::Display::fmt(self.symbol(), f) }
}

impl fmt::Debug for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result
    {
        match f.alternate() {
            false => fmt::Debug::fmt(self.symbol(), f),
            true  => fmt::Debug::fmt(self.name(), f),
        }
    }
}

impl str::FromStr for Element {
    type Err = ElementParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err>
    {
        let &n = DWIM_STR_TO_NUMBER.get(s).ok_or_else(|| ElementParseError::new("element", s))?;
        debug_assert!(Self::is_valid_number(n.into()));
        Ok(Element(n))
    }
}

#[cfg(feature = "serde")]
mod serde_impls {
    use super::*;
    use serde::{Serialize, Deserialize, ser, de};

    impl Serialize for Element {
        fn serialize<S: ser::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            self.symbol().serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Element {
        fn deserialize<D: de::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let raw: &str = <&str>::deserialize(deserializer)?;
            raw.parse().map_err(|_| {
                de::Error::invalid_value(de::Unexpected::Str(raw), &"an element name or symbol")
            })
        }
    }
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
        // TODO: add systematic names for all elements >= 100,
        //       including those who also have official names
        SPECIAL_NAMES.iter()
            .map(|&(num, sym, _)| (sym, num))
            .collect()
    };

    static ref NUMBER_TO_SHORT: HashMap<u16, &'static str> =
    {
        // TODO: add systematic names for missing elements
        //       up to MAX_ATOMIC_NUMBER
        SPECIAL_NAMES.iter()
            .map(|&(num, sym, _)| (num, sym))
            .collect()
    };

    static ref NUMBER_TO_AMERICAN: HashMap<u16, &'static str> =
    {
        // TODO: add systematic names for others up to MAX_ATOMIC_NUMBER
        SPECIAL_NAMES.iter()
            .map(|&(num, _, american)| (num, american))
            .collect()
    };

    static ref DWIM_STR_TO_NUMBER: DwimMap =
    {
        let mut map = DwimMap::new();

        // TODO: add systematic names, alternative spellings
        for &(num, sym, american) in SPECIAL_NAMES {
            map.insert(sym, num);
            map.insert(american, num);
        }
        map
    };
}

use self::dwim::DwimMap;
mod dwim {
    use super::*;
    /// Case-insensitive lookup that allows either the symbol or the name
    pub struct DwimMap(HashMap<String, u16>);

    impl DwimMap {
        pub fn new() -> DwimMap
        { DwimMap(Default::default()) }

        pub fn insert(&mut self, key: &str, value: u16)
        { self.0.insert(Self::canonicalize(key), value); }

        pub fn get(&self, key: &str) -> Option<&u16>
        { self.0.get(&Self::canonicalize(key)) }

        fn canonicalize(s: &str) -> String {
            let mut s = s.to_string();
            s.make_ascii_lowercase();
            s
        }
    }
}

macro_rules! define_consts {
    (
        pub mod $consts:ident {
            $( pub const $NAME:ident: Element = Element($num:expr); )+
        }
    ) => {
        // Define associated constants for convenience
        impl Element {
            $( pub const $NAME: Element = Element($num); )+
        }

        // Also put them in a mod, where they can be imported to be used unqualified.
        pub mod $consts {
            use super::*;

            $( pub const $NAME: Element = Element::$NAME; )+
        }
    };
}

define_consts! {
    pub mod consts {
        pub const HYDROGEN: Element = Element(001);
        pub const HELIUM: Element = Element(002);
        pub const LITHIUM: Element = Element(003);
        pub const BERYLLIUM: Element = Element(004);
        pub const BORON: Element = Element(005);
        pub const CARBON: Element = Element(006);
        pub const NITROGEN: Element = Element(007);
        pub const OXYGEN: Element = Element(008);
        pub const FLUORINE: Element = Element(009);
        pub const NEON: Element = Element(010);
        pub const SODIUM: Element = Element(011);
        pub const MAGNESIUM: Element = Element(012);
        pub const ALUMINUM: Element = Element(013);
        pub const SILICON: Element = Element(014);
        pub const PHOSPHORUS: Element = Element(015);
        pub const SULFUR: Element = Element(016);
        pub const CHLORINE: Element = Element(017);
        pub const ARGON: Element = Element(018);
        pub const POTASSIUM: Element = Element(019);
        pub const CALCIUM: Element = Element(020);
        pub const SCANDIUM: Element = Element(021);
        pub const TITANIUM: Element = Element(022);
        pub const VANADIUM: Element = Element(023);
        pub const CHROMIUM: Element = Element(024);
        pub const MANGANESE: Element = Element(025);
        pub const IRON: Element = Element(026);
        pub const COBALT: Element = Element(027);
        pub const NICKEL: Element = Element(028);
        pub const COPPER: Element = Element(029);
        pub const ZINC: Element = Element(030);
        pub const GALLIUM: Element = Element(031);
        pub const GERMANIUM: Element = Element(032);
        pub const ARSENIC: Element = Element(033);
        pub const SELENIUM: Element = Element(034);
        pub const BROMINE: Element = Element(035);
        pub const KRYPTON: Element = Element(036);
        pub const RUBIDIUM: Element = Element(037);
        pub const STRONTIUM: Element = Element(038);
        pub const YTTRIUM: Element = Element(039);
        pub const ZIRCONIUM: Element = Element(040);
        pub const NIOBIUM: Element = Element(041);
        pub const MOLYBDENUM: Element = Element(042);
        pub const TECHNETIUM: Element = Element(043);
        pub const RUTHENIUM: Element = Element(044);
        pub const RHODIUM: Element = Element(045);
        pub const PALLADIUM: Element = Element(046);
        pub const SILVER: Element = Element(047);
        pub const CADMIUM: Element = Element(048);
        pub const INDIUM: Element = Element(049);
        pub const TIN: Element = Element(050);
        pub const ANTIMONY: Element = Element(051);
        pub const TELLURIUM: Element = Element(052);
        pub const IODINE: Element = Element(053);
        pub const XENON: Element = Element(054);
        pub const CAESIUM: Element = Element(055);
        pub const BARIUM: Element = Element(056);
        pub const LANTHANUM: Element = Element(057);
        pub const CERIUM: Element = Element(058);
        pub const PRASEODYMIUM: Element = Element(059);
        pub const NEODYMIUM: Element = Element(060);
        pub const PROMETHIUM: Element = Element(061);
        pub const SAMARIUM: Element = Element(062);
        pub const EUROPIUM: Element = Element(063);
        pub const GADOLINIUM: Element = Element(064);
        pub const TERBIUM: Element = Element(065);
        pub const DYSPROSIUM: Element = Element(066);
        pub const HOLMIUM: Element = Element(067);
        pub const ERBIUM: Element = Element(068);
        pub const THULIUM: Element = Element(069);
        pub const YTTERBIUM: Element = Element(070);
        pub const LUTETIUM: Element = Element(071);
        pub const HAFNIUM: Element = Element(072);
        pub const TANTALUM: Element = Element(073);
        pub const TUNGSTEN: Element = Element(074);
        pub const RHENIUM: Element = Element(075);
        pub const OSMIUM: Element = Element(076);
        pub const IRIDIUM: Element = Element(077);
        pub const PLATINUM: Element = Element(078);
        pub const GOLD: Element = Element(079);
        pub const MERCURY: Element = Element(080);
        pub const THALLIUM: Element = Element(081);
        pub const LEAD: Element = Element(082);
        pub const BISMUTH: Element = Element(083);
        pub const POLONIUM: Element = Element(084);
        pub const ASTATINE: Element = Element(085);
        pub const RADON: Element = Element(086);
        pub const FRANCIUM: Element = Element(087);
        pub const RADIUM: Element = Element(088);
        pub const ACTINIUM: Element = Element(089);
        pub const THORIUM: Element = Element(090);
        pub const PROTACTINIUM: Element = Element(091);
        pub const URANIUM: Element = Element(092);
        pub const NEPTUNIUM: Element = Element(093);
        pub const PLUTONIUM: Element = Element(094);
        pub const AMERICIUM: Element = Element(095);
        pub const CURIUM: Element = Element(096);
        pub const BERKELIUM: Element = Element(097);
        pub const CALIFORNIUM: Element = Element(098);
        pub const EINSTEINIUM: Element = Element(099);
        pub const FERMIUM: Element = Element(100);
        pub const MENDELEVIUM: Element = Element(101);
        pub const NOBELIUM: Element = Element(102);
        pub const LAWRENCIUM: Element = Element(103);
        pub const RUTHERFORDIUM: Element = Element(104);
        pub const DUBNIUM: Element = Element(105);
        pub const SEABORGIUM: Element = Element(106);
        pub const BOHRIUM: Element = Element(107);
        pub const HASSIUM: Element = Element(108);
        pub const MEITNERIUM: Element = Element(109);
        pub const DARMSTADTIUM: Element = Element(110);
        pub const ROENTGENIUM: Element = Element(111);
        pub const COPERNICIUM: Element = Element(112);
    }
}
