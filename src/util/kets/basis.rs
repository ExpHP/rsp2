
/// Full double-precision rectangular representation,
/// for applications where precision matters.
pub mod lossless {

    pub use self::basis::Basis;
    pub mod basis {
        use super::KetRef;

        pub type Iter<'a> = Box<Iterator<Item=KetRef<'a>> + 'a>;

        #[derive(Debug, Clone)]
        #[derive(PartialEq)]
        pub struct Basis {
            rank: usize,
            // real of ket 1, then imag of ket 1, then real of ket 2...
            data: Vec<f64>,
        }

        impl Basis {
            pub fn new(data: Vec<f64>, rank: usize) -> Basis {
                Raw { data: data, rank: rank }.validate()
            }

            pub fn rank(&self) -> usize { self.rank }
            pub fn ket(&self, i: usize) -> KetRef {
                let r = self.rank;
                KetRef {
                    real: &self.data[r * (2 * i + 0) .. r * (2 * i + 1)],
                    imag: &self.data[r * (2 * i + 1) .. r * (2 * i + 2)],
                }
            }

            pub fn iter(&self) -> Iter {
                Box::new((0..self.rank()).map(move |i| self.ket(i)))
            }

            pub fn lossy_compress(&self) -> ::basis::compact::Basis {
                use ::complex::compact::PhaseTable;

                let rank = self.rank;
                let table = PhaseTable::get();
                let mut norm = vec![];
                let mut phase = vec![];
                for i in 0..rank {
                    let KetRef { real, imag } = self.ket(i);
                    for k in 0..rank {
                        phase.push(table.nearest_phase(imag[k].atan2(real[k])));
                        norm.push((real[k]*real[k] + imag[k]*imag[k]).sqrt() as f32);
                    }
                }
                ::basis::compact::basis::Raw { rank, phase, norm }.validate()
            }
        }

        /// Raw data type with no invariants, for serialization
        #[derive(Debug, Clone)]
        #[derive(Serialize, Deserialize)]
        #[derive(PartialEq)]
        pub struct Raw {
            pub rank: usize,
            pub data: Vec<f64>,
        }

        impl Raw {
            pub fn validate(self) -> Basis {
                let Raw { rank, data } = self;
                assert_eq!(data.len(), 2 * rank * rank);
                Basis { rank, data }
            }
        }

        impl Basis {
            pub fn raw(self) -> Raw {
                let Basis { rank, data } = self;
                Raw { rank, data }
            }
        }
    }

    pub use self::ket::Ket;
    pub use self::ket::KetRef;
    pub use self::ket::AsKetRef;
    pub mod ket {
        use ::complex::lossless::Rect;
        pub type IntoIter = Box<Iterator<Item=Rect>>;
        pub type Iter<'a> = Box<Iterator<Item=Rect> + 'a>;

        /// An owned ket.
        #[derive(Debug, Clone)]
        pub struct Ket {
            pub(crate) real: Vec<f64>,
            pub(crate) imag: Vec<f64>,
        }

        impl Ket {
            pub fn as_ref(&self) -> KetRef {
                KetRef {
                    real: &self.real,
                    imag: &self.imag,
                }
            }

            pub fn at(&self, i: usize) -> Rect { self.as_ref().at(i) }
            pub fn overlap<K: AsKetRef>(self, other: &K) -> f64 { self.as_ref().overlap(other) }
            pub fn iter(&self) -> Iter { self.as_ref().iter() }
        }

        impl IntoIterator for Ket {
            type Item = Rect;
            type IntoIter = IntoIter;
            fn into_iter(self) -> Self::IntoIter {
                let Ket { real, imag } = self;
                Box::new(real.into_iter().zip(imag).map(|(real,imag)| Rect { real, imag }))
            }
        }

        pub trait AsKetRef { fn as_ket_ref(&self) -> KetRef; }
        impl<'a> AsKetRef for KetRef<'a> { fn as_ket_ref(&self) -> KetRef { *self } }
        impl AsKetRef for Ket { fn as_ket_ref(&self) -> KetRef { self.as_ref() } }

        #[derive(Debug, Copy, Clone)]
        pub struct KetRef<'a> {
            pub(crate) real: &'a [f64],
            pub(crate) imag: &'a [f64],
        }

        impl<'a> KetRef<'a> {
            fn at(&self, i: usize) -> Rect {
                Rect {
                    real: self.real[i],
                    imag: self.imag[i],
                }
            }

            pub fn overlap<K: AsKetRef>(self, other: &K) -> f64 {
                let other = other.as_ket_ref();
                assert_eq!(self.real.len(), other.real.len());
                (0..self.real.len())
                    .map(|i| (self.at(i).conj() * other.at(i)))
                    .fold(Rect::zero(), |a,b| a + b)
                    .sqnorm()
            }

            pub fn iter(&self) -> Iter<'a> { Box::new(
                self.imag.iter().zip(self.real).map(|(&real, &imag)| Rect { real, imag })
            ) }
        }
    }
}

/// A lossily-compressed form that is amenable to further compression.
///
/// Suitable for e.g. band uncrossing.
pub mod compact {
    pub use self::basis::Basis;
    pub mod basis {
        use super::KetRef;

        pub type Iter<'a> = Box<Iterator<Item=KetRef<'a>> + 'a>;

        #[derive(Debug, Clone)]
        #[derive(PartialEq)]
        pub struct Basis {
            rank: usize,
            norm:  Vec<f32>,
            phase: Vec<u8>,
        }

        impl Basis {
            pub fn new(norm: Vec<f32>, phase: Vec<u8>, rank: usize) -> Basis {
                Raw { norm, phase, rank }.validate()
            }

            pub fn rank(&self) -> usize { self.rank }
            pub fn ket(&self, i: usize) -> KetRef {
                let r = self.rank;
                KetRef {
                    norm:  &self.norm [r * i .. r * (i + 1)],
                    phase: &self.phase[r * i .. r * (i + 1)],
                }
            }

            pub fn iter(&self) -> Iter {
                Box::new((0..self.rank()).map(move |i| self.ket(i)))
            }
        }

        /// Raw data type with no invariants, for serialization
        #[derive(Debug, Clone)]
        #[derive(Serialize, Deserialize)]
        #[derive(PartialEq)]
        pub struct Raw {
            pub rank: usize,
            pub norm:  Vec<f32>,
            pub phase: Vec<u8>,
        }

        impl Raw {
            pub fn validate(self) -> Basis {
                let Raw { rank, norm, phase } = self;
                assert_eq!(norm.len(),  rank * rank);
                assert_eq!(phase.len(), rank * rank);
                Basis { rank, norm, phase }
            }
        }

        impl Basis {
            pub fn raw(self) -> Raw {
                let Basis { rank, norm, phase } = self;
                Raw { rank, norm, phase }
            }
        }
    }

    pub use self::ket::Ket;
    pub use self::ket::KetRef;
    pub use self::ket::AsKetRef;
    pub mod ket {
        use ::complex::compact::{Rect, Polar, PhaseTable};
        pub type IntoIter = Box<Iterator<Item=Polar>>;
        pub type Iter<'a> = Box<Iterator<Item=Polar> + 'a>;

        /// An owned ket.
        #[derive(Debug, Clone)]
        pub struct Ket {
            pub(crate) norm:  Vec<f32>,
            pub(crate) phase: Vec<u8>,
        }

        impl Ket {
            pub fn as_ref(&self) -> KetRef {
                KetRef {
                    norm: &self.norm,
                    phase: &self.phase,
                }
            }

            pub fn at(&self, i: usize) -> Polar { self.as_ref().at(i) }
            pub fn overlap<K: AsKetRef>(self, other: &K) -> f64 { self.as_ref().overlap(other) }
        }

        impl IntoIterator for Ket {
            type Item = Polar;
            type IntoIter = IntoIter;
            fn into_iter(self) -> Self::IntoIter {
                let Ket { norm, phase } = self;
                Box::new(norm.into_iter().zip(phase)
                    .map(|(norm, phase)| Polar { norm, phase }))
            }
        }

        pub trait AsKetRef { fn as_ket_ref(&self) -> KetRef; }
        impl<'a> AsKetRef for KetRef<'a> { fn as_ket_ref(&self) -> KetRef { *self } }
        impl AsKetRef for Ket { fn as_ket_ref(&self) -> KetRef { self.as_ref() } }

        #[derive(Debug, Copy, Clone)]
        pub struct KetRef<'a> {
            pub(crate) norm:  &'a [f32],
            pub(crate) phase: &'a [u8],
        }

        impl<'a> KetRef<'a> {
            fn at(&self, i: usize) -> Polar {
                Polar {
                    norm: self.norm[i],
                    phase: self.phase[i],
                }
            }

            pub fn overlap<K: AsKetRef>(self, other: &K) -> f64 {
                let other = other.as_ket_ref();
                assert_eq!(self.norm.len(), other.norm.len());
                let table = PhaseTable::get();
                (0..self.norm.len())
                    .map(|i| (self.at(i).conj() * other.at(i)).to_rect(table))
                    .fold(Rect::zero(), |a,b| a + b)
                    .sqnorm() as f64
            }

            pub fn iter(&self) -> Iter<'a> {
                let KetRef { norm, phase } = *self;
                Box::new(norm.into_iter().zip(phase)
                    .map(|(&norm, &phase)| Polar { norm, phase }))
            }
        }
    }
}
