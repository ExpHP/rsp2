

/// Marker trait for a built-in array type, i.e. `[T; n]`.
///
/// This is a workaround for the current lack of generic constants in rust.
/// Its methods are arbitrarily chosen to meet the needs of internal implementations
/// and honestly it would be best if you just forgot that this even exists. Capiche?
///
/// It is unsafe so that various properties can be trusted by unsafe code.
/// For instance, `align_of::<Self>() == align_of<Self::Element>()` and
/// `size_of::<Self>() == Self::array_len() * size_of<Self::Element>().
///
/// The complete set of properties relied on by unsafe code is unspecified
/// and may change; Please don't implement this for your own types.
pub unsafe trait IsArray: Sized
{
    /// `T` from the array type `[T; n]`.
    type Element;

    /// `n` from the array type `[T; n]`.
    fn array_len() -> usize;
    /// Perform the `&[T; n] -> &[T]` coercion.
    fn array_as_slice(&self) -> &[Self::Element];
    /// Perform the `&mut [T; n] -> &mut [T]` coercion.
    fn array_as_mut_slice(&mut self) -> &mut [Self::Element];
}

/// A poor-man's type family that can be used to construct
/// the type `[B; n]` from `[A; n]`.
pub trait WithElement<B>: IsArray {
    /// The type `[B; n]`.
    type Type: IsArray<Element=B> + WithElement<Self::Element, Type=Self>;
}

macro_rules! impl_is_array {
    {$n:expr} => {
        unsafe impl<T> IsArray for [T; $n] {
            type Element = T;

            #[inline(always)] fn array_len() -> usize { $n }
            #[inline(always)] fn array_as_slice(&self) -> &[T] { self }
            #[inline(always)] fn array_as_mut_slice(&mut self) -> &mut [T] { self }
        }

        impl<A, B> WithElement<B> for [A; $n] {
            type Type = [B; $n];
        }
    };
}

each_array_size!{ impl_is_array!{0...32} }

