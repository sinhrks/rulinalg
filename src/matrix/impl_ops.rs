use super::{Matrix, MatrixSlice, MatrixSliceMut};
use super::{Row, RowMut, Column, ColumnMut};
use super::{BaseMatrix, BaseMatrixMut};

use super::super::utils;
use super::super::vector::Vector;

use std::ops::{Mul, Div, Add, Sub, Rem,
               MulAssign, DivAssign, AddAssign, SubAssign, RemAssign,
               Neg, Not,
               BitAnd, BitOr, BitXor, BitAndAssign, BitOrAssign, BitXorAssign,
               Index, IndexMut};
use libnum::Zero;

/// Indexes matrix.
///
/// Takes row index first then column.
impl<T> Index<[usize; 2]> for Matrix<T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe { &*(self.get_unchecked(idx)) }
    }
}

macro_rules! impl_index_slice (
    ($slice_type:ident, $doc:expr) => (
/// Indexes
#[doc=$doc]
/// Takes row index first then column.
impl<'a, T> Index<[usize; 2]> for $slice_type<'a, T> {
    type Output = T;

    fn index(&self, idx: [usize; 2]) -> &T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe {
            &*(self.get_unchecked(idx))
        }
    }
}
    );
);

impl_index_slice!(MatrixSlice, "matrix slice.");
impl_index_slice!(MatrixSliceMut, "mutable matrix slice.");

/// Indexes mutable matrix slice.
///
/// Takes row index first then column.
impl<'a, T> IndexMut<[usize; 2]> for MatrixSliceMut<'a, T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");

        unsafe { &mut *(self.ptr.offset((idx[0] * self.row_stride + idx[1]) as isize)) }
    }
}

/// Indexes mutable matrix.
///
/// Takes row index first then column.
impl<T> IndexMut<[usize; 2]> for Matrix<T> {
    fn index_mut(&mut self, idx: [usize; 2]) -> &mut T {
        assert!(idx[0] < self.rows,
                "Row index is greater than row dimension.");
        assert!(idx[1] < self.cols,
                "Column index is greater than column dimension.");
        let self_cols = self.cols;
        unsafe { self.data.get_unchecked_mut(idx[0] * self_cols + idx[1]) }
    }
}

impl<'a, T> Index<usize> for Row<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.row[[0, idx]]
    }
}

impl<'a, T> Index<usize> for RowMut<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.row[[0, idx]]
    }
}

impl<'a, T> IndexMut<usize> for RowMut<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.row[[0, idx]]
    }
}

impl<'a, T> Index<usize> for Column<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.col[[idx, 0]]
    }
}

impl<'a, T> Index<usize> for ColumnMut<'a, T> {
    type Output = T;
    fn index(&self, idx: usize) -> &T {
        &self.col[[idx, 0]]
    }
}

impl<'a, T> IndexMut<usize> for ColumnMut<'a, T> {
    fn index_mut(&mut self, idx: usize) -> &mut T {
        &mut self.col[[idx, 0]]
    }
}

macro_rules! impl_bin_op_scalar_slice (
    ($trt:ident, $op:ident, $slice:ident, $doc:expr) => (

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, T> $trt<T> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, 'b, T> $trt<&'b T> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        (&self).$op(f)
    }
}

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, 'b, T> $trt<T> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}

/// Scalar
#[doc=$doc]
/// with matrix slice.
impl<'a, 'b, 'c, T> $trt<&'c T> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.iter().map(|v| (*v).$op(*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);
macro_rules! impl_bin_op_scalar_slice_patterns (
    ($trt:ident, $op:ident, $doc:expr) => (
        impl_bin_op_scalar_slice!($trt, $op, MatrixSlice, $doc);
        impl_bin_op_scalar_slice!($trt, $op, MatrixSliceMut, $doc);
    );
);
impl_bin_op_scalar_slice_patterns!(Mul, mul, "multiplication");
impl_bin_op_scalar_slice_patterns!(Div, div, "division");
impl_bin_op_scalar_slice_patterns!(Add, add, "addition");
impl_bin_op_scalar_slice_patterns!(Sub, sub, "subtraction");
impl_bin_op_scalar_slice_patterns!(Rem, rem, "remainder");
impl_bin_op_scalar_slice_patterns!(BitAnd, bitand, "bitwise-and");
impl_bin_op_scalar_slice_patterns!(BitOr, bitor, "bitwise-xor");
impl_bin_op_scalar_slice_patterns!(BitXor, bitxor, "bitwise-xor");

macro_rules! impl_bin_op_scalar_matrix (
    ($trt:ident, $op:ident, $doc:expr) => (

/// Scalar
#[doc=$doc]
/// with matrix.
///
/// Will reuse the memory allocated for the existing matrix.
impl<T> $trt<T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (self).$op(&f)
    }
}


/// Scalar
#[doc=$doc]
/// with matrix.
///
/// Will reuse the memory allocated for the existing matrix.
impl<'a, T> $trt<&'a T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(mut self, f: &T) -> Matrix<T> {
        for val in &mut self.data {
        	*val = (*val).$op(*f)
        }

        self
    }
}


/// Scalar
#[doc=$doc]
/// with matrix.
impl<'a, T> $trt<T> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: T) -> Matrix<T> {
        (&self).$op(&f)
    }
}


/// Scalar
#[doc=$doc]
/// with matrix.
impl<'a, 'b, T> $trt<&'b T> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, f: &T) -> Matrix<T> {
        let new_data: Vec<T> = self.data.iter().map(|v| (*v).$op(*f)).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);

impl_bin_op_scalar_matrix!(Mul, mul, "multiplication");
impl_bin_op_scalar_matrix!(Div, div, "division");
impl_bin_op_scalar_matrix!(Add, add, "addition");
impl_bin_op_scalar_matrix!(Sub, sub, "subtraction");
impl_bin_op_scalar_matrix!(Rem, rem, "remainder");
impl_bin_op_scalar_matrix!(BitAnd, bitand, "bitwise-and");
impl_bin_op_scalar_matrix!(BitOr, bitor, "bitwise-or");
impl_bin_op_scalar_matrix!(BitXor, bitxor, "bitwise-xor");

/// Multiplies matrix by vector.
impl<T> Mul<Vector<T>> for Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        (&self) * (&m)
    }
}

/// Multiplies matrix by vector.
impl<'a, T> Mul<Vector<T>> for &'a Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, m: Vector<T>) -> Vector<T> {
        self * (&m)
    }
}

/// Multiplies matrix by vector.
impl<'a, T> Mul<&'a Vector<T>> for Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, m: &Vector<T>) -> Vector<T> {
        (&self) * m
    }
}

/// Multiplies matrix by vector.
impl<'a, 'b, T> Mul<&'b Vector<T>> for &'a Matrix<T>
    where T: Copy + Zero + Mul<T, Output = T> + Add<T, Output = T>
{
    type Output = Vector<T>;

    fn mul(self, v: &Vector<T>) -> Vector<T> {
        assert!(v.size() == self.cols, "Matrix and Vector dimensions do not agree.");

        let mut new_data = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            new_data.push(utils::dot(&self.data[i * self.cols..(i + 1) * self.cols], v.data()));
        }

        Vector::new(new_data)
    }
}

macro_rules! impl_bin_op_slice (
    ($trt:ident, $op:ident, $slice_1:ident, $slice_2:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, T> $trt<$slice_2<'b, T>> for $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: $slice_2<T>) -> Matrix<T> {
        (&self).$op(&s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, T> $trt<$slice_2<'b, T>> for &'c $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: $slice_2<T>) -> Matrix<T> {
        (self).$op(&s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, T> $trt<&'c $slice_2<'b, T>> for $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: &$slice_2<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between the slices.
impl<'a, 'b, 'c, 'd, T> $trt<&'d $slice_2<'b, T>> for &'c $slice_1<'a, T>
    where T : Copy + $trt<T, Output=T>
{
    type Output = Matrix<T>;

    fn $op(self, s: &$slice_2<T>) -> Matrix<T> {
        assert!(self.cols == s.cols, "Column dimensions do not agree.");
        assert!(self.rows == s.rows, "Row dimensions do not agree.");

        let mut res_data : Vec<T> = self.iter().cloned().collect();
        let s_data : Vec<T> = s.iter().cloned().collect();

        utils::in_place_vec_bin_op(&mut res_data, &s_data, |x, &y| { *x = (*x).$op(y) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: res_data,
        }
    }
}
    );
);

macro_rules! impl_bin_op_slice_patterns (
    ($trt:ident, $op:ident, $doc:expr) => (
        impl_bin_op_slice!($trt, $op, MatrixSlice, MatrixSlice, $doc);
        impl_bin_op_slice!($trt, $op, MatrixSliceMut, MatrixSlice, $doc);
        impl_bin_op_slice!($trt, $op, MatrixSlice, MatrixSliceMut, $doc);
        impl_bin_op_slice!($trt, $op, MatrixSliceMut, MatrixSliceMut, $doc);
    );
);
impl_bin_op_slice_patterns!(Add, add, "addition");
impl_bin_op_slice_patterns!(Sub, sub, "subtraction");
impl_bin_op_slice_patterns!(Rem, rem, "remainder");
impl_bin_op_slice_patterns!(BitAnd, bitand, "bitwise-and");
impl_bin_op_slice_patterns!(BitOr, bitor, "bitwise-or");
impl_bin_op_slice_patterns!(BitXor, bitxor, "bitwise-xor");


macro_rules! impl_bin_op_mat_slice (
    ($trt:ident, $op:ident, $slice:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, T> $trt<Matrix<T>> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        (&self).$op(&m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<Matrix<T>> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        self.$op(&m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<&'b Matrix<T>> for $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        (&self).$op(m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, 'c, T> $trt<&'c Matrix<T>> for &'b $slice<'a, T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = self.iter().cloned().collect();
        utils::in_place_vec_bin_op(&mut new_data, &m.data(), |x, &y| { *x = (*x).$op(y) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, T> $trt<$slice<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<$slice<'a, T>> for &'b Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: $slice<T>) -> Matrix<T> {
        self.$op(&s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, T> $trt<&'b $slice<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice<T>) -> Matrix<T> {
        (&self).$op(s)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between `Matrix` and `MatrixSlice`.
impl<'a, 'b, 'c, T> $trt<&'c $slice<'a, T>> for &'b Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, s: &$slice<T>) -> Matrix<T> {
        assert!(self.cols == s.cols, "Column dimensions do not agree.");
        assert!(self.rows == s.rows, "Row dimensions do not agree.");

        let mut new_data : Vec<T> = s.iter().cloned().collect();
        utils::in_place_vec_bin_op(&mut new_data, self.data(), |x, &y| { *x = (y).$op(*x) });

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);

macro_rules! impl_bin_op_mat_slice_patterns (
    ($trt:ident, $op:ident, $doc:expr) => (
        impl_bin_op_mat_slice!($trt, $op, MatrixSlice, $doc);
        impl_bin_op_mat_slice!($trt, $op, MatrixSliceMut, $doc);
    );
);
impl_bin_op_mat_slice_patterns!(Add, add, "addition");
impl_bin_op_mat_slice_patterns!(Sub, sub, "subtraction");
impl_bin_op_mat_slice_patterns!(Rem, rem, "remainder");
impl_bin_op_mat_slice_patterns!(BitAnd, bitand, "bitwise-and");
impl_bin_op_mat_slice_patterns!(BitOr, bitor, "bitwise-or");
impl_bin_op_mat_slice_patterns!(BitXor, bitxor, "bitwise-xor");

macro_rules! impl_bin_op_mat (
    ($trt:ident, $op:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
///
/// This will reuse allocated memory from `self`.
impl<T> $trt<Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: Matrix<T>) -> Matrix<T> {
        self.$op(&m)
    }
}

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
///
/// This will reuse allocated memory from `m`.
impl<'a, T> $trt<Matrix<T>> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, mut m: Matrix<T>) -> Matrix<T> {
    	assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        utils::in_place_vec_bin_op(&mut m.data, &self.data, |x,&y| {*x = (y).$op(*x)});
        m
    }
}

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
///
/// This will reuse allocated memory from `self`.
impl<'a, T> $trt<&'a Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(mut self, m: &Matrix<T>) -> Matrix<T> {
    	assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        utils::in_place_vec_bin_op(&mut self.data, &m.data, |x,&y| {*x = (*x).$op(y)});
        self
    }
}

/// Performs elementwise
#[doc=$doc]
/// between two matrices.
impl<'a, 'b, T> $trt<&'b Matrix<T>> for &'a Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    type Output = Matrix<T>;

    fn $op(self, m: &Matrix<T>) -> Matrix<T> {
        assert!(self.cols == m.cols, "Column dimensions do not agree.");
        assert!(self.rows == m.rows, "Row dimensions do not agree.");

        let new_data = utils::vec_bin_op(&self.data, &m.data, |x, y| { x.$op(y) });

        Matrix {
        	rows: self.rows,
        	cols: self.cols,
        	data: new_data,
        }
    }
}
    );
);

impl_bin_op_mat!(Add, add, "addition");
impl_bin_op_mat!(Sub, sub, "subtraction");
impl_bin_op_mat!(Rem, rem, "remainder");
impl_bin_op_mat!(BitAnd, bitand, "bitwise-and");
impl_bin_op_mat!(BitOr, bitor, "bitwise-or");
impl_bin_op_mat!(BitXor, bitxor, "bitwise-xor");

macro_rules! impl_op_assign_mat_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs
#[doc=$doc]
/// assignment between a matrix and a scalar.
impl<T> $assign_trt<T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in &mut self.data {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a matrix and a scalar.
impl<'a, T> $assign_trt<&'a T> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in &mut self.data {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);

impl_op_assign_mat_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");
impl_op_assign_mat_scalar!(DivAssign, Div, div, div_assign, "division");
impl_op_assign_mat_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_mat_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_mat_scalar!(RemAssign, Rem, rem, rem_assign, "remainder");
impl_op_assign_mat_scalar!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_op_assign_mat_scalar!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_op_assign_mat_scalar!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

macro_rules! impl_op_assign_slice_scalar (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs
#[doc=$doc]
/// assignment between a mutable matrix slice and a scalar.
impl<'a, T> $assign_trt<T> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: T) {
        for x in self.iter_mut() {
            *x = (*x).$op(_rhs)
        }
    }
}

/// Performs
#[doc=$doc]
/// assignment between a mutable matrix slice and a scalar.
impl<'a, 'b, T> $assign_trt<&'b T> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &T) {
        for x in self.iter_mut() {
            *x = (*x).$op(*_rhs)
        }
    }
}
    );
);

impl_op_assign_slice_scalar!(MulAssign, Mul, mul, mul_assign, "multiplication");
impl_op_assign_slice_scalar!(DivAssign, Div, div, div_assign, "division");
impl_op_assign_slice_scalar!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_slice_scalar!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_slice_scalar!(RemAssign, Rem, rem, rem_assign, "remainder");
impl_op_assign_slice_scalar!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_op_assign_slice_scalar!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_op_assign_slice_scalar!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

macro_rules! impl_op_assign_mat (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<T> $assign_trt<Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T>  {
    fn $op_assign(&mut self, _rhs: Matrix<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, T> $assign_trt<&'a Matrix<T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &Matrix<T>) {
        utils::in_place_vec_bin_op(&mut self.data, &_rhs.data, |x, &y| {*x = (*x).$op(y) });
    }
}
    );
);

impl_op_assign_mat!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_mat!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_mat!(RemAssign, Rem, rem, rem_assign, "remainder");
impl_op_assign_mat!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_op_assign_mat!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_op_assign_mat!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

macro_rules! impl_op_assign_slice_mat (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, T> $assign_trt<Matrix<T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: Matrix<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut().zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, T> $assign_trt<&'b Matrix<T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: &Matrix<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut()
                                        .zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}
    );
);

impl_op_assign_slice_mat!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_slice_mat!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_slice_mat!(RemAssign, Rem, rem, rem_assign, "remainder");
impl_op_assign_slice_mat!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_op_assign_slice_mat!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_op_assign_slice_mat!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

macro_rules! impl_op_assign_slice (
    ($target_slice:ident, $assign_trt:ident,
        $trt:ident, $op:ident,
        $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, T> $assign_trt<$target_slice<'b, T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: $target_slice<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut()
                                            .zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, 'c, T> $assign_trt<&'c $target_slice<'b, T>> for MatrixSliceMut<'a, T>
    where T: Copy + $trt<T, Output=T>
{
    fn $op_assign(&mut self, _rhs: &$target_slice<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut()
                                            .zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}
    );
);

macro_rules! impl_op_assign_slice_patterns (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (
        impl_op_assign_slice!(MatrixSlice, $assign_trt, $trt, $op, $op_assign, $doc);
        impl_op_assign_slice!(MatrixSliceMut, $assign_trt, $trt, $op, $op_assign, $doc);
    );
);
impl_op_assign_slice_patterns!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_slice_patterns!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_slice_patterns!(RemAssign, Rem, rem, rem_assign, "remainder");
impl_op_assign_slice_patterns!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_op_assign_slice_patterns!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_op_assign_slice_patterns!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

macro_rules! impl_op_assign_mat_slice (
    ($target_mat:ident, $assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, T> $assign_trt<$target_mat<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: $target_mat<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut().zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}

/// Performs elementwise
#[doc=$doc]
/// assignment between two matrices.
impl<'a, 'b, T> $assign_trt<&'b $target_mat<'a, T>> for Matrix<T>
    where T: Copy + $trt<T, Output=T> {
    fn $op_assign(&mut self, _rhs: &$target_mat<T>) {
        for (mut slice_row, target_row) in self.row_iter_mut().zip(_rhs.row_iter()) {
            utils::in_place_vec_bin_op(slice_row.raw_slice_mut(),
                                        target_row.raw_slice(),
                                        |x, &y| {*x = (*x).$op(y) });
        }
    }
}
    );
);

macro_rules! impl_op_assign_mat_slice_patterns (
    ($assign_trt:ident, $trt:ident, $op:ident, $op_assign:ident, $doc:expr) => (
        impl_op_assign_mat_slice!(MatrixSlice, $assign_trt, $trt, $op, $op_assign, $doc);
        impl_op_assign_mat_slice!(MatrixSliceMut, $assign_trt, $trt, $op, $op_assign, $doc);
    );
);

impl_op_assign_mat_slice_patterns!(AddAssign, Add, add, add_assign, "addition");
impl_op_assign_mat_slice_patterns!(SubAssign, Sub, sub, sub_assign, "subtraction");
impl_op_assign_mat_slice_patterns!(RemAssign, Rem, rem, rem_assign, "remainder");
impl_op_assign_mat_slice_patterns!(BitAndAssign, BitAnd, bitand, bitand_assign, "bitwise-and");
impl_op_assign_mat_slice_patterns!(BitOrAssign, BitOr, bitor, bitor_assign, "bitwise-or");
impl_op_assign_mat_slice_patterns!(BitXorAssign, BitXor, bitxor, bitxor_assign, "bitwise-xor");

macro_rules! impl_unary_slice (
    ($slice:ident, $trt:ident, $op:ident, $sym:tt, $doc:expr) => (

/// Gets
#[doc=$doc]
/// of matrix slice.
impl<'a, T> $trt for $slice<'a, T>
    where T: $trt<Output=T> + Copy {
    type Output = Matrix<T>;

    fn $op(self) -> Matrix<T> {
        $sym &self
    }
}

/// Gets
#[doc=$doc]
/// of matrix slice.
impl<'a, 'b, T> $trt for &'b $slice<'a, T>
    where T: $trt<Output=T> + Copy {
    type Output = Matrix<T>;

    fn $op(self) -> Matrix<T> {
        let new_data = self.iter().map(|v| $sym *v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);
impl_unary_slice!(MatrixSlice, Neg, neg, -, "nagative");
impl_unary_slice!(MatrixSliceMut, Neg, neg, -, "nagative");
impl_unary_slice!(MatrixSlice, Not, not, !, "not");
impl_unary_slice!(MatrixSliceMut, Not, not, !, "not");

macro_rules! impl_unary_op (
    ($trt:ident, $op:ident, $sym:tt, $doc:expr) => (

/// Gets
#[doc=$doc]
/// of matrix.
impl<T> $trt for Matrix<T>
    where T: $trt<Output=T> + Copy
{
    type Output = Matrix<T>;

    fn $op(mut self) -> Matrix<T> {
        for val in &mut self.data {
            *val = $sym *val;
        }

        self
    }
}

/// Gets
#[doc=$doc]
/// of matrix.
impl<'a, T> $trt for &'a Matrix<T>
    where T: $trt<Output=T> + Copy
{
    type Output = Matrix<T>;

    fn $op(self) -> Matrix<T> {
        let new_data = self.data.iter().map(|v| $sym *v).collect();

        Matrix {
            cols: self.cols,
            rows: self.rows,
            data: new_data,
        }
    }
}
    );
);
impl_unary_op!(Neg, neg, -, "nagative");
impl_unary_op!(Not, not, !, "not");

#[cfg(test)]
mod tests {

    use super::super::Matrix;
    use super::super::MatrixSlice;
    use super::super::MatrixSliceMut;

    #[test]
    fn indexing_mat() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];

        assert_eq!(a[[0, 0]], 1.0);
        assert_eq!(a[[0, 1]], 2.0);
        assert_eq!(a[[1, 0]], 3.0);
        assert_eq!(a[[1, 1]], 4.0);
        assert_eq!(a[[2, 0]], 5.0);
        assert_eq!(a[[2, 1]], 6.0);
    }

    #[test]
    fn index_slice() {
        let mut b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        {
            let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

            assert_eq!(c[[0, 0]], 4);
            assert_eq!(c[[0, 1]], 5);
            assert_eq!(c[[1, 0]], 7);
            assert_eq!(c[[1, 1]], 8);
        }


        let mut c = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        assert_eq!(c[[0, 0]], 4);
        assert_eq!(c[[0, 1]], 5);
        assert_eq!(c[[1, 0]], 7);
        assert_eq!(c[[1, 1]], 8);

        c[[0, 0]] = 9;

        assert_eq!(c[[0, 0]], 9);
        assert_eq!(c[[0, 1]], 5);
        assert_eq!(c[[1, 0]], 7);
        assert_eq!(c[[1, 1]], 8);
    }

    #[test]
    fn matrix_mul_f32_vec() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = vector![4., 7.];

        let c = a * b;

        assert_eq!(c.size(), 3);

        assert_eq!(c[0], 18.0);
        assert_eq!(c[1], 40.0);
        assert_eq!(c[2], 62.0);
    }

    #[test]
    fn matrix_mul_f32_elemwise() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];

        let exp = matrix![2., 4.;
                          6., 8.;
                          10., 12.];

        // Allocating new memory
        let c = &a * &2.0;
        assert_matrix_eq!(c, exp);

        // Allocating new memory
        let c = &a * 2.0;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a.clone() * &2.0;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a * 2.0;
        assert_matrix_eq!(c, exp);
    }

    #[test]
    fn matrix_add_f32_elemwise() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = matrix![2., 3.;
                        4., 5.;
                        6., 7.];

        let exp = matrix![3., 5.;
                          7., 9.;
                          11., 13.];

        // Allocating new memory
        let c = &a + &b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a.clone() + &b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = &a + b.clone();
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a + b;
        assert_matrix_eq!(c, exp);
    }

    #[test]
    fn matrix_add_f32_broadcast() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = 3.0;

        let exp = matrix![4., 5.;
                          6., 7.;
                          8., 9.];

        // Allocating new memory
        let c = &a + &b;
        assert_matrix_eq!(c, exp);

        // Allocating new memory
        let c = &a + b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a.clone() + &b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a + b;
        assert_matrix_eq!(c, exp);
    }

    #[test]
    fn matrix_sub_f32_elemwise() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = matrix![2., 3.;
                        4., 5.;
                        6., 7.];

        let exp = matrix![-1., -1.;
                          -1., -1.;
                          -1., -1.];

        // Allocate new memory
        let c = &a - &b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a.clone() - &b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = &a - b.clone();
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = &a - b;
        assert_matrix_eq!(c, exp);
    }

    #[test]
    fn matrix_sub_f32_broadcast() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = 3.0;

        let exp = matrix![-2., -1.;
                          -0., 1.;
                          2., 3.];

        // Allocating new memory
        let c = &a - &b;
        assert_matrix_eq!(c, exp);

        // Allocating new memory
        let c = &a - b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a.clone() - &b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a - b;
        assert_matrix_eq!(c, exp);
    }

    #[test]
    fn matrix_div_f32_broadcast() {
        let a = matrix![1., 2.;
                        3., 4.;
                        5., 6.];
        let b = 3.0;

        let exp = matrix![1. / 3., 2. / 3.;
                          3. / 3., 4. / 3.;
                          5. / 3., 6. / 3.];

        // Allocating new memory
        let c = &a / &b;
        assert_matrix_eq!(c, exp);

        // Allocating new memory
        let c = &a / b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a.clone() / &b;
        assert_matrix_eq!(c, exp);

        // Reusing memory
        let c = a / b;
        assert_matrix_eq!(c, exp);
    }

    #[test]
    fn matrixslice_add_f32_broadcast() {
        let a = 3.0;
        let mut b = Matrix::ones(3, 3) * 2.;
        let c = Matrix::ones(2, 2);

        {
            let d = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

            let m_1 = &d + a.clone();
            assert_eq!(m_1.into_vec(), vec![5.0; 4]);

            let m_2 = c.clone() + &d;
            assert_eq!(m_2.into_vec(), vec![3.0; 4]);

            let m_3 = &d + c.clone();
            assert_eq!(m_3.into_vec(), vec![3.0; 4]);

            let m_4 = &d + &d;
            assert_eq!(m_4.into_vec(), vec![4.0; 4]);
        }

        let e = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        let m_1 = &e + a.clone();
        assert_eq!(m_1.into_vec(), vec![5.0; 4]);

        let m_2 = c.clone() + &e;
        assert_eq!(m_2.into_vec(), vec![3.0; 4]);

        let m_3 = &e + c;
        assert_eq!(m_3.into_vec(), vec![3.0; 4]);

        let m_4 = &e + &e;
        assert_eq!(m_4.into_vec(), vec![4.0; 4]);
    }

    #[test]
    fn matrixslice_sub_f32_broadcast() {
        let a = 3.0;
        let b = Matrix::ones(2, 2);
        let mut c = Matrix::ones(3, 3) * 2.;

        {
            let d = MatrixSlice::from_matrix(&c, [1, 1], 2, 2);

            let m_1 = &d - a.clone();
            assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

            let m_2 = b.clone() - &d;
            assert_eq!(m_2.into_vec(), vec![-1.0; 4]);

            let m_3 = &d - b.clone();
            assert_eq!(m_3.into_vec(), vec![1.0; 4]);

            let m_4 = &d - &d;
            assert_eq!(m_4.into_vec(), vec![0.0; 4]);
        }

        let e = MatrixSliceMut::from_matrix(&mut c, [1, 1], 2, 2);

        let m_1 = &e - a;
        assert_eq!(m_1.into_vec(), vec![-1.0; 4]);

        let m_2 = b.clone() - &e;
        assert_eq!(m_2.into_vec(), vec![-1.0; 4]);

        let m_3 = &e - b;
        assert_eq!(m_3.into_vec(), vec![1.0; 4]);

        let m_4 = &e - &e;
        assert_eq!(m_4.into_vec(), vec![0.0; 4]);
    }

    #[test]
    fn matrixslice_div_f32_broadcast() {
        let a = 3.0;

        let mut b = Matrix::ones(3, 3) * 2.;

        {
            let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

            let m = c / a;
            assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
        }

        let d = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        let m = d / a;
        assert_eq!(m.into_vec(), vec![2.0/3.0 ;4]);
    }

    #[test]
    fn matrix_add_assign_int_broadcast() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += &2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += 2;
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());
    }

    #[test]
    fn matrix_add_assign_int_elemwise() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += &b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a += b;
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);

            a += &c;
            assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

            let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a += c;
            assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);
        }

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
        a += &c;
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        a += c;
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);
    }

    #[test]
    fn matrix_sub_assign_int_broadcast() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());

        a -= &2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        a -= 2;
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());
    }

    #[test]
    fn matrix_sub_assign_int_elemwise() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a -= &b;
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        a -= b;
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            a -= &c;
            assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

            let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a -= c;
            assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);
        }

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
        a -= &c;
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
        a -= c;
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);
    }

    #[test]
    fn matrix_div_assign_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let mut a = Matrix::new(3, 3, a_data.clone());

        a /= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        a /= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    fn matrix_mul_assign_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![2f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut a = Matrix::new(3, 3, a_data.clone());

        a *= &2f32;
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        a *= 2f32;
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn matrixslice_add_assign_int_broadcast() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &2;
        }
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += 2;
        }
        assert_eq!(a.into_vec(), (2..11).collect::<Vec<_>>());
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn matrixslice_add_assign_int_elemwise() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &b;
        }
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += b;
        }
        assert_eq!(a.into_vec(), (0..9).map(|x| 2 * x).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += &c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice += c;
        }
        assert_eq!(a.into_vec(), vec![0, 2, 4, 7, 9, 11, 14, 16, 18]);
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn matrixslice_sub_assign_int_elemwise() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= &2;
        }
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= 2;
        }
        assert_eq!(a.into_vec(), (-2..7).collect::<Vec<_>>());

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a_slice -= &b;
        }
        assert_eq!(a.into_vec(), vec![0; 9]);
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn matrixslice_sub_assign_int_broadcast() {
        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            let b = Matrix::new(3, 3, (0..9).collect::<Vec<_>>());
            a_slice -= b;
        }
        assert_eq!(a.into_vec(), vec![0; 9]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        let mut b = Matrix::new(4, 4, (0..16).collect::<Vec<_>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= &c;
        }

        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSlice::from_matrix(&b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= c;
        }
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= &c;
        }
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);

        let mut a = Matrix::new(3, 3, (0..9).collect::<Vec<i32>>());
        {
            let c = MatrixSliceMut::from_matrix(&mut b, [0, 0], 3, 3);
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice -= c;
        }
        assert_eq!(a.into_vec(), vec![0, 0, 0, -1, -1, -1, -2, -2, -2]);
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn matrixslice_div_assign_f32_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5];
        let mut a = Matrix::new(3, 3, a_data.clone());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice /= &2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice /= 2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    #[allow(unused_assignments, unused_variables)]
    fn matrixslice_mul_assign_f32_broadcast() {
        let a_data = vec![1f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let res_data = vec![2f32, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
        let mut a = Matrix::new(3, 3, a_data.clone());

        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice *= &2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());

        let mut a = Matrix::new(3, 3, a_data.clone());
        {
            let mut a_slice = MatrixSliceMut::from_matrix(&mut a, [0, 0], 3, 3);
            a_slice *= 2f32;
        }
        assert_eq!(a.into_vec(), res_data.clone());
    }

    #[test]
    fn matrix_neg_f32() {
        let b = matrix![1., 2., 3.;
                        4., 5., 6.];
        let exp = matrix![-1., -2., -3.;
                          -4., -5., -6.];
        assert_matrix_eq!(-&b, exp);
        assert_matrix_eq!(-b, exp);
    }

    #[test]
    fn matrixslice_neg_f32() {
        let b = Matrix::ones(3, 3) * 2.;
        let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

        let m = -c;
        assert_eq!(m.into_vec(), vec![-2.0;4]);

        let mut b = Matrix::ones(3, 3) * 2.;

        let c = MatrixSliceMut::from_matrix(&mut b, [1, 1], 2, 2);

        let m = -c;
        assert_eq!(m.into_vec(), vec![-2.0;4]);
    }

    #[test]
    fn matrix_not_int() {
        let b = matrix![1, 2, 3;
                        4, 5, 6];
        let exp = matrix![!1, !2, !3;
                          !4, !5, !6];
        assert_matrix_eq!(!&b, exp);
        assert_matrix_eq!(!b, exp);
    }

    #[test]
    fn matrix_not_bool() {
        let b = matrix![true, false, true;
                        false, true, false];
        let exp = matrix![false, true, false;
                          true, false, true];
        assert_matrix_eq!(!&b, exp);
        assert_matrix_eq!(!b, exp);
    }

    #[test]
    fn matrixslice_not_bool() {
        let b = matrix![true, false, true;
                        false, true, false;
                        true, false, true];
        let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

        let m = !c;
        assert_eq!(m, matrix![false, true; true, false]);

        let b = matrix![true, false, true;
                        false, true, false;
                        true, false, true];
        let c = MatrixSlice::from_matrix(&b, [1, 1], 2, 2);

        let m = !c;
        assert_eq!(m, matrix![false, true; true, false]);
    }
}
