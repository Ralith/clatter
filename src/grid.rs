#[inline(always)]
pub fn skew_factor(dimension: usize) -> f32 {
    (((dimension + 1) as f32).sqrt() - 1.0) / dimension as f32
}

#[inline(always)]
pub fn unskew_factor(dimension: usize) -> f32 {
    ((1.0 / ((dimension + 1) as f32).sqrt()) - 1.0) / dimension as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::*;

    #[test]
    fn skew_factors() {
        assert_abs_diff_eq!(skew_factor(2), (3.0f32.sqrt() - 1.0) / 2.0);
        assert_abs_diff_eq!(skew_factor(3), 1.0 / 3.0);
        assert_abs_diff_eq!(skew_factor(4), (5.0f32.sqrt() - 1.0) / 4.0);

        assert_abs_diff_eq!(unskew_factor(2), -(3.0 - 3.0f32.sqrt()) / 6.0);
        assert_abs_diff_eq!(unskew_factor(3), -1.0 / 6.0);
        assert_abs_diff_eq!(unskew_factor(4), -(5.0 - 5.0f32.sqrt()) / 20.0);
    }
}
