macro_rules! impl_op {
    ($op:tt |$lhs_name:ident : $lhs:ty, $rhs_name:ident : $rhs:ty| -> $out:path $body:block) => {
        _match_op_name! {$op, $lhs, $rhs, $lhs_name, $rhs_name, $out, $body}
    };
}

macro_rules! impl_op_commutative {
    ($op:tt |$lhs_name:ident : $lhs:ty, $rhs_name:ident : $rhs:ty| -> $out:path $body:block) => {
        _match_op_name! {$op, $lhs, $rhs, $lhs_name, $rhs_name, $out, $body}
        _match_op_name! {$op, $rhs, $lhs, $rhs_name, $lhs_name, $out, $body}
    };
}

// Match to op_name, function_name
macro_rules! _match_op_name {
    (*, $($t:tt)+) => {_internal_op!{Mul, mul, $($t)*}};
    (-, $($t:tt)+) => {_internal_op!{Sub, sub, $($t)*}};
    (+, $($t:tt)+) => {_internal_op!{Add, add, $($t)*}};
}


macro_rules! _internal_op {

    ($op_name:ident, $fn_name:ident, $lhs:ty, $rhs:ty, $lhs_name:ident, $rhs_name:ident, $out:ty, $body:block) => {
        impl $op_name<$rhs> for $lhs{
            type Output = $out;

            fn $fn_name(self, $rhs_name: $rhs) -> Self::Output {
                let $lhs_name = self;
                $body
            }
        }

        impl<'a> $op_name<&'a $rhs> for $lhs{
            type Output = $out;

            fn $fn_name(self, $rhs_name: &$rhs) -> Self::Output {
                let $lhs_name = self;
                $body
            }
        }

        impl<'a> $op_name<$rhs> for &'a $lhs{
            type Output = $out;

            fn $fn_name(self, $rhs_name: $rhs) -> Self::Output {
                let $lhs_name = self;
                $body
            }
        }

        impl<'a, 'b> $op_name<&'a $rhs> for &'b $lhs{
            type Output = $out;

            fn $fn_name(self, $rhs_name: &'a $rhs) -> Self::Output {
                let $lhs_name = self;
                $body
            }
        }
    }
}


#[cfg(test)]
mod tests {
    use std::ops::Mul;

    #[derive(PartialEq, Debug)]
    pub struct Test {
        a: f32,
    }

    impl_op!(* |lhs: Test, rhs: Test| -> Test { Test{ a: lhs.a * rhs.a}});

    #[test]
    pub fn test_mut() {
        let t1 = Test { a: 1. };
        let t2 = Test { a: 2. };

        assert_eq!(t1 * t2, Test { a: 2. })
    }
}
