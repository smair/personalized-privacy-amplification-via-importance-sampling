/// Computes `log(exp(x) - 1)` in a numerically stable way.
pub fn log_exp_minus_one(x: f64) -> f64 {
    let log = (1. - (-x).exp()).ln();
    x + log
}

/// `log(1 + exp(x))` in a numerically stable way.
pub fn log_one_plus_exp(x: f64) -> f64 {
    x + ((-x).exp() + 1.).ln()
}

pub fn optimal_importance_weight_rs(profile: LloydProfile, target_epsilon: f64) -> Result<f64, String> {
    let log_target = log_exp_minus_one(target_epsilon);
    let objective = |weight: f64| {
        log_exp_minus_one(profile.eval(weight)) - weight.ln() - log_target
    };
    let obj_1 = objective(1.);
    if obj_1 > 0. {
        return Err("The chosen target_epsilon is not satisfiable, even without sampling.".to_string());
    }
    if obj_1 == 0. {
        return Ok(1.);
    }

    let (mut lb, mut ub) = (1.0, 2.0);
    while objective(ub) < 0. {
        (lb, ub) = (ub, ub * 2.);
    }

    let num_iter = 11;  // corresponds to relative error of 0.1%
    let mut mid = (lb + ub) / 2.;
    for _ in 0..num_iter {
        if objective(mid) > 0. {
            ub = mid;
        } else {
            lb = mid;
        }
        mid = (lb + ub) / 2.;
    }

    Ok(mid)
}


pub struct LloydProfile {
    pub l1_norm: f64,
    pub beta_count: f64,
    pub beta_sum: f64,
    pub num_iterations: usize,
}

impl LloydProfile {
    fn eval(&self, weight: f64) -> f64 {
        self.num_iterations as f64 * weight * (1. / self.beta_count + self.l1_norm / self.beta_sum)
    }
}
