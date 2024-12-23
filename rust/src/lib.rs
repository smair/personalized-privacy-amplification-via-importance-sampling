mod pais;

use pais::{
    log_one_plus_exp,
    log_exp_minus_one,
    optimal_importance_weight_rs,
    LloydProfile,
};

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;


/// Amplify a privacy profile by a given weight.
/// 
/// This function computes the PDP value of the mechanism when subsampled via Poisson importance sampling with the given weight.
/// 
/// # Arguments
/// 
/// * `epsilon` - The weighted PDP profile evaluated at `weight`.
/// * `weight` - The weight of the privacy profile to amplify.
#[pyfunction]
fn amplify(epsilon: f64, weight: f64) -> f64 {
    if epsilon <= 0. { panic!("Epsilon must be positive"); }
    if weight < 1. { panic!("Weight must be at least 1"); }
    if weight == 1. { return epsilon; }
    let logexpm1 = log_exp_minus_one(epsilon);
    let log_second_term = logexpm1 - weight.ln();
    log_one_plus_exp(log_second_term)
}

#[pyfunction]
fn optimal_weight(l1_norm: f64, beta_count: f64, beta_sum: f64, num_iterations: usize, target_epsilon: f64) -> PyResult<f64> {
    let profile = LloydProfile {
        l1_norm,
        beta_count,
        beta_sum,
        num_iterations,
    };
    let result = optimal_importance_weight_rs(profile, target_epsilon);
    match result {
        Ok(weight) => Ok(weight),
        Err(msg) => Err(PyErr::new::<PyValueError, _>(msg)),
    }
}

#[pyfunction]
fn all_optimal_weights(l1_norms: Vec<f64>, beta_count: f64, beta_sum: f64, num_iterations: usize, target_epsilon: f64) -> Vec<f64> {
    l1_norms.into_iter().map(|l1_norm| {
        let profile = LloydProfile {
            l1_norm,
            beta_count,
            beta_sum,
            num_iterations,
        };
        let result = optimal_importance_weight_rs(profile, target_epsilon);
        match result {
            Ok(weight) => weight,
            Err(msg) => panic!("{} {}", l1_norm, msg),
        }
    }).collect()
}

/// A Python module implemented in Rust.
#[pymodule]
fn pais_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(amplify, m)?)?;
    m.add_function(wrap_pyfunction!(optimal_weight, m)?)?;
    m.add_function(wrap_pyfunction!(all_optimal_weights, m)?)?;
    Ok(())
}
