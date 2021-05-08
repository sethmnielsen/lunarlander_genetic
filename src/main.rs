#![allow(unused_assignments)]
#![allow(dead_code)]

use cpython::PyResult;

extern crate cpython;
extern crate tch;

pub mod gym_env;
// mod vec_gym_env;

pub mod genetic;

fn main() -> PyResult<()>{
    let a: Vec<String> = std::env::args().collect();
    match a.iter().map(|x| x.as_str()).collect::<Vec<_>>().as_slice() {
        [_, "play"] => genetic::play_episode()?,
        _ => genetic::run()?,
    }
    Ok(())
}
