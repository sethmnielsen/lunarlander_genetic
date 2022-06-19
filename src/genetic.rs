// use super::vec_gym_env::VecGymEnv;
use super::gym_env::GymEnv;
use tch::kind::*;
use tch::{nn, Device, Kind::Float, Tensor, IndexOp, nn::Path};
// use std::time::Duration;
// use std::thread;

const ENV_NAME: &'static str = "LunarLander-v2";

// const INT_TYPE: (tch::Kind, Device) = INT64_CPU;
// const FLOAT_TYPE: (tch::Kind, Device) = FLOAT_CPU;
// const DEVICE: Device = Device::Cpu;
const INT_TYPE: (tch::Kind, Device) = INT64_CUDA;
const FLOAT_TYPE: (tch::Kind, Device) = FLOAT_CUDA;
const DEVICE: Device = Device::Cuda(0);

const SAVE_INDEX: i64 = 8;

// const LOAD_INDEX: i64 = 0;
// const LOAD_EPOCH: i64 = 1049;
// const MINFIT: bool = false;

const LOAD_INDEX: i64 = 7;
const LOAD_EPOCH: i64 = 350;
const MINFIT: bool = true;

const LOAD_VS: bool = true;

const EPOCHS: i64 = 1000000;
const POPULATION_SIZE: i64 = 100;    // N
const MUTATION_POWER: f64 = 0.001;    // sigma
const SELECTION_TOP: i64 = 10;  // T
const MIN_INIT_SCORE: f64 = -130.0;
const NUM_VALIDATION_EPS: usize = 10;
const NUM_HIDDEN: i64 = 32;
const NUM_ACTS: i64 = 2;
const NUM_OBS: i64 = 8;
    // state/observation: Box(8,)
    //  - posx
    //  - posy
    //  - velx
    //  - vely
    //  - ang
    //  - ang vel
    //  - contact L
    //  - contact R
const NUM_PARAMS: i64 = NUM_HIDDEN*(NUM_OBS+1) + (NUM_HIDDEN+NUM_ACTS)*(NUM_HIDDEN+1);

struct Model {
    network: nn::Sequential,
    var_store: nn::VarStore,
}

impl Model {
    fn new() -> Self {
        let mut var_store = nn::VarStore::new(DEVICE);
        let p = &var_store.root();
        let network = Model::create_new_network(p);
        var_store.freeze();
        Self {
            network,
            var_store,
        }
    }

    fn forward(&self, obs: &Tensor) -> Tensor {
        obs.to_device(DEVICE).apply(&self.network)
    }

    /// Creates a new feed-forward DNN with two hidden layers and returns it.
    fn create_new_network(p: &Path) -> nn::Sequential {
        nn::seq()
            .add(nn::linear(p / "lin1", NUM_OBS, NUM_HIDDEN, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "lin2", NUM_HIDDEN, NUM_HIDDEN, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::linear(p / "lin3", NUM_HIDDEN, NUM_ACTS, Default::default()))
    }

    /// Drop the current network and replace it with a newly created one.
    fn reinitialize_network(&mut self) {
        self.network = Model::create_new_network(&self.var_store.root());
    }

    /// Literal copy of values from the model's var_store weights to the input tensor.
    fn copy_network_to_tensor(&self, individual: &Tensor) {
        // from 0 to 32, put v0
        // from 32 to 288, put v1
        // from 288 to 320, put v2
        // from 320 to 1344, put v3
        // from 1344 to 1346, put v4
        // from 1346 to 1410, put v5
        let mut idx2_start: i64 = 0;
        let mut idx2_end: i64 = 0;
        for v in self.var_store.trainable_variables() {
            idx2_end += v.numel() as i64;
            individual.i(idx2_start..idx2_end).copy_(&v.flatten(0, -1));
            idx2_start = idx2_end;
        }
    }

    /// Literal copy of values from input tensor to the model's var_store weights.
    fn copy_tensor_to_network(&mut self, individual: &Tensor) {
        // v0 size: 32         NUM_HIDDEN
        // v1 size: (32,8)    (NUM_HIDDEN, NUM_OBS)
        // v2 size: 32         NUM_HIDDEN
        // v3 size: (32,32)   (NUM_HIDDEN, NUM_HIDDEN)
        // v4 size: 2          NUM_ACTS
        // v5 size: (32,2)    (NUM_HIDDEN, NUM_ACTS)
        let mut idx2_start: i64 = 0;
        let mut idx2_end: i64 = 0;
        for v in self.var_store.trainable_variables() {
            idx2_end += v.numel() as i64;
            v.i(..).copy_(&individual.i(idx2_start..idx2_end).view(v.size().as_slice()));
            idx2_start = idx2_end;
        }
    }
}

struct Population {
    individuals: Tensor,
    top_individuals: Tensor,
    rewards: Tensor,
    fitness: Tensor,
    max_fitness: Tensor,
    min_fitness: Tensor,
    noise: Tensor,
    thetas_to_mutate: Tensor,
}

impl Default for Population {
    fn default() -> Self {
        Self {
            individuals: Tensor::zeros(&[POPULATION_SIZE, NUM_PARAMS], FLOAT_TYPE),
            top_individuals: Tensor::zeros(&[SELECTION_TOP, NUM_PARAMS], FLOAT_TYPE),
            rewards: Tensor::zeros(&[POPULATION_SIZE, NUM_VALIDATION_EPS as i64], FLOAT_TYPE),
            fitness: Tensor::zeros(&[POPULATION_SIZE], FLOAT_TYPE),
            max_fitness: Tensor::zeros(&[POPULATION_SIZE], FLOAT_TYPE),
            min_fitness: Tensor::zeros(&[POPULATION_SIZE], FLOAT_TYPE),
            noise: Tensor::randn(&[POPULATION_SIZE-1, NUM_PARAMS], FLOAT_TYPE),
            thetas_to_mutate: Tensor::randint(SELECTION_TOP, &[POPULATION_SIZE-1], INT_TYPE),
        }
    }
}

impl Clone for Population {
    fn clone(&self) -> Self {
        Self {
            individuals: self.individuals.copy(),
            top_individuals: self.top_individuals.copy(),
            rewards: self.rewards.copy(),
            fitness: self.fitness.copy(),
            max_fitness: self.max_fitness.copy(),
            min_fitness: self.max_fitness.copy(),
            noise: self.noise.copy(),
            thetas_to_mutate: self.thetas_to_mutate.copy(),
        }
    }

    fn clone_from(&mut self, source: &Self) {
        *self = source.clone()
    }
}

impl Population {
    /// Take mean of all rewards for each individual, grab indexes of k highest from mean scores
    /// and copy top individuals into self.top_individuals.
    fn compute_fitness_and_reorder(&mut self) {
        // Compute fitness -- not ordered yet
        self.fitness = self.rewards.mean_dim(&[1], false, Float);
        self.min_fitness = self.rewards.min_dim(1, false).0;

        // find highest SELECTION_TOP fitness scores and save the (values, indices)
        let top_vals_idxs = self.min_fitness.topk(SELECTION_TOP, 0, true, true);

        let min_top_vals = self.min_fitness.i(&top_vals_idxs.1);
        let mean_top_vals = self.fitness.i(&top_vals_idxs.1);
        let rewards_top_vals = self.rewards.i(&top_vals_idxs.1);

        // Begin reordering the first SELECTION_TOP values
        self.fitness.i(0..SELECTION_TOP).copy_(&mean_top_vals);
        self.min_fitness.i(0..SELECTION_TOP).copy_(&min_top_vals);
        self.rewards.i(0..SELECTION_TOP).copy_(&rewards_top_vals);
        self.top_individuals.copy_(&self.individuals.i(&top_vals_idxs.1));
        self.individuals.i(0..SELECTION_TOP).copy_(&self.top_individuals);
    }

    fn set_all_individuals(&mut self, tensor: Tensor) {
        for i in 0..POPULATION_SIZE {
            self.individuals.i(i).copy_(&tensor.copy());
        }
        self.top_individuals.copy_(&self.individuals.i(0..SELECTION_TOP));
    }

    /// Sample some noise, randomly choose a POPULATION_SIZE's amount of indices from self.top_individuals
    /// and add the noise to all, then copy mutated top individuals into self.individuals.
    fn mutate_population(&mut self) {
        self.noise = self.noise.normal_(0., 1.);
        self.thetas_to_mutate = self.thetas_to_mutate.random_to_(SELECTION_TOP);
        self.individuals.i(1..).copy_(&(self.top_individuals.i(&self.thetas_to_mutate) + &self.noise*MUTATION_POWER));
    }
}

struct Trainer {
    // models: Vec<Model>,
    model: Model,
    population: Population,
    env: GymEnv,
    top_mean_vec: Vec<f64>,
    top_min_vec: Vec<f64>,
}

impl Trainer {
    fn new(env: GymEnv) -> Self {
        println!("Creating model");
        // let models = vec![Model::new()];
        Self {
            model: Model::new(),
            population: Population::default(),
            env,
            top_mean_vec: Vec::<f64>::new(),
            top_min_vec: Vec::<f64>::new(),
        }
    }

    fn find_good_initial_weights(&mut self) -> cpython::PyResult<()> {
        loop {
            println!("Testing performance of init weights...");
            let rewards_vec = self.test_performance_of_network()?;
            let mean_score = f64::from(&Tensor::of_slice(&rewards_vec.as_slice()).mean(Float));
            println!("Mean score of network's init weights: {}", mean_score);
            match mean_score > MIN_INIT_SCORE {
                true => break,
                false => println!("Score too low, reinitializing weights and trying again\n"),
            }
            self.model.reinitialize_network();
        }
        println!("Init weights are OK\n");
        self.initialize_population_from_network();
        Ok(())
    }

    fn load_weights_from_file(&mut self) {
        let minfit_str = match MINFIT {
            true => "-minfit",
            false => "",
        };
        let filename = format!("vs_saves/ga_weights-{}-{}{}.ot", LOAD_INDEX, LOAD_EPOCH, minfit_str);
        println!("Loading weights from file {}", filename);
        self.model.var_store.load(filename).unwrap();
        self.model.var_store.freeze();
        self.initialize_population_from_network();
    }

    /// Copy elite weights to network, then save the network's var_store to file.
    fn save_weights(&mut self, epoch_idx: i64) {
        self.model.copy_tensor_to_network(&self.population.individuals.i(0));
        if let Err(err) = self.model.var_store.save(
            format!("vs_saves/ga_weights-{}-{}-minfit.ot",
                    SAVE_INDEX,
                    epoch_idx+1)) {
            println!("error while saving file, error msg:\n{}", err)
        }
    }

    fn initialize_population_from_network(&mut self) {
        let initial_weights = Tensor::zeros(&[NUM_PARAMS], FLOAT_TYPE);
        self.model.copy_network_to_tensor(&initial_weights);
        self.population.set_all_individuals(initial_weights);
        self.population.mutate_population();
    }

    fn train_epoch(&mut self, epoch_idx: i64) -> cpython::PyResult<()> {
        for i in 0..POPULATION_SIZE {
            self.evaluate_individual(&i)?;
        }
        self.population.compute_fitness_and_reorder();
        self.population.mutate_population();

        let elite_mean = f64::from(self.population.fitness.i(0));
        let elite_max = f64::from(self.population.rewards.i(0).max());
        let elite_min = f64::from(self.population.rewards.i(0).min());
        let top_mean = f64::from(self.population.fitness.max());
        println!("finished epoch {:>4} --- elite | mean:{:>4.0} | max:{:>4.0} | min:{:>4.0} | top mean:{:>4.0}",
            epoch_idx+1,
            elite_mean,
            elite_max,
            elite_min,
            top_mean,
        );

        self.top_mean_vec.push(top_mean);
        self.top_min_vec.push(elite_min);
        if (epoch_idx+1) % 50 == 0 {
            let mean_past_50 = f64::from(&Tensor::of_slice(&self.top_mean_vec.as_slice()).mean(Float));
            let min_past_50 = f64::from(&Tensor::of_slice(&self.top_min_vec.as_slice()).mean(Float));
            self.top_mean_vec.clear();
            self.top_min_vec.clear();
            println!("\n ---- Average of previous 50 | top mean: {:>7.1} | top min: {:>7.1} ----\n", mean_past_50, min_past_50);
            self.save_weights(epoch_idx);
        }

        Ok(())
    }

    /// Copy individual weights to network, test performance of network and
    /// save each episode's total reward inside self.population.rewards.
    fn evaluate_individual(&mut self, indv_idx: &i64) -> cpython::PyResult<()> {
        self.model.copy_tensor_to_network(&self.population.individuals.i(*indv_idx));
        let rewards_vec = self.test_performance_of_network()?;
        self.population.rewards.i(*indv_idx).copy_(&Tensor::of_slice(&rewards_vec.as_slice()));
        Ok(())
    }

    /// Now that the network's weights have been set, test performance of the network over
    /// NUM_VALIDATION_EPS episodes and return the vec of total rewards for each episode.
    fn test_performance_of_network(&mut self) -> cpython::PyResult<Vec<f64>> {
        let mut rewards_vec = vec![0.; NUM_VALIDATION_EPS];
        for j in 0..NUM_VALIDATION_EPS {
            rewards_vec[j] = self.run_episode()?;
        }
        Ok(rewards_vec)
    }

    // NOTE: PARALLELIZING - start by doing these episodes in parallel
    fn run_episode(&mut self) -> cpython::PyResult<f64> {
        let mut obs = self.env.reset()?;
        let mut reward_total = 0.;
        let mut is_done = false;

        // Play episode
        while !is_done {
            let actions = self.model.forward(&obs);
            let actions_vec = Vec::<f64>::from(&actions);

            let step = self.env.step(&actions_vec)?;

            obs = step.obs;
            reward_total += step.reward;
            is_done = step.is_done;
        }

        Ok(reward_total)
    }
}

struct Player {
    model: Model,
    env: GymEnv,
}

impl Player {
    fn new(env: GymEnv) -> Self {
        println!("Creating model");
        Self {
            model: Model::new(),
            env,
        }
    }
    fn load_weights_from_file(&mut self) {
        let minfit_str = match MINFIT {
            true => "-minfit",
            false => "",
        };
        let filename = format!("vs_saves/ga_weights-{}-{}{}.ot", LOAD_INDEX, LOAD_EPOCH, minfit_str);
        println!("Loading weights from file {}", filename);
        self.model.var_store.load(filename).unwrap();
        self.model.var_store.freeze();
    }

    fn play_episodes(&mut self, n_eps: i64) -> cpython::PyResult<()> {
        let mut rewards = vec![0.; n_eps as usize];

        for i in 0..n_eps {
            let mut obs = self.env.reset()?;
            let mut reward_total = 0.;
            let mut is_done = false;
            // Play episode
            while !is_done {
                self.env.render()?;
                let actions = self.model.forward(&obs);
                let actions_vec = Vec::<f64>::from(&actions);
                println!("{:?}", actions_vec);

                let step = self.env.step(&actions_vec)?;

                obs = step.obs;
                reward_total += step.reward;
                rewards[i as usize] = reward_total;
                is_done = step.is_done;
                // thread::sleep(Duration::from_millis(10))
            }
            for _ in 0..54 {
                self.env.render()?;
                let actions = self.model.forward(&obs);
                let actions_vec = Vec::<f64>::from(&actions);

                let step = self.env.step(&actions_vec)?;

                obs = step.obs;
                // thread::sleep(Duration::from_millis(10));
            }
            println!("Episode {} score: {}", i, reward_total);
        }


        let rewards_tens = Tensor::of_slice(&rewards.as_slice());
        let mean = f64::from(rewards_tens.mean(Float));
        let min = f64::from(rewards_tens.min());
        println!("\nDone!\n");
        println!("Average score: {}", mean);
        println!("Min score: {}\n", min);

        Ok(())
    }
}

/// Trains an agent using GA.
pub fn run() -> cpython::PyResult<()> {
    println!("Loading gym env");
    let env = GymEnv::new(ENV_NAME)?;
    println!("action space: {}", env.action_space());
    println!("observation space: {:?}", env.observation_space());

    let mut trainer = Trainer::new(env);

    match LOAD_VS {
        true => trainer.load_weights_from_file(),
        false => trainer.find_good_initial_weights()?,
    }

    println!("\n---- TRAINING START ----\n\n");
    for epoch_idx in 0..EPOCHS {
        trainer.train_epoch(epoch_idx)?;
    }
    Ok(())
}

/// Play an episode with visualization and without training.
pub fn play_episode() -> cpython::PyResult<()> {
    println!("Playing episode");
    println!("Loading gym env");
    let env = GymEnv::new(ENV_NAME)?;

    let mut player = Player::new(env);
    player.load_weights_from_file();
    let n_eps: i64 = 10;

    player.play_episodes(n_eps)?;

    Ok(())
}