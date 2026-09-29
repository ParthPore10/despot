// An experiment driver around the existing DESPOT tree construction algorithm.
// Sampling follows the library's systematic resampler with an explicitly seeded
// permutation, not nested scenario sets.
#include "pocman.h"

#include <despot/interface/default_policy.h>
#include <despot/interface/upper_bound.h>
#include <despot/solver/despot.h>
#include <despot/util/seeds.h>

#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {
using namespace despot;

struct Options {
  std::string mode = "episode";
  std::vector<int> ks = {64};
  unsigned seed = 1;
  int steps = 30;
  double time = 0.05;
  int depth = 30;
  double prune = 0.01;
  double discount = 0.95;
  int belief_particles = 5000;
  int repeats = 5;
  int probe_every = 5;
  int reference_k = 512;
  std::string output = ".";
};

void Usage(std::ostream& out) {
  out << "Pocman scenario-count experiment (full map, SMART bounds)\n"
      << "  --mode episode|probe     Episode return or fixed-belief action probes\n"
      << "  --k 16,64,256           One K in episode mode; a sweep in probe mode\n"
      << "  --seed N                Nonnegative 32-bit experiment seed (default 1)\n"
      << "  --steps N               Maximum trajectory steps (default 30)\n"
      << "  --time SECONDS          Tree-search CPU budget per action (default .05)\n"
      << "  --depth N               Search and rollout depth (default 30)\n"
      << "  --prune X               Nonnegative regularization constant (default .01)\n"
      << "  --discount X            Strictly between 0 and 1 (default .95)\n"
      << "  --belief-particles N    At least 2500 (default 5000)\n"
      << "  --repeats N             Independent plans per K and probe (default 5)\n"
      << "  --probe-every N         Probe at steps 0,N,2N,... (default 5)\n"
      << "  --reference-k N         K for separate probe trajectory policy (default 512)\n"
      << "  --output DIRECTORY      Destination for CSV files (default .)\n";
}

unsigned long long Integer(const std::string& s, const std::string& option,
                           unsigned long long max) {
  if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
    throw std::runtime_error(option + " must be an integer");
  errno = 0;
  char* end = NULL;
  unsigned long long value = std::strtoull(s.c_str(), &end, 10);
  if (errno == ERANGE || *end != '\0' || value > max)
    throw std::runtime_error(option + " is out of range");
  return value;
}

int Positive(const std::string& s, const std::string& option) {
  int value = static_cast<int>(Integer(s, option, INT_MAX));
  if (value == 0) throw std::runtime_error(option + " must be positive");
  return value;
}

double Number(const std::string& s, const std::string& option) {
  char* end = NULL;
  errno = 0;
  double value = std::strtod(s.c_str(), &end);
  if (s.empty() || end == s.c_str() || *end != '\0' || errno == ERANGE ||
      !std::isfinite(value))
    throw std::runtime_error(option + " must be a finite number");
  return value;
}

Options Parse(int argc, char** argv) {
  Options o;
  std::set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") { Usage(std::cout); std::exit(0); }
    if (!seen.insert(key).second) throw std::runtime_error("Repeated option " + key);
    if (i + 1 >= argc) throw std::runtime_error("Missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--mode") o.mode = value;
    else if (key == "--k") {
      o.ks.clear();
      std::set<int> unique;
      std::size_t start = 0;
      do {
        std::size_t end = value.find(',', start);
        int k = Positive(value.substr(start, end - start), key);
        if (!unique.insert(k).second) throw std::runtime_error("Duplicate K in --k");
        o.ks.push_back(k);
        if (end == std::string::npos) break;
        start = end + 1;
      } while (true);
    } else if (key == "--seed") {
      o.seed = static_cast<unsigned>(Integer(value, key, UINT_MAX));
    } else if (key == "--steps") o.steps = Positive(value, key);
    else if (key == "--time") o.time = Number(value, key);
    else if (key == "--depth") o.depth = Positive(value, key);
    else if (key == "--prune") o.prune = Number(value, key);
    else if (key == "--discount") o.discount = Number(value, key);
    else if (key == "--belief-particles") o.belief_particles = Positive(value, key);
    else if (key == "--repeats") o.repeats = Positive(value, key);
    else if (key == "--probe-every") o.probe_every = Positive(value, key);
    else if (key == "--reference-k") o.reference_k = Positive(value, key);
    else if (key == "--output") o.output = value;
    else throw std::runtime_error("Unknown option " + key);
  }
  if (o.mode != "episode" && o.mode != "probe")
    throw std::runtime_error("--mode must be episode or probe");
  if (o.mode == "episode" && o.ks.size() != 1)
    throw std::runtime_error("Episode mode requires exactly one K");
  if (o.time <= 0) throw std::runtime_error("--time must be positive");
  if (o.prune < 0) throw std::runtime_error("--prune must be nonnegative");
  if (!(o.discount > 0 && o.discount < 1))
    throw std::runtime_error("--discount must be strictly between 0 and 1");
  // The stock ParticleBelief constructor splits smaller populations and retains
  // pointers to the freed originals. Avoid that legacy path without changing it.
  if (o.belief_particles < 2500 || o.belief_particles > INT_MAX / 10)
    throw std::runtime_error("--belief-particles must be between 2500 and INT_MAX/10");
  if (o.output.empty()) throw std::runtime_error("--output cannot be empty");
  return o;
}

// SplitMix64 finalizer: stable seed derivation, independent of call order and K.
uint64_t Mix(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  return x ^ (x >> 31);
}

unsigned Seed(unsigned base, unsigned domain, unsigned step = 0,
              unsigned repeat = 0) {
  uint64_t value = Mix(static_cast<uint64_t>(base) + UINT64_C(0x9e3779b97f4a7c15));
  value = Mix(value ^ domain);
  value = Mix(value ^ (static_cast<uint64_t>(step) << 32) ^ repeat);
  return static_cast<unsigned>(value ^ (value >> 32));
}

void ResetRng(unsigned seed) {
  Random::RANDOM = Random(seed);
  Seeds::root_seed(seed);
  std::srand(seed);  // Seed legacy code on standard libraries that use rand().
  Globals::config.root_seed = seed;
}

void Directory(const std::string& path) {
  for (std::size_t i = 1; i <= path.size(); ++i) {
    if (i < path.size() && path[i] != '/') continue;
    const std::string prefix = path.substr(0, i);
    if (::mkdir(prefix.c_str(), 0755) != 0 && errno != EEXIST)
      throw std::runtime_error("Cannot create " + prefix + ": " + std::strerror(errno));
    struct stat info;
    if (::stat(prefix.c_str(), &info) != 0 || !S_ISDIR(info.st_mode))
      throw std::runtime_error("Not a directory: " + prefix);
  }
}

void Open(std::ofstream& out, const std::string& path) {
  struct stat info;
  if (::stat(path.c_str(), &info) == 0)
    throw std::runtime_error("Refusing to overwrite existing CSV: " + path);
  if (errno != ENOENT)
    throw std::runtime_error("Cannot inspect output file: " + path);
  out.exceptions(std::ios::failbit | std::ios::badbit);
  out.open(path.c_str(), std::ios::out | std::ios::trunc);
  out << std::setprecision(17);
}

int BeliefSize(const ParticleBelief& belief) {
  int n = static_cast<int>(belief.particles().size());
  if (n == 0)
    throw std::runtime_error("Pocman particle belief is empty; increase --belief-particles");
  return n;
}

void Shuffle(std::vector<State*>& particles) {
  for (int n = static_cast<int>(particles.size()); n > 1; --n)
    std::swap(particles[n - 1], particles[Random::RANDOM.NextInt(n)]);
}

// libc++ random_shuffle uses a private generator that srand cannot reset. Keep
// Pocman's filter unchanged, but give both permutations an explicit seed source.
class ReproduciblePocmanBelief : public PocmanBelief {
 public:
  ReproduciblePocmanBelief(std::vector<State*> particles, const Pocman* model)
      : PocmanBelief(particles, model) {
    // Undo the base constructor's opaque permutation before seeded shuffling.
    std::sort(particles_.begin(), particles_.end(),
              [](const State* a, const State* b) { return a->state_id < b->state_id; });
    Shuffle(particles_);
  }

  std::vector<State*> Sample(int num) const {
    if (num <= 0 || particles_.empty())
      throw std::runtime_error("Cannot sample an empty Pocman belief");
    // Same systematic resampling and equal weights as ParticleBelief::Sample.
    const double unit = 1.0 / num;
    double mass = Random::RANDOM.NextDouble(0, unit);
    std::size_t pos = 0;
    double cumulative = particles_[0]->weight;
    std::vector<State*> sample;
    sample.reserve(num);
    for (int i = 0; i < num; ++i) {
      while (mass > cumulative) {
        pos = (pos + 1) % particles_.size();
        cumulative += particles_[pos]->weight;
      }
      mass += unit;
      State* particle = model_->Copy(particles_[pos]);
      particle->weight = unit;
      sample.push_back(particle);
    }
    Shuffle(sample);
    return sample;
  }
};

struct PlanResult {
  ACT_TYPE action;
  double wall;
  double cpu;
  SearchStatistics statistics;
  int belief_particles;
  bool chosen_default;
};

class MeasuredDESPOT : public DESPOT {
 public:
  MeasuredDESPOT(const DSPOMDP* model, ScenarioLowerBound* lb,
                 ScenarioUpperBound* ub, Belief* belief)
      : DESPOT(model, lb, ub, belief) {}

  PlanResult Plan(int k, unsigned seed, const History& history) {
    PlanResult result;
    result.belief_particles = BeliefSize(*static_cast<ParticleBelief*>(belief_));
    history_ = history;
    Globals::config.num_scenarios = k;
    const int active_before = model_->NumActiveParticles();
    const std::chrono::steady_clock::time_point wall_start = std::chrono::steady_clock::now();
    const std::clock_t cpu_start = std::clock();

    ResetRng(Seed(seed, 1));
    std::vector<State*> particles = belief_->Sample(k);
    ResetRng(Seed(seed, 2));
    RandomStreams streams(k, Globals::config.search_depth);
    lower_bound_->Init(streams);
    upper_bound_->Init(streams);
    // K changes the number of shuffle and stream draws. Reset SMART-policy randomness separately so every K starts with the same rollout RNG seed.
    ResetRng(Seed(seed, 3));
    statistics_ = SearchStatistics();
    root_ = ConstructTree(particles, streams, lower_bound_, upper_bound_, model_,
                          history_, Globals::config.time_per_move, &statistics_);

    double best_child = Globals::NEG_INFTY;
    for (std::size_t a = 0; a < root_->children().size(); ++a)
      best_child = std::max(best_child, root_->Child(static_cast<int>(a))->lower_bound());
    result.chosen_default = root_->default_move().value > best_child;
    result.action = OptimalAction(root_).action;
    result.statistics = statistics_;
    root_->Free(*model_);
    delete root_;
    root_ = NULL;

    result.cpu = static_cast<double>(std::clock() - cpu_start) / CLOCKS_PER_SEC;
    result.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count();
    if (model_->NumActiveParticles() != active_before)
      throw std::runtime_error("Particle leak during planning");
    if (result.action < 0 || result.action >= model_->NumActions())
      throw std::runtime_error("Planner returned an invalid action");
    return result;
  }
};

const char* MetricsHeader() {
  return "planning_wall_s,planning_cpu_s,search_cpu_s,tree_nodes,policy_nodes,"
         "expanded_nodes,trials,initial_gap,final_gap,belief_particles,chosen_default,"
         "longest_trial_length";
}

void Metrics(std::ostream& out, const PlanResult& result) {
  const SearchStatistics& s = result.statistics;
  out << result.wall << ',' << result.cpu << ',' << s.time_search << ','
      << s.num_tree_nodes << ',' << s.num_policy_nodes << ','
      << s.num_expanded_nodes << ',' << s.num_trials << ','
      << s.initial_ub - s.initial_lb << ',' << s.final_ub - s.final_lb << ','
      << result.belief_particles << ',' << result.chosen_default << ','
      << s.longest_trial_length << '\n';
}

struct StateDeleter {
  const DSPOMDP* model;
  void operator()(State* state) const { if (state) model->Free(state); }
};

void Run(const Options& o, FullPocman& model) {
  Globals::config.time_per_move = o.time;
  Globals::config.search_depth = o.depth;
  Globals::config.max_policy_sim_len = o.depth;
  Globals::config.pruning_constant = o.prune;
  Globals::config.discount = o.discount;
  Globals::config.sim_len = o.steps;
  Globals::config.silence = true;
  logging::level(logging::ERROR);
  PocmanBelief::num_particles = o.belief_particles;

  ResetRng(Seed(o.seed, 10));
  std::unique_ptr<State, StateDeleter> state(model.CreateStartState(), StateDeleter{&model});
  ResetRng(Seed(o.seed, 20));
  // Pocman::InitialBelief ignores the actual start state and samples this prior.
  // Assign stable insertion indices solely to undo the stock opaque shuffle.
  std::vector<State*> initial_particles;
  initial_particles.reserve(o.belief_particles);
  for (int i = 0; i < o.belief_particles; ++i) {
    State* particle = model.CreateStartState();
    particle->state_id = i;
    particle->weight = 1.0 / o.belief_particles;
    initial_particles.push_back(particle);
  }
  std::unique_ptr<Belief> belief(new ReproduciblePocmanBelief(initial_particles, &model));
  ParticleBelief& particle_belief = *static_cast<ParticleBelief*>(belief.get());
  BeliefSize(particle_belief);

  std::unique_ptr<ScenarioLowerBound> lower(model.CreateScenarioLowerBound("SMART", "LEGAL"));
  // DefaultPolicy does not own/delete this nested bound in the upstream library.
  std::unique_ptr<ParticleLowerBound> particle_lower(
      static_cast<DefaultPolicy*>(lower.get())->particle_lower_bound());
  std::unique_ptr<ScenarioUpperBound> upper(model.CreateScenarioUpperBound("SMART"));
  MeasuredDESPOT planner(&model, lower.get(), upper.get(), belief.get());
  History history;

  Directory(o.output);
  std::ofstream steps, episodes, probes;
  if (o.mode == "episode") {
    Open(steps, o.output + "/steps.csv");
    steps << "seed,k,step,action,observation,reward,terminal," << MetricsHeader() << '\n';
    Open(episodes, o.output + "/episodes.csv");
    episodes << "seed,k,steps,return,discounted_return,terminal\n";
  } else {
    Open(probes, o.output + "/probes.csv");
    probes << "seed,step,k,repeat,planner_seed,action," << MetricsHeader() << '\n';
  }

  int completed = 0;
  bool terminal = false;
  double total = 0, discounted = 0, discount = 1;
  for (int step = 0; step < o.steps; ++step) {
    PlanResult result;
    if (o.mode == "episode") {
      result = planner.Plan(o.ks[0], Seed(o.seed, 40, step), history);
    } else {
      if (step % o.probe_every == 0) {
        // Every plan sees this exact belief and history; Sample returns copies.
        for (std::size_t i = 0; i < o.ks.size(); ++i) {
          for (int repeat = 0; repeat < o.repeats; ++repeat) {
            unsigned plan_seed = Seed(o.seed, 50, step, repeat);
            PlanResult probe = planner.Plan(o.ks[i], plan_seed, history);
            probes << o.seed << ',' << step << ',' << o.ks[i] << ',' << repeat
                   << ',' << plan_seed << ',' << probe.action << ',';
            Metrics(probes, probe);
          }
        }
        probes.flush();
      }
      // Probe actions never affect the trajectory. Use an independent planner seed domain to choose the next reference-trajectory action.
      result = planner.Plan(o.reference_k, Seed(o.seed, 60, step), history);
    }

    ResetRng(Seed(o.seed, 30, step));
    double reward = 0;
    OBS_TYPE observation = 0;
    terminal = model.Step(*state, Random::RANDOM.NextDouble(), result.action, reward, observation);
    ++completed;
    total += reward;
    discounted += discount * reward;
    discount *= o.discount;
    if (o.mode == "episode") {
      steps << o.seed << ',' << o.ks[0] << ',' << step << ',' << result.action << ','
            << observation << ',' << reward << ',' << terminal << ',';
      Metrics(steps, result);
      steps.flush();
    }
    if (terminal || completed == o.steps) break;
    history.Add(result.action, observation);
    ResetRng(Seed(o.seed, 21, step));
    BeliefSize(particle_belief);  // Update itself assumes a nonempty population.
    belief->Update(result.action, observation);
    BeliefSize(particle_belief);
  }
  if (o.mode == "episode") {
    episodes << o.seed << ',' << o.ks[0] << ',' << completed << ',' << total << ','
             << discounted << ',' << terminal << '\n';
    episodes.close();
    steps.close();
  } else {
    probes.close();
  }
  std::cout << o.mode << " seed=" << o.seed << " steps=" << completed
            << " terminal=" << terminal << " output=" << o.output << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = Parse(argc, argv);
    FullPocman model;
    Run(options, model);
    if (model.NumActiveParticles() != 0)
      throw std::runtime_error("Particle leak after experiment cleanup");
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "pocman_k: " << error.what() << '\n';
    return 1;
  }
}
