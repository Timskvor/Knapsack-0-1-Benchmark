/*
Compile with OpenMP and pthread support:
  g++ -O2 -std=c++17 knap_all_with_sweep.cpp -o knap_all_with_sweep -fopenmp -pthread

Usage:
  ./knap_all_with_sweep <mode> <input_file_or_directory>

Modes:
  local-omp      : DP 0/1 knapsack with OpenMP
  genetic-omp    : Genetic algorithm with OpenMP
  local-tp       : DP 0/1 knapsack with thread pool
  genetic-tp     : Genetic algorithm with thread pool
  sweep-omp      : Parameter sweep for genetic-omp on a single file
  sweep-tp       : Parameter sweep for genetic-tp on a single file
*/

#include <bits/stdc++.h>
#include <chrono>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <filesystem>
#ifdef _OPENMP
#endif

using namespace std;
using namespace std::chrono;
namespace fs = std::filesystem;

// Thread pool implementation
template<typename F>
class ThreadPool {
    vector<thread> workers;
    queue<F> tasks;
    mutex mtx;
    condition_variable cv;
    bool stop = false;
public:
    ThreadPool(size_t n) {
        for(size_t i = 0; i < n; ++i) {
            workers.emplace_back([this](){
                while(true) {
                    F task;
                    {
                        unique_lock<mutex> lk(mtx);
                        cv.wait(lk, [this]{ return stop || !tasks.empty(); });
                        if(stop && tasks.empty()) return;
                        task = move(tasks.front()); tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    ~ThreadPool() {
        { unique_lock<mutex> lk(mtx); stop = true; }
        cv.notify_all();
        for(auto &w : workers) w.join();
    }
    void enqueue(F f) {
        { unique_lock<mutex> lk(mtx); tasks.push(move(f)); }
        cv.notify_one();
    }
};

// Read file and populate values, weights, return capacity W
int processFile(const string &path, vector<int> &values, vector<int> &weights) {
    ifstream file(path);
    if(!file.is_open()) throw runtime_error("Cannot open " + path);
    int N, W;
    file >> N >> W;
    values.resize(N);
    weights.resize(N);
    for(int i = 0; i < N; ++i) file >> values[i] >> weights[i];
    return W;
}

// Local DP + OpenMP
template<typename T = long long>
pair<double,T> solve_local_omp(const vector<int>& val, const vector<int>& wt, int W) {
    int n = (int)val.size();
    vector<long long> prev(W+1,0), curr(W+1,0);
    auto t0 = high_resolution_clock::now();
    for(int i = 0; i < n; ++i) {
        #pragma omp parallel for
        for(int w = 0; w <= W; ++w) {
            curr[w] = (w >= wt[i] ? max(prev[w], prev[w - wt[i]] + val[i]) : prev[w]);
        }
        swap(prev, curr);
    }
    auto t1 = high_resolution_clock::now();
    return { duration<double,milli>(t1 - t0).count(), prev[W] };
}

// Genetic algorithm + OpenMP
template<typename T = long long>
pair<double,T> solve_genetic_omp(const vector<int>& val, const vector<int>& wt, int W,
                                  int POP=150, int GENS=300,
                                  double MUT_RATE=0.01, double CROSS_RATE=0.85) {
    int n = (int)val.size();
    const int PENALTY = 100;
    struct Chrom { vector<bool> genes; long long fitness; long long weight; };
    vector<Chrom> pop(POP);
    mt19937 rng(random_device{}());
    // Greedy init
    for(auto &ch : pop) {
        ch.genes.assign(n, false);
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);
        ch.weight = ch.fitness = 0;
        for(int i : idx) if(ch.weight + wt[i] <= W) {
            ch.genes[i] = true;
            ch.weight += wt[i];
            ch.fitness += val[i];
        }
    }
    auto calc = [&](Chrom &ch) {
        ch.weight = ch.fitness = 0;
        for(int i = 0; i < n; ++i) if(ch.genes[i]) {
            ch.weight += wt[i];
            ch.fitness += val[i];
        }
        if(ch.weight > W) ch.fitness -= PENALTY * (ch.weight - W);
        if(ch.fitness < 0) ch.fitness = 0;
    };
    auto t0 = high_resolution_clock::now();
    for(int gen = 0; gen < GENS; ++gen) {
        #pragma omp parallel for
        for(int i = 0; i < POP; ++i) calc(pop[i]);
        sort(pop.begin(), pop.end(), [](auto &a, auto &b){ return a.fitness > b.fitness; });
        vector<Chrom> new_pop;
        new_pop.reserve(POP);
        new_pop.push_back(pop[0]); new_pop.push_back(pop[1]);
        uniform_int_distribution<int> elite_dist(0, min(9, POP-1));
        uniform_real_distribution<double> prob(0.0, 1.0);
        uniform_int_distribution<int> cut_dist(1, n-1);
        uniform_int_distribution<int> gene_dist(0, n-1);
        while((int)new_pop.size() < POP) {
            Chrom p1 = pop[elite_dist(rng)];
            Chrom p2; int tries = 0;
            do { p2 = pop[elite_dist(rng)]; } while(p2.genes == p1.genes && ++tries < 5);
            Chrom c1 = p1, c2 = p2;
            if(prob(rng) < CROSS_RATE) {
                int cut = cut_dist(rng);
                for(int i = cut; i < n; ++i) swap(c1.genes[i], c2.genes[i]);
            }
            int mcount = uniform_int_distribution<int>(0, max(1, (int)(n * MUT_RATE)))(rng);
            for(int m = 0; m < mcount; ++m) c1.genes[gene_dist(rng)] = !c1.genes[gene_dist(rng)];
            for(int m = 0; m < mcount; ++m) c2.genes[gene_dist(rng)] = !c2.genes[gene_dist(rng)];
            calc(c1); calc(c2);
            new_pop.push_back(move(c1));
            if((int)new_pop.size() < POP) new_pop.push_back(move(c2));
        }
        pop.swap(new_pop);
    }
    #pragma omp parallel for
    for(int i = 0; i < POP; ++i) calc(pop[i]);
    auto best = max_element(pop.begin(), pop.end(), [](auto &a, auto &b){ return a.fitness < b.fitness; });
    auto t1 = high_resolution_clock::now();
    return { duration<double,milli>(t1 - t0).count(), best->fitness };
}

// Local DP + ThreadPool
template<typename T = long long>
pair<double,T> solve_local_tp(const vector<int>& val, const vector<int>& wt, int W) {
    int n = val.size();
    vector<long long> prev(W+1,0), curr(W+1,0);
    ThreadPool<function<void()>> pool(thread::hardware_concurrency());
    auto t0 = high_resolution_clock::now();
    int threads = thread::hardware_concurrency();
    for(int i=0;i<n;++i) {
        int chunk = W/threads + 1;
        for(int t=0;t<threads;++t) {
            int L = t*chunk, R = min(W, (t+1)*chunk-1);
            pool.enqueue([&,L,R,i](){ for(int w=L;w<=R;++w) curr[w] = (w>=wt[i]? max(prev[w], prev[w-wt[i]]+val[i]) : prev[w]); });
        }
        this_thread::sleep_for(milliseconds(1));
        swap(prev, curr);
    }
    auto t1 = high_resolution_clock::now();
    return { duration<double,milli>(t1-t0).count(), prev[W] };
}

// Genetic algorithm + ThreadPool
template<typename T = long long>
pair<double,T> solve_genetic_tp(const vector<int>& val, const vector<int>& wt, int W,
                                  int POP=150, int GENS=300,
                                  double MUT_RATE=0.01, double CROSS_RATE=0.85) {
    int n = (int)val.size();
    const int PENALTY = 100;
    struct Chrom { vector<bool> genes; long long fitness; long long weight; };
    vector<Chrom> pop(POP);
    mt19937 rng(random_device{}());
    ThreadPool<function<void()>> pool(thread::hardware_concurrency());
    for(auto &ch : pop) {
        ch.genes.assign(n, false);
        vector<int> idx(n);
        iota(idx.begin(), idx.end(), 0);
        shuffle(idx.begin(), idx.end(), rng);
        ch.weight = ch.fitness = 0;
        for(int i : idx) if(ch.weight + wt[i] <= W) {
            ch.genes[i] = true;
            ch.weight += wt[i];
            ch.fitness += val[i];
        }
    }
    auto calc = [&](Chrom &ch) {
        ch.weight = ch.fitness = 0;
        for(int i = 0; i < n; ++i) if(ch.genes[i]) {
            ch.weight += wt[i];
            ch.fitness += val[i];
        }
        if(ch.weight > W) ch.fitness -= PENALTY * (ch.weight - W);
        if(ch.fitness < 0) ch.fitness = 0;
    };
    auto t0 = high_resolution_clock::now();
    for(int gen = 0; gen < GENS; ++gen) {
        mutex fm; condition_variable fcv; int done = 0;
        for(int i = 0; i < POP; ++i) {
            pool.enqueue([&, i]() {
                calc(pop[i]);
                lock_guard<mutex> lk(fm);
                if(++done == POP) fcv.notify_one();
            });
        }
        unique_lock<mutex> lk(fm);
        fcv.wait(lk, [&]{ return done == POP; });
        sort(pop.begin(), pop.end(), [](auto &a, auto &b) { return a.fitness > b.fitness; });
        vector<Chrom> new_pop; new_pop.reserve(POP);
        new_pop.push_back(pop[0]); new_pop.push_back(pop[1]);
        uniform_int_distribution<int> elite_dist(0, min(9, POP-1));
        uniform_real_distribution<double> prob(0.0, 1.0);
        uniform_int_distribution<int> cut_dist(1, n-1);
        uniform_int_distribution<int> gene_dist(0, n-1);
        while((int)new_pop.size() < POP) {
            Chrom p1 = pop[elite_dist(rng)]; Chrom p2; int tries = 0;
            do { p2 = pop[elite_dist(rng)]; } while(p2.genes == p1.genes && ++tries < 5);
            Chrom c1 = p1, c2 = p2;
            if(prob(rng) < CROSS_RATE) {
                int cut = cut_dist(rng);
                for(int i = cut; i < n; ++i) swap(c1.genes[i], c2.genes[i]);
            }
            int mcount = uniform_int_distribution<int>(0, max(1, (int)(n * MUT_RATE)))(rng);
            for(int m = 0; m < mcount; ++m) c1.genes[gene_dist(rng)] = !c1.genes[gene_dist(rng)];
            for(int m = 0; m < mcount; ++m) c2.genes[gene_dist(rng)] = !c2.genes[gene_dist(rng)];
            calc(c1); calc(c2);
            new_pop.push_back(move(c1));
            if((int)new_pop.size() < POP) new_pop.push_back(move(c2));
        }
        pop.swap(new_pop);
    }
    mutex fm2; condition_variable fcv2; int done2 = 0;
    for(int i = 0; i < POP; ++i) {
        pool.enqueue([&, i]() {
            calc(pop[i]);
            lock_guard<mutex> lk(fm2);
            if(++done2 == POP) fcv2.notify_one();
        });
    }
    unique_lock<mutex> lk2(fm2);
    fcv2.wait(lk2, [&]{ return done2 == POP; });
    auto best = max_element(pop.begin(), pop.end(), [](auto &a, auto &b){ return a.fitness < b.fitness; });
    auto t1 = high_resolution_clock::now();
    return { duration<double,milli>(t1 - t0).count(), best->fitness };
}

// Parameter sweep for genetic algorithm, now reporting optimal parameters
template<typename Solver>
void parameterSweep(const string &mode, Solver solver, const vector<int> &val, const vector<int> &wt, int W) {
    vector<int> pops    = {50, 100, 150, 200};
    vector<int> gens    = {100, 300, 500};
    vector<double> muts  = {0.005, 0.01, 0.02};
    vector<double> crosses = {0.7, 0.85, 0.95};

    cout << "Parameter sweep for mode: " << mode << "\n";
    cout << "POP,GENS,MUT_RATE,CROSS_RATE,Time(ms),BestValue\n";

    int bestP = 0, bestG = 0;
    double bestM = 0, bestC = 0;
    double bestVal = -1;
    double bestTime = 0;

    for (int P : pops) {
        for (int G : gens) {
            for (double m : muts) {
                for (double c : crosses) {
                    auto res = solver(val, wt, W, P, G, m, c);
                    double t = res.first;
                    double v = res.second;
                    cout << P << "," << G << "," << m << "," << c
                         << "," << t << "," << v << "\n";

                    if (v > bestVal) {
                        bestVal  = v;
                        bestTime = t;
                        bestP    = P;
                        bestG    = G;
                        bestM    = m;
                        bestC    = c;
                    }
                }
            }
        }
    }

    // Print the optimal set
    cout << "\nOptimal parameters:\n"
         << "  POP = "       << bestP    << "\n"
         << "  GENS = "      << bestG    << "\n"
         << "  MUT_RATE = "  << bestM    << "\n"
         << "  CROSS_RATE = "<< bestC    << "\n"
         << "Resulting best value = " << bestVal
         << " (Time = " << bestTime << " ms)\n";
}


int main(int argc,char* argv[]) {
    if(argc<3){ cerr<<"Usage: "<<argv[0]<<" <mode> <input>\n"; return 1;}
    string mode=argv[1], path=argv[2]; vector<int> val, wt;
    auto runFile = [&](const string &file){
        cout<<"File: "<<fs::path(file).filename()<<"\n";
        int W = processFile(file, val, wt);
        if(mode=="sweep-omp") {
            parameterSweep(mode, solve_genetic_omp<>, val, wt, W);
        } else if(mode=="sweep-tp") {
            parameterSweep(mode, solve_genetic_tp<>, val, wt, W);
        } else {
            auto res = (mode=="local-omp"? solve_local_omp(val,wt,W)
                      :mode=="genetic-omp"? solve_genetic_omp(val,wt,W)
                      :mode=="local-tp"? solve_local_tp(val,wt,W)
                      : solve_genetic_tp(val,wt,W));
            cout<<"Time: "<<res.first<<" ms\n";
            cout<<"Maximum value: "<<res.second<<"\n";
        }
    };
    if(fs::is_directory(path)){
        for(auto &e:fs::directory_iterator(path)) if(fs::is_regular_file(e)) runFile(e.path().string());
    } else runFile(path);
    return 0;
}
