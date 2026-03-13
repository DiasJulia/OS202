#include <vector>
#include <iostream>
#include <random>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"
#include <omp.h>
#include <chrono>
#include <mpi.h>
#include <memory>
#include <algorithm> 

void advance_time( const fractal_land& land, pheronome& phen,
                              const position_t& pos_nest, const position_t& pos_food,
                              std::vector<int>& ant_pos_x, std::vector<int>& ant_pos_y,
                              std::vector<ant::state>& ant_states, std::vector<std::size_t>& ant_seeds,
                              std::size_t& cpteur, double eps,
                              std::chrono::duration<double, std::milli>& evaporation_time,
                              std::chrono::duration<double, std::milli>& update_time )
{
    using clock = std::chrono::steady_clock;
    #pragma omp parallel for
    for ( std::size_t i = 0; i < ant_pos_x.size(); ++i ) {
        auto ant_choice = [&ant_seeds, i]() mutable { return rand_double( 0., 1., ant_seeds[i] ); };
        auto dir_choice = [&ant_seeds, i]() mutable { return rand_int32( 1, 4, ant_seeds[i] ); };
        double consumed_time = 0.;
        while ( consumed_time < 1. ) {
            int ind_pher = ( ant_states[i] == ant::loaded ? 1 : 0 );
            double choix = ant_choice();
            position_t old_pos_ant{ant_pos_x[i], ant_pos_y[i]};
            position_t new_pos_ant = old_pos_ant;
            double max_phen = std::max( {phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher],
                                         phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher],
                                         phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher],
                                         phen( new_pos_ant.x, new_pos_ant.y + 1 )[ind_pher]} );
            if ( ( choix > eps ) || ( max_phen <= 0. ) ) {
                do {
                    new_pos_ant = old_pos_ant;
                    int d = dir_choice();
                    if ( d == 1 ) new_pos_ant.x -= 1;
                    if ( d == 2 ) new_pos_ant.y -= 1;
                    if ( d == 3 ) new_pos_ant.x += 1;
                    if ( d == 4 ) new_pos_ant.y += 1;
                } while ( phen[new_pos_ant][ind_pher] == -1 );
            } else {
                if ( phen( new_pos_ant.x - 1, new_pos_ant.y )[ind_pher] == max_phen )
                    new_pos_ant.x -= 1;
                else if ( phen( new_pos_ant.x + 1, new_pos_ant.y )[ind_pher] == max_phen )
                    new_pos_ant.x += 1;
                else if ( phen( new_pos_ant.x, new_pos_ant.y - 1 )[ind_pher] == max_phen )
                    new_pos_ant.y -= 1;
                else
                    new_pos_ant.y += 1;
            }
            consumed_time += land( new_pos_ant.x, new_pos_ant.y );
            
            #pragma omp critical
            {
                phen.mark_pheronome( new_pos_ant );
            }
            
            ant_pos_x[i] = new_pos_ant.x;
            ant_pos_y[i] = new_pos_ant.y;
            if ( new_pos_ant == pos_nest ) {
                if ( ant_states[i] == ant::loaded )
                    #pragma omp atomic
                    cpteur += 1;
                ant_states[i] = ant::unloaded;
            }
            if ( new_pos_ant == pos_food )
                ant_states[i] = ant::loaded;
        }
    }

    const auto evaporation_start = clock::now();
    phen.do_evaporation();
    evaporation_time += clock::now() - evaporation_start;

    const auto update_start = clock::now();
    phen.update();
    update_time += clock::now() - update_start;
}

void sync_pheromones( pheronome& phen, MPI_Comm comm,
                      std::chrono::duration<double, std::milli>& sync_time,
                      long long& sync_bytes )
{
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    int total_size = phen.map_data_count();
    std::vector<double> global_pheromones(total_size);

    MPI_Allreduce(phen.map_data(), global_pheromones.data(), total_size, MPI_DOUBLE, MPI_MAX, comm);

    std::copy(global_pheromones.begin(), global_pheromones.end(), phen.map_data());
    phen.restore_borders();

    sync_bytes += static_cast<long long>(total_size) * 8LL * 2;
    sync_time  += clock::now() - t0;
}

int main(int nargs, char* argv[])
{
    MPI_Init(&nargs, &argv);
    int rank = 0;
    int world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (rank == 0)
        SDL_Init( SDL_INIT_VIDEO );
    std::size_t seed = 2026; 
    const int nb_ants = 5000; 
    const double eps = 0.8;  
    const double alpha=0.7; 
    const double beta=0.999; 
    position_t pos_nest{256,256};
    position_t pos_food{500,500};
    fractal_land land(8,2,1.,1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            land(i,j) = (land(i,j)-min_val)/delta;
        }
    ant::set_exploration_coef(eps);

    std::vector<ant> ants;
    const std::size_t ants_begin = static_cast<std::size_t>(rank) * static_cast<std::size_t>(nb_ants) / static_cast<std::size_t>(world_size);
    const std::size_t ants_end = static_cast<std::size_t>(rank + 1) * static_cast<std::size_t>(nb_ants) / static_cast<std::size_t>(world_size);
    const std::size_t local_nb_ants = ants_end - ants_begin;
    ants.reserve(local_nb_ants);

    std::vector<int> ant_pos_x, ant_pos_y;
    std::vector<ant::state> ant_states;
    std::vector<std::size_t> ant_seeds;

    ant_pos_x.resize(local_nb_ants);
    ant_pos_y.resize(local_nb_ants);
    ant_states.assign(local_nb_ants, ant::unloaded);
    ant_seeds.resize(local_nb_ants);

    auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, land.dimensions()-1, seed); };

    for ( std::size_t local_i = 0; local_i < local_nb_ants; ++local_i ) {
        ant_pos_x[local_i] = static_cast<int>(gen_ant_pos());
        ant_pos_y[local_i] = static_cast<int>(gen_ant_pos());
        ant_seeds[local_i] = seed + ants_begin + local_i;
        ants.emplace_back(position_t{ant_pos_x[local_i], ant_pos_y[local_i]}, ant_seeds[local_i]);
    }

    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    std::unique_ptr<Window> win;
    std::unique_ptr<Renderer> renderer;
    if (rank == 0) {
        win = std::make_unique<Window>("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
        renderer = std::make_unique<Renderer>(land, phen, pos_nest, pos_food, ants);
    }
    
    std::size_t food_quantity_local = 0;
    std::size_t food_quantity_global = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;

    int local_n = static_cast<int>(local_nb_ants);
    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size);
    MPI_Gather(&local_n, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> all_ant_x;
    std::vector<int> all_ant_y;
    if (rank == 0) {
        all_ant_x.resize(nb_ants);
        all_ant_y.resize(nb_ants);
        displs[0] = 0;
        for (int i = 1; i < world_size; ++i) {
            displs[i] = displs[i - 1] + counts[i - 1];
        }
    }

    using clock = std::chrono::steady_clock;
    std::chrono::duration<double, std::milli> total_advance_ms{0};
    std::chrono::duration<double, std::milli> total_evaporation_ms{0};
    std::chrono::duration<double, std::milli> total_update_ms{0};
    std::chrono::duration<double, std::milli> total_display_ms{0};
    std::chrono::duration<double, std::milli> total_iter_ms{0};
    std::chrono::duration<double, std::milli> total_comm_ms{0};
    std::chrono::duration<double, std::milli> total_sync_ms{0};
    long long total_sync_bytes = 0;
    const auto max_simulation_time = std::chrono::seconds(120);
    const auto simulation_start = clock::now();

    while (cont_loop) {
        const auto iter_start = clock::now();
        ++it;
        if (rank == 0) {
            while (SDL_PollEvent(&event)) {
                if (event.type == SDL_QUIT)
                    cont_loop = false;
            }
        }

        const bool local_time_ok = (clock::now() - simulation_start) < max_simulation_time;
        int local_continue = (cont_loop && local_time_ok) ? 1 : 0;
        int global_continue = 0;
        {
            const auto comm_start = clock::now();
            MPI_Allreduce(&local_continue, &global_continue, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            total_comm_ms += clock::now() - comm_start;
        }
        if (!global_continue) {
            if (rank == 0 && !local_time_ok) {
                std::cout << "Simulation stopped: max execution time reached ("
                          << max_simulation_time.count() << " s)." << std::endl;
            }
            break;
        }

        const auto advance_start = clock::now();
        advance_time( land, phen, pos_nest, pos_food,
                                     ant_pos_x, ant_pos_y, ant_states, ant_seeds, food_quantity_local, eps,
                                     total_evaporation_ms, total_update_ms );
        total_advance_ms += clock::now() - advance_start;

        sync_pheromones(phen, MPI_COMM_WORLD, total_sync_ms, total_sync_bytes);

        unsigned long long local_food_ull = static_cast<unsigned long long>(food_quantity_local);
        unsigned long long global_food_ull = 0;
        {
            const auto comm_start = clock::now();
            MPI_Allreduce(&local_food_ull, &global_food_ull, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
            total_comm_ms += clock::now() - comm_start;
        }
        food_quantity_global = static_cast<std::size_t>(global_food_ull);

        MPI_Gatherv(ant_pos_x.data(), local_n, MPI_INT,
                    all_ant_x.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);
        
        MPI_Gatherv(ant_pos_y.data(), local_n, MPI_INT,
                    all_ant_y.data(), counts.data(), displs.data(), MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            ants.clear();
            ants.reserve(nb_ants);
            for ( std::size_t i = 0; i < nb_ants; ++i ) {
                ants.emplace_back(position_t{all_ant_x[i], all_ant_y[i]}, 0);
            }

            const auto display_start = clock::now();
            renderer->display(*win, food_quantity_global);
            total_display_ms += clock::now() - display_start;
            win->blit();
        }

        if ( rank == 0 && not_food_in_nest && food_quantity_global > 0 ) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }
        total_iter_ms += clock::now() - iter_start;
    }

    double local_compute_avg = 0.0;
    if (it > 0) {
        local_compute_avg = (total_advance_ms.count() - total_evaporation_ms.count() - total_update_ms.count()) / it;
    }
    
    std::vector<double> all_compute_avg(world_size);
    MPI_Gather(&local_compute_avg, 1, MPI_DOUBLE, all_compute_avg.data(), 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int local_threads = omp_get_max_threads();
    std::vector<int> all_threads(world_size);
    MPI_Gather(&local_threads, 1, MPI_INT, all_threads.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && it > 0) {
        std::cout << "\n--- Détails par Processus (MPI Rank) ---\n";
        std::cout << "Rank | Threads OMP | Charge (Fourmis) | Temps Calcul/iter (ms)\n";
        std::cout << "--------------------------------------------------------------\n";
        for(int i = 0; i < world_size; ++i) {
            std::cout << i << "    | " 
                      << all_threads[i] << "           | " 
                      << counts[i] << "              | " 
                      << all_compute_avg[i] << "\n";
        }
        std::cout << "--------------------------------------------------------------\n\n";

        const double avg_compute = total_advance_ms.count() / it
                                   - total_evaporation_ms.count() / it
                                   - total_update_ms.count() / it;
        const double avg_comm    = total_comm_ms.count() / it;
        const double avg_sync    = total_sync_ms.count() / it;
        const double total_comm_all = avg_comm + avg_sync;
        const double granularity = (total_comm_all > 0.0) ? (avg_compute / total_comm_all) : -1.0;
        const double sync_mb_per_call = total_sync_bytes / static_cast<double>(it) / 1e6;
        std::cout << "Benchmark (ms)\n"
        << "  MPI processes:            " << world_size << "\n"
        << "  avg advance_time (ms):    " << (total_advance_ms.count() / it) << "\n"
        << "    avg ant compute (ms):   " << avg_compute << "\n"
        << "    avg evaporation (ms):   " << (total_evaporation_ms.count() / it) << "\n"
        << "    avg pheromone upd (ms): " << (total_update_ms.count() / it) << "\n"
        << "  avg pheromone sync (ms):  " << avg_sync << "\n"
        << "    sync data/call (MB):    " << sync_mb_per_call << "\n"
        << "  avg other MPI comm (ms):  " << avg_comm << "\n"
        << "  compute/comm ratio:       " << granularity << "\n"
        << "  avg display (ms):         " << (total_display_ms.count() / it) << "\n"
        << "  avg iteration (ms):       " << (total_iter_ms.count() / it) << "\n"
        << "  total iterations:         " << it << std::endl;
    }

    if (rank == 0)
        SDL_Quit();
    MPI_Finalize();
    return 0;
}