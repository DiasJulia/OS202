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
            phen.mark_pheronome( new_pos_ant );
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

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int total_size = phen.map_data_count();

    std::vector<double> all_pheromones;
    if (rank == 0)
        all_pheromones.resize(static_cast<std::size_t>(total_size) * size);

    std::vector<int> sendcounts(size, total_size);
    std::vector<int> displs(size, 0);
    for (int i = 1; i < size; ++i)
        displs[i] = displs[i-1] + total_size;

    MPI_Gatherv(phen.map_data(), total_size, MPI_DOUBLE,
                rank == 0 ? all_pheromones.data() : nullptr,
                sendcounts.data(), displs.data(), MPI_DOUBLE, 0, comm);

    if (rank == 0) {
        double* local = phen.map_data();
        std::fill(local, local + total_size, 0.0);
        for (int proc = 0; proc < size; ++proc) {
            const double* proc_data = all_pheromones.data() + proc * total_size;
            for (int i = 0; i < total_size; ++i)
                local[i] += proc_data[i] / size;
        }
        phen.restore_borders();
    }

    MPI_Bcast(phen.map_data(), total_size, MPI_DOUBLE, 0, comm);

    // Gather: size*total_size*8 bytes + Bcast: total_size*8 bytes
    sync_bytes += static_cast<long long>(total_size) * 8LL * (size + 1);
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
    std::size_t seed = 2026; // Graine pour la génération aléatoire ( reproductible )
    const int nb_ants = 5000; // Nombre de fourmis
    const double eps = 0.8;  // Coefficient d'exploration
    const double alpha=0.7; // Coefficient de chaos
    //const double beta=0.9999; // Coefficient d'évaporation
    const double beta=0.999; // Coefficient d'évaporation
    // Location du nid
    position_t pos_nest{256,256};
    // Location de la nourriture
    position_t pos_food{500,500};
    //const int i_food = 500, j_food = 500;    
    // Génération du territoire 512 x 512 ( 2*(2^8) par direction )
    fractal_land land(8,2,1.,1024);
    double max_val = 0.0;
    double min_val = 0.0;
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j ) {
            max_val = std::max(max_val, land(i,j));
            min_val = std::min(min_val, land(i,j));
        }
    double delta = max_val - min_val;
    /* On redimensionne les valeurs de fractal_land de sorte que les valeurs
    soient comprises entre zéro et un */
    for ( fractal_land::dim_t i = 0; i < land.dimensions(); ++i )
        for ( fractal_land::dim_t j = 0; j < land.dimensions(); ++j )  {
            land(i,j) = (land(i,j)-min_val)/delta;
        }
    // Définition du coefficient d'exploration de toutes les fourmis.
    ant::set_exploration_coef(eps);

    // On va créer des fourmis un peu partout sur la carte :
    std::vector<ant> ants;
    const std::size_t ants_begin = static_cast<std::size_t>(rank) * static_cast<std::size_t>(nb_ants) / static_cast<std::size_t>(world_size);
    const std::size_t ants_end = static_cast<std::size_t>(rank + 1) * static_cast<std::size_t>(nb_ants) / static_cast<std::size_t>(world_size);
    const std::size_t local_nb_ants = ants_end - ants_begin;
    ants.reserve(local_nb_ants);

    // Tableaux pour la version vectorisée
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

    // On crée toutes les fourmis dans la fourmilière.
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    std::unique_ptr<Window> win;
    std::unique_ptr<Renderer> renderer;
    if (rank == 0) {
        win = std::make_unique<Window>("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
        renderer = std::make_unique<Renderer>(land, phen, pos_nest, pos_food, ants);
    }
    
    // Compteur de la quantité de nourriture apportée au nid par les fourmis
    std::size_t food_quantity_local = 0;
    std::size_t food_quantity_global = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;

    using clock = std::chrono::steady_clock;
    std::chrono::duration<double, std::milli> total_advance_ms{0};
    std::chrono::duration<double, std::milli> total_evaporation_ms{0};
    std::chrono::duration<double, std::milli> total_update_ms{0};
    std::chrono::duration<double, std::milli> total_display_ms{0};
    std::chrono::duration<double, std::milli> total_iter_ms{0};
    std::chrono::duration<double, std::milli> total_comm_ms{0}; // tempo total em MPI_Allreduce
    std::chrono::duration<double, std::milli> total_sync_ms{0};  // tempo em sync_pheromones
    long long total_sync_bytes = 0;                              // bytes transferidos em sync
    const auto max_simulation_time = std::chrono::seconds(60);
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

        ants.clear();
        ants.reserve(local_nb_ants);
        for ( std::size_t i = 0; i < ant_pos_x.size(); ++i )
            ants.emplace_back(position_t{ant_pos_x[i], ant_pos_y[i]}, ant_seeds[i]);

        if (rank == 0) {
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
        //SDL_Delay(10);
    }

    if (rank == 0 && it > 0) {
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
        << "  ants/process:             " << local_nb_ants << "\n"
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