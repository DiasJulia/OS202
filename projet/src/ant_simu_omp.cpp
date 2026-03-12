#include <vector>
#include <iostream>
#include <random>
#include <algorithm>
#include <string>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"
#include <chrono>
#include <omp.h>

void advance_time( const fractal_land& land, pheronome& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur )
{
    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);
    phen.do_evaporation();
    phen.update();
}

void advance_time_vectorized( const fractal_land& land, pheronome& phen,
                              const position_t& pos_nest, const position_t& pos_food,
                              std::vector<int>& ant_pos_x, std::vector<int>& ant_pos_y,
                              std::vector<ant::state>& ant_states, std::vector<std::size_t>& ant_seeds,
                              std::size_t& cpteur, double eps, bool parallel = false )
{
    const std::size_t dim = land.dimensions();
    const int nthreads = parallel ? omp_get_max_threads() : 1;
    std::vector<std::vector<unsigned char>> local_touched(
        nthreads, std::vector<unsigned char>(dim * dim, 0));

    #pragma omp parallel if(parallel) reduction(+:cpteur)
    {
        const int tid = omp_get_thread_num();
        auto& touched = local_touched[tid];

        #pragma omp for
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
                touched[static_cast<std::size_t>(new_pos_ant.x) + static_cast<std::size_t>(new_pos_ant.y) * dim] = 1;
                ant_pos_x[i] = new_pos_ant.x;
                ant_pos_y[i] = new_pos_ant.y;
                if ( new_pos_ant == pos_nest ) {
                    if ( ant_states[i] == ant::loaded )
                        cpteur += 1;
                    ant_states[i] = ant::unloaded;
                }
                if ( new_pos_ant == pos_food )
                    ant_states[i] = ant::loaded;
            }
        }
    }

    for ( std::size_t idx = 0; idx < dim * dim; ++idx ) {
        for ( int t = 0; t < nthreads; ++t ) {
            if ( local_touched[t][idx] ) {
                phen.mark_pheronome( {static_cast<int>(idx % dim), static_cast<int>(idx / dim)} );
                break;
            }
        }
    }

    phen.do_evaporation();
    phen.update();
}

int main(int nargs, char* argv[])
{
    SDL_Init( SDL_INIT_VIDEO );
    bool use_oop_mode = false;
    bool benchmark_mode = false;
    bool parallel_mode = false;
    std::size_t benchmark_iters = 5000;
    for ( int arg = 1; arg < nargs; ++arg ) {
        if ( std::string(argv[arg]) == "--oop" )
            use_oop_mode = true;
        else if ( std::string(argv[arg]) == "--benchmark" )
            benchmark_mode = true;
        else if ( std::string(argv[arg]) == "--parallel" )
            parallel_mode = true;
    }
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
    ants.reserve(nb_ants);

    // Tableaux pour la version vectorisée
    std::vector<int> ant_pos_x, ant_pos_y;
    std::vector<ant::state> ant_states;
    std::vector<std::size_t> ant_seeds;

    if ( use_oop_mode ) {
        std::cout << "Mode POO (--oop)" << std::endl;

        auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, land.dimensions()-1, seed); };
        for ( size_t i = 0; i < nb_ants; ++i )
            ants.emplace_back(position_t{gen_ant_pos(), gen_ant_pos()}, seed + static_cast<std::size_t>(i));
    } else {
        std::cout << "Mode vectorisé (defaut)" << std::endl;
        ant_pos_x.resize(nb_ants);
        ant_pos_y.resize(nb_ants);
        ant_states.assign(nb_ants, ant::unloaded);
        ant_seeds.resize(nb_ants);

        auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, land.dimensions()-1, seed); };

        for ( size_t i = 0; i < nb_ants; ++i ) {
            ant_pos_x[i] = static_cast<int>(gen_ant_pos());
            ant_pos_y[i] = static_cast<int>(gen_ant_pos());
            ant_seeds[i] = seed + static_cast<std::size_t>(i);
            ants.emplace_back(position_t{ant_pos_x[i], ant_pos_y[i]}, ant_seeds[i]);
        }
    }
    // On crée toutes les fourmis dans la fourmilière.
    pheronome phen(land.dimensions(), pos_food, pos_nest, alpha, beta);

    Window win("Ant Simulation", 2*land.dimensions()+10, land.dimensions()+266);
    Renderer renderer( land, phen, pos_nest, pos_food, ants );
    // Compteur de la quantité de nourriture apportée au nid par les fourmis
    size_t food_quantity = 0;
    SDL_Event event;
    bool cont_loop = true;
    bool not_food_in_nest = true;
    std::size_t it = 0;
    
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point start_display, start_advance, end_display, end_advance;
    auto avg_display_time = std::chrono::duration<double, std::milli>(0.);
    auto avg_advance_time = std::chrono::duration<double, std::milli>(0.);
    while (cont_loop) {
        ++it;
        if ( benchmark_mode && it > benchmark_iters )
                cont_loop = false;
        
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                cont_loop = false;
        }
        if ( benchmark_mode )
            start_advance = std::chrono::high_resolution_clock::now();
        if ( use_oop_mode ) {
            advance_time( land, phen, pos_nest, pos_food, ants, food_quantity );
        } else {
            advance_time_vectorized( land, phen, pos_nest, pos_food,
                                     ant_pos_x, ant_pos_y, ant_states, ant_seeds, food_quantity, eps, parallel_mode );
            ants.clear();
            ants.reserve(nb_ants);
            for ( std::size_t i = 0; i < ant_pos_x.size(); ++i )
                ants.emplace_back(position_t{ant_pos_x[i], ant_pos_y[i]}, ant_seeds[i]);
        }
        if ( benchmark_mode ) {
            end_advance = std::chrono::high_resolution_clock::now();
            avg_advance_time += end_advance - start_advance;
            
            start_display = std::chrono::high_resolution_clock::now();
        }
        renderer.display( win, food_quantity );
        win.blit();
        if ( benchmark_mode ) {
            end_display = std::chrono::high_resolution_clock::now();
            avg_display_time += end_display - start_display;
        }
        if ( not_food_in_nest && food_quantity > 0 ) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }
        //SDL_Delay(10);
    }

    if ( benchmark_mode ) {
        const auto elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - start ).count();
        std::cout << "[benchmark] mode=" << (use_oop_mode ? "oop" : "vectorized")
                  << " iters=" << benchmark_iters << "\n"
                  << " total_ms=" << elapsed << "\n"
                  << " ms_per_iter=" << (benchmark_iters == 0 ? 0. : elapsed / benchmark_iters) << "\n"
                  << " avg_advance_ms=" << (avg_advance_time.count() / benchmark_iters) << "\n"
                  << " avg_display_ms=" << (avg_display_time.count() / benchmark_iters) << "\n"
                  << " food=" << food_quantity << std::endl;
    } 
    SDL_Quit();
    return 0;
}