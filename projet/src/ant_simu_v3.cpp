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

void advance_time( const fractal_land& land, pheronome& phen,
                              const position_t& pos_nest, const position_t& pos_food,
                              std::vector<int>& ant_pos_x, std::vector<int>& ant_pos_y,
                              std::vector<ant::state>& ant_states, std::vector<std::size_t>& ant_seeds,
                              std::size_t& cpteur, double eps )
{
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
    phen.do_evaporation();
    phen.update();
}

int main(int nargs, char* argv[])
{
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
    ants.reserve(nb_ants);

    // Tableaux pour la version vectorisée
    std::vector<int> ant_pos_x, ant_pos_y;
    std::vector<ant::state> ant_states;
    std::vector<std::size_t> ant_seeds;

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

    using clock = std::chrono::steady_clock;
    std::chrono::duration<double, std::milli> total_advance_ms{0};
    std::chrono::duration<double, std::milli> total_display_ms{0};
    std::chrono::duration<double, std::milli> total_iter_ms{0};
    const auto max_simulation_time = std::chrono::seconds(60);
    const auto simulation_start = clock::now();

    while (cont_loop) {
        if (clock::now() - simulation_start >= max_simulation_time) {
            std::cout << "Simulation stopped: max execution time reached ("
                      << max_simulation_time.count() << " s)." << std::endl;
            break;
        }

        const auto iter_start = clock::now();
        ++it;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT)
                cont_loop = false;
        }

        const auto advance_start = clock::now();
        advance_time( land, phen, pos_nest, pos_food,
                                     ant_pos_x, ant_pos_y, ant_states, ant_seeds, food_quantity, eps );
        ants.clear();
        ants.reserve(nb_ants);
        for ( std::size_t i = 0; i < ant_pos_x.size(); ++i )
            ants.emplace_back(position_t{ant_pos_x[i], ant_pos_y[i]}, ant_seeds[i]);
        
        total_advance_ms += clock::now() - advance_start;

        const auto display_start = clock::now();
        renderer.display( win, food_quantity );
        total_display_ms += clock::now() - display_start;
        win.blit();
        if ( not_food_in_nest && food_quantity > 0 ) {
            std::cout << "La première nourriture est arrivée au nid a l'iteration " << it << std::endl;
            not_food_in_nest = false;
        }
        total_iter_ms += clock::now() - iter_start;
        //SDL_Delay(10);
    }

    if (it > 0) {
        std::cout << "Benchmark (ms)\n" 
        << "avg advance_time (ms): " << (total_advance_ms.count() / it)
        << "\navg display (ms): " << (total_display_ms.count() / it)
        << "\navg iteration (ms): " << (total_iter_ms.count() / it)
        << "\ntotal amount of iterations: " << it << std::endl;
    }

    SDL_Quit();
    return 0;
}