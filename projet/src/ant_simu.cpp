#include <vector>
#include <iostream>
#include <random>
#include "fractal_land.hpp"
#include "ant.hpp"
#include "pheronome.hpp"
# include "renderer.hpp"
# include "window.hpp"
# include "rand_generator.hpp"
#include <chrono>

void advance_time( const fractal_land& land, pheronome& phen, 
                   const position_t& pos_nest, const position_t& pos_food,
                   std::vector<ant>& ants, std::size_t& cpteur,
                   std::chrono::duration<double, std::milli>& evaporation_time,
                   std::chrono::duration<double, std::milli>& update_time )
{
    using clock = std::chrono::steady_clock;
    for ( size_t i = 0; i < ants.size(); ++i )
        ants[i].advance(phen, land, pos_food, pos_nest, cpteur);

    const auto evaporation_start = clock::now();
    phen.do_evaporation();
    evaporation_time += clock::now() - evaporation_start;

    const auto update_start = clock::now();
    phen.update();
    update_time += clock::now() - update_start;
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
    auto gen_ant_pos = [&land, &seed] () { return rand_int32(0, land.dimensions()-1, seed); };
    for ( size_t i = 0; i < nb_ants; ++i )
        ants.emplace_back(position_t{gen_ant_pos(),gen_ant_pos()}, seed);
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
    std::chrono::duration<double, std::milli> total_evaporation_ms{0};
    std::chrono::duration<double, std::milli> total_update_ms{0};
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
        advance_time( land, phen, pos_nest, pos_food, ants, food_quantity,
                  total_evaporation_ms, total_update_ms );
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
        << "\navg evaporation (ms): " << (total_evaporation_ms.count() / it)
        << "\navg pheromone update (ms): " << (total_update_ms.count() / it)
        << "\navg display (ms): " << (total_display_ms.count() / it)
        << "\navg iteration (ms): " << (total_iter_ms.count() / it)
        << "\ntotal amount of iterations: " << it << std::endl;
    }

    SDL_Quit();
    return 0;
}