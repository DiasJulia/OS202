#ifndef _PHERONOME_HPP_
#define _PHERONOME_HPP_
#include <algorithm>
#include <array>
#include <cassert>
#include <iostream>
#include <utility>
#include <vector>
#include "basic_types.hpp"

/**
 * @brief Carte des phéronomes
 * @details Gère une carte des phéronomes avec leurs mis à jour ( dont l'évaporation )
 *
 */
class pheronome {
public:
    using size_t      = unsigned long;
    using pheronome_t = std::array< double, 2 >;

    /**
     * @brief Construit une carte initiale des phéronomes
     * @details La carte des phéronomes est initialisées à zéro ( neutre )
     *          sauf pour les bords qui sont marqués comme indésirables
     *
     * @param dim Nombre de cellule dans chaque direction
     * @param alpha Paramètre de bruit
     * @param beta Paramêtre d'évaporation
     */
    pheronome( size_t dim, const position_t& pos_food, const position_t& pos_nest,
               double alpha = 0.7, double beta = 0.9999 )
        : m_dim( dim ),
          m_stride( dim + 2 ),
          m_alpha(alpha), m_beta(beta),
          m_map_of_pheronome( m_stride * m_stride, {{0., 0.}} ),
          m_buffer_pheronome( ),
          m_pos_nest( pos_nest ),
          m_pos_food( pos_food ) 
          {
        m_map_of_pheronome[index(pos_food)][0] = 1.;
        m_map_of_pheronome[index(pos_nest)][1] = 1.;
        cl_update( );
        m_buffer_pheronome = m_map_of_pheronome;
    }
    pheronome( const pheronome& ) = delete;
    pheronome( pheronome&& )      = delete;
    ~pheronome( )                 = default;

    pheronome_t& operator( )( size_t i, size_t j ) {
        return m_map_of_pheronome[( i + 1 ) * m_stride + ( j + 1 )];
    }

    const pheronome_t& operator( )( size_t i, size_t j ) const {
        return m_map_of_pheronome[( i + 1 ) * m_stride + ( j + 1 )];
    }

    pheronome_t& operator[] ( const position_t& pos ) {
      return m_map_of_pheronome[index(pos)];
    }

    const pheronome_t& operator[] ( const position_t& pos ) const {
      return m_map_of_pheronome[index(pos)];
    }

    void do_evaporation( ) {
        for ( std::size_t i = 1; i <= m_dim; ++i )
            for ( std::size_t j = 1; j <= m_dim; ++j ) {
                m_buffer_pheronome[i * m_stride + j][0] *= m_beta;
                m_buffer_pheronome[i * m_stride + j][1] *= m_beta;
            }
    }

    void mark_pheronome( const position_t& pos ) {
      std::size_t i = pos.x;
      std::size_t j = pos.y;
        assert( i >= 0 );
        assert( j >= 0 );
        assert( i < m_dim );
        assert( j < m_dim );
        pheronome&         phen        = *this;
        const pheronome_t& left_cell   = phen( i - 1, j );
        const pheronome_t& right_cell  = phen( i + 1, j );
        const pheronome_t& upper_cell  = phen( i, j - 1 );
        const pheronome_t& bottom_cell = phen( i, j + 1 );
        double             v1_left     = std::max( left_cell[0], 0. );
        double             v2_left     = std::max( left_cell[1], 0. );
        double             v1_right    = std::max( right_cell[0], 0. );
        double             v2_right    = std::max( right_cell[1], 0. );
        double             v1_upper    = std::max( upper_cell[0], 0. );
        double             v2_upper    = std::max( upper_cell[1], 0. );
        double             v1_bottom   = std::max( bottom_cell[0], 0. );
        double             v2_bottom   = std::max( bottom_cell[1], 0. );
        m_buffer_pheronome[( i + 1 ) * m_stride + ( j + 1 )][0] =
            m_alpha * std::max( {v1_left, v1_right, v1_upper, v1_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v1_left + v1_right + v1_upper + v1_bottom );
        m_buffer_pheronome[( i + 1 ) * m_stride + ( j + 1 )][1] =
            m_alpha * std::max( {v2_left, v2_right, v2_upper, v2_bottom} ) +
            ( 1 - m_alpha ) * 0.25 * ( v2_left + v2_right + v2_upper + v2_bottom );
    }

    void update( ) {
        m_map_of_pheronome.swap( m_buffer_pheronome );
        cl_update( );
        m_map_of_pheronome[( m_pos_food.x + 1 ) * m_stride + m_pos_food.y + 1][0] = 1;
        m_map_of_pheronome[( m_pos_nest.x + 1 ) * m_stride + m_pos_nest.y + 1][1] = 1;
    }

    // Acesso ao buffer interno para sincronização MPI (ex.: sync_pheromones)
    double* map_data() { return reinterpret_cast<double*>(m_map_of_pheronome.data()); }
    int     map_data_count() const { return static_cast<int>(m_map_of_pheronome.size() * 2); }
    void    restore_borders() { cl_update(); }

private:
    size_t index( const position_t& pos ) const
    {
      return (pos.x+1)*m_stride + pos.y + 1;
    }
    /**
     * @brief Mets à jour les conditions limites sur les cellules fantômes
     * @details Mets à jour les conditions limites sur les cellules fantômes :
     *     pour l'instant, on se contente simplement de mettre ces cellules avec
     *     des valeurs à -1 pour être sûr que les fourmis évitent ces cellules
     */
    void cl_update( ) {
        // On mets tous les bords à -1 pour les marquer comme indésirables :
        for ( unsigned long j = 0; j < m_stride; ++j ) {
            m_map_of_pheronome[j]                            = {{-1., -1.}};
            m_map_of_pheronome[j + m_stride * ( m_dim + 1 )] = {{-1., -1.}};
            m_map_of_pheronome[j * m_stride]                 = {{-1., -1.}};
            m_map_of_pheronome[j * m_stride + m_dim + 1]     = {{-1., -1.}};
        }
    }
    unsigned long              m_dim, m_stride;
    double                     m_alpha, m_beta;
    std::vector< pheronome_t > m_map_of_pheronome, m_buffer_pheronome;
    position_t m_pos_nest, m_pos_food;
};

#endif