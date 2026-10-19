// Tabulated BK-evolved dipole amplitude N(r, Y), as written by the rcbk solver
// (https://github.com/hejajama/rcbkdipole).
//
// This is only the data on the grid. Interpolating it is the job of a separate
// interpolator class (see interpolation.hpp), so the two can be changed
// independently.
//
// File format: a comment header of '#' lines is followed by four '###' lines
// giving MinR, RMultiplier, RPoints and X0, and then one block per rapidity:
// a '### Y' line followed by RPoints values of N on the geometric grid
// r_i = MinR * RMultiplier^i. Parsing follows BKDipole._parse in
// building_blocks/correlators/Dipoles/BK_dipole.py.

#ifndef DIS_RCBK_TABLE_HPP
#define DIS_RCBK_TABLE_HPP

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace dis {

struct RcbkTable {
    std::string filename;

    // Grid definition from the '###' lines.
    double MinR = 0;
    double RMultiplier = 0;
    double x0 = 0;  // from the fourth '###' line, which wins over the header

    // r_grid[i] = MinR * RMultiplier^i, strictly increasing.
    std::vector<double> r_grid;
    // Rapidities of the blocks, sorted increasingly.
    std::vector<double> Y_grid;
    // N_grid[iY * r_grid.size() + ir] = N(r_grid[ir], Y_grid[iY]).
    std::vector<double> N_grid;

    // Initial-condition parameters from the comment header, when present.
    std::optional<double> Qs0_sq, gamma, ec, LambdaQCD;

    std::size_t n_r() const { return r_grid.size(); }
    std::size_t n_Y() const { return Y_grid.size(); }
    double N(std::size_t iY, std::size_t ir) const { return N_grid[iY * n_r() + ir]; }

    double r_min() const { return r_grid.front(); }
    double r_max() const { return r_grid.back(); }
    double Y_min() const { return Y_grid.front(); }
    double Y_max() const { return Y_grid.back(); }

    // Reads an rcbk output file. Throws std::runtime_error on a malformed file.
    static RcbkTable read(const std::string &filename);
};

}  // namespace dis

#endif  // DIS_RCBK_TABLE_HPP
