// Interpolators of the tabulated dipole amplitude N(r, Y).
//
// The interpolation is a template parameter of BKDipole (dipole.hpp), so a new
// scheme is added by writing a class with the interface below; nothing else
// in the code needs to change. main.cpp selects one at run time by name
// (--interpolation), which only requires adding it to the dispatch there.
//
// Interface of an interpolator class `I`:
//
//   explicit I(const RcbkTable &table);
//       Builds the interpolant from the grid data.
//
//   static constexpr const char *name;
//       Short name used on the command line.
//
//   class Slice { public: double operator()(double r) const; };
//   Slice at(double Y) const;
//       N(r, Y) at a fixed rapidity. at() is called once per integrand
//       evaluation (several dipoles share one Y), so put the Y-dependent work
//       there and keep operator() cheap. Slices must be cheap to copy and must
//       not outlive the interpolator.
//
// Both r and Y passed to an interpolator are already inside the grid,
// [r_grid.front(), r_grid.back()] x [Y_grid.front(), Y_grid.back()]. What
// happens outside it (clamping), and clipping N to [0, 1], is decided by
// BKDipole, not by the interpolator.

#ifndef DIS_INTERPOLATION_HPP
#define DIS_INTERPOLATION_HPP

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "rcbk_table.hpp"

namespace dis {

// Index i of the grid cell [nodes[i], nodes[i+1]) containing x, clipped to
// [0, nodes.size() - 2]. The rcbk grids are uniform in Y and geometric in r,
// so the index comes from one floor() (as _CellLocator in BK_dipole.py);
// other grids use a binary search.
//
// For a grid that is uniform only to round-off, floor() may put a query that
// sits on a node into the neighbouring cell. As in the Python code this is
// harmless for a continuous interpolant: the neighbouring cell's polynomial
// evaluated a round-off distance outside its cell gives the same value.
class CellLocator {
public:
    CellLocator() = default;
    explicit CellLocator(const std::vector<double> &nodes);

    std::size_t operator()(double x) const;

private:
    enum class Kind { uniform, geometric, general };
    Kind kind_ = Kind::general;
    double origin_ = 0;
    double inv_step_ = 0;
    std::vector<double> nodes_;
    long last_ = 0;
};

// Bicubic spline, the same function as scipy's RectBivariateSpline(kx=ky=3,
// s=0) used by BK_dipole.py: the unique C^2 tensor-product cubic interpolant
// with not-a-knot end conditions, linear in r (not log r).
//
// It is stored as one polynomial per grid cell, 16 coefficients of
// (Y - Y_iY)^a (r - r_ir)^b, built from Hermite data (values and r, Y and
// rY derivatives at the nodes) given by 1D not-a-knot splines along each axis.
class BicubicSpline {
public:
    static constexpr const char *name = "bicubic";

    explicit BicubicSpline(const RcbkTable &table);

    class Slice {
    public:
        double operator()(double r) const;

    private:
        friend class BicubicSpline;
        const BicubicSpline *spline_ = nullptr;
        const double *cells_ = nullptr;  // first cell of the Y row
        double dY_ = 0;
    };

    Slice at(double Y) const;

private:
    std::vector<double> r_nodes_, Y_nodes_;
    CellLocator locate_r_, locate_Y_;
    std::size_t n_r_cells_ = 0;
    std::vector<double> coeffs_;  // 16 per cell, cell index iY * n_r_cells_ + ir
};

// Bilinear interpolation in (r, Y). Not used by the Python code; it is here
// as a simple alternative to the spline, and as an example of the interface.
class Bilinear {
public:
    static constexpr const char *name = "bilinear";

    explicit Bilinear(const RcbkTable &table);

    class Slice {
    public:
        double operator()(double r) const;

    private:
        friend class Bilinear;
        const Bilinear *interp_ = nullptr;
        const double *row0_ = nullptr;  // N at Y_grid[iY]
        const double *row1_ = nullptr;  // N at Y_grid[iY + 1]
        double w_ = 0;                  // weight of row1_
    };

    Slice at(double Y) const;

private:
    std::vector<double> r_nodes_, Y_nodes_, N_;
    CellLocator locate_r_, locate_Y_;
};

// Slopes of the not-a-knot cubic spline through (x[i], y[i]), i.e. its first
// derivative at each node. Same end conditions as scipy's CubicSpline
// (bc_type="not-a-knot"). Needs at least 4 points. Exposed for testing.
std::vector<double> not_a_knot_slopes(const std::vector<double> &x, const std::vector<double> &y);

// ---------------------------------------------------------------------------
// Evaluation, inline because it is the innermost loop of every integrand.
// ---------------------------------------------------------------------------

inline std::size_t CellLocator::operator()(double x) const {
    long i;
    switch (kind_) {
        case Kind::uniform:
            i = static_cast<long>(std::floor((x - origin_) * inv_step_));
            break;
        case Kind::geometric:
            i = static_cast<long>(std::floor((std::log(x) - origin_) * inv_step_));
            break;
        default:
            i = static_cast<long>(std::upper_bound(nodes_.begin(), nodes_.end(), x) -
                                  nodes_.begin()) - 1;
    }
    return static_cast<std::size_t>(std::clamp(i, 0L, last_));
}

inline BicubicSpline::Slice BicubicSpline::at(double Y) const {
    const std::size_t iY = locate_Y_(Y);
    Slice s;
    s.spline_ = this;
    s.cells_ = coeffs_.data() + 16 * iY * n_r_cells_;
    s.dY_ = Y - Y_nodes_[iY];
    return s;
}

inline double BicubicSpline::Slice::operator()(double r) const {
    const std::size_t ir = spline_->locate_r_(r);
    const double dr = r - spline_->r_nodes_[ir];
    const double *c = cells_ + 16 * ir;  // c[4 * a + b] multiplies dY^a dr^b
    double rows[4];
    for (int a = 0; a < 4; ++a) {
        const double *ca = c + 4 * a;
        rows[a] = ((ca[3] * dr + ca[2]) * dr + ca[1]) * dr + ca[0];
    }
    return ((rows[3] * dY_ + rows[2]) * dY_ + rows[1]) * dY_ + rows[0];
}

inline Bilinear::Slice Bilinear::at(double Y) const {
    const std::size_t iY = locate_Y_(Y);
    const std::size_t n_r = r_nodes_.size();
    Slice s;
    s.interp_ = this;
    s.row0_ = N_.data() + iY * n_r;
    s.row1_ = s.row0_ + n_r;
    s.w_ = (Y - Y_nodes_[iY]) / (Y_nodes_[iY + 1] - Y_nodes_[iY]);
    return s;
}

inline double Bilinear::Slice::operator()(double r) const {
    const std::size_t ir = interp_->locate_r_(r);
    const double *rn = interp_->r_nodes_.data();
    const double t = (r - rn[ir]) / (rn[ir + 1] - rn[ir]);
    const double at_r0 = (1 - w_) * row0_[ir] + w_ * row1_[ir];
    const double at_r1 = (1 - w_) * row0_[ir + 1] + w_ * row1_[ir + 1];
    return (1 - t) * at_r0 + t * at_r1;
}

}  // namespace dis

#endif  // DIS_INTERPOLATION_HPP
