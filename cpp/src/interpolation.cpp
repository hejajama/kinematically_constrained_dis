#include "interpolation.hpp"

#include <array>
#include <cmath>
#include <stdexcept>
#include <string>

namespace dis {

namespace {

// Same test as np.allclose(steps, steps[0], rtol=1e-6, atol=0) in BK_dipole.py.
bool all_close_to_first(const std::vector<double> &steps) {
    for (double s : steps)
        if (std::abs(s - steps[0]) > 1e-6 * std::abs(steps[0])) return false;
    return true;
}

// Coefficients of the cubic p(t) = sum_k c_k t^k on [0, h] with p(0) = f0,
// p(h) = f1, p'(0) = m0, p'(h) = m1, as a matrix acting on (f0, f1, m0, m1).
std::array<std::array<double, 4>, 4> hermite_to_power(double h) {
    const double h2 = h * h, h3 = h2 * h;
    return {{
        {1, 0, 0, 0},
        {0, 0, 1, 0},
        {-3 / h2, 3 / h2, -2 / h, -1 / h},
        {2 / h3, -2 / h3, 1 / h2, 1 / h2},
    }};
}

void require_grid(const std::vector<double> &nodes, std::size_t min_points, const char *what,
                  const std::string &filename) {
    if (nodes.size() < min_points)
        throw std::runtime_error(filename + ": need at least " + std::to_string(min_points) +
                                 " " + what + " points to interpolate, got " +
                                 std::to_string(nodes.size()));
    for (std::size_t i = 1; i < nodes.size(); ++i)
        if (!(nodes[i] > nodes[i - 1]))
            throw std::runtime_error(filename + ": " + what + " grid is not strictly increasing");
}

}  // namespace

// ---------------------------------------------------------------------------
// CellLocator
// ---------------------------------------------------------------------------

CellLocator::CellLocator(const std::vector<double> &nodes)
    : nodes_(nodes), last_(static_cast<long>(nodes.size()) - 2) {
    std::vector<double> steps, log_steps;
    for (std::size_t i = 1; i < nodes.size(); ++i) {
        steps.push_back(nodes[i] - nodes[i - 1]);
        if (nodes[0] > 0) log_steps.push_back(std::log(nodes[i]) - std::log(nodes[i - 1]));
    }
    if (all_close_to_first(steps)) {
        kind_ = Kind::uniform;
        origin_ = nodes[0];
        inv_step_ = 1 / steps[0];
    } else if (!log_steps.empty() && all_close_to_first(log_steps)) {
        kind_ = Kind::geometric;
        origin_ = std::log(nodes[0]);
        inv_step_ = 1 / log_steps[0];
    } else {
        kind_ = Kind::general;
    }
}

// ---------------------------------------------------------------------------
// Not-a-knot cubic spline slopes
// ---------------------------------------------------------------------------

std::vector<double> not_a_knot_slopes(const std::vector<double> &x, const std::vector<double> &y) {
    const std::size_t n = x.size();
    if (n < 4 || y.size() != n)
        throw std::invalid_argument("not_a_knot_slopes needs at least 4 points");

    std::vector<double> dx(n - 1), slope(n - 1);
    for (std::size_t i = 0; i + 1 < n; ++i) {
        dx[i] = x[i + 1] - x[i];
        slope[i] = (y[i + 1] - y[i]) / dx[i];
    }

    // Tridiagonal system lower[i] m[i-1] + diag[i] m[i] + upper[i] m[i+1] = rhs[i],
    // set up exactly as scipy.interpolate.CubicSpline does for not-a-knot ends.
    std::vector<double> lower(n, 0), diag(n), upper(n, 0), rhs(n);
    for (std::size_t i = 1; i + 1 < n; ++i) {
        lower[i] = dx[i];
        diag[i] = 2 * (dx[i - 1] + dx[i]);
        upper[i] = dx[i - 1];
        rhs[i] = 3 * (dx[i] * slope[i - 1] + dx[i - 1] * slope[i]);
    }
    {
        const double d = x[2] - x[0];
        diag[0] = dx[1];
        upper[0] = d;
        rhs[0] = ((dx[0] + 2 * d) * dx[1] * slope[0] + dx[0] * dx[0] * slope[1]) / d;
    }
    {
        const double d = x[n - 1] - x[n - 3];
        lower[n - 1] = d;
        diag[n - 1] = dx[n - 3];
        rhs[n - 1] = (dx[n - 2] * dx[n - 2] * slope[n - 3] +
                      (2 * d + dx[n - 2]) * dx[n - 3] * slope[n - 2]) / d;
    }

    // Thomas algorithm. The interior rows are diagonally dominant and
    // eliminating the first row into the second leaves a positive pivot, so
    // no pivoting is needed.
    for (std::size_t i = 1; i < n; ++i) {
        const double w = lower[i] / diag[i - 1];
        diag[i] -= w * upper[i - 1];
        rhs[i] -= w * rhs[i - 1];
    }
    std::vector<double> m(n);
    m[n - 1] = rhs[n - 1] / diag[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) m[i] = (rhs[i] - upper[i] * m[i + 1]) / diag[i];
    return m;
}

// ---------------------------------------------------------------------------
// BicubicSpline
// ---------------------------------------------------------------------------

BicubicSpline::BicubicSpline(const RcbkTable &table)
    : r_nodes_(table.r_grid), Y_nodes_(table.Y_grid) {
    // The tensor-product not-a-knot spline needs 4 nodes per axis. (scipy
    // lowers the degree for shorter axes; rcbk files are never that short.)
    require_grid(r_nodes_, 4, "r", table.filename);
    require_grid(Y_nodes_, 4, "Y", table.filename);

    const std::size_t n_r = r_nodes_.size(), n_Y = Y_nodes_.size();
    locate_r_ = CellLocator(r_nodes_);
    locate_Y_ = CellLocator(Y_nodes_);
    n_r_cells_ = n_r - 1;

    // Hermite data of the tensor-product spline at every node. Restricted to a
    // grid line the spline is the 1D spline of the data on that line, so
    //   f_r  = slopes along r of each Y row,
    //   f_Y  = slopes along Y of each r column,
    //   f_rY = slopes along Y of f_r.
    const auto idx = [n_r](std::size_t iY, std::size_t ir) { return iY * n_r + ir; };
    std::vector<double> f(table.N_grid), f_r(n_Y * n_r), f_Y(n_Y * n_r), f_rY(n_Y * n_r);

    std::vector<double> line(n_r);
    for (std::size_t iY = 0; iY < n_Y; ++iY) {
        for (std::size_t ir = 0; ir < n_r; ++ir) line[ir] = f[idx(iY, ir)];
        const auto m = not_a_knot_slopes(r_nodes_, line);
        for (std::size_t ir = 0; ir < n_r; ++ir) f_r[idx(iY, ir)] = m[ir];
    }
    std::vector<double> column(n_Y), column_r(n_Y);
    for (std::size_t ir = 0; ir < n_r; ++ir) {
        for (std::size_t iY = 0; iY < n_Y; ++iY) {
            column[iY] = f[idx(iY, ir)];
            column_r[iY] = f_r[idx(iY, ir)];
        }
        const auto mY = not_a_knot_slopes(Y_nodes_, column);
        const auto mrY = not_a_knot_slopes(Y_nodes_, column_r);
        for (std::size_t iY = 0; iY < n_Y; ++iY) {
            f_Y[idx(iY, ir)] = mY[iY];
            f_rY[idx(iY, ir)] = mrY[iY];
        }
    }

    // Per-cell power-form coefficients: C = H_Y G H_r^T, where G holds the
    // Hermite data of the cell's corners. G[p][q]: p = value at Y0, value at
    // Y1, d/dY at Y0, d/dY at Y1; q the same for r.
    coeffs_.resize(16 * (n_Y - 1) * n_r_cells_);
    for (std::size_t iY = 0; iY + 1 < n_Y; ++iY) {
        const auto HY = hermite_to_power(Y_nodes_[iY + 1] - Y_nodes_[iY]);
        for (std::size_t ir = 0; ir + 1 < n_r; ++ir) {
            const auto Hr = hermite_to_power(r_nodes_[ir + 1] - r_nodes_[ir]);

            double G[4][4];
            for (int p = 0; p < 4; ++p) {
                for (int q = 0; q < 4; ++q) {
                    const std::vector<double> &source =
                        p < 2 ? (q < 2 ? f : f_r) : (q < 2 ? f_Y : f_rY);
                    G[p][q] = source[idx(iY + p % 2, ir + q % 2)];
                }
            }
            double HG[4][4];
            for (int a = 0; a < 4; ++a)
                for (int q = 0; q < 4; ++q) {
                    HG[a][q] = 0;
                    for (int p = 0; p < 4; ++p) HG[a][q] += HY[a][p] * G[p][q];
                }
            double *c = coeffs_.data() + 16 * (iY * n_r_cells_ + ir);
            for (int a = 0; a < 4; ++a)
                for (int b = 0; b < 4; ++b) {
                    double sum = 0;
                    for (int q = 0; q < 4; ++q) sum += HG[a][q] * Hr[b][q];
                    c[4 * a + b] = sum;
                }
        }
    }
}

// ---------------------------------------------------------------------------
// Bilinear
// ---------------------------------------------------------------------------

Bilinear::Bilinear(const RcbkTable &table)
    : r_nodes_(table.r_grid), Y_nodes_(table.Y_grid), N_(table.N_grid) {
    require_grid(r_nodes_, 2, "r", table.filename);
    require_grid(Y_nodes_, 2, "Y", table.filename);
    locate_r_ = CellLocator(r_nodes_);
    locate_Y_ = CellLocator(Y_nodes_);
}

}  // namespace dis
