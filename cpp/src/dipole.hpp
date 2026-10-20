// Dipole S-matrix models.
//
// The quadrupole, the integrands and the cross sections are templates on the
// dipole model, which only has to provide
//
//   AtRapidity at(double Y) const;            // with
//   double AtRapidity::S(double r) const;     // S(r) at that rapidity
//
// at() is called once per integrand evaluation and S() for each of the
// dipoles at that point, so per-rapidity work belongs in at(). It must also
// provide
//
//   double Y_min() const, Y_max() const;
//   std::vector<double> Y_breakpoints() const;
//
// [Y_min, Y_max] is the range in which S depends on Y; outside it S is
// constant in Y. Between consecutive breakpoints S is a smooth function of Y;
// at a breakpoint its Y derivative may jump (e.g. where a tabulated solution
// is clamped). The FE 5D cross section uses these to restrict the M_qq^2
// range where dS/dY != 0 and to place its interpolation nodes.
//
// BKDipole wraps a tabulated rcbk solution and an interpolator (see
// interpolation.hpp for how to add one). LargeXExtrapolation extends a dipole
// to x > x0. GBWDipole is an analytic model used to test the quadrupole
// independently of any interpolation.

#ifndef DIS_DIPOLE_HPP
#define DIS_DIPOLE_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "interpolation.hpp"
#include "rcbk_table.hpp"

namespace dis {

struct DipoleOptions {
    // Clip the interpolated N to [0, 1] (BKDipole clamp_N in Python).
    bool clip_N = true;
    // Lower bound on S = 1 - N, so that log(S) in the quadrupole is finite.
    double S_floor = 1e-14;
};

// BK-evolved dipole N(r, Y) from an rcbk file, as BKDipole in BK_dipole.py
// with its default settings: a query outside the tabulated grid is evaluated
// at the nearest boundary (r and Y clamped separately), N is clipped to
// [0, 1], and S = max(1 - N, S_floor).
template <class Interpolator>
class BKDipole {
public:
    explicit BKDipole(RcbkTable table, DipoleOptions options = {})
        : table_(std::move(table)), interp_(table_), options_(options) {}

    explicit BKDipole(const std::string &filename, DipoleOptions options = {})
        : BKDipole(RcbkTable::read(filename), options) {}

    class AtRapidity {
    public:
        double N(double r) const {
            double value = slice_(std::clamp(r, r_min_, r_max_));
            if (clip_N_) value = std::clamp(value, 0.0, 1.0);
            return value;
        }
        double S(double r) const { return std::max(1.0 - N(r), S_floor_); }

    private:
        friend class BKDipole;
        typename Interpolator::Slice slice_;
        double r_min_, r_max_, S_floor_;
        bool clip_N_;
    };

    AtRapidity at(double Y) const {
        AtRapidity a;
        a.slice_ = interp_.at(std::clamp(Y, table_.Y_min(), table_.Y_max()));
        a.r_min_ = table_.r_min();
        a.r_max_ = table_.r_max();
        a.S_floor_ = options_.S_floor;
        a.clip_N_ = options_.clip_N;
        return a;
    }

    double N(double r, double Y) const { return at(Y).N(r); }
    double S(double r, double Y) const { return at(Y).S(r); }

    // Outside the tabulated rapidities Y is clamped, so S does not change.
    double Y_min() const { return table_.Y_min(); }
    double Y_max() const { return table_.Y_max(); }
    std::vector<double> Y_breakpoints() const { return {Y_min(), Y_max()}; }

    const RcbkTable &table() const { return table_; }
    const Interpolator &interpolator() const { return interp_; }

private:
    RcbkTable table_;
    Interpolator interp_;
    DipoleOptions options_;
};

// GBW dipole S(r) = exp(-r^2 Qs^2 / 4), independent of Y.
class GBWDipole {
public:
    explicit GBWDipole(double Qs2) : Qs2_(Qs2) {}

    class AtRapidity {
    public:
        double S(double r) const { return std::exp(-r * r * Qs2_ / 4); }

    private:
        friend class GBWDipole;
        double Qs2_ = 0;
    };

    AtRapidity at(double /*Y*/) const {
        AtRapidity a;
        a.Qs2_ = Qs2_;
        return a;
    }

    double Y_min() const { return -INFINITY; }
    double Y_max() const { return INFINITY; }
    std::vector<double> Y_breakpoints() const { return {}; }

private:
    double Qs2_;
};

// Extends a dipole to x > x0, i.e. to Y = ln(x0 / x) < 0, as
//
//   N(r, x) = N(r, x0) [(1 - x) / (1 - x0)]^power,
//
// which is continuous at x = x0 and makes the target transparent (N = 0,
// S = 1) at x = 1. For Y >= 0 it is the base dipole unchanged. The base dipole's AtRapidity must provide
// N(r) as well as S(r) (BKDipole does). x0 must be the x0 that defines Y in
// the cross sections (Kinematics::x0).
template <class Base>
class LargeXExtrapolation {
public:
    LargeXExtrapolation(const Base &base, double x0, double power, double S_floor = 1e-14)
        : base_(base), x0_(x0), power_(power), S_floor_(S_floor) {}

    class AtRapidity {
    public:
        double N(double r) const { return factor_ * at_.N(r); }
        double S(double r) const {
            return factor_ == 1 ? at_.S(r) : std::max(1.0 - N(r), S_floor_);
        }

    private:
        friend class LargeXExtrapolation;
        typename Base::AtRapidity at_;
        double factor_ = 1, S_floor_ = 0;
    };

    AtRapidity at(double Y) const {
        AtRapidity a;
        a.S_floor_ = S_floor_;
        if (Y >= 0) {
            a.at_ = base_.at(Y);
            return a;
        }
        const double x = x0_ * std::exp(-Y);
        a.at_ = base_.at(0.0);
        a.factor_ = x < 1 ? std::pow((1 - x) / (1 - x0_), power_) : 0.0;
        return a;
    }

    // S depends on Y down to x = 1; beyond, N = 0.
    double Y_min() const { return std::log(x0_); }
    double Y_max() const { return base_.Y_max(); }
    // Y = 0 joins the extrapolation to the base dipole (dS/dY jumps there).
    std::vector<double> Y_breakpoints() const {
        std::vector<double> b = {Y_min(), 0.0};
        for (double Y : base_.Y_breakpoints())
            if (Y > 0) b.push_back(Y);
        return b;
    }

    double x0() const { return x0_; }
    double power() const { return power_; }

private:
    const Base &base_;
    double x0_, power_, S_floor_;
};

}  // namespace dis

#endif  // DIS_DIPOLE_HPP
